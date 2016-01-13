-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.
require('xlua')
require('paths')
require('mobdebug').start()
local tds = require('tds')
paths.dofile('data.lua')
paths.dofile('model.lua')

--the train function
local function train(images, images_q)
    local N = math.ceil(images:size(1) / g_params.batchsize)
    local cost = 0
    local y = torch.ones(1)
    --define the tensors for query(input), memory(context), target(quantifier)
    local input = torch.CudaTensor(g_params.batchsize, g_params.vector_size)
    local target = torch.CudaTensor(g_params.batchsize)
    local context = torch.CudaTensor(g_params.batchsize, g_params.memsize, g_params.vector_size)
    -- structure the train dataset in batches and substitute each symbol with its associated vector
    for n = 1, N do
        if g_params.show then xlua.progress(n, N) end
        for b = 1, g_params.batchsize do
            local start = (n - 1) * g_params.batchsize
            input[b] = g_vectors[images_q[start + b][1]]
            target[b] = images_q[start + b][2]
            for i = 1, 16 do --change the parameter
              context[b][i] = g_vectors[images[start + b][i]]
            end
        end
        local x = {input, target, context}
        local out = g_model:forward(x)
        cost = cost + out[1]
        g_paramdx:zero()
        g_model:backward(x, y)
        local gn = g_paramdx:norm()
        if gn > g_params.maxgradnorm then
            g_paramdx:mul(g_params.maxgradnorm / gn)
        end
        g_paramx:add(g_paramdx:mul(-g_params.dt))
    end
    return cost/N/g_params.batchsize
end

--the test function
local function test(images, images_q)
    local N = math.ceil(images:size(1) / g_params.batchsize)
    local cost = 0
    --define the tensors for query(input), memory(context), target(quantifier index (from 1 to 3))
    local input = torch.CudaTensor(g_params.batchsize, g_params.vector_size)
    local target = torch.CudaTensor(g_params.batchsize)
    local context = torch.CudaTensor(g_params.batchsize, g_params.memsize, g_params.vector_size)
    -- structure the test dataset in batches and substitute each symbol with its associated vector
    for n = 1, N do
        if g_params.show then xlua.progress(n, N) end
        for b = 1, g_params.batchsize do
            local start = (n - 1) * g_params.batchsize
            input[b] = g_vectors[images_q[start + b][1]]
            target[b] = images_q[start + b][2]
            for i = 1, 16 do --change the parameter
              context[b][i] = g_vectors[images[start + b][i]]
            end
        end
        local x = {input, target, context}
        local out = g_model:forward(x)
        cost = cost + out[1]
    end
    return cost/N/g_params.batchsize
end

--the main function which runs the train, validate and test
local function run(epochs)
    for i = 1, epochs do
        local c, ct
        c = train(g_img_train, g_img_q_train)
        ct = test(g_img_valid, g_img_q_valid)

        -- Logging
        local m = #g_log_cost+1
        g_log_cost[m] = {m, c, ct}
        g_log_perp[m] = {m, math.exp(c), math.exp(ct)}
        local stat = {perplexity = math.exp(c) , epoch = m,
                valid_perplexity = math.exp(ct), LR = g_params.dt}
        if g_params.test then
            local ctt = test(g_img_test, g_img_q_test)
            table.insert(g_log_cost[m], ctt)
            table.insert(g_log_perp[m], math.exp(ctt))
            stat['test_perplexity'] = math.exp(ctt)
        end
        print(stat)

        -- Learning rate annealing
        if m > 1 and g_log_cost[m][3] > g_log_cost[m-1][3] * 0.9999 then
            g_params.dt = g_params.dt / 1.5
            if g_params.dt < 1e-5 then break end
        end
    end
end

local function save(path)
    local d = {}
    d.params = g_params
    d.paramx = g_paramx:float()
    d.log_cost = g_log_cost
    d.log_perp = g_log_perp
    torch.save(path, d)
end

--------------------------------------------------------------------
--------------------------------------------------------------------
-- model params:
local cmd = torch.CmdLine()
cmd:option('--gpu', 1, 'GPU id to use')
cmd:option('--init_std', 0.05, 'weight initialization std')
cmd:option('--sdt', 0.1, 'initial learning rate')
cmd:option('--maxgradnorm', 50, 'maximum gradient norm')
cmd:option('--memsize', 16, 'memory size')
cmd:option('--nhop', 1, 'number of hops')
cmd:option('--batchsize', 5)
cmd:option('--show', true, 'print progress')
cmd:option('--load', '', 'model file to load')
cmd:option('--save', '', 'path to save model')
cmd:option('--epochs', 100)
cmd:option('--test', true, 'enable testing')
cmd:option('--vector_size', 15, 'size of the vectors of the symbols')
cmd:option('--train_size', 3500)
cmd:option('--test_size', 1000)
cmd:option('--valid_size', 500)
g_params = cmd:parse(arg or {})

print(g_params)
cutorch.setDevice(g_params.gpu)

g_vocab =  tds.hash()
g_ivocab =  tds.hash()
g_ivocab[#g_vocab + 1] = 'some'
g_vocab['some'] = #g_vocab + 1
g_ivocab[#g_vocab + 1] = 'all'
g_vocab['all'] = #g_vocab + 1
g_ivocab[#g_vocab + 1] = 'no'
g_vocab['no'] = #g_vocab + 1
g_vectors = tds.hash()


g_img_train, g_img_q_train = g_read_images('data/quant.train.txt', 'data/vectors.txt', 
                            g_vocab, g_ivocab, g_vectors, g_params.train_size)
g_img_valid, g_img_q_valid = g_read_images('data/quant.valid.txt', 'data/vectors.txt', 
                            g_vocab, g_ivocab, g_vectors, g_params.valid_size)
g_img_test, g_img_q_test = g_read_images('data/quant.test.txt', 'data/vectors.txt', 
                            g_vocab, g_ivocab, g_vectors, g_params.test_size)
g_params.nwords = 3
print('vocabulary size ' .. #g_vocab)

g_model = g_build_model(g_params)
g_paramx, g_paramdx = g_model:getParameters()
g_paramx:normal(0, g_params.init_std)
if g_params.load ~= '' then
    local f = torch.load(g_params.load)
    g_paramx:copy(f.paramx)
end

g_log_cost = {}
g_log_perp = {}
g_params.dt = g_params.sdt

print('starting to run....')
run(g_params.epochs)

if g_params.save ~= '' then
    save(g_params.save)
end
