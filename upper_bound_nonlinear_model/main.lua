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
local file_log = require('pl.file')

local function prepare_output(images_ind, queries_ind, n, preds, max_inds)
  local start = (n - 1) * g_params.batchsize
  for b=1, g_params.batchsize do
    local image_ind = images_ind[start + b]
    local query_ind = queries_ind[start + b]
    local max_ind = max_inds[b]
    local pred = preds[b]
    local reverse_image = torch.CudaTensor(4,4)
    for i=1, 4 do
      for j=1, 4 do
        reverse_image[i][j] = image_ind[(i - 1) * 4 + j]
      end
    end
    local stat_log = {image = tostring(reverse_image), 
      query = tostring(query_ind), class_distribution = tostring(pred),
      prediction = tostring(max_ind)}
    print(stat_log)
--    file_log.write('error_analysis.txt', pred)  
  end
end

--the train function
local function train(images, images_q)
    local N = math.ceil(images:size(1) / g_params.batchsize)
    local cost = 0
    local y1 = torch.CudaTensor(1)
    local y2 = torch.CudaTensor(g_params.batchsize, g_params.nwords)
    y1:fill(1.0)
    y2:fill(1.0)
    local correct = torch.CudaTensor(g_params.nwords)
    --define the tensors for query(input), memory(context), target(quantifier)
    local input = torch.CudaTensor(g_params.batchsize, #g_vocab - 3)
    local target = torch.CudaTensor(g_params.batchsize)
    local context = torch.CudaTensor(g_params.batchsize, #g_vocab - 3)
    -- structure the train dataset in batches and substitute each symbol with its associated vector
    for n = 1, N do
        if g_params.show then xlua.progress(n, N) end
        for b = 1, g_params.batchsize do
            local start = (n - 1) * g_params.batchsize
            input[b]:fill(0.0)
            input[b][images_q[start + b][1] - 3] = 1.0
            context[b]:fill(0.0)
            target[b] = images_q[start + b][2]
            for i = 1, 16 do --change the parameter
              if images[start + b][i] >= 1.0 then
                context[b][images[start + b][i] - 3] = context[b][images[start + b][i] -3] + 1.0
              end
            end
        end
        local x = {input, target, context}
        local out = g_model:forward(x)
        cost = cost + out[1][1]
        g_paramdx:zero()
        g_model:backward(x, {y1, y2})
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
    local input = torch.CudaTensor(g_params.batchsize, #g_vocab - 3)
    local target = torch.CudaTensor(g_params.batchsize)
    local context = torch.CudaTensor(g_params.batchsize, #g_vocab - 3)
    local correct = 0
    local confusion_matrix = torch.CudaTensor(3, 3)
    confusion_matrix:fill(0.0)
    -- structure the test dataset in batches and substitute each symbol with its associated vector
    for n = 1, N do
        if g_params.show then xlua.progress(n, N) end
        for b = 1, g_params.batchsize do
            local start = (n - 1) * g_params.batchsize
            input[b]:fill(0.0)
            input[b][images_q[start + b][1] - 3] = 1.0
            context[b]:fill(0.0)
            target[b] = images_q[start + b][2]
            for i = 1, 16 do --change the parameter
              if images[start + b][i] >= 1.0 then
                context[b][images[start + b][i] - 3] = context[b][images[start + b][i] -3] + 1.0
              end
            end
        end
        local x = {input, target, context}
        local out = g_model:forward(x)
        cost = cost + out[1][1]
        local max_els = torch.CudaTensor(g_params.batchsize)
        local max_ind = torch.CudaTensor(g_params.batchsize)
        max_els, max_ind = torch.max(out[2], 2)
        --prepare_output(images, images_q, n, out[2], max_ind)
        for b = 1, g_params.batchsize do
            if target[b] == max_ind[b][1] then
                correct = correct + 1
            end
            confusion_matrix[target[b]][max_ind[b][1]] = confusion_matrix[target[b]][max_ind[b][1]] + 1
        end
    end
    return cost/N/g_params.batchsize, correct, confusion_matrix
end

--the main function which runs the train, validate and test
local function run(epochs)
    for i = 1, epochs do
        local c, ct
        local correct_valid = 0.0
        local correct_test = 0.0
        local conf_matrix_valid = torch.CudaTensor(3,3)
        local conf_matrix_test = torch.CudaTensor(3,3)
        conf_matrix_test:fill(0.0)
        conf_matrix_valid:fill(0.0)
        c = train(g_img_train, g_img_q_train)
        ct, correct_valid, conf_matrix_valid = test(g_img_valid, g_img_q_valid)
        -- perplexity- exponential compared with cost function
        -- Logging
        local m = #g_log_cost+1
        g_log_cost[m] = {m, c, ct}
        g_log_perp[m] = {m, math.exp(c), math.exp(ct)}
        --local stat = {perplexity = math.exp(c) , epoch = m,
        --        valid_perplexity = math.exp(ct), LR = g_params.dt}
        local stat = {epoch = m}
        if g_params.test then
            local ctt, correct_test, conf_matrix_test = test(g_img_test, g_img_q_test)
            table.insert(g_log_cost[m], ctt)
            table.insert(g_log_perp[m], math.exp(ctt))
            --stat['test_perplexity'] = math.exp(ctt)
            stat['test_precision'] = correct_test / g_params.test_size
            stat['valid_precision'] = correct_valid / g_params.valid_size
            stat['confusion_matrix_valid'] = tostring(conf_matrix_valid)
            stat['confusion_matrix_test'] = tostring(conf_matrix_test)
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
cmd:option('--batchsize', 10)
cmd:option('--show', false, 'print progress')
cmd:option('--load', '', 'model file to load')
cmd:option('--save', '', 'path to save model')
cmd:option('--epochs', 100)
cmd:option('--test', true, 'enable testing')
cmd:option('--vector_size', 20, 'size of the vectors of the symbols')
cmd:option('--hidden_size', 10, 'size of the hidden state')
cmd:option('--train_size', 3500)
cmd:option('--test_size', 1000)
cmd:option('--valid_size', 500)
cmd:option('--vector_file', 'data/vectors-20-threshold-055.txt')
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


g_img_train, g_img_q_train = g_read_images('data/quant.train.txt', 
                            g_vocab, g_ivocab, g_params.train_size)
g_img_valid, g_img_q_valid = g_read_images('data/quant.valid.txt', 
                            g_vocab, g_ivocab, g_params.valid_size)
g_img_test, g_img_q_test = g_read_images('data/quant.test.txt', 
                            g_vocab, g_ivocab, g_params.test_size)
g_params.nwords = 3
print('vocabulary size ' .. #g_vocab)
g_params.vocab_size = #g_vocab - 3
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
