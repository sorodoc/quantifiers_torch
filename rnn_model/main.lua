require('rnn')
require('nn')
require('cunn')
require('nngraph')
require('mobdebug').start()
local tds = require('tds')
paths.dofile('Peek.lua')
paths.dofile('data.lua')




-- build simple recurrent neural network

-- build dummy dataset (task is to predict class given rho words)
-- similar to sentiment analysis datasets

-- training
local function train(images, images_q)
    local N = math.ceil(images:size(1) / g_params.batchsize)
    local input = torch.DoubleTensor(g_params.batchsize, g_params.memsize + 1, g_params.vector_size)
    local target = torch.DoubleTensor(g_params.batchsize)
    local context = torch.DoubleTensor(g_params.batchsize, g_params.memsize, g_params.vector_size)
    local cost = 0
    for n = 1, N do
        if g_params.show then xlua.progress(n, N) end
        for b = 1, g_params.batchsize do
          local start = (n - 1) * g_params.batchsize
          input[b][1] = g_vectors[images_q[start + b][1]]
          target[b] = images_q[start + b][2]
          for i = 1, 16 do --change the parameter
            input[b][i + 1] = g_vectors[images[start + b][i]]
          end
        end
        rnn:zeroGradParameters() 
        local output = rnn:forward(input)
        local err = criterion:forward(output, target)
        cost = cost + err
        local gradOutputs = criterion:backward(output, target)
        local gradInputs = rnn:backward(input, gradOutputs)
        rnn:updateParameters(g_params.dt)
    end
    return cost/N/g_params.batchsize
end

local function test(images, images_q)
    local N = math.ceil(images:size(1) / g_params.batchsize)
    local input = torch.DoubleTensor(g_params.batchsize, g_params.memsize + 1, g_params.vector_size)
    local target = torch.DoubleTensor(g_params.batchsize)
    local context = torch.DoubleTensor(g_params.batchsize, g_params.memsize, g_params.vector_size)
    local correct = 0
    local confusion_matrix = torch.CudaTensor(3, 3)
    confusion_matrix:fill(0.0)
    local cost = 0
    for n = 1, N do
        if g_params.show then xlua.progress(n, N) end
        for b = 1, g_params.batchsize do
          local start = (n - 1) * g_params.batchsize
          input[b][1] = g_vectors[images_q[start + b][1]]
          target[b] = images_q[start + b][2]
          for i = 1, 16 do --change the parameter
            input[b][i + 1] = g_vectors[images[start + b][i]]
          end
        end
        rnn:zeroGradParameters() 
        local output = rnn:forward(input)
        local err = criterion:forward(output, target)
        cost = cost + err
        local max_els = torch.CudaTensor(g_params.batchsize)
        local max_ind = torch.CudaTensor(g_params.batchsize)
        max_els, max_ind = torch.max(output, 2)
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


g_img_train, g_img_q_train = g_read_images('data/quant.train.txt', g_params.vector_file, 
                            g_vocab, g_ivocab, g_vectors, g_params.train_size)
g_img_valid, g_img_q_valid = g_read_images('data/quant.valid.txt', g_params.vector_file, 
                            g_vocab, g_ivocab, g_vectors, g_params.valid_size)
g_img_test, g_img_q_test = g_read_images('data/quant.test.txt', g_params.vector_file, 
                            g_vocab, g_ivocab, g_vectors, g_params.test_size)
g_params.nwords = 3

print('vocabulary size ' .. #g_vocab)
print('starting to run....')

g_log_cost = {}
g_log_perp = {}
g_params.dt = g_params.sdt

r = nn.Recurrent(
   g_params.vector_size, nn.Identity(), 
   nn.Linear(g_params.vector_size, g_params.vector_size), nn.Sigmoid(), 
   g_params.memsize
)

rnn = nn.Sequential()
   :add(nn.Identity())
   :add(nn.SplitTable(1,2))
   :add(nn.Sequencer(r))
   :add(nn.SelectTable(-1)) -- this selects the last time-step of the rnn output sequence
   :add(nn.Linear(g_params.vector_size, g_params.nwords))
   :add(nn.SoftMax())

-- build criterion

criterion = nn.CrossEntropyCriterion()

g_params.dt = g_params.sdt

run(g_params.epochs)

if g_params.save ~= '' then
    save(g_params.save)
end
