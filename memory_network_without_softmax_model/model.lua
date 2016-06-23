require('mobdebug').start()
require('nn')
--require('cunn')
require('nngraph')
paths.dofile('Peek.lua')
paths.dofile('LinearNB.lua')

local function build_memory(params, input, context)
    local hid = {}
    -- instantiate hid[0] with query vector 
    hid[0] = input  
    -- instantiate Ain and Bin with the memory(matrix with symbols vectors)
    local Ain = context
    local Bin = context
    -- hid3dim has 3 dimensions(batch * 1 * vector),hid[0] has 2(batch * vector) 
    local hid3dim = nn.View(1, -1):setNumInputDims(1)(hid[0])
    -- Aout - matrix product between hid3dim(query) and Ain(memory)
    local MMaout = nn.MM(false, true)
    local Aout = MMaout({hid3dim, Ain})
    local Aout2dim = nn.View(-1):setNumInputDims(2)(Aout)
    -- apply softmax on the product of query and memory
    --local P = nn.SoftMax()(Aout2dim) --softmax function
    local P = Aout2dim
    -- Bout - product between probability distribution and memory
    --local probs3dim = nn.View(1, -1):setNumInputDims(1)(nn.Peek()(P))
    local probs3dim = nn.View(1, -1):setNumInputDims(1)(P)
    local MMbout = nn.MM(false, false)
    local Bout = MMbout({probs3dim, Bin})
    -- C - apply LinearNB over the query
    --local C = nn.LinearNB(params.vector_size, params.vector_size)(hid[0])
    -- D - sum between C and Bout
    --local D = nn.CAddTable()({C, Bout}) -- for sum
    Bout2 = nn.View(-1):setNumInputDims(2)(Bout) -- for concatenation
    local D = nn.JoinTable(2)({hid[0], Bout2}) -- for concatenation
    hid[1] = D
    return hid
end

function g_build_model(params)
    local input = nn.Identity()()
    local target = nn.Identity()()
    local context = nn.Identity()()
    -- call the memory function
    local hid = build_memory(params, input, context)
    -- apply LinearNB on the output of the memory function
    local z1 = nn.LinearNB(params.vector_size * 2, params.vector_size)(hid[#hid])
    local non_linear = nn.ReLU()(z1)
    local z = nn.LinearNB(params.vector_size, params.nwords)(non_linear)
    -- apply SoftMax on the result
    local pred = nn.SoftMax()(z)
    -- calculate the cross entropy between the class distribution and target index
    local costl = nn.CrossEntropyCriterion()
    costl.sizeAverage = false
    local cost = costl({pred, target})
    local model = nn.gModule({input, target, context}, {cost, pred})
    --model:cuda()
    return model
end
