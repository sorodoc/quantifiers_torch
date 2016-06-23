-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.
require('mobdebug').start()
require('nn')
require('cunn')
require('nngraph')
paths.dofile('Peek.lua')
paths.dofile('LinearNB.lua')

      
function g_build_model(params)
    local input = nn.Identity()()
    local target = nn.Identity()()
    local context = nn.Identity()()
    -- call the memory function
    --local context_table = nn.SplitTable(1,2)(context)
    local r = nn.Recurrent(g_params.vector_size, context, 
            nn.Linear(g_params.vector_size, g_params.vector_size), nn.Sigmoid(), 
            g_params.memsize)
    local res = nn.Sequencer(r)()
    local hid = nn.SelectTable(-1)(res)
    local hid1 = nn.View(1, -1):setNumInputDims(1)(hid)
    local hid2 = nn.JoinTable(2)({hid1, input})
    -- apply LinearNB on the output of the memory function
    local z1 = nn.LinearNB(params.vector_size * 2, params.vector_size)(hid2)
    local non_linear = nn.ReLU()(z1)
    local z = nn.LinearNB(params.vector_size, params.nwords)(non_linear)
    -- apply SoftMax on the result
    local pred = nn.SoftMax()(z)
    -- calculate the cross entropy between the class distribution and target index
    local costl = nn.CrossEntropyCriterion()
    costl.sizeAverage = false
    local cost = costl({pred, target})
    local model = nn.gModule({input, target, context}, {cost, pred})
    model:cuda()
    return model
end
