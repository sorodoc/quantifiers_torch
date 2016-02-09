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
    -- apply LinearNB on the output of the memory function
    local D = nn.JoinTable(2)({context, input})
    local y = nn.LinearNB(params.vocab_size * 2, params.hidden_size)(D)
    local y_nonlin = nn.ReLU()(y)
    -- apply SoftMax on the result
    local z = nn.LinearNB(params.hidden_size, params.nwords)(y_nonlin)
    local pred = nn.SoftMax()(z)
    -- calculate the cross entropy between the class distribution and target index
    local costl = nn.CrossEntropyCriterion()
    costl.sizeAverage = false
    local cost = costl({pred, target})
    local model = nn.gModule({input, target, context}, {cost, pred})
    model:cuda()
    return model
end
