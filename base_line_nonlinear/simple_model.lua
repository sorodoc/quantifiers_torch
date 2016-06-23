-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.
require('mobdebug').start()
require('nn')
--require('cunn')
require('nngraph')
paths.dofile('Peek.lua')
paths.dofile('LinearNB.lua')


function g_build_model(params)
    local freq_vector = nn.Identity()()
    local one_hot_vector = nn.Identity()()
    local target = nn.Identity()()
    -- call the memory function
    -- apply LinearNB on the output of the memory function
    local D = nn.JoinTable(2)({freq_vector, one_hot_vector})
    local z1 = nn.LinearNB(params.vocab_size * 2, params.vocab_size)(D)
    local non_linear = nn.ReLU()(z1)
    local z = nn.LinearNB(params.vocab_size, params.nwords)(non_linear)
    -- apply SoftMax on the result
    local pred = nn.SoftMax()(z)
    -- calculate the cross entropy between the class distribution and target index
    local costl = nn.CrossEntropyCriterion()
    costl.sizeAverage = false
    local cost = costl({pred, target})
    local model = nn.gModule({freq_vector, one_hot_vector, target}, {cost, pred})
    return model
end
