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
paths.dofile('LinearNB.lua')

local function build_memory(params, input, context)
    local hid = {}
    hid[0] = input
    
    local Ain = context
--    local Ain_t = nn.LookupTable(params.memsize, params.edim)(time)
--    local Ain = nn.CAddTable()({Ain_c, Ain_t})
--    local Ain = torch.CudaTensor(params.batchsize, params.memsize, params.vector_size)
--    for t = 1, params.batchsize do
--      for s = 1, params.memsize do
--        Ain[t][s] = q_vectors[context[t][s]]
--      end
--    end
--    local Bin_c = nn.LookupTable(params.nwords, params.edim)(context)
--    local Bin_t = nn.LookupTable(params.memsize, params.edim)(time)
--    local Bin = nn.CAddTable()({Bin_c, Bin_t})
    local Bin = context
    local hid3dim = nn.View(1, -1):setNumInputDims(1)(hid[0])
    local MMaout = nn.MM(false, true):cuda()
    local Aout = MMaout({hid3dim, Ain})
    local Aout2dim = nn.View(-1):setNumInputDims(2)(Aout)
    local P = nn.SoftMax()(Aout2dim)
    local probs3dim = nn.View(1, -1):setNumInputDims(1)(P)
    local MMbout = nn.MM(false, false):cuda()
    local Bout = MMbout({probs3dim, Bin})
--    local C = time
--    local D = nn.CAddTable()({C, Bout})
    local D = Bout
    if params.lindim == params.edim then
      hid[1] = D
    elseif params.lindim == 0 then
      hid[1] = nn.ReLU()(D)
    else
      local F = nn.Narrow(2,1,params.lindim)(D)
      local G = nn.Narrow(2,1+params.lindim,params.edim-params.lindim)(D)
      local K = nn.ReLU()(G)
      hid[1] = nn.JoinTable(2)({F,K})
    end
    return hid
end

function g_build_model(params)
    local input = nn.Identity()()
    local target = nn.Identity()()
    local context = nn.Identity()()
--    local time = nn.Identity()()
    local hid = build_memory(params, input, context)
    local z = nn.LinearNB(params.edim, params.vector_size)(hid[#hid])
    local pred = nn.LogSoftMax()(z)
    local costl = nn.ClassNLLCriterion()
    costl.sizeAverage = false
    local cost = costl({pred, target})
    local model = nn.gModule({input, target, context}, {cost})
    model:cuda()
    -- IMPORTANT! do weight sharing after model is in cuda
    return model
end
