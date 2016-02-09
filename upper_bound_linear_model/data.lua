-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.
require('mobdebug').start()
require('paths')

local stringx = require('pl.stringx')
local file = require('pl.file')

-- read the images and the vectors associated with the symbols
function g_read_images(fname, vocab, ivocab, size)
    local data = file.read(fname)
    local lines = stringx.splitlines(data)
    local c = 0
    local image_ind = 1
    -- define the size of the dataset(5000)
    local images = torch.Tensor(size, 16)
    local images_q = torch.Tensor(size, 2)
    for n = 1,#lines do
        local w = stringx.split(lines[n])
        if w[2] ~= '?' then 
          for i = 1,#w do
              c = c + 1
              if w[i] ~= 'null' then
                if not vocab[w[i]] then
                    ivocab[#vocab+1] = w[i]
                    vocab[w[i]] = #vocab+1
                end
                images[image_ind][c] = vocab[w[i]]
              end
          end
        elseif w[2] == '?' then
          if not vocab[w[1]] then
            ivocab[#vocab + 1] = w[1]
            vocab[w[1]] = #vocab + 1
          end
          images_q[image_ind][1] = vocab[w[1]]
          images_q[image_ind][2] = vocab[w[3]]
          image_ind = image_ind + 1
          c = 0
        end
    end
    print('Read ' .. image_ind - 1 .. ' images from ' .. fname)
    return images, images_q
end
