function [block_base, block_mask] = build_clt(basis, D_full)
%FTLM.BUILD_CLT  Compressed lookup table (CLT) of a sector basis.
%   [BLOCK_BASE, BLOCK_MASK] = FTLM.BUILD_CLT(BASIS, D_FULL) partitions
%   the label space {0, ..., D_FULL-1} into blocks of 32 labels and
%   stores for each block b
%       BLOCK_MASK(b)  occupancy mask (bit j set iff label 32b+j is in
%                      the sector)                         [uint32]
%       BLOCK_BASE(b)  sector index of the first in-sector label of the
%                      block, i.e. the prefix count C_b; -1 for empty
%                      blocks (never accessed)             [int32]
%   BASIS must be sorted increasingly (see FTLM.ENUMERATE_SECTOR).
%   Memory: 8 bytes per block, i.e. D_FULL/4 bytes (16x less than a
%   full int32 lookup table).

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%     http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.
% ================================================================

BLOCK_SIZE = 32;
n_blocks   = ceil(D_full / BLOCK_SIZE);
states     = double(basis(:));

blks = floor(states / BLOCK_SIZE) + 1;
bits = mod(states, BLOCK_SIZE);

block_base     = int32(-ones(n_blocks, 1));
[ub, fi]       = unique(blks, 'first');
block_base(ub) = int32(fi - 1);                 % 0-based sector index

mask_sums  = accumarray(blks, pow2(bits), [n_blocks, 1]);   % exact in double
block_mask = uint32(mask_sums);
end
