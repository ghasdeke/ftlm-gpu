function cfg = kernel_config(model, A, lookup, precision, B, basis)
%FTLM.KERNEL_CONFIG  Initialization struct for ftlm_gpu_mex / ftlm_cpu_mex.
%   CFG = FTLM.KERNEL_CONFIG(MODEL, A, LOOKUP, PRECISION, B, BASIS)
%   collects the model and sector data needed by the MEX kernels:
%
%     MODEL      struct from FTLM.MODEL
%     A          digit sum of the sector (A = S_max + M)
%     LOOKUP     'clt' (compressed lookup table) or 'cr' (combinatorial
%                ranking; GPU only)
%     PRECISION  'double' | 'single' | 'half' | 'bfloat16'
%     B          block size (number of chains per block Lanczos call)
%     BASIS      sorted sector basis (int32, FTLM.ENUMERATE_SECTOR);
%                required for 'clt', ignored for 'cr'
%
%   Usage:
%       ftlm_gpu_mex('init', cfg)   or   ftlm_cpu_mex('init', cfg)

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

cfg = struct();
cfg.lookup    = char(lookup);
cfg.precision = char(precision);
cfg.N         = model.N;
cfg.B         = B;
cfg.two_s     = int32(model.two_s);
cfg.ci        = int32(model.couplings(:, 1) - 1);     % 0-based
cfg.cj        = int32(model.couplings(:, 2) - 1);
cfg.J         = double(model.couplings(:, 3));

switch cfg.lookup
    case 'clt'
        assert(nargin >= 6 && ~isempty(basis), 'ftlm:kernel_config', ...
               'lookup ''clt'' requires the sector basis.');
        [block_base, block_mask] = ftlm.build_clt(basis, model.D_full);
        cfg.dim        = numel(basis);
        cfg.power      = int32(model.power);
        cfg.basis      = int32(basis);
        cfg.block_base = block_base;
        cfg.block_mask = block_mask;
    case 'cr'
        assert(model.cr_ok, 'ftlm:kernel_config', ...
               'lookup ''cr'': the packed state needs %d > 64 bits.', sum(model.bits));
        [dcum, pstride, astride, dim] = ftlm.cr_tables(model, A);
        cfg.dim          = dim;
        cfg.shift        = int32(model.shift);
        cfg.A_total      = A;
        cfg.dcum         = dcum;
        cfg.dcum_pstride = pstride;
        cfg.dcum_astride = astride;
    otherwise
        error('ftlm:kernel_config', 'lookup must be ''clt'' or ''cr''.');
end
end
