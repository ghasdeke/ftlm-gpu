function opts = defaults(opts)
%FTLM.DEFAULTS  Complete and validate the FTLM run options.
%   OPTS = FTLM.DEFAULTS(OPTS) fills in defaults for all optional fields
%   and checks the values.
%
%   Model (see FTLM.MODEL):
%     geometry, N_ring, s_val, spins, J, couplings, N_sites
%
%   Required:
%     R             random vectors per sector
%     M_lz          Lanczos steps per random vector (N_L)
%     T_range       temperatures (positive, units of J/k_B)
%
%   Optional (default):
%     backend       'gpu'      'gpu' | 'cpu'
%     precision     'single'   storage/arithmetic precision of the main
%                              run. GPU: 'single' | 'double' | 'half' |
%                              'bfloat16' (16-bit formats: storage only,
%                              FP32 arithmetic). CPU: 'double' | 'single'
%     lookup        'clt'      'clt' | 'cr' (GPU only)
%     use_cpu_reference false  repeat the run with the CPU kernel
%     cpu_precision 'double'   precision of the CPU reference run
%     only_M0       false      lowest magnetization sector only
%     ed_thresh     1000       sectors with dim <= ed_thresh: exact
%                              diagonalization instead of FTLM
%     seed          0          0: v1-compatible start vectors;
%                              k > 0: independent set number k
%     B_gpu         0          GPU block size (0 = adaptive, else 1..16)
%     B_cpu         8          CPU block size (1..32)
%     L2_cache_bytes 48e6      threshold of the adaptive B_gpu (paper value; the
%                              RTX 4000 SFF Ada has a 40 MB L2 cache)
%     save_ritz     false      keep Ritz values/weights and Lanczos
%                              coefficients per sector in the output
%     output_dir    '.'        directory of the output .mat file
%     output_name   ''         file name (default ftlm_<tag>.mat)
%     verbose       true

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

for f = {'R', 'M_lz', 'T_range'}
    assert(isfield(opts, f{1}) && ~isempty(opts.(f{1})), 'ftlm:options', ...
           'Required input missing: %s', f{1});
end

d = struct('backend', 'gpu', 'precision', 'single', 'lookup', 'clt', ...
           'use_cpu_reference', false, 'cpu_precision', 'double', ...
           'only_M0', false, 'ed_thresh', 1000, 'seed', 0, ...
           'B_gpu', 0, 'B_cpu', 8, 'L2_cache_bytes', 48e6, ...
           'save_ritz', false, 'output_dir', '.', 'output_name', '', ...
           'verbose', true, 'geometry', '', 'N_ring', [], 's_val', [], ...
           'spins', [], 'J', [], 'couplings', [], 'N_sites', []);
fn = fieldnames(d);
for k = 1 : numel(fn)
    if ~isfield(opts, fn{k}) || (isempty(opts.(fn{k})) && ~isempty(d.(fn{k})))
        opts.(fn{k}) = d.(fn{k});
    end
end

opts.backend       = lower(char(opts.backend));
opts.precision     = lower(char(opts.precision));
opts.lookup        = lower(char(opts.lookup));
opts.cpu_precision = lower(char(opts.cpu_precision));
opts.geometry      = char(opts.geometry);

is_int = @(x, lo) isnumeric(x) && isscalar(x) && x == round(x) && x >= lo;
assert(is_int(opts.R, 1),    'ftlm:options', 'R must be a positive integer.');
assert(is_int(opts.M_lz, 1), 'ftlm:options', 'M_lz must be a positive integer.');
T = opts.T_range;
assert(isnumeric(T) && isvector(T) && all(T > 0) && all(isfinite(T)), 'ftlm:options', ...
       'T_range must be a vector of positive finite numbers.');
opts.T_range = double(T(:)');

assert(any(strcmp(opts.backend, {'gpu', 'cpu'})), 'ftlm:options', ...
       'backend must be ''gpu'' or ''cpu''.');
assert(any(strcmp(opts.lookup, {'clt', 'cr'})), 'ftlm:options', ...
       'lookup must be ''clt'' or ''cr''.');
if strcmp(opts.backend, 'gpu')
    assert(any(strcmp(opts.precision, {'single', 'double', 'half', 'bfloat16'})), ...
           'ftlm:options', 'GPU precision must be single, double, half or bfloat16.');
else
    assert(any(strcmp(opts.precision, {'single', 'double'})), 'ftlm:options', ...
           'CPU precision must be single or double.');
    assert(strcmp(opts.lookup, 'clt'), 'ftlm:options', ...
           'The CPU backend supports lookup = ''clt'' only.');
end
assert(any(strcmp(opts.cpu_precision, {'single', 'double'})), 'ftlm:options', ...
       'cpu_precision must be single or double.');
assert(is_int(opts.ed_thresh, 0), 'ftlm:options', 'ed_thresh must be a non-negative integer.');
assert(is_int(opts.seed, 0), 'ftlm:options', 'seed must be a non-negative integer.');
assert(is_int(opts.B_gpu, 0) && opts.B_gpu <= 16, 'ftlm:options', ...
       'B_gpu must be an integer in [0, 16] (0 = adaptive).');
assert(is_int(opts.B_cpu, 1) && opts.B_cpu <= 32, 'ftlm:options', ...
       'B_cpu must be an integer in [1, 32].');
assert(isnumeric(opts.L2_cache_bytes) && isscalar(opts.L2_cache_bytes) && ...
       opts.L2_cache_bytes > 0, 'ftlm:options', 'L2_cache_bytes must be positive.');
opts.only_M0           = logical(opts.only_M0);
opts.use_cpu_reference = logical(opts.use_cpu_reference);
opts.save_ritz         = logical(opts.save_ritz);
opts.verbose           = logical(opts.verbose);
end
