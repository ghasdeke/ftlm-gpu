function out = study_precision(varargin)
%STUDY_PRECISION  Same FTLM calculation in all precisions (paper Figs. 1, 3).
%   OUT = STUDY_PRECISION() runs, for each system, the FTLM calculation
%   with identical start vectors (seed = 0, ed_thresh = 0 as in the paper)
%   with the variants
%       GPU FP64, GPU FP32, GPU FP16, GPU BF16   (CLT kernel)
%       CPU FP64, CPU FP32                      (OpenMP kernel)
%   and stores C(T), chi(T), Z_eff(T) and timings of every variant.
%   Differences between GPU FP32 and GPU FP64 isolate the arithmetic
%   precision (identical kernel and reduction order); CPU FP64 vs GPU
%   FP64 shows the reduction-order effect alone.
%
%   Systems (keys):
%     ico_s1, ico_s3o2        icosahedron s = 1, 3/2 (Fig. 1), R = 100
%     ring12_s1, ring20_s1o2  rings (Fig. 3 a-d), R = 100
%     dodeca_s1o2             dodecahedron s = 1/2 (Fig. 3 e, f), R = 100
%     icosid_M0               icosidodecahedron s = 1/2, M = 0, R = 8 (Fig. 3 g)
%
%   Name-value options: 'Systems', 'Variants', 'OutDir' (default '.').
%   Output: study_precision_<key>.mat per system.

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

addpath(fileparts(fileparts(fileparts(mfilename('fullpath')))));

T_std = linspace(0.02, 10, 500);
sys_all = {
    'ico_s1',      struct('geometry', 'ico',    's_val', 1.0, 'J', 1), false, 100, T_std;
    'ico_s3o2',    struct('geometry', 'ico',    's_val', 1.5, 'J', 1), false, 100, T_std;
    'ring12_s1',   struct('geometry', 'ring', 'N_ring', 12, 's_val', 1.0, 'J', 1), false, 100, T_std;
    'ring20_s1o2', struct('geometry', 'ring', 'N_ring', 20, 's_val', 0.5, 'J', 1), false, 100, T_std;
    'dodeca_s1o2', struct('geometry', 'dodeca', 's_val', 0.5, 'J', 1), false, 100, T_std;
    'icosid_M0',   struct('geometry', 'icosid', 's_val', 0.5, 'J', 1), true, 8, linspace(0.02, 2, 400)};
var_all = {'gpu_double', 'gpu_single', 'gpu_half', 'gpu_bfloat16', 'cpu_double', 'cpu_single'};

p = inputParser;
p.addParameter('Systems', sys_all(:, 1)');
p.addParameter('Variants', var_all);
p.addParameter('OutDir', '.');
p.parse(varargin{:});
o = p.Results;

out = struct();
for is = 1 : size(sys_all, 1)
    key = sys_all{is, 1};
    if ~any(strcmp(o.Systems, key)), continue; end
    [mopts, only_M0, R, T] = sys_all{is, 2:5};
    fprintf('\n=== %s (R = %d, N_L = 100) ===\n', key, R);
    S = struct('key', key, 'T_range', T, 'R', R, 'M_lz', 100, 'only_M0', only_M0);
    for iv = 1 : numel(var_all)
        v = var_all{iv};
        if ~any(strcmp(o.Variants, v)), continue; end
        parts = strsplit(v, '_');
        opts = mopts;
        opts.backend = parts{1};  opts.precision = parts{2};  opts.lookup = 'clt';
        opts.R = R;  opts.M_lz = 100;  opts.T_range = T;  opts.only_M0 = only_M0;
        opts.ed_thresh = 0;  opts.seed = 0;  opts.verbose = false;
        t0 = tic;
        res = ftlm.run(opts);
        S.(v) = struct('C', res.C_T, 'chi', res.chi_T, 'Z', res.Z_eff, ...
                       't_wall', toc(t0), 't_lanczos', res.t_lanczos);
        fprintf('  %-13s t = %7.1f s (Lanczos %7.1f s)\n', v, S.(v).t_wall, res.t_lanczos);
        save(fullfile(o.OutDir, sprintf('study_precision_%s.mat', key)), '-struct', 'S');
    end
    report(S);
    out.(key) = S;
end
end

function report(S)
    if ~isfield(S, 'gpu_double'), return; end
    ref = S.gpu_double;
    fn = setdiff(fieldnames(S), {'key', 'T_range', 'R', 'M_lz', 'only_M0', 'gpu_double'});
    fprintf('  max_T |dC| / max C,  max_T |dchi| / max chi   (reference: GPU FP64)\n');
    for k = 1 : numel(fn)
        x = S.(fn{k});
        dC = max(abs(x.C - ref.C)) / max(ref.C);
        dX = 0;
        if max(ref.chi) > 0, dX = max(abs(x.chi - ref.chi)) / max(ref.chi); end
        fprintf('    %-13s %.2e   %.2e\n', fn{k}, dC, dX);
    end
end
