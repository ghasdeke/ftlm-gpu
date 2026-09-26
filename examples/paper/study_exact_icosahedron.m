function S = study_exact_icosahedron(varargin)
%STUDY_EXACT_ICOSAHEDRON  FTLM vs. exact C(T) for the s = 3/2 icosahedron.
%   S = STUDY_EXACT_ICOSAHEDRON() compares the heat capacity of the
%   s = 3/2 icosahedron from study_precision (R = 100, N_L = 100, GPU FP64,
%   FP32, FP16, BF16; same start vectors) with the exact result obtained
%   by full diagonalization with simultaneous spin and point-group
%   adaptation (S. Ghassemi Tabrizi, T. D. Kuehne, Magnetism 5 (2025) 8),
%   stored in data/ico_s3o2_exact_heat_capacity.csv (columns T, C).
%
%   Reports max_T |C_FTLM - C_exact| (total FTLM error at R = 100, which is
%   dominated by the stochastic trace estimate) and the precision
%   deviations |C_x - C_FP64| on T in [TMin, TMax].
%
%   Name-value options: 'DataDir' (study_precision output, default '.'),
%   'TMin' (0.02), 'TMax' (5).

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

p = inputParser;
p.addParameter('DataDir', '.');
p.addParameter('TMin', 0.02);
p.addParameter('TMax', 5);
p.parse(varargin{:});
o = p.Results;

here = fileparts(mfilename('fullpath'));
X = readmatrix(fullfile(here, 'data', 'ico_s3o2_exact_heat_capacity.csv'));
P = load(fullfile(o.DataDir, 'study_precision_ico_s3o2.mat'));
T = P.T_range(:);
sel = T >= o.TMin & T <= o.TMax;
C_ex = interp1(X(:, 1), X(:, 2), T(sel), 'pchip');

S = struct('T', T(sel), 'C_exact', C_ex, 'C_max', max(X(:, 2)));
ref = P.gpu_double.C(:);
S.err_total_fp64 = abs(ref(sel) - C_ex);
[S.max_err_total, k] = max(S.err_total_fp64);
S.T_at_max = S.T(k);
fprintf('max |C_FP64 - C_exact| = %.3e at T = %.3f (%.2f %% of C_max)\n', ...
    S.max_err_total, S.T_at_max, 100 * S.max_err_total / S.C_max);
for v = {'gpu_single', 'gpu_half', 'gpu_bfloat16', 'cpu_double'}
    if ~isfield(P, v{1}), continue; end
    d = abs(P.(v{1}).C(:) - ref);
    S.(['max_dev_' v{1}]) = max(d(sel));
    fprintf('max |C_%s - C_FP64| = %.3e\n', v{1}, max(d(sel)));
end
save(fullfile(o.DataDir, 'study_exact_icosahedron.mat'), '-struct', 'S');
end
