function out = study_lanczos_steps(varargin)
%STUDY_LANCZOS_STEPS  Convergence of the FTLM observables with N_L.
%   OUT = STUDY_LANCZOS_STEPS() repeats the FTLM calculation with fixed
%   start vectors (seed = 0, R = 100, ed_thresh = 0) for a list of
%   Lanczos step numbers N_L on the GPU in FP64 and FP32 and records the
%   observables and the Lanczos time.  With identical start vectors the
%   stochastic error is the same for all N_L, so that
%       |O(N_L) - O(N_L,ref)|   (FP64)  is the Lanczos truncation error,
%       |O_32(N_L) - O_64(N_L)|         the precision error at given N_L.
%
%   Name-value options:
%     'Systems' {'ico_s1', 'ico_s3o2'}
%     'NL'      [10 20 30 40 50 60 75 100 150 200 300]  (last = reference)
%     'OutDir'  '.'
%   Output: study_lanczos_steps_<key>.mat

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

p = inputParser;
p.addParameter('Systems', {'ico_s1', 'ico_s3o2'});
p.addParameter('NL', [10 20 30 40 50 60 75 100 150 200 300]);
p.addParameter('OutDir', '.');
p.parse(varargin{:});
o = p.Results;

T = unique([linspace(0.005, 0.2, 196), linspace(0.2, 10, 491)]);
out = struct();
for ks = 1 : numel(o.Systems)
    key = o.Systems{ks};
    switch key
        case 'ico_s1',   mopts = struct('geometry', 'ico', 's_val', 1.0, 'J', 1);
        case 'ico_s3o2', mopts = struct('geometry', 'ico', 's_val', 1.5, 'J', 1);
        otherwise, error('unknown system %s', key);
    end
    NL = o.NL;
    S = struct('key', key, 'T_range', T, 'R', 100, 'NL', NL);
    for pr = {'double', 'single'}
        C = zeros(numel(NL), numel(T));  X = C;  t_lz = zeros(1, numel(NL));
        for q = 1 : numel(NL)
            opts = mopts;
            opts.R = 100;  opts.M_lz = NL(q);  opts.T_range = T;
            opts.ed_thresh = 0;  opts.seed = 0;  opts.precision = pr{1};
            opts.verbose = false;
            r = ftlm.run(opts);
            C(q, :) = r.C_T;  X(q, :) = r.chi_T;  t_lz(q) = r.t_lanczos;
            fprintf('%s %s N_L = %3d: t_lanczos = %.1f s\n', key, pr{1}, NL(q), t_lz(q));
        end
        S.(pr{1}) = struct('C', C, 'chi', X, 't_lanczos', t_lz);
    end
    % truncation error (FP64, reference = largest N_L) and FP32 error per N_L
    S.dC_trunc   = abs(S.double.C - S.double.C(end, :));
    S.dchi_trunc = abs(S.double.chi - S.double.chi(end, :));
    S.dC_fp32    = abs(S.single.C - S.double.C);
    S.dchi_fp32  = abs(S.single.chi - S.double.chi);
    fprintf('  N_L   max|dC_trunc|  max|dchi_trunc|  max|dC_fp32|  max|dchi_fp32|\n');
    for q = 1 : numel(NL)
        fprintf('  %4d   %.2e       %.2e         %.2e      %.2e\n', NL(q), ...
            max(S.dC_trunc(q, :)), max(S.dchi_trunc(q, :)), ...
            max(S.dC_fp32(q, :)), max(S.dchi_fp32(q, :)));
    end
    save(fullfile(o.OutDir, sprintf('study_lanczos_steps_%s.mat', key)), '-struct', 'S');
    out.(key) = S;
end
end
