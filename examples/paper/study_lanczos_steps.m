function out = study_lanczos_steps(varargin)
%STUDY_LANCZOS_STEPS  Convergence of the FTLM observables with N_L.
%   OUT = STUDY_LANCZOS_STEPS() runs the FTLM calculation with fixed start
%   vectors (seed = 0, R = 100, ed_thresh = 0) once with N_L,max Lanczos
%   steps on the GPU in FP64 and FP32 and evaluates the observables for
%   every N_L of the list by truncating the recorded Lanczos coefficients
%   of each chain to its first N_L steps.  This is identical to separate
%   runs with N_L steps: the recursion, the batches, the start vectors and
%   the per-chain termination do not depend on N_L,max.  With identical
%   start vectors the stochastic error is the same for all N_L, so that
%       |O(N_L) - O(N_L,max)|  (FP64)  is the Lanczos truncation error,
%       |O_32(N_L) - O_64(N_L)|        the precision error at given N_L.
%   The Lanczos time is measured in separate runs with the N_L of 'NLTime'.
%
%   Name-value options:
%     'Systems' {'ico_s1', 'ico_s3o2'}
%     'NL'      [10:10:100 120 150 200 250 300]  (last = reference N_L,max)
%     'NLTime'  [50 150]  (plus N_L,max from the main run)
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
p.addParameter('NL', [10:10:100, 120, 150, 200, 250, 300]);
p.addParameter('NLTime', [50 150]);
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
    NL = o.NL(:)';
    NLmax = NL(end);
    NLt = unique([o.NLTime(o.NLTime < NLmax), NLmax]);
    S = struct('key', key, 'T_range', T, 'R', 100, 'NL', NL, 'NL_time', NLt);
    for pr = {'double', 'single'}
        opts = mopts;
        opts.R = 100;  opts.T_range = T;  opts.ed_thresh = 0;  opts.seed = 0;
        opts.precision = pr{1};  opts.verbose = false;
        % main run with N_L,max steps; Lanczos coefficients of all chains kept
        opts.M_lz = NLmax;  opts.save_ritz = true;
        r = ftlm.run(opts);
        [C, X] = truncated_observables(r, NL, T);
        dev = max(abs(C(end, :) - r.C_T(:)'));
        assert(dev <= 1e-12 * max(abs(r.C_T)), 'truncation check failed (%.2e)', dev);
        % Lanczos time for a few N_L (separate runs, only the time is used)
        t_lz = zeros(1, numel(NLt));
        t_lz(end) = r.t_lanczos;
        opts.save_ritz = false;
        for q = 1 : numel(NLt) - 1
            opts.M_lz = NLt(q);
            t_lz(q) = ftlm.run(opts).t_lanczos;
        end
        fprintf('%s %s: N_L,max = %d, t_lanczos = %s s\n', key, pr{1}, NLmax, mat2str(t_lz, 3));
        S.(pr{1}) = struct('C', C, 'chi', X, 't_lanczos', t_lz);
    end
    % truncation error (FP64, reference = N_L,max) and FP32 error per N_L
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

function [C, X] = truncated_observables(r, NL, T)
%TRUNCATED_OBSERVABLES  C(T), chi(T) from the first N_L steps of every chain.
    C = zeros(numel(NL), numel(T));  X = C;
    for q = 1 : numel(NL)
        E = [];  w = [];  M = [];
        for s = 1 : numel(r.ritz)
            z = r.ritz{s};
            dim = r.sector_dims(s);
            R_eff = size(z.alpha, 2);
            for c = 1 : R_eff
                n = min(z.nsteps(c), NL(q));
                [theta, q1] = ftlm.solve_tridiag(z.alpha(1:n, c), z.beta(1:n-1, c));
                E = [E; theta];                                          %#ok<AGROW>
                w = [w; r.sector_mult(s) * ((dim / R_eff) * q1)];        %#ok<AGROW>
                M = [M; r.sector_M(s) * ones(n, 1)];                     %#ok<AGROW>
            end
        end
        [C(q, :), X(q, :)] = ftlm.observables(E, w, M, T);
    end
end
