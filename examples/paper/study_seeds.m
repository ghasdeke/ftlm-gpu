function out = study_seeds(varargin)
%STUDY_SEEDS  Stochastic error vs. precision error (paper Fig. 2, R*).
%   OUT = STUDY_SEEDS() runs N_s independent FTLM calculations (seed =
%   1..N_s, R = 100 random vectors per sector each, N_L = 100,
%   ed_thresh = 0) on the GPU in FP32 and, for the first N_64 seeds, in
%   FP64 with the same start vectors, and evaluates
%
%     sigma_emp(T)   empirical standard deviation over the N_s FP32 runs
%     sigma_theo(T)  O / sqrt(R Z_eff), the expected standard deviation of a
%                    single run with R random vectors per sector (Ref. [13]),
%                    O and Z_eff from the pooled estimate
%     Delta_k(T)     |O_32 - O_64| of the estimate pooled over k seeds
%                    (R_eff = k R), k = 1, 2, 4, ..., N_64
%     R*             break-even number of random vectors,
%                    R* = R (sigma_emp / Delta)^2 with Delta at R_eff
%
%   The FP32 error of the pooled estimate does not grow with R_eff,
%   whereas sigma decreases as 1/sqrt(R); R* quantifies where the two
%   would cross.
%
%   Name-value options:
%     'Systems' {'ico_s1', 'ico_s3o2'}   'NSeeds' 50   'NSeeds64' 50 (ico_s1)
%     / 16 (ico_s3o2)   'OutDir' '.'
%   Output: study_seeds_<key>.mat

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
p.addParameter('NSeeds', 50);
p.addParameter('NSeeds64', []);
p.addParameter('OutDir', '.');
p.parse(varargin{:});
o = p.Results;

T = unique([linspace(0.005, 0.2, 196), linspace(0.2, 10, 491)]);
R = 100;
out = struct();
for ks = 1 : numel(o.Systems)
    key = o.Systems{ks};
    switch key
        case 'ico_s1',   mopts = struct('geometry', 'ico', 's_val', 1.0, 'J', 1); n64 = o.NSeeds;
        case 'ico_s3o2', mopts = struct('geometry', 'ico', 's_val', 1.5, 'J', 1); n64 = 16;
        otherwise, error('unknown system %s', key);
    end
    if ~isempty(o.NSeeds64), n64 = o.NSeeds64; end
    Ns = o.NSeeds;
    fprintf('\n=== %s: %d seeds (FP32), %d seeds (FP64) ===\n', key, Ns, n64);

    runs32 = cell(Ns, 1);  runs64 = cell(n64, 1);
    C32 = zeros(Ns, numel(T));  X32 = C32;
    t0 = tic;
    for s = 1 : Ns
        opts = mopts;
        opts.R = R;  opts.M_lz = 100;  opts.T_range = T;  opts.ed_thresh = 0;
        opts.seed = s;  opts.save_ritz = true;  opts.verbose = false;
        opts.precision = 'single';
        r = ftlm.run(opts);
        [E, w, M] = ftlm.collect_ritz(r);
        runs32{s} = struct('E', E, 'w', w, 'M', M);
        C32(s, :) = r.C_T;  X32(s, :) = r.chi_T;
        if s <= n64
            opts.precision = 'double';
            r = ftlm.run(opts);
            [E, w, M] = ftlm.collect_ritz(r);
            runs64{s} = struct('E', E, 'w', w, 'M', M);
        end
        fprintf('  seed %3d done (%.0f s)\n', s, toc(t0));
    end

    % empirical standard deviation over seeds (FP32), Eq. (22)
    S = struct('key', key, 'T_range', T, 'R', R, 'Ns', Ns, 'N64', n64);
    S.C_runs = C32;  S.chi_runs = X32;
    S.sigma_emp_C   = std(C32, 1, 1);
    S.sigma_emp_chi = std(X32, 1, 1);

    % pooled estimates
    [S.C_pool, S.chi_pool, Zp] = pooled(runs32(1:Ns), T);
    S.sigma_theo_C   = S.C_pool ./ sqrt(R * Zp);
    S.sigma_theo_chi = S.chi_pool ./ sqrt(R * Zp);

    % FP32 - FP64 difference of pooled estimates, k = 1, 2, 4, ...
    ks_list = unique([2 .^ (0 : floor(log2(n64))), n64]);
    S.k_list = ks_list;
    S.dC_k = zeros(numel(ks_list), numel(T));  S.dchi_k = S.dC_k;
    for q = 1 : numel(ks_list)
        k = ks_list(q);
        [C3, X3] = pooled(runs32(1:k), T);
        [C6, X6] = pooled(runs64(1:k), T);
        S.dC_k(q, :)   = abs(C3 - C6);
        S.dchi_k(q, :) = abs(X3 - X6);
    end
    % break-even R* = R * (sigma_emp(R) / Delta)^2, Delta of the largest pool
    S.Rstar_C   = R * (S.sigma_emp_C   ./ S.dC_k(end, :)).^2;
    S.Rstar_chi = R * (S.sigma_emp_chi ./ S.dchi_k(end, :)).^2;
    fprintf('  min_T R*(C) = %.2e, min_T R*(chi) = %.2e\n', ...
        min(S.Rstar_C(S.dC_k(end, :) > 0)), min(S.Rstar_chi(S.dchi_k(end, :) > 0)));
    save(fullfile(o.OutDir, sprintf('study_seeds_%s.mat', key)), '-struct', 'S');
    out.(key) = S;
end
end

function [C, chi, Z] = pooled(runs, T)
%POOLED  Observables of the estimate pooled over the runs (weights / k).
    k = numel(runs);
    E = cellfun(@(r) r.E, runs, 'UniformOutput', false);
    w = cellfun(@(r) r.w / k, runs, 'UniformOutput', false);
    M = cellfun(@(r) r.M, runs, 'UniformOutput', false);
    [C, chi, Z] = ftlm.observables(vertcat(E{:}), vertcat(w{:}), vertcat(M{:}), T);
end
