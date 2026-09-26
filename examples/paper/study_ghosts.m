function S = study_ghosts(varargin)
%STUDY_GHOSTS  Ghost Ritz values and cluster weights (paper Figs. 4, 5).
%   S = STUDY_GHOSTS() runs one Lanczos chain (N_L = 100) from the same
%   start vector in the M = 0 sector of the s = 1 icosahedron in FP64 and
%   FP32 (GPU, identical kernel) and applies the Cullum-Willoughby test:
%   with T1 the tridiagonal matrix without its first row and column, a
%   Ritz value theta_k is flagged as ghost if
%       l_k = min_j |theta_k - theta1_j| < tau,   tau = C_tau u_FP32 W_T,
%   W_T = theta_max - theta_min.  Ritz values within tau of each other are
%   grouped into clusters, and the total cluster weights are compared
%   between FP64 and FP32.
%
%   Name-value options:
%     'Ctau'      numerical factor C_tau (default 4)
%     'NL'        Lanczos steps (default 100)
%     'Vector'    index of the start vector in the seed-0 sequence (1)
%     'Reference' 'gpu' (default, FP64 GPU kernel) or 'cpu' (FP64 CPU)
%     'OutDir'    '.'
%   Output: study_ghosts.mat

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
p.addParameter('Ctau', 4);
p.addParameter('NL', 100);
p.addParameter('Vector', 1);
p.addParameter('Reference', 'gpu');
p.addParameter('OutDir', '.');
p.parse(varargin{:});
o = p.Results;

model = ftlm.model(struct('geometry', 'ico', 's_val', 1.0, 'J', 1));
sec   = ftlm.sectors(model, true);
basis = ftlm.enumerate_sector(model, sec.A);
rng(sec.dim, 'twister');
V = randn(sec.dim, o.Vector);
v0 = V(:, end);

runs = {'fp64', o.Reference, 'double', v0;  'fp32', 'gpu', 'single', single(v0)};
S = struct('Ctau', o.Ctau, 'NL', o.NL, 'dim', sec.dim);
u32 = 2^-24;
for k = 1 : size(runs, 1)
    [name, be, pr, v] = runs{k, :};
    cfg = ftlm.kernel_config(model, sec.A, 'clt', pr, 1, basis);
    f = str2func(sprintf('ftlm_%s_mex', be));
    f('init', cfg);
    [AL, BE, ns] = f('block_lanczos', v, o.NL);
    f('cleanup');
    n = ns(1);
    a = AL(1:n);  b = BE(1:n - 1);
    [theta, w] = ftlm.solve_tridiag(a, b);
    theta1 = eig(diag(a(2:end)) + diag(b(2:end), 1) + diag(b(2:end), -1));
    ell = min(abs(theta - theta1'), [], 2);          % deflation distance, Eq. (25)
    X = struct('alpha', a, 'beta', b, 'theta', theta, 'w', w, 'ell', ell);
    S.(name) = X;
end

% common tolerance from the FP32 backward-error scale
W_T  = max(S.fp32.theta) - min(S.fp32.theta);
tau  = o.Ctau * u32 * W_T;
S.tau = tau;
for name = {'fp64', 'fp32'}
    X = S.(name{1});
    X.ghost = X.ell < tau;
    % clusters: consecutive (sorted) Ritz values closer than tau
    lab = cumsum([1; diff(X.theta) > tau]);
    X.cluster = lab;
    cl = unique(lab(X.ghost));
    X.clusters = struct('theta', {}, 'members', {}, 'w_total', {});
    for c = cl'
        idx = find(lab == c);
        if numel(idx) < 2, continue; end      % isolated ghost (not a cluster)
        X.clusters(end + 1) = struct('theta', mean(X.theta(idx)), 'members', numel(idx), ...
                                     'w_total', sum(X.w(idx)));
    end
    S.(name{1}) = X;
    fprintf('%s: %d Ritz values, %d ghosts, %d clusters\n', name{1}, numel(X.theta), ...
        nnz(X.ghost), numel(X.clusters));
    for c = 1 : numel(X.clusters)
        fprintf('   cluster at theta = %9.4f: %d members, total weight %.6e\n', ...
            X.clusters(c).theta, X.clusters(c).members, X.clusters(c).w_total);
    end
    iso = find(X.ghost & arrayfun(@(i) nnz(lab == lab(i)) == 1, (1:numel(lab))'));
    for i = iso'
        fprintf('   isolated ghost: theta = %.4f, w = %.2e\n', X.theta(i), X.w(i));
    end
end
fprintf('tau = %.3e (C_tau = %g, W_T = %.3f)\n', tau, o.Ctau, W_T);
save(fullfile(o.OutDir, 'study_ghosts.mat'), '-struct', 'S');
end
