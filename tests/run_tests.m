function run_tests(varargin)
%RUN_TESTS  Correctness tests of the FTLM kernels and MATLAB front end.
%   RUN_TESTS runs all tests (GPU tests are skipped without a GPU).
%   RUN_TESTS('cpu') skips the GPU tests.
%
%   Tests:
%     1. basis enumeration and CR tables vs. brute force (mixed spins)
%     2. SpMV of every kernel variant vs. the explicit sparse Hamiltonian
%        (random couplings between arbitrary pairs, mixed spins), and
%        identical basis order of CLT and CR
%     3. Lanczos/weights pipeline: with N_L = dim the FTLM estimate of
%        sum_r <r|exp(-beta H)|r> equals the exact value for the same
%        start vectors
%     4. sum rule sum(w) = prod(2 s_i + 1) including half-integer S_max
%     5. Lanczos breakdown in a degenerate sector (no NaN, early stop)
%     6. full run: FTLM vs. exact diagonalization (mixed-spin model)
%     7. input validation

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

root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
use_gpu = ~any(strcmp(varargin, 'cpu')) && gpu_available();
if ~use_gpu
    fprintf('GPU tests skipped.\n');
end

tests = {@test_basis, @test_spmv, @test_lanczos_exact, @test_sum_rule, ...
         @test_breakdown, @test_ftlm_vs_ed, @test_validation};
n_fail = 0;
for k = 1 : numel(tests)
    name = func2str(tests{k});
    try
        tests{k}(use_gpu);
        fprintf('[PASS] %s\n', name);
    catch ME
        n_fail = n_fail + 1;
        fprintf('[FAIL] %s: %s\n', name, ME.message);
    end
end
if n_fail > 0
    error('run_tests:failed', '%d of %d tests failed.', n_fail, numel(tests));
end
fprintf('All %d tests passed.\n', numel(tests));
end

%% ------------------------------------------------------------------------
function m = mixed_model()
% 7 sites with mixed spins, couplings between all pairs (random signs)
    rs = RandStream('mt19937ar', 'Seed', 1);
    N = 7;
    spins = [0.5, 1, 1.5, 0.5, 1, 2, 0.5];
    [I, J] = find(triu(ones(N), 1));
    Jv = randn(rs, numel(I), 1);
    m = ftlm.model(struct('spins', spins, 'couplings', [I, J, Jv]));
end

function test_basis(~)
    m = mixed_model();
    labels = (0 : m.D_full - 1)';
    digsum = zeros(size(labels));
    tmp = labels;
    for k = 1 : m.N
        dk = mod(tmp, m.radix(k));
        digsum = digsum + dk;
        tmp = (tmp - dk) / m.radix(k);
    end
    secs = ftlm.sectors(m);
    for q = 1 : numel(secs)
        b = ftlm.enumerate_sector(m, secs(q).A);
        ref = labels(digsum == secs(q).A);
        assert(isequal(double(b), ref), 'basis mismatch in sector A = %d', secs(q).A);
        assert(numel(b) == secs(q).dim, 'sector dimension mismatch');
        [~, ~, ~, dim] = ftlm.cr_tables(m, secs(q).A);
        assert(dim == secs(q).dim, 'CR dimension mismatch');
    end
end

function test_spmv(use_gpu)
    m   = mixed_model();
    sec = ftlm.sectors(m);
    sec = sec(1);                                   % largest sector
    basis = ftlm.enumerate_sector(m, sec.A);
    H = ftlm.hamiltonian(m, basis);
    assert(norm(H - H', 1) < 1e-12, 'explicit H not symmetric');
    rs = RandStream('mt19937ar', 'Seed', 2);
    V  = randn(rs, sec.dim, 3);
    W0 = H * V;
    nrm = norm(W0, 'fro');

    % tolerance: c * u * ||H||_1 * ||V|| / ||HV||
    scale = norm(H, 1) * norm(V, 'fro') / nrm;
    variants = {'cpu', 'clt', 'double', 2^-53;  'cpu', 'clt', 'single', 2^-24};
    if use_gpu
        for lk = {'clt', 'cr'}
            variants(end+1, :) = {'gpu', lk{1}, 'double',   2^-53}; %#ok<AGROW>
            variants(end+1, :) = {'gpu', lk{1}, 'single',   2^-24}; %#ok<AGROW>
            variants(end+1, :) = {'gpu', lk{1}, 'half',     2^-11}; %#ok<AGROW>
            variants(end+1, :) = {'gpu', lk{1}, 'bfloat16', 2^-8};  %#ok<AGROW>
        end
    end
    W_clt = struct();
    for v = 1 : size(variants, 1)
        [be, lk, pr, u] = variants{v, :};
        cfg = ftlm.kernel_config(m, sec.A, lk, pr, 3, basis);
        f = str2func(sprintf('ftlm_%s_mex', be));
        f('init', cfg);
        W = f('spmv', V);
        f('cleanup');
        err = norm(W - W0, 'fro') / nrm;
        assert(err < 50 * u * scale, '%s/%s/%s: relative SpMV error %.2e', be, lk, pr, err);
        key = [be '_' pr];
        if strcmp(lk, 'clt')
            W_clt.(key) = W;
        elseif isfield(W_clt, key)
            % CR must act on the identically ordered basis
            d = norm(W - W_clt.(key), 'fro') / nrm;
            assert(d < 50 * u * scale, 'CLT/CR mismatch (%s): %.2e', pr, d);
        end
    end
end

function test_lanczos_exact(use_gpu)
% With N_L >= dim (full Krylov space) the Gauss quadrature is exact:
% sum_k w_k exp(-beta theta_k) = <r|exp(-beta H)|r> / <r|r>.
    m   = mixed_model();
    secs = ftlm.sectors(m);
    sec = secs(end - 2);                            % small sector
    basis = ftlm.enumerate_sector(m, sec.A);
    [U, E] = eig(full(ftlm.hamiltonian(m, basis)), 'vector');
    R = 5;
    rng(sec.dim, 'twister');                        % seed = 0 convention
    V = randn(sec.dim, R);
    betas = [0.1, 1, 5];
    exact = zeros(size(betas));
    for ib = 1 : numel(betas)
        c2 = (U' * V).^2 ./ sum(V.^2, 1);
        exact(ib) = sum(c2' * exp(-betas(ib) * E));
    end
    backends = {'cpu'};
    if use_gpu, backends{end+1} = 'gpu'; end
    for be = backends
        o = ftlm.defaults(struct('R', R, 'M_lz', sec.dim, 'T_range', 1, ...
            'backend', be{1}, 'precision', 'double', 'B_gpu', 2, 'B_cpu', 2));
        r = ftlm.sector_ftlm(m, sec, basis, o);
        for ib = 1 : numel(betas)
            est = (R / sec.dim) * sum(r.w .* exp(-betas(ib) * r.E));
            assert(abs(est - exact(ib)) < 1e-10 * exact(ib), ...
                '%s: beta = %g: %.15g vs %.15g', be{1}, betas(ib), est, exact(ib));
        end
    end
end

function test_sum_rule(use_gpu)
% sum of all weights = Z(beta = 0) = prod(2 s_i + 1); ring with odd N and
% s = 1/2 has half-integer S_max (all sectors M = 1/2, 3/2, ...).
    be = 'cpu';
    if use_gpu, be = 'gpu'; end
    cases = {struct('geometry', 'ring', 'N_ring', 9, 's_val', 0.5, 'J', 1), ...
             struct('spins', [0.5 1 0.5 1 0.5 1], ...
                    'couplings', [1 2 1; 2 3 1; 3 4 1; 4 5 1; 5 6 1; 6 1 1; 1 4 -0.3])};
    for c = 1 : numel(cases)
        o = cases{c};
        o.R = 3;  o.M_lz = 20;  o.T_range = [1e3, 1e9];  o.backend = be;
        o.ed_thresh = 20;  o.verbose = false;
        res = ftlm.run(o);
        m = res.model;
        Z_inf = res.Z_eff(end);        % exp(-beta (E - E0)) = 1 - O(1e-8)
        assert(abs(Z_inf - m.D_full) < 1e-6 * m.D_full, ...
            'case %d: sum rule %.10g vs %d', c, Z_inf, m.D_full);
        if c == 1
            assert(all(mod(res.sector_M, 1) == 0.5), 'half-integer M expected');
        end
    end
end

function test_breakdown(use_gpu)
% s = 1 icosahedron, M = 11: dim = 12, but the Krylov space of a random
% vector is limited by the number of distinct eigenvalues (< 12).
    m = ftlm.model(struct('geometry', 'ico', 's_val', 1, 'J', 1));
    secs = ftlm.sectors(m);
    sec = secs([secs.M] == 11);
    basis = ftlm.enumerate_sector(m, sec.A);
    n_dist = numel(uniquetol(eig(full(ftlm.hamiltonian(m, basis))), 1e-9));
    variants = {'cpu', 'double'; 'cpu', 'single'};
    if use_gpu
        variants = [variants; {'gpu', 'double'; 'gpu', 'single'; 'gpu', 'half'}];
    end
    for v = 1 : size(variants, 1)
        o = ftlm.defaults(struct('R', 12, 'M_lz', 100, 'T_range', 1, ...
            'backend', variants{v, 1}, 'precision', variants{v, 2}));
        r = ftlm.sector_ftlm(m, sec, basis, o);
        assert(all(isfinite(r.E)) && all(isfinite(r.w)), '%s/%s: non-finite output', variants{v, :});
        assert(max(r.nsteps) <= n_dist + 1, '%s/%s: no early stop (%d steps, %d distinct)', ...
               variants{v, 1}, variants{v, 2}, max(r.nsteps), n_dist);
        assert(abs(sum(r.w) - sec.dim) < 1e-6 * sec.dim, 'weight sum');
    end
end

function test_ftlm_vs_ed(use_gpu)
% statistical test: the stochastic FTLM error (R_eff = min(R, dim)) stays
% below ~3 % for T >= 0.5 (it is larger at lower T)
    base = struct('spins', [1 0.5 1 0.5 1 0.5 1 0.5], 'R', 200, 'M_lz', 60, ...
                  'T_range', linspace(0.5, 5, 25), 'verbose', false, ...
                  'couplings', [1 2 1; 2 3 1; 3 4 1; 4 5 1; 5 6 1; 6 7 1; 7 8 1; 8 1 1; ...
                                1 3 0.4; 5 7 0.4; 2 6 -0.2]);
    ed = base;  ed.ed_thresh = 1e4;  ed.backend = 'cpu';
    r_ed = ftlm.run(ed);
    ft = base;  ft.ed_thresh = 0;
    ft.backend = 'cpu';
    if use_gpu, ft.backend = 'gpu'; end
    r_ft = ftlm.run(ft);
    err = max(abs(r_ft.C_T - r_ed.C_T)) / max(r_ed.C_T);
    assert(err < 0.05, 'FTLM vs ED: relative C error %.3f', err);
    err = max(abs(r_ft.chi_T - r_ed.chi_T)) / max(r_ed.chi_T);
    assert(err < 0.05, 'FTLM vs ED: relative chi error %.3f', err);
end

function test_validation(~)
    bad = {struct('couplings', [1 1 1], 's_val', 0.5), ...
           struct('geometry', 'ico', 'spins', [0.5 0.5], 'J', 1), ...
           struct('couplings', [1 2 1], 'spins', [0.5 0.7]), ...
           struct('geometry', 'ico', 's_val', 1)};
    for k = 1 : numel(bad)
        threw = false;
        try
            ftlm.model(bad{k});
        catch
            threw = true;
        end
        assert(threw, 'invalid input %d accepted', k);
    end
    m = ftlm.model(struct('couplings', [1 2 1; 2 1 0.5; 2 3 0; 3 1 2], 's_val', 1));
    assert(isequal(m.couplings, [1 2 1.5; 3 1 2]), 'coupling merge/drop failed');
end

function ok = gpu_available()
    ok = false;
    try
        ok = gpuDeviceCount > 0 && exist('ftlm_gpu_mex', 'file') == 3;
    catch
    end
end
