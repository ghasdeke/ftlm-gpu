function out = sector_ftlm(model, sec, basis, opts)
%FTLM.SECTOR_FTLM  FTLM (block Lanczos) for one magnetization sector.
%   OUT = FTLM.SECTOR_FTLM(MODEL, SEC, BASIS, OPTS) runs R_eff =
%   min(R, dim) Lanczos chains of N_L,eff = min(M_lz, dim) steps in the
%   sector SEC (element of FTLM.SECTORS) and returns
%
%     OUT.E, OUT.w   Ritz values and FTLM weights (dim/R_eff)*|s_k1|^2
%                    (the M-multiplicity is NOT included)
%     OUT.B          block size used
%     OUT.t_lanczos  wall time of the Lanczos calls (s)
%     OUT.nsteps     1 x R_eff number of Lanczos steps per chain
%     OUT.alpha, OUT.beta   Lanczos coefficients (M_lz x R_eff, only if
%                    OPTS.save_ritz)
%
%   BASIS is the sorted sector basis (required for lookup 'clt' and for
%   the CPU backend; may be empty for 'cr').  OPTS fields used:
%     backend       'gpu' | 'cpu'
%     precision     GPU: 'single' | 'double' | 'half' | 'bfloat16'
%                   CPU: 'double' | 'single'
%     lookup        'clt' | 'cr'  (CPU: 'clt' only)
%     R, M_lz, seed, B_gpu, B_cpu, L2_cache_bytes, save_ritz
%
%   Start vectors: normally distributed, drawn on the host.  seed = 0
%   reproduces v1 (rng(dim, 'twister') per sector); seed > 0 uses
%   substream SEED of an mrg32k3a stream seeded with dim, which gives
%   statistically independent vectors for different seeds.

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

dim   = sec.dim;
R_eff = min(opts.R, dim);
M_eff = min(opts.M_lz, dim);
gpu   = strcmp(opts.backend, 'gpu');

%% ---- block size -------------------------------------------------------
elem = precision_bytes(opts.precision);
if gpu
    if opts.B_gpu == 0
        % keep three blocks of B = 8 vectors in L2 if possible (v1 heuristic)
        if 3 * dim * 8 * elem <= opts.L2_cache_bytes
            B = 8;
        else
            B = 4;
        end
    else
        B = opts.B_gpu;
    end
    B = min(B, R_eff);
    B = fit_gpu_memory(B, dim, elem, opts.lookup, model);
    mex_fun = @ftlm_gpu_mex;
else
    B = min(opts.B_cpu, R_eff);
    mex_fun = @ftlm_cpu_mex;
end

cfg = ftlm.kernel_config(model, sec.A, opts.lookup, opts.precision, B, basis);
assert(cfg.dim == dim, 'ftlm:sector_ftlm', 'dimension mismatch (%d vs %d).', cfg.dim, dim);
mex_fun('init', cfg);
% release the device/host buffers also on errors and Ctrl+C
guard = onCleanup(@() mex_fun('cleanup')); %#ok<NASGU>
clear cfg

%% ---- random start vectors ----------------------------------------------
if opts.seed == 0
    rng(dim, 'twister');
    draw = @(n, b) randn(n, b);
else
    rs = RandStream('mrg32k3a', 'Seed', dim);
    rs.Substream = opts.seed;
    draw = @(n, b) randn(rs, n, b);
end
single_in = gpu && ~strcmp(opts.precision, 'double');

E_all  = zeros(M_eff * R_eff, 1);
w_all  = zeros(M_eff * R_eff, 1);
nsteps = zeros(1, R_eff);
if opts.save_ritz
    out.alpha = zeros(M_eff, R_eff);
    out.beta  = zeros(M_eff, R_eff);
end
idx = 0;
t_lanczos = 0;

for r0 = 1 : B : R_eff
    r1 = min(r0 + B - 1, R_eff);
    V0 = draw(dim, r1 - r0 + 1);
    if single_in
        V0 = single(V0);
    end
    t0 = tic;
    [AL, BE, ns] = mex_fun('block_lanczos', V0, M_eff);
    t_lanczos = t_lanczos + toc(t0);
    clear V0

    for b = 1 : size(AL, 2)
        n_b = ns(b);
        [theta, q1] = ftlm.solve_tridiag(AL(1:n_b, b), BE(1:n_b-1, b));
        E_all(idx+1 : idx+n_b) = theta;
        w_all(idx+1 : idx+n_b) = (dim / R_eff) * q1;
        idx = idx + n_b;
        r = r0 + b - 1;
        nsteps(r) = n_b;
        if opts.save_ritz
            out.alpha(1:n_b, r) = AL(1:n_b, b);
            out.beta(1:n_b, r)  = BE(1:n_b, b);
        end
    end
end
mex_fun('cleanup');

out.E         = E_all(1:idx);
out.w         = w_all(1:idx);
out.B         = B;
out.t_lanczos = t_lanczos;
out.nsteps    = nsteps;
end

function n = precision_bytes(p)
    switch p
        case 'double',            n = 8;
        case 'single',            n = 4;
        case {'half', 'bfloat16'}, n = 2;
        otherwise, error('ftlm:precision', 'unknown precision ''%s''.', p);
    end
end

function B = fit_gpu_memory(B, dim, elem, lookup, model)
%FIT_GPU_MEMORY  Reduce B until the kernel buffers fit into free VRAM.
    g = gpuDevice;
    avail = 0.95 * g.AvailableMemory;
    fixed = 8 * dim;                                  % staging buffer
    if strcmp(lookup, 'clt')
        fixed = fixed + 4 * dim + 8 * ceil(model.D_full / 32);
    end
    while B > 1 && fixed + 3 * dim * B * elem > avail
        B = floor(B / 2);
    end
    need = fixed + 3 * dim * B * elem;
    assert(need <= avail, 'ftlm:memory', ...
        'Sector (dim = %d) needs %.2f GB of GPU memory, %.2f GB available.', ...
        dim, need / 1e9, avail / 1e9);
end
