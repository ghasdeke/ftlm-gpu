%% benchmark_ico_v1.m
%  ================================================================
%  Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
%  and Helmholtz-Zentrum Dresden-Rossendorf e.V.
%
%  Licensed under the Apache License, Version 2.0 (the "License");
%  you may not use this file except in compliance with the License.
%  You may obtain a copy of the License at
%
%      http://www.apache.org/licenses/LICENSE-2.0
%
%  Unless required by applicable law or agreed to in writing, software
%  distributed under the License is distributed on an "AS IS" BASIS,
%  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
%  See the License for the specific language governing permissions and
%  limitations under the License.
%  ================================================================
%  Performance measurement: icosahedron N=12, generic in the spin
%  quantum number s.  Full FTLM (all sectors M = 0..N*s).
%
%  Spin selector (s_val in the CONFIGURATION block below):
%    s_val = 0.5  -> d_loc=2,  n_total = 4 096
%    s_val = 1.0  -> d_loc=3,  n_total = 531 441         <- "S1"
%    s_val = 1.5  -> d_loc=4,  n_total = 16 777 216      <- "S2", intermediate-size regime
%    s_val = 2.0  -> d_loc=5,  n_total = 244 140 625     <- "S4", large sector
%  Constraint: N * ceil(log2(2s+1)) <= 64.  All OK for N=12.
%
%  5 methods, n_runs runs per method (first run discarded as warmup):
%    M1  CPU-blk-CLUT          cpu_lanczos_omp                FP64, OMP, B=8
%    M2  GPU-CLT-single        cuda_lanczos_clut              FP32
%    M3  GPU-CLT-block         cuda_lanczos_clut_block        FP32, B=adaptive
%    M4  GPU-CRank-single      cuda_lanczos_crank_Sr_general  FP32, B=1
%    M5  GPU-CRank-block       cuda_lanczos_crank_Sr_general  FP32, B=adaptive
%
%  For each run the following are recorded:
%    - wall-clock time of the hot Lanczos loop
%    - ground state energy E_0 = min(all_E)  (numerical sanity check)
%    - GPU memory peak (VRAM peak) via gpuDevice.AvailableMemory
%      directly after each kernel 'init' call (CPU methods: NaN)
%
%  Aggregation: median and standard deviation over the runs after
%  discarding the warmup.
%
%  Output:  benchmark_ico_s<tag>_v1_<timestamp>.mat
%           spin tag: '1o2' (s=1/2), '1', '3o2' (s=3/2), '2'.
%           Struct array results(im) with fields method/label/B/R/M_lz/n_runs/
%           t_wall_runs/t_wall_median/t_wall_std/E0_runs/vram_peak_mb_runs/vram_peak_mb
%
%  Required MEX files (on the MATLAB path):
%    cpu_lanczos_omp.mexw64
%    cuda_lanczos_clut.mexw64
%    cuda_lanczos_clut_block.mexw64
%    cuda_lanczos_crank_Sr_general.mexw64
%  Required helpers (defined as local functions at the end of this file):
%    icosahedron_adjacency, build_sector_basis, enumerate_popcount_states,
%    build_lookup, build_clut_arrays, build_heisenberg_sparse,
%    process_sector_cpu, solve_tridiag, lanczos_run, ftlm_heat_capacity
%  Required standalone function: format_num.m
%
%  History:
%    v1  21.05.2026  Renamed from benchmark_ico_s1_v1.m; s_val promoted
%                    to a proper selector (s in {1/2, 1, 3/2, 2}); output
%                    filename now contains the spin tag.  Includes the
%                    bug fix of 21.05.2026 in benchmark_omp_block_clut_seq
%                    (basis_omp = int32(basis) instead of int32(basis)-1).
%  ================================================================

clear; close all; clc;

%% ============================================================
%  CONFIGURATION
%  ============================================================
geometry       = 'ico';
s_val          = 1.5;          % 0.5 | 1.0 | 1.5 | 2.0  (spin quantum number)
J              = 1.0;
M_lanczos      = 100;
R_samples      = 24;
ed_thresh      = 1;          % sectors with dim<=ed_thresh handled by full ED (default 1: only dim=1)
n_runs         = 1;          % 1 warmup + 2 measurement runs (with n_runs=1: no warmup discard, a single measurement run)
L2_cache_bytes = 48e6;       % RTX 4000 Ada: 48 MB L2 cache
M0_only        = true;       % true: only M=0 sector (showcase mode, useful for large s/dim)
                             % false: full FTLM over all M = 0..N*s

% Spin tag for logs and output filename
if abs(s_val - 0.5) < 1e-9
    s_tag = '1o2';
elseif abs(s_val - 1.5) < 1e-9
    s_tag = '3o2';
else
    s_tag = sprintf('%g', s_val);   % '1', '2'
end

fprintf('=== Benchmark icosahedron s=%s, full FTLM, 5 methods ===\n\n', s_tag);

%% ============================================================
%  GEOMETRY + SPIN
%  ============================================================
N           = 12;
bonds       = icosahedron_adjacency();
geo_label   = 'Icosahedron (N=12, 30 bonds, 5-regular)';
n_bonds_geo = size(bonds, 1);
bonds_flat  = int32(reshape(bonds' - 1, [], 1));

two_s         = round(2 * s_val);
d_loc         = two_s + 1;             % 2 (s=1/2) | 3 (s=1) | 4 (s=3/2) | 5 (s=2)
n_total       = double(d_loc)^N;       % d_loc^12
bits_per_site = ceil(log2(d_loc));
n_bits_total  = N * bits_per_site;
assert(n_bits_total <= 64, 'Bit-packing constraint violated.');

fprintf('Geometry: %s\n', geo_label);
fprintf('Spin:     s = %g, d_loc = %d, n_total = %d^%d = %s\n', ...
    s_val, d_loc, d_loc, N, format_num(n_total));
fprintf('R = %d, M_lz = %d, runs per method = %d (warmup discarded)\n\n', ...
    R_samples, M_lanczos, n_runs);

%% ============================================================
%  GPU + MEX
%  ============================================================
assert(gpuDeviceCount > 0, 'No GPU found!');
gpu = gpuDevice;
fprintf('GPU: %s  (%.1f GB VRAM, Compute %s)\n', ...
    gpu.Name, gpu.TotalMemory/1e9, gpu.ComputeCapability);

req_mex = {'cpu_lanczos_omp', 'cuda_lanczos_clut', ...
           'cuda_lanczos_clut_block', 'cuda_lanczos_crank_Sr_general'};
for k = 1:length(req_mex)
    assert(exist(req_mex{k}, 'file') == 3, ...
        'Required MEX file missing: %s', req_mex{k});
end
n_omp = cpu_lanczos_omp('info');
fprintf('CPU OMP threads: %d\n\n', n_omp);

%% ============================================================
%  METHODS
%  ============================================================
%  methods_cfg: {mname, label, B_arg}
%    mname    = key for the switch statement
%    label    = string for output/storage
%    B_arg    = numeric block parameter (0 = adaptive, 1 = single vector,
%               NaN = not applicable / method sets B internally)
methods_cfg = {
    'cpu_omp_clut',       sprintf('CPU-blk-CLUT (B=8, FP64, %dT)', n_omp),  NaN ;
    'cuda_clut',          'GPU-CLT-single (FP32)',                          NaN ;
    'cuda_clut_blk',      'GPU-CLT-block (B=auto, FP32)',                   0   ;
    'cuda_crank_Sr_gen',  'GPU-CRank-single (B=1, FP32)',                   1   ;
    'cuda_crank_Sr_gen',  'GPU-CRank-block (B=auto, FP32)',                 0   ;
};
n_methods = size(methods_cfg, 1);

%% ============================================================
%  SECTORS
%  ============================================================
M_max = round(N * s_val);
if M0_only
    M_range = 0;
    fprintf('Preparing sectors (M=0 only, showcase mode)...');
else
    M_range = 0:M_max;
    fprintf('Preparing sectors (full FTLM, all M = 0..%d)...', M_max);
end
tic;

sec_M     = [];
sec_basis = {};
sec_dim   = [];
sec_mult  = [];

for M = M_range
    basis_M = build_sector_basis(N, s_val, d_loc, n_total, M);
    dim_M = length(basis_M);
    if dim_M == 0, continue; end
    sec_M(end+1)     = M;            %#ok<SAGROW>
    sec_basis{end+1} = basis_M;      %#ok<SAGROW>
    sec_dim(end+1)   = dim_M;        %#ok<SAGROW>
    sec_mult(end+1)  = 1 + (M > 0);  %#ok<SAGROW>
end
n_sec = length(sec_M);

fprintf(' %d sectors, dim_max = %s, sum(dim) = %s  (%.1f s)\n\n', ...
    n_sec, format_num(max(sec_dim)), format_num(sum(sec_dim)), toc);

%% ============================================================
%  MAIN LOOP: METHODS x RUNS
%  ============================================================
results(n_methods) = struct( ...
    'method', '', 'label', '', 'B', NaN, 'R', NaN, 'M_lz', NaN, ...
    'n_runs', NaN, 't_wall_runs', [], 't_wall_median', NaN, ...
    't_wall_std', NaN, 'E0_runs', [], 'vram_peak_mb_runs', [], ...
    'vram_peak_mb', NaN);

for im = 1:n_methods
    mname  = methods_cfg{im, 1};
    mlabel = methods_cfg{im, 2};
    B_arg  = methods_cfg{im, 3};

    fprintf('============================================================\n');
    fprintf('  M%d: %s\n', im, mlabel);
    fprintf('============================================================\n');

    t_runs    = zeros(n_runs, 1);
    E0_runs   = zeros(n_runs, 1);
    vram_runs = zeros(n_runs, 1);

    for r = 1:n_runs
        fprintf('  Run %d/%d ', r, n_runs);

        switch mname
            case 'cpu_omp_clut'
                [all_E, all_w, t, peak_vram] = ...
                    benchmark_omp_block_clut_seq( ...
                        sec_basis, sec_dim, sec_mult, n_sec, ...
                        bonds, bonds_flat, s_val, J, N, n_total, d_loc, ...
                        M_lanczos, R_samples, ed_thresh);

            case 'cuda_clut'
                [all_E, all_w, t, peak_vram] = ...
                    benchmark_cuda_clut_seq( ...
                        sec_basis, sec_dim, sec_mult, n_sec, ...
                        bonds, bonds_flat, s_val, J, N, n_total, d_loc, ...
                        M_lanczos, R_samples, ed_thresh);

            case 'cuda_clut_blk'
                [all_E, all_w, t, peak_vram] = ...
                    benchmark_cuda_clut_block_seq( ...
                        sec_basis, sec_dim, sec_mult, n_sec, ...
                        bonds, bonds_flat, s_val, J, N, n_total, d_loc, ...
                        M_lanczos, R_samples, ed_thresh, B_arg, L2_cache_bytes);

            case 'cuda_crank_Sr_gen'
                [all_E, all_w, t, peak_vram] = ...
                    benchmark_cuda_crank_Sr_gen_seq( ...
                        sec_basis, sec_dim, sec_mult, sec_M, n_sec, ...
                        bonds, bonds_flat, s_val, J, N, n_total, ...
                        M_lanczos, R_samples, ed_thresh, B_arg, L2_cache_bytes);
        end

        t_runs(r)   = t;
        E0_runs(r)  = min(all_E);
        if isnan(peak_vram)
            vram_runs(r) = NaN;
        else
            vram_runs(r) = peak_vram / 1e6;   % bytes -> MB
        end

        fprintf('  -> t = %.2f s, E0 = %.6f, VRAM = ', t, E0_runs(r));
        if isnan(vram_runs(r)), fprintf('—\n'); else, fprintf('%.0f MB\n', vram_runs(r)); end
    end

    % Aggregation (warmup = run 1 discarded)
    t_keep    = t_runs(2:end);
    vram_keep = vram_runs(2:end);

    results(im).method            = mname;
    results(im).label             = mlabel;
    results(im).B                 = B_arg;
    results(im).R                 = R_samples;
    results(im).M_lz              = M_lanczos;
    results(im).n_runs            = n_runs;
    results(im).t_wall_runs       = t_runs;
    results(im).t_wall_median     = median(t_keep);
    results(im).t_wall_std        = std(t_keep);
    results(im).E0_runs           = E0_runs;
    results(im).vram_peak_mb_runs = vram_runs;
    if all(isnan(vram_keep))
        results(im).vram_peak_mb  = NaN;
    else
        results(im).vram_peak_mb  = max(vram_keep);
    end

    fprintf('  Aggregate: median = %.2f s, std = %.2f s, E0 = %.6f, VRAM_peak = ', ...
        results(im).t_wall_median, results(im).t_wall_std, E0_runs(end));
    if isnan(results(im).vram_peak_mb)
        fprintf('—\n\n');
    else
        fprintf('%.0f MB\n\n', results(im).vram_peak_mb);
    end
end

%% ============================================================
%  E_0 SANITY CHECK (CPU FP64 vs. GPU FP32)
%  ============================================================
fprintf('============================================================\n');
fprintf('  E_0 consistency (CPU FP64 vs. GPU FP32)\n');
fprintf('============================================================\n');
E0_cpu = results(1).E0_runs(end);
fprintf('  CPU FP64 reference:          E0 = %.10f\n', E0_cpu);
for im = 2:n_methods
    E0_m = results(im).E0_runs(end);
    rel  = abs(E0_m - E0_cpu) / max(abs(E0_cpu), 1e-12);
    flag = '';
    if rel > 1e-5, flag = '  <-- above FP32 threshold, please check!'; end
    fprintf('  %-32s E0 = %.10f, rel.dev. = %.2e%s\n', ...
        results(im).label, E0_m, rel, flag);
end
fprintf('\n');

%% ============================================================
%  OUTPUT
%  ============================================================
ts = datestr(now, 'yyyymmdd_HHMM');
outfile = sprintf('benchmark_ico_s%s_v1_%s.mat', s_tag, ts);
save(outfile, 'results', 'geometry', 's_val', 's_tag', 'N', 'J', ...
              'R_samples', 'M_lanczos', 'L2_cache_bytes', 'M0_only', ...
              'sec_M', 'sec_dim', 'sec_mult', 'n_sec', 'n_runs');
fprintf('Results saved: %s\n\n', outfile);

%% ============================================================
%  SUMMARY
%  ============================================================
fprintf('============================================================\n');
fprintf('  SUMMARY  (all sectors, dim_max = %s)\n', ...
    format_num(max(sec_dim)));
fprintf('============================================================\n');
fprintf('  %-32s  %10s  %9s  %10s\n', 'Method', 't_med (s)', 't_std', 'VRAM (MB)');
fprintf('  %s\n', repmat('-', 1, 66));
for im = 1:n_methods
    if isnan(results(im).vram_peak_mb)
        fprintf('  %-32s  %10.2f  %9.2f  %10s\n', ...
            results(im).label, results(im).t_wall_median, ...
            results(im).t_wall_std, '—');
    else
        fprintf('  %-32s  %10.2f  %9.2f  %10.0f\n', ...
            results(im).label, results(im).t_wall_median, ...
            results(im).t_wall_std, results(im).vram_peak_mb);
    end
end
fprintf('  %s\n', repmat('-', 1, 66));
fprintf('\nDone.\n');

%% ============================================================
%  ============================================================
%  LOCAL FUNCTIONS  (taken from polyhedra_cpu_vs_gpu_v1.m,
%                    benchmark_* functions with VRAM peak capture)
%  ============================================================
%  ============================================================

function [all_E, all_w, t_total, peak_vram_bytes] = benchmark_omp_block_clut_seq( ...
    sec_basis, sec_dim, sec_mult, n_sec, ...
    bonds, bonds_flat, s, J, N, n_total, d_loc, ...
    M_lanczos, R_samples, ed_thresh)
%BENCHMARK_OMP_BLOCK_CLUT_SEQ  CPU OMP block Lanczos B=8, CLUT (FP64)

    BLOCK_SIZE = 32;
    B_batch = 8;
    n_large = sum(sec_dim > ed_thresh);
    i_large = 0;

    E_cell = cell(n_sec, 1);
    w_cell = cell(n_sec, 1);

    peak_vram_bytes = NaN;  % CPU method, no VRAM

    tic_total = tic;
    for k = 1:n_sec
        basis = sec_basis{k};
        dim   = sec_dim(k);
        mult  = sec_mult(k);

        M_lz = min(M_lanczos, dim);
        R    = min(R_samples, dim);
        rng(dim);

        if dim <= ed_thresh
            lookup_1b = build_lookup(basis, dim, n_total);
            H = build_heisenberg_sparse(basis, lookup_1b, bonds, ...
                s, J, N, n_total);
            E_sec = sort(eig(full(H)));
            E_cell{k} = E_sec;
            w_cell{k} = mult * ones(dim, 1);
        else
            i_large = i_large + 1;
            tic_sec = tic;
            n_blocks_lz = ceil(R / B_batch);
            fprintf('\n    Sec. %d/%d (dim=%s): ', i_large, n_large, ...
                format_num(dim));

            % Build CLT
            [block_base, block_mask] = build_clut_arrays(basis, n_total, BLOCK_SIZE);

            basis_omp = int32(basis);   % FIX 2026-05-21: removed -1 (off-by-one in MEX cpu_lanczos_omp SpMV during radix-d decoding)

            cpu_lanczos_omp('init_clut', ...
                int32(block_base), block_mask, basis_omp, ...
                bonds_flat, N, d_loc, s, J, dim);

            V0_all = randn(dim, R);

            E_k = zeros(M_lz * R, 1);
            w_k = zeros(M_lz * R, 1);
            idx = 0;

            i_blk = 0;
            for r_start = 1:B_batch:R
                r_end = min(r_start + B_batch - 1, R);
                B = r_end - r_start + 1;
                i_blk = i_blk + 1;

                V0_blk = V0_all(:, r_start:r_end);
                [AL, BE] = cpu_lanczos_omp('block_lanczos_clut', V0_blk, M_lz);

                for b = 1:B
                    [eps_l, Q1sq] = solve_tridiag(AL(:,b), BE(:,b));
                    n_l = length(eps_l);
                    E_k(idx+1 : idx+n_l) = eps_l;
                    w_k(idx+1 : idx+n_l) = mult * (dim / R) * Q1sq;
                    idx = idx + n_l;
                end

                fprintf('Blk %d/%d ', i_blk, n_blocks_lz);
            end

            cpu_lanczos_omp('cleanup');
            fprintf(' (%.1f s, cum. %.0f s)', toc(tic_sec), toc(tic_total));

            E_cell{k} = E_k(1:idx);
            w_cell{k} = w_k(1:idx);
        end
    end
    t_total = toc(tic_total);
    if n_large > 0, fprintf('\n  '); end

    all_E = vertcat(E_cell{:});
    all_w = vertcat(w_cell{:});
end

%% ================================================================

function [all_E, all_w, t_total, peak_vram_bytes] = benchmark_cuda_clut_seq( ...
    sec_basis, sec_dim, sec_mult, n_sec, ...
    bonds, bonds_flat, s, J, N, n_total, d_loc, ...
    M_lanczos, R_samples, ed_thresh)
%BENCHMARK_CUDA_CLUT_SEQ  CUDA single vector with CLUT

    BLOCK_SIZE = 32;

    E_cell = cell(n_sec, 1);
    w_cell = cell(n_sec, 1);

    n_large = sum(sec_dim > ed_thresh);
    i_large = 0;

    peak_vram_bytes = 0;
    gpu_h = gpuDevice;
    tot_vram = gpu_h.TotalMemory;

    tic_total = tic;
    for k = 1:n_sec
        basis = sec_basis{k};
        dim   = sec_dim(k);
        mult  = sec_mult(k);

        M_lz = min(M_lanczos, dim);
        R    = min(R_samples, dim);
        rng(dim);

        if dim <= ed_thresh
            lookup_1b = build_lookup(basis, dim, n_total);
            H = build_heisenberg_sparse(basis, lookup_1b, bonds, ...
                s, J, N, n_total);
            E_sec = sort(eig(full(H)));
            E_cell{k} = E_sec;
            w_cell{k} = mult * ones(dim, 1);
        else
            i_large = i_large + 1;
            tic_sec = tic;
            fprintf('\n    Sec. %d/%d (dim=%s): ', i_large, n_large, ...
                format_num(dim));

            % Build CLT
            [block_base, block_mask] = build_clut_arrays(basis, n_total, BLOCK_SIZE);

            basis_cuda = int32(basis);

            cuda_lanczos_clut('init', ...
                gpuArray(block_base), gpuArray(block_mask), ...
                gpuArray(basis_cuda), ...
                bonds_flat, N, d_loc, s, J, dim);
            wait(gpu_h);
            used_now = tot_vram - gpu_h.AvailableMemory;
            if used_now > peak_vram_bytes, peak_vram_bytes = used_now; end

            E_k = zeros(M_lz * R, 1);
            w_k = zeros(M_lz * R, 1);
            idx = 0;

            for r = 1:R
                v0 = gpuArray(single(randn(dim, 1)));
                [al, be] = cuda_lanczos_clut('lanczos', v0, M_lz);
                [eps_l, Q1sq] = solve_tridiag(al, be);
                n_l = length(eps_l);

                E_k(idx+1 : idx+n_l) = eps_l;
                w_k(idx+1 : idx+n_l) = mult * (dim / R) * Q1sq;
                idx = idx + n_l;

                if mod(r, 10) == 0 || r == R
                    fprintf('%d/%d ', r, R);
                end
            end

            cuda_lanczos_clut('cleanup');
            fprintf(' (%.1f s, cum. %.0f s)', toc(tic_sec), toc(tic_total));

            E_cell{k} = E_k(1:idx);
            w_cell{k} = w_k(1:idx);
        end
    end
    wait(gpuDevice);
    t_total = toc(tic_total);
    if n_large > 0, fprintf('\n  '); end

    all_E = vertcat(E_cell{:});
    all_w = vertcat(w_cell{:});
end

%% ================================================================

function [all_E, all_w, t_total, peak_vram_bytes] = benchmark_cuda_clut_block_seq( ...
    sec_basis, sec_dim, sec_mult, n_sec, ...
    bonds, bonds_flat, s, J, N, n_total, d_loc, ...
    M_lanczos, R_samples, ed_thresh, B_batch, L2_cache)
%BENCHMARK_CUDA_CLUT_BLOCK_SEQ  GPU block Lanczos with CLT (generic)

    if nargin < 17 || isempty(L2_cache), L2_cache = 48e6; end
    adaptive_B = (B_batch == 0);

    BLOCK_SIZE = 32;

    E_cell = cell(n_sec, 1);
    w_cell = cell(n_sec, 1);

    n_large = sum(sec_dim > ed_thresh);
    i_large = 0;

    peak_vram_bytes = 0;
    gpu_h = gpuDevice;
    tot_vram = gpu_h.TotalMemory;

    tic_total = tic;
    for k = 1:n_sec
        basis = sec_basis{k};
        dim   = sec_dim(k);
        mult  = sec_mult(k);

        M_lz = min(M_lanczos, dim);
        R    = min(R_samples, dim);
        rng(dim);

        if dim <= ed_thresh
            lookup_1b = build_lookup(basis, dim, n_total);
            H = build_heisenberg_sparse(basis, lookup_1b, bonds, ...
                s, J, N, n_total);
            E_sec = sort(eig(full(H)));
            E_cell{k} = E_sec;
            w_cell{k} = mult * ones(dim, 1);
        else
            i_large = i_large + 1;
            tic_sec = tic;

            % Adaptive choice of B
            if adaptive_B
                mem_B8 = 3 * dim * 8 * 4;
                if mem_B8 <= L2_cache
                    B_batch_k = 8;
                else
                    B_batch_k = 4;
                end
                B_batch_k = min(B_batch_k, R);
            else
                B_batch_k = min(B_batch, R);
            end

            n_blocks_lz = ceil(R / B_batch_k);
            fprintf('\n    Sec. %d/%d (dim=%s, B=%d): ', i_large, n_large, ...
                format_num(dim), B_batch_k);

            % Build CLT
            [block_base, block_mask] = build_clut_arrays(basis, n_total, BLOCK_SIZE);

            basis_cuda = int32(basis);

            cuda_lanczos_clut_block('init', ...
                gpuArray(block_base), gpuArray(block_mask), ...
                gpuArray(basis_cuda), ...
                bonds_flat, N, d_loc, s, J, dim, B_batch_k);
            wait(gpu_h);
            used_now = tot_vram - gpu_h.AvailableMemory;
            if used_now > peak_vram_bytes, peak_vram_bytes = used_now; end

            V0_all = single(randn(dim, R));

            E_k = zeros(M_lz * R, 1);
            w_k = zeros(M_lz * R, 1);
            idx = 0;

            i_blk = 0;
            for r_start = 1:B_batch_k:R
                r_end = min(r_start + B_batch_k - 1, R);
                B = r_end - r_start + 1;
                i_blk = i_blk + 1;

                V0_blk = gpuArray(V0_all(:, r_start:r_end));
                [AL, BE] = cuda_lanczos_clut_block('block_lanczos', ...
                    V0_blk, M_lz);

                for b = 1:B
                    [eps_l, Q1sq] = solve_tridiag(AL(:,b), BE(:,b));
                    n_l = length(eps_l);
                    E_k(idx+1 : idx+n_l) = eps_l;
                    w_k(idx+1 : idx+n_l) = mult * (dim / R) * Q1sq;
                    idx = idx + n_l;
                end

                fprintf('Blk %d/%d ', i_blk, n_blocks_lz);
            end

            cuda_lanczos_clut_block('cleanup');
            fprintf(' (%.1f s, cum. %.0f s)', toc(tic_sec), toc(tic_total));

            E_cell{k} = E_k(1:idx);
            w_cell{k} = w_k(1:idx);
        end
    end
    wait(gpuDevice);
    t_total = toc(tic_total);
    if n_large > 0, fprintf('\n  '); end

    all_E = vertcat(E_cell{:});
    all_w = vertcat(w_cell{:});
end

%% ================================================================

function [all_E, all_w, t_total, peak_vram_bytes] = benchmark_cuda_crank_Sr_gen_seq( ...
    sec_basis, sec_dim, sec_mult, sec_M, n_sec, ...
    bonds, bonds_flat, s, J, N, n_total, ...
    M_lanczos, R_samples, ed_thresh, B_batch, L2_cache)
%BENCHMARK_CUDA_CRANK_SR_GEN_SEQ  GPU block Lanczos, generalized crank (s>=1/2)

    if nargin < 16 || isempty(L2_cache), L2_cache = 48e6; end
    adaptive_B = (B_batch == 0);

    two_s = round(2 * s);

    E_cell = cell(n_sec, 1);
    w_cell = cell(n_sec, 1);

    n_large = sum(sec_dim > ed_thresh);
    i_large = 0;

    peak_vram_bytes = 0;
    gpu_h = gpuDevice;
    tot_vram = gpu_h.TotalMemory;

    tic_total = tic;
    for k = 1:n_sec
        basis = sec_basis{k};
        dim   = sec_dim(k);
        mult  = sec_mult(k);
        M_val = sec_M(k);

        % A_total = N*s - M  (sum of digits in the sector)
        A_total = round(N * s - M_val);

        M_lz = min(M_lanczos, dim);
        R    = min(R_samples, dim);
        rng(dim);

        if dim <= ed_thresh
            lookup_1b = build_lookup(basis, dim, n_total);
            H = build_heisenberg_sparse(basis, lookup_1b, bonds, ...
                s, J, N, n_total);
            E_sec = sort(eig(full(H)));
            E_cell{k} = E_sec;
            w_cell{k} = mult * ones(dim, 1);
            continue;
        end

        i_large = i_large + 1;

        if adaptive_B
            mem_B8 = 3 * dim * 8 * 4;
            if mem_B8 <= L2_cache
                B_batch_k = 8;
            else
                vram_B8 = 4 * dim * 8 * 4;
                if vram_B8 <= 17e9
                    B_batch_k = 8;
                else
                    B_batch_k = 4;
                end
            end
            B_batch_k = min(B_batch_k, R);
        else
            B_batch_k = min(B_batch, R);
        end

        n_blocks_lz = ceil(R / B_batch_k);
        fprintf('\n    Sec. %d/%d (dim=%s, B=%d, CRank-Gen 2s=%d): ', ...
            i_large, n_large, format_num(dim), B_batch_k, two_s);

        cuda_lanczos_crank_Sr_general('init', N, two_s, A_total, ...
            bonds_flat, J, dim, B_batch_k);
        wait(gpu_h);
        used_now = tot_vram - gpu_h.AvailableMemory;
        if used_now > peak_vram_bytes, peak_vram_bytes = used_now; end

        V0_all = single(randn(dim, R));

        E_k = zeros(M_lz * R, 1);
        w_k = zeros(M_lz * R, 1);
        idx = 0;

        i_blk = 0;
        for r_start = 1:B_batch_k:R
            r_end = min(r_start + B_batch_k - 1, R);
            B = r_end - r_start + 1;
            i_blk = i_blk + 1;

            V0_blk = gpuArray(V0_all(:, r_start:r_end));
            [AL, BE] = cuda_lanczos_crank_Sr_general('block_lanczos', ...
                V0_blk, M_lz);

            for b = 1:B
                [eps_l, Q1sq] = solve_tridiag(AL(:,b), BE(:,b));
                n_l = length(eps_l);
                E_k(idx+1 : idx+n_l) = eps_l;
                w_k(idx+1 : idx+n_l) = mult * (dim / R) * Q1sq;
                idx = idx + n_l;
            end

            fprintf('Blk %d/%d ', i_blk, n_blocks_lz);
        end

        cuda_lanczos_crank_Sr_general('cleanup');
        t_sec = toc(tic_total);
        fprintf(' (cum. %.0f s)', t_sec);

        E_cell{k} = E_k(1:idx);
        w_cell{k} = w_k(1:idx);
    end
    wait(gpuDevice);
    t_total = toc(tic_total);
    if n_large > 0, fprintf('\n  '); end

    all_E = vertcat(E_cell{:});
    all_w = vertcat(w_cell{:});
end

%% ================================================================
%  ================================================================
%  PHYSICS FUNCTIONS
%  ================================================================
%  ================================================================

%% ================================================================

function [E, w] = process_sector_cpu(basis, dim, mult, ...
    bonds, s, J, N, n_total, M_lanczos, R_samples, ed_thresh)
%PROCESS_SECTOR_CPU  Process a single sector on the CPU (sparse H, FP64)

    lookup = build_lookup(basis, dim, n_total);

    M_lz = min(M_lanczos, dim);
    R    = min(R_samples, dim);
    rng(dim);

    H = build_heisenberg_sparse(basis, lookup, bonds, s, J, N, n_total);

    if dim <= ed_thresh
        E_sec = sort(eig(full(H)));
        E = E_sec;
        w = mult * ones(dim, 1);
    else
        E = zeros(M_lz * R, 1);
        w = zeros(M_lz * R, 1);
        idx = 0;
        for r = 1:R
            v0 = randn(dim, 1);
            [eps_l, Q1sq] = lanczos_run(H, v0, M_lz);
            n_l = length(eps_l);
            E(idx+1 : idx+n_l) = eps_l;
            w(idx+1 : idx+n_l) = mult * (dim / R) * Q1sq;
            idx = idx + n_l;
        end
        E = E(1:idx);
        w = w(1:idx);
    end
end

%% ================================================================

function [eps_l, Q1sq] = solve_tridiag(alpha, beta)
%SOLVE_TRIDIAG  Solve the tridiagonal eigenvalue problem (CPU, FP64)

    n = length(alpha);
    T = diag(alpha(1:n));
    if n > 1
        T = T + diag(beta(1:n-1), 1) + diag(beta(1:n-1), -1);
    end
    [Q, D] = eig(T, 'vector');
    eps_l = D;
    Q1sq = abs(Q(1,:)').^2;
end

%% ================================================================

function [eps_l, Q1sq] = lanczos_run(H, v0, M_lz)
%LANCZOS_RUN  Lanczos iteration (sparse H, CPU, FP64)

    dim = length(v0);
    M_lz = min(M_lz, dim);
    v = v0 / norm(v0);
    v_prev = zeros(dim, 1);
    alpha = zeros(M_lz, 1);
    beta_off = zeros(M_lz, 1);
    actual_steps = M_lz;

    for j = 1:M_lz
        w = H * v;
        alpha(j) = v' * w;
        w = w - alpha(j) * v;
        if j > 1
            w = w - beta_off(j-1) * v_prev;
        end
        b = norm(w);
        if b < 1e-14
            actual_steps = j;
            break;
        end
        if j < M_lz
            beta_off(j) = b;
            v_prev = v;
            v = w / b;
        end
    end

    [eps_l, Q1sq] = solve_tridiag(alpha(1:actual_steps), ...
        beta_off(1:actual_steps));
end

%% ================================================================

function C = ftlm_heat_capacity(all_E, all_w, T_range)
%FTLM_HEAT_CAPACITY  Heat capacity from (E, w) pairs

    E_min = min(all_E);
    dE = all_E - E_min;
    nT = length(T_range);
    C = zeros(nT, 1);

    for iT = 1:nT
        beta = 1.0 / T_range(iT);
        boltz = all_w .* exp(-beta * dE);
        Z = sum(boltz);
        if Z < 1e-300, C(iT) = 0; continue; end
        E_avg  = sum(all_E .* boltz) / Z;
        E2_avg = sum(all_E.^2 .* boltz) / Z;
        C(iT) = beta^2 * (E2_avg - E_avg^2);
    end
end

%% ================================================================

function basis = build_sector_basis(N, s, d_loc, n_total, M_target)
%BUILD_SECTOR_BASIS  Basis states for sector M (Sz = M_target)
%
%  For s=1/2 a popcount-based enumeration is used (efficient even for
%  large N=30 via Gosper's hack).
%  For s>1/2 the state space is scanned block-wise and filtered by
%  digit sum.  For large n_total (s>1/2, N>~16) this becomes slow,
%  but here it is only needed once for the sector setup.

    if abs(s - 0.5) < 1e-9
        % s=1/2:  M = n_up - N/2  =>  n_up = N/2 + M
        n_up = N/2 + M_target;
        if n_up < 0 || n_up > N || abs(n_up - round(n_up)) > 1e-9
            basis = int64([]);
            return;
        end
        n_up = round(n_up);
        dim_M = nchoosek(N, n_up);
        basis = enumerate_popcount_states(N, n_up, dim_M);
        return;
    end

    % General case (s > 1/2): scan digit sums
    %   sector: sum_i digit_i = N*s - M_target  (digit in 0..2s)
    %   Sz = sum_i (digit_i - s) = sum digit - N*s
    A_total = round(N * s - M_target);
    if A_total < 0 || A_total > N * (d_loc - 1)
        basis = int64([]);
        return;
    end

    n_total_int = int64(n_total);  % careful with large values
    if double(n_total_int) ~= n_total
        error('n_total = %g too large for int64 encoding.', n_total);
    end

    blk = int64(5e6);
    basis_parts = {};
    for start = int64(0):blk:n_total_int-1
        stop = min(start + blk - 1, n_total_int - 1);
        states = (start:stop)';
        ns = length(states);
        Av = zeros(ns, 1);
        tmp = states;
        d_int = int64(d_loc);
        for ii = 1:N
            dg = double(mod(tmp, d_int));
            Av = Av + dg;
            tmp = (tmp - int64(dg)) / d_int;
        end
        idx = (round(Av) == A_total);
        if any(idx)
            basis_parts{end+1} = states(idx); %#ok<AGROW>
        end
    end
    if isempty(basis_parts)
        basis = int64([]);
    else
        basis = cat(1, basis_parts{:});
    end
end

%% ================================================================

function basis = enumerate_popcount_states(N, n_up, dim_expected)
%ENUMERATE_POPCOUNT_STATES  All N-bit integers with exactly n_up "1" bits
%  (s=1/2 only)

    if n_up == 0
        basis = int64(0);
        return;
    end
    if n_up == N
        basis = int64(2^N - 1);
        return;
    end

    if nargin < 3 || isempty(dim_expected)
        dim_expected = nchoosek(N, n_up);
    end

    pow2_vec = int64(2).^int64(0:N-1);

    if dim_expected <= 5e6
        % Small sectors: nchoosek + vectorized
        combos = nchoosek(1:N, n_up);
        basis = sum(pow2_vec(combos), 2);
        basis = sort(basis);
    else
        % Large sectors: Gosper's hack
        fprintf('[Gosper %s] ', format_num(dim_expected));
        basis = zeros(dim_expected, 1, 'int64');
        state = int64(2^n_up - 1);

        for i = 1:dim_expected
            basis(i) = state;
            if i == dim_expected, break; end

            c = bitand(state, -state);
            r = state + c;
            xr = bitxor(r, state);
            ctz_val = round(log2(double(c)));
            rest = bitshift(xr, -(ctz_val + 2));
            state = bitor(r, rest);
        end
    end
end

%% ================================================================

function lookup = build_lookup(basis, dim, lookup_size)
%BUILD_LOOKUP  Build state->index lookup table (1-based)
    lookup = zeros(lookup_size, 1, 'int32');
    if dim < 1e6
        for i = 1:dim
            lookup(basis(i) + 1) = int32(i);
        end
    else
        lookup(double(basis) + 1) = int32((1:dim)');
    end
end

%% ================================================================

function [block_base, block_mask] = build_clut_arrays(basis, n_total, BLOCK_SIZE)
%BUILD_CLUT_ARRAYS  Build the compressed lookup table (CLUT)
    n_blocks = ceil(n_total / BLOCK_SIZE);
    states = double(basis(:));

    blks = floor(states / BLOCK_SIZE) + 1;
    bits = mod(states, BLOCK_SIZE);

    block_base = int32(-ones(n_blocks, 1));
    [ub, fi] = unique(blks, 'first');
    block_base(ub) = int32(fi - 1);

    bit_vals = pow2(bits);
    mask_sums = accumarray(blks, bit_vals, [n_blocks, 1]);
    block_mask = uint32(mask_sums);
end

%% ================================================================

function H = build_heisenberg_sparse(basis, lookup, bonds, s, J, N, ...
    lookup_size)
%BUILD_HEISENBERG_SPARSE  Sparse Heisenberg Hamiltonian (generic in s)

    d_loc = round(2*s + 1);
    dim = length(basis);
    n_bonds = size(bonds, 1);
    powers = int64(d_loc).^int64((0:N-1)');

    mi = zeros(dim, N);
    temp = int64(basis);
    for site = 1:N
        digit = double(mod(temp, int64(d_loc)));
        mi(:, site) = digit - s;
        temp = (temp - int64(digit)) / int64(d_loc);
    end

    diag_vals = zeros(dim, 1);
    for b = 1:n_bonds
        diag_vals = diag_vals + J * mi(:, bonds(b,1)) .* mi(:, bonds(b,2));
    end

    row_list = (1:dim)';
    col_list = (1:dim)';
    val_list = diag_vals;

    for b = 1:n_bonds
        si = bonds(b,1); sj = bonds(b,2);

        % S+_i S-_j
        can_flip = (mi(:,si) < s - 1e-10) & (mi(:,sj) > -s + 1e-10);
        idx_from = find(can_flip);
        if ~isempty(idx_from)
            mi_i = mi(idx_from, si); mi_j = mi(idx_from, sj);
            coeffs = 0.5*J * sqrt(s*(s+1) - mi_i.*(mi_i+1)) .* ...
                             sqrt(s*(s+1) - mi_j.*(mi_j-1));
            new_states = basis(idx_from) + powers(si) - powers(sj);
            ns1 = double(new_states) + 1;
            ok = (ns1 >= 1) & (ns1 <= lookup_size);
            ni = zeros(length(idx_from), 1, 'int32');
            ni(ok) = lookup(ns1(ok));
            v = ni > 0;
            row_list = [row_list; double(ni(v))]; %#ok<AGROW>
            col_list = [col_list; double(idx_from(v))]; %#ok<AGROW>
            val_list = [val_list; coeffs(v)]; %#ok<AGROW>
        end

        % S-_i S+_j
        can_flip = (mi(:,si) > -s + 1e-10) & (mi(:,sj) < s - 1e-10);
        idx_from = find(can_flip);
        if ~isempty(idx_from)
            mi_i = mi(idx_from, si); mi_j = mi(idx_from, sj);
            coeffs = 0.5*J * sqrt(s*(s+1) - mi_i.*(mi_i-1)) .* ...
                             sqrt(s*(s+1) - mi_j.*(mi_j+1));
            new_states = basis(idx_from) - powers(si) + powers(sj);
            ns1 = double(new_states) + 1;
            ok = (ns1 >= 1) & (ns1 <= lookup_size);
            ni = zeros(length(idx_from), 1, 'int32');
            ni(ok) = lookup(ns1(ok));
            v = ni > 0;
            row_list = [row_list; double(ni(v))]; %#ok<AGROW>
            col_list = [col_list; double(idx_from(v))]; %#ok<AGROW>
            val_list = [val_list; coeffs(v)]; %#ok<AGROW>
        end
    end

    H = sparse(row_list, col_list, val_list, dim, dim);
end

%% ================================================================
%  GEOMETRY FUNCTIONS
%  ================================================================

%% ================================================================

function bonds = icosahedron_adjacency()
%ICOSAHEDRON_ADJACENCY  30 edges of the icosahedron.
    phi = (1+sqrt(5))/2;
    V = [0,1,phi; 0,1,-phi; 0,-1,phi; 0,-1,-phi;
         1,phi,0; 1,-phi,0; -1,phi,0; -1,-phi,0;
         phi,0,1; phi,0,-1; -phi,0,1; -phi,0,-1];
    bonds = [];
    for i = 1:12
        for j = i+1:12
            if abs(norm(V(i,:)-V(j,:))-2) < .01
                bonds = [bonds; i, j]; %#ok<AGROW>
            end
        end
    end
    assert(size(bonds,1) == 30);
end

%% ================================================================

