function ftlm_observables(input_file)
%FTLM_OBSERVABLES  Sector-FTLM thermodynamics of Heisenberg spin clusters.
%   FTLM_OBSERVABLES(INPUT_FILE) computes the specific heat C(T), the
%   magnetic susceptibility chi(T), and the effective partition function
%   Z_eff(T) = Z(T) * exp(beta * E0) of a Heisenberg spin cluster on a
%   user-supplied temperature grid, using the Finite-Temperature Lanczos
%   Method (FTLM) with sector decomposition by total S^z.
%
%   The matrix-vector product H * v is matrix-free, using the compressed
%   lookup table (CLT) representation of the sector basis. The Lanczos
%   iteration runs on the GPU in single precision
%   (cuda_lanczos_clut_block). An optional FP64 CPU reference run
%   (cpu_lanczos_omp) is available for cross-check.
%
%   INPUT_FILE is a plain MATLAB script (.m) that sets the input
%   parameters as variable assignments. See input_ico_s1_example.m for a
%   commented template.
%
%   Required input variables:
%       geometry, s_val, J, R, M_lz, T_range
%       (plus N_ring if geometry='ring')
%
%   Optional input variables (with defaults):
%       only_M0           = false
%       use_cpu_reference = false
%       output_dir        = '.'
%       B_cpu             = 8
%       B_gpu             = 0       (0 = adaptive L2 heuristic)
%       L2_cache_bytes    = 48e6    (used by B_gpu=0 heuristic)
%       ed_thresh         = 0       (sectors with dim<=ed_thresh use
%                                    exact diagonalization instead of
%                                    FTLM; 0 disables this)
%
%   Example:
%       ftlm_observables('input_ico_s1_example.m')
%
%   Output:
%       A single .mat file 'ftlm_<geom>_s<s_str>.mat' in OUTPUT_DIR
%       containing T_range, C_T, chi_T, Z_eff (GPU FP32), optional *_cpu
%       counterparts, plus the full configuration and timing.
%
%   Requirements:
%       MATLAB R2022a or newer with Parallel Computing Toolbox
%       (gpuArray). MEX files cuda_lanczos_clut_block (CUDA) and
%       optionally cpu_lanczos_omp (OpenMP) must be on the path. Build
%       them with the accompanying build_all.m script.
%
%   Citation:
%       If you use this code, please cite
%       S. Ghassemi Tabrizi and T. D. Kuehne, "GPU-accelerated
%       finite-temperature Lanczos method for spin Hamiltonians"
%       (see CITATION.cff for the up-to-date reference).

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

%% ================================================================
%  Argument handling
%  ================================================================
if nargin < 1 || isempty(input_file)
    error('ftlm_observables:NoInput', ...
          ['ftlm_observables requires an input file. Usage:\n', ...
           '    ftlm_observables(''input.m'')\n', ...
           'See input_ico_s1_example.m for a commented template.']);
end
if exist(input_file, 'file') ~= 2
    error('ftlm_observables:InputNotFound', ...
          'Input file not found: %s', input_file);
end

fprintf('=== ftlm_observables: sector-FTLM thermodynamics (GPU FP32) ===\n');
fprintf('Input file: %s\n\n', input_file);

% Source the input file into this function's workspace. Any variables
% assigned in the input file become local variables here.
run(input_file);

%% ================================================================
%  Required inputs: presence check
%  ================================================================
required_vars = {'geometry', 's_val', 'J', 'R', 'M_lz', 'T_range'};
for k = 1:numel(required_vars)
    if ~exist(required_vars{k}, 'var')
        error('ftlm_observables:MissingInput', ...
              'Required input variable missing in %s: %s', ...
              input_file, required_vars{k});
    end
end

%% ================================================================
%  Optional inputs: apply defaults if absent
%  ================================================================
if ~exist('only_M0',           'var'), only_M0           = false; end
if ~exist('use_cpu_reference', 'var'), use_cpu_reference = false; end
if ~exist('output_dir',        'var'), output_dir        = '.';   end
if ~exist('B_cpu',             'var'), B_cpu             = 8;     end
if ~exist('B_gpu',             'var'), B_gpu             = 0;     end
if ~exist('L2_cache_bytes',    'var'), L2_cache_bytes    = 48e6;  end
if ~exist('N_ring',            'var'), N_ring            = [];    end
if ~exist('ed_thresh',         'var'), ed_thresh         = 0;     end

%% ================================================================
%  Input validation
%  ================================================================
assert(ischar(geometry) || (isstring(geometry) && isscalar(geometry)), ...
       'geometry must be a character vector or string scalar.');
geometry = char(geometry);
assert(any(strcmp(geometry, {'ico','cubo','cube','dodeca','icosid','ring'})), ...
       'Unknown geometry: %s (allowed: ico, cubo, cube, dodeca, icosid, ring)', geometry);
assert(isnumeric(s_val) && isscalar(s_val) && s_val > 0 && ...
       abs(2*s_val - round(2*s_val)) < 1e-12, ...
       's_val must be a positive half-integer or integer (0.5, 1, 1.5, ...).');
assert(isnumeric(J) && isscalar(J) && isfinite(J), 'J must be a finite scalar.');
assert(isnumeric(R) && isscalar(R) && R >= 1 && R == round(R), ...
       'R must be a positive integer.');
assert(isnumeric(M_lz) && isscalar(M_lz) && M_lz >= 1 && M_lz == round(M_lz), ...
       'M_lz must be a positive integer.');
assert(isnumeric(T_range) && isvector(T_range) && all(T_range > 0) && ...
       all(isfinite(T_range)), ...
       'T_range must be a vector of positive finite numbers.');
T_range = T_range(:).';   % canonicalize to row vector
assert(isnumeric(B_cpu) && isscalar(B_cpu) && B_cpu >= 1 && B_cpu <= 32 && ...
       B_cpu == round(B_cpu), ...
       'B_cpu must be an integer in [1, 32].');
assert(isnumeric(B_gpu) && isscalar(B_gpu) && B_gpu >= 0 && B_gpu <= 32 && ...
       B_gpu == round(B_gpu), ...
       'B_gpu must be an integer in [0, 32] (0 = adaptive).');
assert(isnumeric(L2_cache_bytes) && isscalar(L2_cache_bytes) && ...
       L2_cache_bytes > 0, 'L2_cache_bytes must be a positive scalar.');
assert(isnumeric(ed_thresh) && isscalar(ed_thresh) && ed_thresh >= 0 && ...
       ed_thresh == round(ed_thresh), ...
       'ed_thresh must be a non-negative integer (0 disables exact diagonalization).');
if strcmp(geometry, 'ring')
    assert(~isempty(N_ring) && isnumeric(N_ring) && isscalar(N_ring) && ...
           N_ring == round(N_ring) && N_ring >= 3, ...
           'geometry=''ring'' requires N_ring (integer >= 3).');
end

%% ================================================================
%  GPU initialization
%  ================================================================
assert(gpuDeviceCount > 0, 'No CUDA-capable GPU found.');
gpu_h = gpuDevice;
reset(gpu_h);
gpu_h = gpuDevice;
fprintf('GPU: %s  (%.1f GB VRAM)\n', gpu_h.Name, gpu_h.TotalMemory/1e9);

assert(exist('cuda_lanczos_clut_block', 'file') == 3, ...
    'MEX file cuda_lanczos_clut_block not found on the path.');
if use_cpu_reference
    assert(exist('cpu_lanczos_omp', 'file') == 3, ...
        'MEX file cpu_lanczos_omp not found (required for CPU reference).');
end

%% ================================================================
%  Geometry dispatch
%  ================================================================
switch geometry
    case 'ico'
        bonds = adjacency_icosahedron();
        N = 12;  geom_name = 'Icosahedron';      geom_short = 'ico';
    case 'cubo'
        bonds = adjacency_cuboctahedron();
        N = 12;  geom_name = 'Cuboctahedron';    geom_short = 'cubo';
    case 'cube'
        bonds = adjacency_cube();
        N = 8;   geom_name = 'Cube';             geom_short = 'cube';
    case 'dodeca'
        bonds = adjacency_dodecahedron();
        N = 20;  geom_name = 'Dodecahedron';     geom_short = 'dodeca';
    case 'icosid'
        bonds = adjacency_icosidodecahedron();
        N = 30;  geom_name = 'Icosidodecahedron'; geom_short = 'icosid';
    case 'ring'
        N = N_ring;
        bonds = adjacency_ring(N);
        geom_name  = sprintf('%d-Ring', N);
        geom_short = sprintf('ring_%d', N);
    otherwise
        error('Unknown geometry: %s (allowed: ico, cubo, cube, dodeca, icosid, ring)', geometry);
end
N_b        = size(bonds, 1);
bonds_flat = int32(reshape(bonds' - 1, [], 1));   % 0-based, interleaved (i,j)

%% ================================================================
%  Derived sizes
%  ================================================================
d_loc   = round(2*s_val + 1);   % local Hilbert space dimension
n_total = d_loc^N;              % full Hilbert space dimension
M_max   = round(N * s_val);     % maximum total S^z

% The MEX kernels encode state labels as signed 32-bit integers and
% support at most 32 sites (MAX_SITES). Reject out-of-range systems
% here rather than risking silent integer overflow downstream.
assert(N <= 32, ...
    'N = %d exceeds the kernel limit of 32 sites.', N);
assert(n_total <= 2^31, ...
    ['d_loc^N = %d^%d = %.3g exceeds 2^31: state labels do not fit ', ...
     'into the int32 encoding used by the MEX kernels. Reduce N or s_val.'], ...
    d_loc, N, n_total);

% Compact spin label for filenames: '1', '1o2', '3o2', ...
two_s = round(2 * s_val);
if mod(two_s, 2) == 0
    s_str = sprintf('%d', two_s/2);
else
    s_str = sprintf('%do2', two_s);
end

fprintf('System:     s = %s, geometry = %s (N=%d, N_b=%d)\n', s_str, geom_name, N, N_b);
fprintf('Hilbert:    d_loc = %d, n_total = %d, M_max = %d\n', d_loc, n_total, M_max);
fprintf('FTLM:       R = %d, M_lz = %d, T-grid: %d points in [%.3g, %.3g]\n', ...
    R, M_lz, numel(T_range), min(T_range), max(T_range));
if only_M0
    fprintf('Sector mode: M=0 only (chi(T) will be 0)\n');
end
fprintf('\n');

%% ================================================================
%  Output filename
%  ================================================================
mat_name = sprintf('ftlm_%s_s%s.mat', geom_short, s_str);
mat_path = fullfile(output_dir, mat_name);

%% ================================================================
%  MAIN: sector loop (GPU FP32)
%  ================================================================
all_E       = zeros(0, 1);
all_w       = zeros(0, 1);
all_M       = zeros(0, 1);
sector_M    = zeros(0, 1);
sector_dims = zeros(0, 1);

t_gpu_start    = tic;

for M = 0 : M_max
    if only_M0 && M ~= 0, continue; end

    [basis, dim_sector] = enumerate_sector(N, s_val, d_loc, n_total, M);
    if dim_sector == 0
        continue;
    end

    sector_M(end+1, 1)    = M;          %#ok<SAGROW>
    sector_dims(end+1, 1) = dim_sector; %#ok<SAGROW>

    mult = 1 + (M > 0);   % M <-> -M symmetry: +M and -M both contribute

    t_sec = tic;
    if dim_sector <= ed_thresh
        % --- Exact diagonalization branch (small sectors) ---
        [E_sec, w_sec] = run_ed_sector(basis, dim_sector, bonds, ...
                                       s_val, J, N, n_total);
        method_str = 'ED';
    else
        % --- FTLM block-Lanczos branch on GPU ---
        [block_base, block_mask] = build_CLT(basis, n_total);
        basis_int = int32(basis);
        [E_sec, w_sec, B_used] = run_ftlm_sector_gpu( ...
            block_base, block_mask, basis_int, ...
            bonds_flat, N, d_loc, s_val, J, dim_sector, ...
            R, M_lz, L2_cache_bytes, B_gpu, gpu_h);
        method_str = sprintf('Lanczos B=%d, R=%d', B_used, R);
    end
    dt_sec = toc(t_sec);

    % Apply the M-multiplicity factor only; the per-vector (dim/R_eff)
    % normalisation has already been applied inside the FTLM runner,
    % and the ED branch yields per-eigenstate weights of unity.
    w_sec = w_sec * mult;

    all_E = [all_E; E_sec];   %#ok<AGROW>
    all_w = [all_w; w_sec];   %#ok<AGROW>
    all_M = [all_M; M * ones(numel(E_sec), 1)]; %#ok<AGROW>

    fprintf('Sector M=%2d: dim=%8d, %-20s t=%.2fs\n', ...
        M, dim_sector, method_str, dt_sec);
end
t_wall_gpu = toc(t_gpu_start);
fprintf('\nGPU total wall time: %.2f s\n', t_wall_gpu);

%% ================================================================
%  Observables (GPU FP32)
%  ================================================================
[C_T, chi_T, Z_eff] = compute_observables(all_E, all_w, all_M, T_range);

%% ================================================================
%  Optional: CPU FP64 reference run
%  ================================================================
C_T_cpu    = [];
chi_T_cpu  = [];
Z_eff_cpu  = [];
t_wall_cpu = NaN;

if use_cpu_reference
    fprintf('\nCPU FP64 reference run...\n');
    all_E_cpu = zeros(0, 1);
    all_w_cpu = zeros(0, 1);
    all_M_cpu = zeros(0, 1);
    t_cpu_start = tic;

    for M = 0 : M_max
        if only_M0 && M ~= 0, continue; end
        [basis, dim_sector] = enumerate_sector(N, s_val, d_loc, n_total, M);
        if dim_sector == 0, continue; end
        mult = 1 + (M > 0);

        t_sec = tic;
        if dim_sector <= ed_thresh
            [E_sec, w_sec] = run_ed_sector(basis, dim_sector, bonds, ...
                                           s_val, J, N, n_total);
            method_str = 'ED';
        else
            [block_base, block_mask] = build_CLT(basis, n_total);
            basis_int = int32(basis);
            [E_sec, w_sec] = run_ftlm_sector_cpu( ...
                block_base, block_mask, basis_int, ...
                bonds_flat, N, d_loc, s_val, J, dim_sector, R, M_lz, B_cpu);
            method_str = sprintf('Lanczos B=%d, R=%d', B_cpu, R);
        end
        dt_sec = toc(t_sec);

        % Apply M-multiplicity (FTLM normalisation is in-runner; ED yields w=1).
        w_sec = w_sec * mult;
        all_E_cpu = [all_E_cpu; E_sec];   %#ok<AGROW>
        all_w_cpu = [all_w_cpu; w_sec];   %#ok<AGROW>
        all_M_cpu = [all_M_cpu; M * ones(numel(E_sec), 1)]; %#ok<AGROW>

        fprintf('Sector M=%2d (CPU): dim=%8d, %-20s t=%.2fs\n', ...
            M, dim_sector, method_str, dt_sec);
    end
    t_wall_cpu = toc(t_cpu_start);
    fprintf('CPU total wall time: %.2f s\n', t_wall_cpu);

    [C_T_cpu, chi_T_cpu, Z_eff_cpu] = compute_observables( ...
        all_E_cpu, all_w_cpu, all_M_cpu, T_range);

    denom = max(abs(C_T_cpu));
    if denom > 0
        rel_err_C = max(abs(C_T - C_T_cpu)) / denom;
    else
        rel_err_C = 0;
    end
    fprintf('Max |C_T_gpu - C_T_cpu| / max(|C_T_cpu|) = %.3e\n', rel_err_C);
end

%% ================================================================
%  Save results
%  ================================================================
n_total_save = double(n_total);   % store as double to avoid int overflow on load
save(mat_path, ...
    'T_range', 'C_T', 'chi_T', 'Z_eff', ...
    'C_T_cpu', 'chi_T_cpu', 'Z_eff_cpu', ...
    'geometry', 's_val', 'J', 'R', 'M_lz', 'N', 'n_total_save', ...
    'M_max', 'only_M0', 'use_cpu_reference', ...
    'B_cpu', 'B_gpu', 'ed_thresh', ...
    'sector_dims', 'sector_M', ...
    't_wall_gpu', 't_wall_cpu', '-v7.3');
fprintf('\nResults saved to: %s\n', mat_path);

end  % ftlm_observables

%% ================================================================
%  LOCAL FUNCTIONS
%  ================================================================

function [E_sec, w_sec, B_used] = run_ftlm_sector_gpu( ...
    block_base, block_mask, basis_int, bonds_flat, ...
    N, d_loc, s_val, J, dim_sector, R, M_lz, L2_cache_bytes, B_gpu, gpu_h)
%RUN_FTLM_SECTOR_GPU  Block-Lanczos FTLM on the GPU (FP32).
%
%  Block size B:
%     B_gpu == 0  -> adaptive: three FP32 vectors of dim_sector each
%                   are kept hot in cache;
%                   B = 8 if 3 * dim_sector * 8 * 4 bytes fits into L2,
%                   else B = 4.
%     B_gpu  > 0  -> fixed value (in [1, 32]).
%     The final B is capped at min(B, R_eff).

    R_eff       = min(R, dim_sector);
    M_lz_actual = min(M_lz, dim_sector);

    if B_gpu == 0
        mem_B8 = 3 * dim_sector * 8 * 4;     % bytes for 3 vectors x B=8 x FP32
        if mem_B8 <= L2_cache_bytes
            B_used = 8;
        else
            B_used = 4;
        end
    else
        B_used = B_gpu;
    end
    B_used = min(B_used, R_eff);

    cuda_lanczos_clut_block('init', ...
        gpuArray(block_base), gpuArray(block_mask), ...
        gpuArray(basis_int), ...
        bonds_flat, N, d_loc, s_val, J, dim_sector, B_used);
    wait(gpu_h);

    E_sec = zeros(M_lz_actual * R_eff, 1);
    w_sec = zeros(M_lz_actual * R_eff, 1);
    idx   = 0;

    rng(dim_sector);   % deterministic seed per sector (reproducible)

    for r_start = 1 : B_used : R_eff
        r_end = min(r_start + B_used - 1, R_eff);
        B_cur = r_end - r_start + 1;

        V0_blk = single(randn(dim_sector, B_cur));
        V0_gpu = gpuArray(V0_blk);

        [AL, BE] = cuda_lanczos_clut_block('block_lanczos', V0_gpu, M_lz_actual);

        for b = 1 : B_cur
            [E_r, q1_r] = solve_tridiag(AL(:, b), BE(:, b));
            n_l = numel(E_r);
            E_sec(idx+1 : idx+n_l) = E_r;
            % FTLM single-vector weight: (dim_sector / R_eff) * |<e_k|v0>|^2.
            % R_eff (not R) is the actual number of random vectors drawn;
            % they differ when dim_sector < R.
            w_sec(idx+1 : idx+n_l) = (dim_sector / R_eff) * q1_r;
            idx = idx + n_l;
        end
    end

    cuda_lanczos_clut_block('cleanup');

    E_sec = E_sec(1:idx);
    w_sec = w_sec(1:idx);
end

function [E_sec, w_sec] = run_ftlm_sector_cpu( ...
    block_base, block_mask, basis_int, bonds_flat, ...
    N, d_loc, s_val, J, dim_sector, R, M_lz, B_cpu)
%RUN_FTLM_SECTOR_CPU  Block-Lanczos FTLM on the CPU (FP64, OpenMP).
%  Reference path. Block size B_cpu is caller-supplied (default 8;
%  one x86 cache line of FP64).

    R_eff       = min(R, dim_sector);
    M_lz_actual = min(M_lz, dim_sector);
    B_cpu       = min(B_cpu, R_eff);

    cpu_lanczos_omp('init_clut', ...
        int32(block_base), block_mask, basis_int, ...
        bonds_flat, N, d_loc, s_val, J, dim_sector);

    E_sec = zeros(M_lz_actual * R_eff, 1);
    w_sec = zeros(M_lz_actual * R_eff, 1);
    idx   = 0;

    rng(dim_sector);

    for r_start = 1 : B_cpu : R_eff
        r_end = min(r_start + B_cpu - 1, R_eff);
        B_cur = r_end - r_start + 1;

        V0_blk = randn(dim_sector, B_cur);

        [AL, BE] = cpu_lanczos_omp('block_lanczos_clut', V0_blk, M_lz_actual);

        for b = 1 : B_cur
            [E_r, q1_r] = solve_tridiag(AL(:, b), BE(:, b));
            n_l = numel(E_r);
            E_sec(idx+1 : idx+n_l) = E_r;
            % FTLM single-vector weight (see GPU runner for explanation):
            w_sec(idx+1 : idx+n_l) = (dim_sector / R_eff) * q1_r;
            idx = idx + n_l;
        end
    end

    cpu_lanczos_omp('cleanup');

    E_sec = E_sec(1:idx);
    w_sec = w_sec(1:idx);
end

function [C_T, chi_T, Z_eff] = compute_observables(all_E, all_w, all_M, T_range)
%COMPUTE_OBSERVABLES  C(T), chi(T), Z_eff(T) from sector-FTLM data.
%
%  Inputs:
%     all_E   - all Ritz values (column vector)
%     all_w   - associated weights mult * (dim_M / R) * |q_k^(1)|^2
%     all_M   - magnetic quantum number (M >= 0) per entry
%     T_range - temperatures, row or column vector
%
%  Outputs:
%     C_T   - specific heat C(T)              [1 x n_T]
%     chi_T - magnetic susceptibility chi(T)  [1 x n_T]
%     Z_eff - Z_eff(T) = Z(T) * exp(beta*E0)  [1 x n_T]
%
%  Formulas:
%     Z(beta)    = sum_i  w_i * exp(-beta * E_i)
%     <E>(beta)  = sum_i  w_i * E_i * exp(-beta * E_i) / Z
%     C(beta)    = beta^2 * Var(E)
%     <M^2>(beta)= sum_i  w_i * M_i^2 * exp(-beta * E_i) / Z
%     chi(beta)  = beta * <M^2>
%
%  Centred-variance form Var(E) = <(E - <E>)^2> is used (a sum of
%  non-negative terms), which is mathematically identical to
%  <E^2> - <E>^2 but free from catastrophic cancellation at low T.

    all_E    = double(all_E(:));
    all_w    = double(all_w(:));
    all_M    = double(all_M(:));
    T_range  = double(T_range(:)');
    n_T      = length(T_range);
    beta_arr = 1.0 ./ T_range;

    E_min = min(all_E);
    dE    = all_E - E_min;
    M2    = all_M .^ 2;

    C_T   = zeros(1, n_T);
    chi_T = zeros(1, n_T);
    Z_eff = zeros(1, n_T);

    for iT = 1 : n_T
        bet   = beta_arr(iT);
        boltz = all_w .* exp(-bet * dE);

        Z = sum(boltz);
        if Z < 1e-250
            continue;   % protect against underflow at very large beta
        end

        dE_avg = sum(dE .* boltz) / Z;
        E_var  = sum((dE - dE_avg).^2 .* boltz) / Z;
        M2_avg = sum(M2 .* boltz) / Z;

        C_T(iT)   = bet^2 * E_var;
        chi_T(iT) = bet * M2_avg;
        Z_eff(iT) = Z;
    end
end

function [basis, dim_sector] = enumerate_sector(N, s_val, d_loc, n_total, M_target)
%ENUMERATE_SECTOR  Enumerate the basis of the sector with total S^z = M_target.
%
%  For s = 1/2 a vectorized popcount path is used.
%  For s >= 1 the radix-d digit decomposition is applied in chunks.
%  Returns the sorted 0-based basis as int64.

    if s_val == 0.5
        k = N/2 + M_target;
        if k < 0 || k > N || abs(k - round(k)) > 0.01
            basis = int64([]);
            dim_sector = 0;
            return;
        end
        k = round(k);
        dim_expected = nchoosek(N, k);
        basis = enumerate_half_spin_basis(N, k, dim_expected);
    else
        chunk = 10000000;
        basis_parts = {};
        for start = 0 : chunk : n_total - 1
            stop = min(start + chunk - 1, n_total - 1);
            states = int64(start:stop)';
            Mv  = zeros(length(states), 1);
            tmp = states;
            for kk = 1 : N
                dg  = double(mod(tmp, int64(d_loc)));
                Mv  = Mv + dg - s_val;
                tmp = (tmp - int64(dg)) / int64(d_loc);
            end
            Mv  = round(Mv * 2) / 2;
            sel = (Mv == M_target);
            if any(sel)
                basis_parts{end+1} = states(sel); %#ok<AGROW>
            end
        end
        basis = cat(1, basis_parts{:});
    end
    dim_sector = length(basis);
end

function basis = enumerate_half_spin_basis(N, k, dim_expected)
%ENUMERATE_HALF_SPIN_BASIS  All N-bit integers with exactly k bits set.
%  Vectorized popcount over byte lookup table in chunks.

    n_total = int64(2)^N;

    lut = uint8(zeros(256, 1));
    for i = 0 : 255
        lut(i+1) = uint8(sum(bitget(uint8(i), 1:8)));
    end

    basis      = zeros(dim_expected, 1, 'int64');
    idx        = 0;
    chunk_size = int64(2^22);

    for cs = int64(0) : chunk_size : (n_total - 1)
        ce    = min(cs + chunk_size - 1, n_total - 1);
        range = (cs:ce)';

        b0 = uint8(bitand(range,                     int64(255)));
        b1 = uint8(bitand(bitshift(range,  -8),      int64(255)));
        b2 = uint8(bitand(bitshift(range, -16),      int64(255)));
        b3 = uint8(bitand(bitshift(range, -24),      int64(255)));

        pop = lut(double(b0)+1) + lut(double(b1)+1) + ...
              lut(double(b2)+1) + lut(double(b3)+1);

        valid   = pop == uint8(k);
        n_valid = sum(valid);
        if n_valid > 0
            basis(idx+1 : idx+n_valid) = range(valid);
            idx = idx + n_valid;
        end
    end

    basis = basis(1:idx);
    assert(length(basis) == dim_expected, ...
        'enumerate_half_spin_basis: expected %d, got %d', dim_expected, length(basis));
end

function [block_base, block_mask] = build_CLT(basis, n_total)
%BUILD_CLT  Build the compressed lookup table (CLT) from the basis array.
%
%  Inputs:
%     basis    - sorted basis array (0-based state indices) [dim x 1]
%     n_total  - total number of states (d^N)
%
%  Outputs:
%     block_base - sector index of the first valid state per block
%                  [n_blocks x 1, int32]
%     block_mask - bitmask (bit j = 1 if state 32*b+j is in the sector)
%                  [n_blocks x 1, uint32]
%
%  Memory: n_blocks * 8 bytes (vs n_total * 4 bytes for the full lookup);
%  exact compression factor 16x.

    BLOCK_SIZE = 32;
    n_blocks   = ceil(n_total / BLOCK_SIZE);
    states     = double(basis(:));

    blks = floor(states / BLOCK_SIZE) + 1;
    bits = mod(states, BLOCK_SIZE);

    block_base      = int32(-ones(n_blocks, 1));
    [ub, fi]        = unique(blks, 'first');
    block_base(ub)  = int32(fi - 1);     % 0-based sector index

    bit_vals   = pow2(bits);             % 2^bit as double (exact for bit <= 52)
    mask_sums  = accumarray(blks, bit_vals, [n_blocks, 1]);
    block_mask = uint32(mask_sums);
end

function [ep, q1] = solve_tridiag(alpha, beta)
%SOLVE_TRIDIAG  Diagonalize the Lanczos tridiagonal matrix.
%
%  Returns the Ritz values ep and the squared first components of
%  the Ritz vectors q1 = |<e_k|v0>|^2. Explicit double() cast is
%  used so that eig() does not silently work in single precision
%  when alpha/beta come from a FP32 MEX call.

    alpha = double(alpha(:));
    beta  = double(beta(:));
    n     = length(alpha);
    T     = diag(alpha(1:n));
    if n > 1
        T = T + diag(beta(1:n-1), 1) + diag(beta(1:n-1), -1);
    end
    [Q, D] = eig(T, 'vector');
    ep     = D;
    q1     = abs(Q(1, :)').^2;
end

function [E_sec, w_sec] = run_ed_sector(basis, dim_sector, bonds, ...
                                        s_val, J, N, n_total)
%RUN_ED_SECTOR  Exact diagonalization of one S^z sector.
%
%  Builds the sparse Heisenberg Hamiltonian on the sector basis and
%  computes its full spectrum with eig(full(H)). Used for sectors with
%  dim_sector <= ed_thresh, where the cost of dense diagonalization is
%  smaller than the FTLM overhead (and where the ED result is exact
%  rather than statistical).
%
%  Output convention matches the FTLM runners: per-eigenstate weights
%  are 1 (the per-vector dim/R_eff normalisation does not apply here);
%  the caller multiplies by mult to account for the M <-> -M symmetry.

    lookup = build_lookup(basis, dim_sector, n_total);
    H = build_heisenberg_sparse(basis, lookup, bonds, s_val, J, N, n_total);
    E_sec = sort(eig(full(H)));
    w_sec = ones(dim_sector, 1);
end

function lookup = build_lookup(basis, dim, lookup_size)
%BUILD_LOOKUP  State -> sector-index lookup (1-based, int32).
%  Returns lookup(state+1) = i if basis(i) == state, 0 otherwise.

    lookup = zeros(lookup_size, 1, 'int32');
    if dim < 1e6
        for i = 1:dim
            lookup(basis(i) + 1) = int32(i);
        end
    else
        lookup(double(basis) + 1) = int32((1:dim)');
    end
end

function H = build_heisenberg_sparse(basis, lookup, bonds, s, J, N, lookup_size)
%BUILD_HEISENBERG_SPARSE  Sparse Heisenberg Hamiltonian on a sector basis.
%
%  Diagonal:    H_kk = J * sum_{<i,j>} m_i(k) * m_j(k)
%  Off-diag:    1/2 * J * ( S+_i S-_j + S-_i S+_j )
%
%  Generic in the local spin s (0.5, 1, 1.5, 2, ...). The full
%  spectrum can then be obtained with eig(full(H)).

    d_loc   = round(2*s + 1);
    dim     = length(basis);
    n_bonds = size(bonds, 1);
    powers  = int64(d_loc).^int64((0:N-1)');

    % Digit-encode each basis state at each site (m_z values, shifted).
    mi = zeros(dim, N);
    temp = int64(basis);
    for site = 1:N
        digit = double(mod(temp, int64(d_loc)));
        mi(:, site) = digit - s;
        temp = (temp - int64(digit)) / int64(d_loc);
    end

    % Diagonal contribution: J * sum_{<i,j>} m_i * m_j.
    diag_vals = zeros(dim, 1);
    for b = 1:n_bonds
        diag_vals = diag_vals + J * mi(:, bonds(b,1)) .* mi(:, bonds(b,2));
    end

    row_list = (1:dim)';
    col_list = (1:dim)';
    val_list = diag_vals;

    % Off-diagonal contributions: S+_i S-_j and S-_i S+_j for each bond.
    for b = 1:n_bonds
        si = bonds(b,1);  sj = bonds(b,2);

        % S+_i S-_j  (raise i, lower j)
        can_flip = (mi(:,si) < s - 1e-10) & (mi(:,sj) > -s + 1e-10);
        idx_from = find(can_flip);
        if ~isempty(idx_from)
            mi_i = mi(idx_from, si);
            mi_j = mi(idx_from, sj);
            coeffs = 0.5*J * sqrt(s*(s+1) - mi_i.*(mi_i+1)) .* ...
                             sqrt(s*(s+1) - mi_j.*(mi_j-1));
            new_states = basis(idx_from) + powers(si) - powers(sj);
            ns1 = double(new_states) + 1;
            ok  = (ns1 >= 1) & (ns1 <= lookup_size);
            ni  = zeros(length(idx_from), 1, 'int32');
            ni(ok) = lookup(ns1(ok));
            v = ni > 0;
            row_list = [row_list; double(ni(v))]; %#ok<AGROW>
            col_list = [col_list; double(idx_from(v))]; %#ok<AGROW>
            val_list = [val_list; coeffs(v)]; %#ok<AGROW>
        end

        % S-_i S+_j  (lower i, raise j)
        can_flip = (mi(:,si) > -s + 1e-10) & (mi(:,sj) < s - 1e-10);
        idx_from = find(can_flip);
        if ~isempty(idx_from)
            mi_i = mi(idx_from, si);
            mi_j = mi(idx_from, sj);
            coeffs = 0.5*J * sqrt(s*(s+1) - mi_i.*(mi_i-1)) .* ...
                             sqrt(s*(s+1) - mi_j.*(mi_j+1));
            new_states = basis(idx_from) - powers(si) + powers(sj);
            ns1 = double(new_states) + 1;
            ok  = (ns1 >= 1) & (ns1 <= lookup_size);
            ni  = zeros(length(idx_from), 1, 'int32');
            ni(ok) = lookup(ns1(ok));
            v = ni > 0;
            row_list = [row_list; double(ni(v))]; %#ok<AGROW>
            col_list = [col_list; double(idx_from(v))]; %#ok<AGROW>
            val_list = [val_list; coeffs(v)]; %#ok<AGROW>
        end
    end

    H = sparse(row_list, col_list, val_list, dim, dim);
end

function bonds = adjacency_icosahedron()
%ADJACENCY_ICOSAHEDRON  30 edges of the icosahedron (12 vertices, z=5).
    phi = (1 + sqrt(5)) / 2;
    V = [0,1,phi;  0,1,-phi;  0,-1,phi;  0,-1,-phi;
         1,phi,0;  1,-phi,0; -1,phi,0;  -1,-phi,0;
         phi,0,1;  phi,0,-1; -phi,0,1;  -phi,0,-1];
    bonds = [];
    for i = 1 : 12
        for j = i+1 : 12
            if abs(norm(V(i,:) - V(j,:)) - 2) < 0.01
                bonds = [bonds; i, j]; %#ok<AGROW>
            end
        end
    end
    assert(size(bonds, 1) == 30);
end

function bonds = adjacency_cuboctahedron()
%ADJACENCY_CUBOCTAHEDRON  24 edges of the cuboctahedron (12 vertices, z=4).
    V = [ 1, 1, 0;  1,-1, 0; -1, 1, 0; -1,-1, 0;
          1, 0, 1;  1, 0,-1; -1, 0, 1; -1, 0,-1;
          0, 1, 1;  0, 1,-1;  0,-1, 1;  0,-1,-1];
    bonds = [];
    for i = 1 : 12
        for j = i+1 : 12
            if abs(norm(V(i,:) - V(j,:)) - sqrt(2)) < 0.01
                bonds = [bonds; i, j]; %#ok<AGROW>
            end
        end
    end
    assert(size(bonds, 1) == 24, ...
        'Cuboctahedron: expected 24 edges, found %d', size(bonds, 1));
end

function bonds = adjacency_cube()
%ADJACENCY_CUBE  12 edges of the cube (8 vertices, z=3).
    V = [-1,-1,-1; 1,-1,-1; -1,1,-1; 1,1,-1;
         -1,-1, 1; 1,-1, 1; -1,1, 1; 1,1, 1];
    bonds = [];
    for i = 1 : 8
        for j = i+1 : 8
            if abs(norm(V(i,:) - V(j,:)) - 2) < 0.01
                bonds = [bonds; i, j]; %#ok<AGROW>
            end
        end
    end
    assert(size(bonds, 1) == 12, ...
        'Cube: expected 12 edges, found %d', size(bonds, 1));
end

function bonds = adjacency_dodecahedron()
%ADJACENCY_DODECAHEDRON  30 edges of the dodecahedron (20 vertices, z=3).
    phi = (1 + sqrt(5)) / 2;
    V = [-1,-1,-1;  1,-1,-1; -1, 1,-1;  1, 1,-1;
         -1,-1, 1;  1,-1, 1; -1, 1, 1;  1, 1, 1;
          0, 1/phi, phi;   0,-1/phi, phi;   0, 1/phi,-phi;   0,-1/phi,-phi;
          1/phi, phi, 0;  -1/phi, phi, 0;   1/phi,-phi, 0;  -1/phi,-phi, 0;
          phi, 0, 1/phi;   phi, 0,-1/phi;  -phi, 0, 1/phi;  -phi, 0,-1/phi];
    target_d = 2 / phi;
    bonds = [];
    for i = 1 : 20
        for j = i+1 : 20
            if abs(norm(V(i,:) - V(j,:)) - target_d) < 0.1
                bonds = [bonds; i, j]; %#ok<AGROW>
            end
        end
    end
    assert(size(bonds, 1) == 30, ...
        'Dodecahedron: expected 30 edges, found %d', size(bonds, 1));
end

function bonds = adjacency_icosidodecahedron()
%ADJACENCY_ICOSIDODECAHEDRON  60 edges of the icosidodecahedron (30 vertices, z=4).
    phi   = (1 + sqrt(5)) / 2;
    polar = [0,0,phi; 0,0,-phi; phi,0,0; -phi,0,0; 0,phi,0; 0,-phi,0];
    sc    = (1 + phi) / 2;
    equat = [
        1/2, phi/2, sc;   -1/2, phi/2, sc;    1/2,-phi/2, sc;   -1/2,-phi/2, sc;
        1/2, phi/2,-sc;   -1/2, phi/2,-sc;    1/2,-phi/2,-sc;   -1/2,-phi/2,-sc;
        phi/2, sc, 1/2;   -phi/2, sc, 1/2;    phi/2,-sc, 1/2;   -phi/2,-sc, 1/2;
        phi/2, sc,-1/2;   -phi/2, sc,-1/2;    phi/2,-sc,-1/2;   -phi/2,-sc,-1/2;
        sc, 1/2, phi/2;   -sc, 1/2, phi/2;    sc,-1/2, phi/2;   -sc,-1/2, phi/2;
        sc, 1/2,-phi/2;   -sc, 1/2,-phi/2;    sc,-1/2,-phi/2;   -sc,-1/2,-phi/2];
    V = [polar; equat];
    bonds = [];
    for i = 1 : 30
        for j = i+1 : 30
            d = norm(V(i,:) - V(j,:));
            if d > 0.9 && d < 1.1
                bonds = [bonds; i, j]; %#ok<AGROW>
            end
        end
    end
    assert(size(bonds, 1) == 60, ...
        'Icosidodecahedron: expected 60 edges, found %d', size(bonds, 1));
end

function bonds = adjacency_ring(N)
%ADJACENCY_RING  N edges of a periodic 1D chain (z=2, bipartite for even N).
    assert(N >= 2, 'Ring requires at least 2 sites.');
    bonds = zeros(N, 2);
    for i = 1 : N - 1
        bonds(i, :) = [i, i+1];
    end
    bonds(N, :) = [N, 1];   % periodic boundary
end
