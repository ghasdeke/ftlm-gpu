function res = run(opts)
%FTLM.RUN  Sector-resolved FTLM thermodynamics of an isotropic spin model.
%   RES = FTLM.RUN(OPTS) computes C(T), chi(T) and Z_eff(T) for the model
%   defined by OPTS (see FTLM.DEFAULTS for all fields and defaults) and
%   returns a struct with the observables, the configuration, per-sector
%   diagnostics and timings.  With OPTS.use_cpu_reference, the same
%   calculation is repeated with the CPU kernel (precision
%   OPTS.cpu_precision, same start vectors) and stored in the *_cpu
%   fields.
%
%   Sectors with dim <= OPTS.ed_thresh are diagonalized exactly (dense
%   eig in FP64 on the host) instead of FTLM, provided that
%   prod(2 s_k + 1) <= 2^31 (the exact diagonalization uses the int32
%   sector basis); for larger label spaces (lookup = 'cr'), all sectors
%   are treated by FTLM.

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

opts  = ftlm.defaults(opts);
model = ftlm.model(opts);
assert(~opts.use_cpu_reference || model.clt_ok, 'ftlm:options', ...
    ['use_cpu_reference requires prod(2 s_k + 1) <= 2^31 ', ...
     '(the CPU kernel uses the CLT).']);
secs  = ftlm.sectors(model, opts.only_M0);
vb    = opts.verbose;

if vb
    fprintf('System:     %s, N = %d, %d couplings, spins: %s\n', model.name, model.N, ...
        size(model.couplings, 1), spin_summary(model.spins));
    fprintf('Hilbert:    prod(2s+1) = %.6g, S_max = %g, %d sector(s) with M >= 0\n', ...
        model.D_full, model.S_max, numel(secs));
    fprintf('FTLM:       R = %d, M_lz = %d, T-grid: %d points in [%.3g, %.3g]\n', ...
        opts.R, opts.M_lz, numel(opts.T_range), min(opts.T_range), max(opts.T_range));
    fprintf('Main run:   backend = %s, precision = %s, lookup = %s, ed_thresh = %d\n\n', ...
        opts.backend, opts.precision, opts.lookup, opts.ed_thresh);
end

if strcmp(opts.backend, 'gpu')
    assert(gpuDeviceCount > 0, 'ftlm:gpu', 'No CUDA-capable GPU found.');
    gpu_h = gpuDevice;
    reset(gpu_h);
    if vb
        gpu_h = gpuDevice;
        fprintf('GPU: %s  (%.1f GB VRAM)\n\n', gpu_h.Name, gpu_h.TotalMemory / 1e9);
    end
end

main = run_all_sectors(model, secs, opts, vb, '');
[C_T, chi_T, Z_eff] = ftlm.observables(main.E, main.w, main.M, opts.T_range);

res = struct();
res.T_range = opts.T_range;
res.C_T     = C_T;
res.chi_T   = chi_T;
res.Z_eff   = Z_eff;

res.E0      = min(main.E);         % lowest Ritz value / eigenvalue

res.C_T_cpu = [];  res.chi_T_cpu = [];  res.Z_eff_cpu = [];  res.E0_cpu = NaN;
res.t_wall_cpu = NaN;
if opts.use_cpu_reference
    ref_opts = opts;
    ref_opts.backend   = 'cpu';
    ref_opts.precision = opts.cpu_precision;
    ref_opts.lookup    = 'clt';
    if vb
        fprintf('\nCPU reference run (precision = %s)...\n', ref_opts.precision);
    end
    ref = run_all_sectors(model, secs, ref_opts, vb, ' (CPU)');
    [res.C_T_cpu, res.chi_T_cpu, res.Z_eff_cpu] = ...
        ftlm.observables(ref.E, ref.w, ref.M, opts.T_range);
    res.t_wall_cpu = ref.t_wall;
    res.E0_cpu     = min(ref.E);
    denom = max(abs(res.C_T_cpu));
    rel = 0;
    if denom > 0, rel = max(abs(res.C_T - res.C_T_cpu)) / denom; end
    res.rel_err_C = rel;
    if vb
        fprintf('Max |C_T - C_T_cpu| / max(|C_T_cpu|) = %.3e\n', rel);
    end
    if opts.save_ritz, res.ritz_cpu = ref.ritz; end
end

res.model        = model;
res.opts         = opts;
res.sector_M     = [secs.M]';
res.sector_A     = [secs.A]';
res.sector_dims  = [secs.dim]';
res.sector_mult  = [secs.mult]';
res.sector_method = main.method;
res.sector_B     = main.B;
res.sector_t     = main.t_sec;
res.t_wall_gpu   = main.t_wall;        % wall time of the main run (any backend)
res.t_lanczos    = main.t_lanczos;     % Lanczos calls only
if opts.save_ritz, res.ritz = main.ritz; end
end

%% ========================================================================
function out = run_all_sectors(model, secs, opts, vb, label)
    n = numel(secs);
    out.E = [];  out.w = [];  out.M = [];
    out.method = cell(n, 1);  out.B = zeros(n, 1);  out.t_sec = zeros(n, 1);
    out.ritz = cell(n, 1);
    t_start = tic;
    t_lz = 0;
    need_basis = strcmp(opts.lookup, 'clt') || strcmp(opts.backend, 'cpu');
    for q = 1 : n
        sec = secs(q);
        t_sec = tic;
        if sec.dim <= opts.ed_thresh && model.clt_ok   % ED needs int32 labels
            basis = ftlm.enumerate_sector(model, sec.A);
            E = sort(eig(full(ftlm.hamiltonian(model, basis))));
            w = ones(sec.dim, 1);
            out.method{q} = 'ED';
            if opts.save_ritz
                out.ritz{q} = struct('E', E, 'w', w, 'method', 'ED');
            end
        else
            if need_basis
                basis = ftlm.enumerate_sector(model, sec.A);
            else
                basis = [];
            end
            r = ftlm.sector_ftlm(model, sec, basis, opts);
            E = r.E;  w = r.w;
            out.B(q) = r.B;
            t_lz = t_lz + r.t_lanczos;
            out.method{q} = sprintf('Lanczos B=%d, R=%d', r.B, min(opts.R, sec.dim));
            if opts.save_ritz
                r.method = 'FTLM';
                out.ritz{q} = r;
            end
        end
        clear basis
        out.t_sec(q) = toc(t_sec);
        out.E = [out.E; E];                           %#ok<AGROW>
        out.w = [out.w; sec.mult * w];                %#ok<AGROW>
        out.M = [out.M; sec.M * ones(numel(E), 1)];   %#ok<AGROW>
        if vb
            fprintf('Sector M=%4g%s: dim=%10d, %-22s t=%.2fs\n', ...
                sec.M, label, sec.dim, out.method{q}, out.t_sec(q));
        end
    end
    out.t_wall    = toc(t_start);
    out.t_lanczos = t_lz;
    if vb
        fprintf('Total wall time%s: %.2f s (Lanczos: %.2f s)\n', label, out.t_wall, t_lz);
    end
end

function s = spin_summary(spins)
    u = unique(spins);
    parts = arrayfun(@(x) sprintf('%d x s=%s', sum(spins == x), rats(x)), u, ...
                     'UniformOutput', false);
    s = strjoin(strtrim(parts), ', ');
end
