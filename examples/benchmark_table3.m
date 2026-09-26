function results = benchmark_table3(varargin)
%BENCHMARK_TABLE3  CPU/GPU x precision x lookup timings (paper Table 3).
%   RESULTS = BENCHMARK_TABLE3() times the FTLM workloads of the paper's
%   scaling study for all kernel variants:
%
%     CPU-CLT-FP64, CPU-CLT-FP32                  (OpenMP, B = 8)
%     GPU-CLT-FP64/FP32/FP16/BF16                  (B adaptive)
%     GPU-CR-FP64/FP32/FP16/BF16                   (B adaptive)
%     GPU-CLT-FP32-B1, GPU-CR-FP32-B1              (single-vector Lanczos)
%
%   on the systems
%     ico_s1       icosahedron s = 1,   all sectors M >= 0, R = 24
%     ico_s3o2     icosahedron s = 3/2, all sectors M >= 0, R = 24
%     ico_s2_M0    icosahedron s = 2,   M = 0 only,         R = 24
%     icosid_M0    icosidodecahedron s = 1/2, M = 0 only,   R = 8
%   with N_L = 100 Lanczos steps and ed_thresh = 1 (only the dim = 1
%   sector is diagonalized exactly), as in the paper.
%
%   Two times are recorded per run (sum over sectors):
%     t_lanczos  block Lanczos calls only (upload of the start vectors +
%                recursion); the kernel time compared in the paper
%     t_sector   everything per sector except the basis enumeration
%                (CLT construction, start vectors, Lanczos, tridiagonal
%                eigenproblems) -- the scope of the v1 benchmark scripts
%   Median over n_runs runs after discarding the first (warm-up) run.
%
%   Name-value options:
%     'Systems'  cell array with a subset of the system keys above
%     'Methods'  cell array with a subset of the method labels above
%     'NRuns'    runs per method (default 3; 2 for icosid_M0)
%     'Out'      output .mat file (default benchmark_table3_<date>.mat)

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

here = fileparts(mfilename('fullpath'));
addpath(fileparts(here));

sys_all = {
    'ico_s1',    struct('geometry', 'ico',    's_val', 1.0, 'J', 1), false, 24;
    'ico_s3o2',  struct('geometry', 'ico',    's_val', 1.5, 'J', 1), false, 24;
    'ico_s2_M0', struct('geometry', 'ico',    's_val', 2.0, 'J', 1), true,  24;
    'icosid_M0', struct('geometry', 'icosid', 's_val', 0.5, 'J', 1), true,  8};
meth_all = {
    'CPU-CLT-FP64',   'cpu', 'clt', 'double',   0;
    'CPU-CLT-FP32',   'cpu', 'clt', 'single',   0;
    'GPU-CLT-FP64',   'gpu', 'clt', 'double',   0;
    'GPU-CLT-FP32',   'gpu', 'clt', 'single',   0;
    'GPU-CLT-FP16',   'gpu', 'clt', 'half',     0;
    'GPU-CLT-BF16',   'gpu', 'clt', 'bfloat16', 0;
    'GPU-CR-FP64',    'gpu', 'cr',  'double',   0;
    'GPU-CR-FP32',    'gpu', 'cr',  'single',   0;
    'GPU-CR-FP16',    'gpu', 'cr',  'half',     0;
    'GPU-CR-BF16',    'gpu', 'cr',  'bfloat16', 0;
    'GPU-CLT-FP32-B1','gpu', 'clt', 'single',   1;
    'GPU-CR-FP32-B1', 'gpu', 'cr',  'single',   1};

p = inputParser;
p.addParameter('Systems', sys_all(:, 1)');
p.addParameter('Methods', meth_all(:, 1)');
p.addParameter('NRuns', []);
p.addParameter('Out', sprintf('benchmark_table3_%s.mat', datestr(now, 'yyyymmdd_HHMM'))); %#ok<TNOW1,DATST>
p.parse(varargin{:});
o = p.Results;

M_lz = 100;
ed_thresh = 1;
g = gpuDevice;
n_threads = ftlm_cpu_mex('info');
fprintf('GPU: %s, CPU threads: %d\n', g.Name, n_threads);

results = struct('system', {}, 'method', {}, 'dim_max', {}, 'R', {}, 'B', {}, ...
                 't_lanczos', {}, 't_sector', {}, 't_lanczos_runs', {}, ...
                 't_sector_runs', {}, 'E0', {});
for is = 1 : size(sys_all, 1)
    key = sys_all{is, 1};
    if ~any(strcmp(o.Systems, key)), continue; end
    mopts = sys_all{is, 2};
    only_M0 = sys_all{is, 3};
    R = sys_all{is, 4};
    n_runs = o.NRuns;
    if isempty(n_runs), n_runs = 3 - strcmp(key, 'icosid_M0'); end

    model = ftlm.model(mopts);
    secs  = ftlm.sectors(model, only_M0);
    secs  = secs([secs.dim] > ed_thresh);
    bases = arrayfun(@(s) ftlm.enumerate_sector(model, s.A), secs, 'UniformOutput', false);
    fprintf('\n=== %s: %d sector(s), dim_max = %d, R = %d, N_L = %d ===\n', ...
        key, numel(secs), max([secs.dim]), R, M_lz);

    for im = 1 : size(meth_all, 1)
        [label, backend, lookup, prec, single_vec] = meth_all{im, :};
        if ~any(strcmp(o.Methods, label)), continue; end
        run_opts = ftlm.defaults(struct('R', R, 'M_lz', M_lz, 'T_range', 1, ...
            'backend', backend, 'lookup', lookup, 'precision', prec, ...
            'B_gpu', single_vec, 'B_cpu', 8, 'verbose', false));
        t_lz = zeros(1, n_runs);  t_sec = zeros(1, n_runs);  B_used = 0;  E0 = NaN;
        try
            for irun = 1 : n_runs
                for q = 1 : numel(secs)
                    basis = [];
                    if strcmp(lookup, 'clt'), basis = bases{q}; end
                    t0 = tic;
                    r = ftlm.sector_ftlm(model, secs(q), basis, run_opts);
                    t_sec(irun) = t_sec(irun) + toc(t0);
                    t_lz(irun)  = t_lz(irun) + r.t_lanczos;
                    if q == 1, B_used = r.B; E0 = min(r.E); end
                end
            end
        catch ME
            fprintf('  %-16s FAILED: %s\n', label, ME.message);
            continue;
        end
        keep = 2 : n_runs;
        if n_runs == 1, keep = 1; end
        res = struct('system', key, 'method', label, 'dim_max', max([secs.dim]), ...
            'R', R, 'B', B_used, 't_lanczos', median(t_lz(keep)), ...
            't_sector', median(t_sec(keep)), 't_lanczos_runs', t_lz, ...
            't_sector_runs', t_sec, 'E0', E0);
        results(end + 1) = res; %#ok<AGROW>
        fprintf('  %-16s B=%2d  t_lanczos = %8.2f s   t_sector = %8.2f s   E0 = %.8f\n', ...
            label, B_used, res.t_lanczos, res.t_sector, E0);
        save(o.Out, 'results', 'M_lz', 'ed_thresh', 'n_threads');
    end
end
fprintf('\nSaved: %s\n', o.Out);
end
