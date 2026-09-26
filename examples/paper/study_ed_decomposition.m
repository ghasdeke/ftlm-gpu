function out = study_ed_decomposition(varargin)
%STUDY_ED_DECOMPOSITION  Separation of the FTLM error contributions by ED.
%   OUT = STUDY_ED_DECOMPOSITION() compares FTLM with exact
%   diagonalization for systems small enough for a full diagonalization
%   of every magnetization sector (with eigenvectors).  Using the SAME
%   start vectors |r> as the FTLM runs (seed = 0), the error of the FTLM
%   estimate is split into
%
%     stochastic   O_stoch - O_ED        O_stoch: trace estimator with the
%                                        exact <r|f(H)|r> (N_L -> inf)
%     truncation   O_64 - O_stoch        O_64: FTLM, GPU FP64, N_L steps
%     precision    O_32 - O_64           O_32: FTLM, GPU FP32 (same for
%                                        FP16 / BF16)
%     implement.   O_cpu64 - O_64        CPU vs. GPU kernel (FP64)
%
%   and the SpMV of the GPU FP64 kernel is compared with the explicit
%   sparse Hamiltonian.
%
%   Name-value options:
%     'Systems' {'cube_s3o2', 'cubo_s1o2'}   'R' 100   'NL' 100
%     'OutDir' '.'
%   Output: study_ed_<key>.mat

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
p.addParameter('Systems', {'cube_s3o2', 'cubo_s1o2'});
p.addParameter('R', 100);
p.addParameter('NL', 100);
p.addParameter('OutDir', '.');
p.parse(varargin{:});
o = p.Results;

T = unique([linspace(0.005, 0.2, 196), linspace(0.2, 10, 491)]);
out = struct();
for ks = 1 : numel(o.Systems)
    key = o.Systems{ks};
    switch key
        case 'cube_s3o2', mopts = struct('geometry', 'cube', 's_val', 1.5, 'J', 1);
        case 'cubo_s1o2', mopts = struct('geometry', 'cubo', 's_val', 0.5, 'J', 1);
        otherwise, error('unknown system %s', key);
    end
    model = ftlm.model(mopts);
    secs  = ftlm.sectors(model);
    fprintf('\n=== %s: %d sectors, dim_max = %d, R = %d, N_L = %d ===\n', ...
        key, numel(secs), max([secs.dim]), o.R, o.NL);

    % ---- exact diagonalization and exact per-vector quadrature ----------
    E_ed = {};  w_ed = {};  M_ed = {};
    E_st = {};  w_st = {};  M_st = {};
    spmv_err = 0;
    t0 = tic;
    for q = 1 : numel(secs)
        sec = secs(q);
        basis = ftlm.enumerate_sector(model, sec.A);
        H = ftlm.hamiltonian(model, basis);
        [U, E] = eig(full(H), 'vector');
        E_ed{end+1} = E;  w_ed{end+1} = sec.mult * ones(sec.dim, 1);   %#ok<AGROW>
        M_ed{end+1} = sec.M * ones(sec.dim, 1);                         %#ok<AGROW>
        % start vectors exactly as in ftlm.sector_ftlm (seed = 0)
        R_eff = min(o.R, sec.dim);
        rng(sec.dim, 'twister');
        V = randn(sec.dim, R_eff);
        c2 = (U' * V).^2 ./ sum(V.^2, 1);                   % dim x R_eff
        E_st{end+1} = repmat(E, R_eff, 1);                                %#ok<AGROW>
        w_st{end+1} = sec.mult * (sec.dim / R_eff) * c2(:);              %#ok<AGROW>
        M_st{end+1} = sec.M * ones(sec.dim * R_eff, 1);                   %#ok<AGROW>
        % kernel SpMV vs explicit H (GPU FP64)
        if sec.dim > 1
            cfg = ftlm.kernel_config(model, sec.A, 'clt', 'double', 2, basis);
            ftlm_gpu_mex('init', cfg);
            W = ftlm_gpu_mex('spmv', V(:, 1:min(2, R_eff)));
            ftlm_gpu_mex('cleanup');
            W0 = H * V(:, 1:min(2, R_eff));
            spmv_err = max(spmv_err, norm(W - W0, 'fro') / norm(W0, 'fro'));
        end
    end
    fprintf('  ED + exact quadrature: %.1f s, max rel. SpMV error (GPU FP64) = %.2e\n', ...
        toc(t0), spmv_err);
    S = struct('key', key, 'T_range', T, 'R', o.R, 'NL', o.NL, 'spmv_err', spmv_err);
    [S.C_ed, S.chi_ed] = ftlm.observables(vertcat(E_ed{:}), vertcat(w_ed{:}), vertcat(M_ed{:}), T);
    [S.C_st, S.chi_st] = ftlm.observables(vertcat(E_st{:}), vertcat(w_st{:}), vertcat(M_st{:}), T);

    % ---- FTLM runs with the same start vectors ----------------------------
    variants = {'gpu', 'double'; 'gpu', 'single'; 'gpu', 'half'; 'gpu', 'bfloat16'; ...
                'cpu', 'double'};
    for iv = 1 : size(variants, 1)
        opts = mopts;
        opts.R = o.R;  opts.M_lz = o.NL;  opts.T_range = T;  opts.ed_thresh = 0;
        opts.seed = 0;  opts.backend = variants{iv, 1};  opts.precision = variants{iv, 2};
        opts.verbose = false;
        r = ftlm.run(opts);
        S.([variants{iv, 1} '_' variants{iv, 2}]) = struct('C', r.C_T, 'chi', r.chi_T);
    end
    g64 = S.gpu_double;
    S.err_stoch = struct('C', abs(S.C_st - S.C_ed),       'chi', abs(S.chi_st - S.chi_ed));
    S.err_trunc = struct('C', abs(g64.C - S.C_st),        'chi', abs(g64.chi - S.chi_st));
    S.err_fp32  = struct('C', abs(S.gpu_single.C - g64.C), 'chi', abs(S.gpu_single.chi - g64.chi));
    S.err_fp16  = struct('C', abs(S.gpu_half.C - g64.C),   'chi', abs(S.gpu_half.chi - g64.chi));
    S.err_bf16  = struct('C', abs(S.gpu_bfloat16.C - g64.C), 'chi', abs(S.gpu_bfloat16.chi - g64.chi));
    S.err_impl  = struct('C', abs(S.cpu_double.C - g64.C), 'chi', abs(S.cpu_double.chi - g64.chi));
    S.err_total = struct('C', abs(S.gpu_single.C - S.C_ed), 'chi', abs(S.gpu_single.chi - S.chi_ed));
    fprintf('  max_T error of C (chi):\n');
    for f = {'err_stoch', 'err_trunc', 'err_fp32', 'err_fp16', 'err_bf16', 'err_impl', 'err_total'}
        fprintf('    %-10s %.2e  (%.2e)\n', f{1}, max(S.(f{1}).C), max(S.(f{1}).chi));
    end
    save(fullfile(o.OutDir, sprintf('study_ed_%s.mat', key)), '-struct', 'S');
    out.(key) = S;
end
end
