function R = summarize_results(study_dir, bench_file, out_json)
%SUMMARIZE_RESULTS  Numbers quoted in the paper, from the study and benchmark files.
%   R = SUMMARIZE_RESULTS(STUDY_DIR, BENCH_FILE, OUT_JSON) reads the
%   study_*.mat files and memory_traffic.json (from memory_traffic.py) in
%   STUDY_DIR and the benchmark_table3 output BENCH_FILE and writes the
%   quoted numbers and the rows of Tables 3 and 5 as JSON (strings;
%   '^^...^^' marks superscripts).  Temperature-resolved diagnostics are
%   written to STUDY_DIR/diagnostics.txt.
%
%   Precision statements refer to the temperature window in which the
%   observable is not exponentially small, C(T) >= CFRAC max_T C with
%   CFRAC = 1e-6 (key Tlo: lowest temperature of this window over all
%   systems).  Below Tlo, where C itself is exponentially small, relative
%   deviations can be large although the absolute deviations are tiny;
%   these are reported separately (keys *_lowT).

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
addpath(here, fileparts(fileparts(here)));
R = struct();
CFRAC = 1e-6;
u32 = 2^-24;  u16 = 2^-11;  ubf = 2^-8;

%% ---------------- common temperature window --------------------------------
keys = {'ico_s1', 'Icosahedron, $s=1$'; 'ico_s3o2', 'Icosahedron, $s=3∕2$';
        'ring12_s1', 'Ring $N=12$, $s=1$'; 'ring20_s1o2', 'Ring $N=20$, $s=1∕2$';
        'dodeca_s1o2', 'Dodecahedron, $s=1∕2$'; 'icosid_M0', 'Icosidodecahedron, $s=1∕2$ ($M=0$)'};
Tlo = precision_window(study_dir, CFRAC);
R.Tlo = sprintf('%.3g', Tlo);
win = @(T) T(:)' >= Tlo;

%% ---------------- Table 3 (benchmark) ------------------------------------
B = load(bench_file);
res = B.results;
t = @(sys, m) pick(res, sys, m, 't_lanczos');
ts = @(sys, m) pick(res, sys, m, 't_sector');
sysdef = {'ico_s1', 'Icosahedron, $s=1$', '7.4⋅10^^4^^';
          'ico_s3o2', 'Icosahedron, $s=3∕2$', '1.7⋅10^^6^^';
          'ico_s2_M0', 'Icosahedron, $s=2$ ($M=0$)', '2.0⋅10^^7^^';
          'icosid_M0', 'Icosidodecahedron, $s=1∕2$ ($M=0$)', '1.55⋅10^^8^^'};
rows = {{'System', '$𝒟_(M=0)$', 'CPU FP64', 'CPU FP32', 'GPU FP64 (CLT)', 'GPU FP32 (CLT)', ...
         'GPU FP32 (CR)', 'GPU FP16 (CLT)', 'Speedup FP64', 'Speedup FP32'}};
sp64 = []; sp32 = []; cpu_gain = []; gpu_ratio = []; f16_gain = []; overhead = [];
for k = 1 : size(sysdef, 1)
    s = sysdef{k, 1};
    c64 = t(s, 'CPU-CLT-FP64');  c32 = t(s, 'CPU-CLT-FP32');
    g64c = t(s, 'GPU-CLT-FP64');  g64 = min(g64c, t(s, 'GPU-CR-FP64'));
    g32c = t(s, 'GPU-CLT-FP32');  g32r = t(s, 'GPU-CR-FP32');  g16 = t(s, 'GPU-CLT-FP16');
    s64 = c64 / g64;  s32 = c32 / min(g32c, g32r);
    rows{end+1} = {sysdef{k, 2}, sysdef{k, 3}, f3(c64), f3(c32), f3(g64c), f3(g32c), f3(g32r), ...
                   f3(g16), sprintf('%.1f×', s64), sprintf('%.1f×', s32)}; %#ok<AGROW>
    if k == 1, R.sp_s1 = rng_txt([s64 s32], '%.1f'); end
    if k >= 2   % sector dimensions >= 1e6
        sp64(end+1) = s64; sp32(end+1) = s32; %#ok<AGROW>
        cpu_gain(end+1) = 1 - c32 / c64; %#ok<AGROW>
        gpu_ratio(end+1) = g64c / g32c; %#ok<AGROW>
        f16_gain(end+1) = 1 - g16 / g32c; %#ok<AGROW>
        overhead(end+1) = ts(s, 'GPU-CLT-FP32') / g32c - 1; %#ok<AGROW>
    end
end
R.table3_rows = rows;
crd = arrayfun(@(k) t(sysdef{k, 1}, 'GPU-CR-FP32') / t(sysdef{k, 1}, 'GPU-CLT-FP32') - 1, 1 : 3);
R.cr_clt_ico = [rng_txt(100 * abs(crd), '%.0f') ' %'];
R.cr_icosid_pct = sprintf('%.0f %%', 100 * (t('icosid_M0', 'GPU-CR-FP32') / t('icosid_M0', 'GPU-CLT-FP32') - 1));
R.sp64_range = rng_txt(sp64, '%.1f');
R.sp32_range = rng_txt(sp32, '%.1f');
R.sp_min = sprintf('%.1f', min([sp64 sp32]));
R.sp_max = sprintf('%.1f', max([sp64 sp32]));
R.cpu32_gain = [rng_txt(100 * cpu_gain, '%.0f') ' %'];
R.gpu_ratio = rng_txt(gpu_ratio, '%.1f');
R.fp16_gain = [rng_txt(100 * f16_gain, '%.0f') ' %'];
R.setup_overhead = [rng_txt(100 * overhead, '%.0f') ' %'];
nr = arrayfun(@(r) numel(r.t_lanczos_runs), res);
R.n_runs_text = sprintf('%d (icosidodecahedron: %d)', max(nr) - 1, min(nr) - 1);
R.ic_clt32 = f3(t('icosid_M0', 'GPU-CLT-FP32'));  R.ic_cr32 = f3(t('icosid_M0', 'GPU-CR-FP32'));
R.ic_clt64 = f3(t('icosid_M0', 'GPU-CLT-FP64'));
R.ic_cpu64 = f3(t('icosid_M0', 'CPU-CLT-FP64'));  R.ic_cpu32 = f3(t('icosid_M0', 'CPU-CLT-FP32'));
R.hist_cpu = sprintf('%.0f', 2.5 * ts('icosid_M0', 'CPU-CLT-FP64') / 60);
R.hist_gpu = sprintf('%.0f', 2.5 * ts('icosid_M0', 'GPU-CLT-FP32') / 60);
R.b1_text = sprintf('%s s vs. %s s for the $s=3∕2$ icosahedron and %s s vs. %s s for the $s=2$ icosahedron', ...
    f3(t('ico_s3o2', 'GPU-CLT-FP32-B1')), f3(t('ico_s3o2', 'GPU-CR-FP32-B1')), ...
    f3(t('ico_s2_M0', 'GPU-CLT-FP32-B1')), f3(t('ico_s2_M0', 'GPU-CR-FP32-B1')));

%% ---------------- memory traffic (memory_traffic.py) ------------------------
f = fullfile(study_dir, 'memory_traffic.json');
if isfile(f)
    MT = jsondecode(fileread(f));
    rr = MT.runs;
    sel = @(sys, lk, pr) rr(strcmp({rr.system}, sys) & strcmp({rr.lookup}, lk) & strcmp({rr.precision}, pr));
    vec = [];  frac = [];
    for k = 1 : numel(rr)
        if ~strcmp(rr(k).precision, 'half'), vec(end+1) = rr(k).vec_GBs; end %#ok<AGROW>
        frac(end+1) = rr(k).spmv_fraction; %#ok<AGROW>
    end
    i32 = sel('icosid_M0', 'clt', 'single');  i64 = sel('icosid_M0', 'clt', 'double');
    i16 = sel('icosid_M0', 'clt', 'half');   s32 = sel('ico_s2_M0', 'clt', 'single');
    s64 = sel('ico_s2_M0', 'clt', 'double');  s16 = sel('ico_s2_M0', 'clt', 'half');
    R.bw_copy = sprintf('%.0f', MT.bw_copy_GBs);
    R.bw_nominal = sprintf('%.0f', MT.bw_nominal_GBs);
    R.bw_l2 = sprintf('%.0f', MT.l2_MB);
    R.mt_vec = rng_txt(vec, '%.0f');
    R.mt_spmv_frac = [rng_txt(100 * frac, '%.0f') ' %'];
    R.mt_host = [rng_txt(100 * [rr.host_overhead], '%.0f') ' %'];
    R.mt_offdiag = sprintf('%.1f', i32.offdiag_per_row);
    R.mt_noreuse_GB = sprintf('%.0f', i32.spmv_noreuse_GB);
    R.mt_noreuse_GBs = sprintf('%.0f', i32.spmv_noreuse_GBs);
    R.mt_sector_GBs = sprintf('%.0f', i32.spmv_sectors_GBs);
    R.mt_comp_GBs = sprintf('%.0f', i32.spmv_compulsory_GBs);
    R.mt_ratio64 = rng_txt([i64.t_spmv / i32.t_spmv, s64.t_spmv / s32.t_spmv], '%.1f');
    R.mt_fp16_gain = [rng_txt(100 * [1 - i16.t_spmv / i32.t_spmv, 1 - s16.t_spmv / s32.t_spmv], '%.0f') ' %'];
end

%% ---------------- precision study (Figs. 1, 3; Table 5) -------------------
rows5 = {{'System', '$𝒟_\"max\"$', '$W$', 'FP32 $|ΔC|$', 'FP32 $|Δχ|$', 'FP16 $|ΔC|$', ...
          'FP16 $|Δχ|$', 'BF16 $|ΔC|$', 'CPU FP32 $|ΔC|$', 'CPU–GPU FP64 $|ΔC|$'}};
dc32 = [];
W = []; d32 = []; d16 = []; d16x = []; dbf = []; dcg = []; ord1 = [];
d16low = []; dbflow = []; d32low = []; rel32 = []; dE16 = []; dEbf = [];
for k = 1 : size(keys, 1)
    f = fullfile(study_dir, sprintf('study_precision_%s.mat', keys{k, 1}));
    if ~isfile(f), continue; end
    S = load(f);
    w = win(S.T_range);
    ref = S.gpu_double;
    Wk = S.N_B * S.s^2 - ref.E0;
    dC = @(x) abs(x.C(:)' - ref.C(:)');
    dX = @(x) abs(x.chi(:)' - ref.chi(:)');
    mC = @(x) max(dC(x) .* w);                       % maximum inside the window
    mX = @(x) max(dX(x) .* w);
    lowC = @(x) max([0, dC(x) .* ~w]);               % maximum below the window
    has_chi = any(ref.chi ~= 0);
    xs = @(x) iff(has_chi, sci(mX(x)), '–');
    cgC = NaN; cg = NaN;
    if isfield(S, 'cpu_double')
        cgC = max(dC(S.cpu_double));                  % all T: rounding level
        cg = max([cgC, iff(has_chi, max(dX(S.cpu_double)), 0)]);
    end
    c32 = NaN;
    if isfield(S, 'cpu_single'), c32 = mC(S.cpu_single); dc32(end+1) = c32; end %#ok<AGROW>
    rows5{end+1} = {keys{k, 2}, sci(S.dim_max), sprintf('%.1f', Wk), sci(mC(S.gpu_single)), ...
        xs(S.gpu_single), sci(mC(S.gpu_half)), xs(S.gpu_half), sci(mC(S.gpu_bfloat16)), ...
        sci_or_dash(c32), sci_or_dash(cgC)}; %#ok<AGROW>
    W(end+1) = Wk; d32(end+1) = mC(S.gpu_single); d16(end+1) = mC(S.gpu_half); %#ok<AGROW>
    if has_chi, d16x(end+1) = mX(S.gpu_half); end %#ok<AGROW>
    dbf(end+1) = mC(S.gpu_bfloat16); dcg(end+1) = cg; %#ok<AGROW>
    d32low(end+1) = lowC(S.gpu_single); d16low(end+1) = lowC(S.gpu_half); %#ok<AGROW>
    dbflow(end+1) = lowC(S.gpu_bfloat16); %#ok<AGROW>
    rel32(end+1) = abs(S.gpu_single.E0 / ref.E0 - 1); %#ok<AGROW>
    dE16(end+1) = abs(S.gpu_half.E0 - ref.E0); dEbf(end+1) = abs(S.gpu_bfloat16.E0 - ref.E0); %#ok<AGROW>
    if any(strcmp(keys{k, 1}, {'ico_s1', 'ico_s3o2'}))
        ord1(end+1) = log10(max(ref.C) / mC(S.gpu_single)); %#ok<AGROW>
        if has_chi, ord1(end+1) = log10(max(ref.chi) / mX(S.gpu_single)); end %#ok<AGROW>
    end
end
R.table5_rows = rows5;
R.W_min = sprintf('%.0f', min(W));  R.W_max = sprintf('%.0f', max(W));
R.uW_min = sci(u32 * min(W));  R.uW_max = sci(u32 * max(W));
R.uW16_min = sci(u16 * min(W));  R.uWbf_max = sci(ubf * max(W));
R.fp32_dC_min = sci(min(d32));  R.fp32_dC_max = sci(max(d32));
R.fp32_lowT = sci(max(d32low));
R.fp16_dC_range = rng_sci(d16);
R.fp16_dchi_max = sci(max(d16x));
R.fp16_lowT = sci(max(d16low));
R.bf16_dC_max = sci(max(dbf));
R.bf16_lowT = sci(max(dbflow));
R.cpu_gpu64_max = sci(max(dcg));
if ~isempty(dc32), R.cpu32_dC_range = rng_sci(dc32); end
R.fig1_orders = orders_txt(ord1);
R.E0_rel_fp32 = rng_sci(rel32);
R.fp16_dE0 = sci(max(dE16));
R.bf16_dE0 = sci(max(dEbf));

%% ---------------- seeds (Fig. 2, R*) --------------------------------------
Rs = []; ord2 = []; pool_txt = {}; lowseed = [];
names = {'ico_s1', '$s=1$'; 'ico_s3o2', '$s=3∕2$'};
for k = 1 : 2
    f = fullfile(study_dir, sprintf('study_seeds_%s.mat', names{k, 1}));
    if ~isfile(f), continue; end
    S = load(f);
    for fn = {'sigma_emp_C', 'sigma_theo_C', 'sigma_emp_chi', 'sigma_theo_chi', 'Rstar_C', 'Rstar_chi'}
        S.(fn{1}) = S.(fn{1})(:)';                   % row vectors over T
    end
    w = win(S.T_range);
    okC = w & S.dC_k(1, :) > 0;  okX = w & S.dchi_k(1, :) > 0;
    Rs(end+1) = min(S.Rstar_C(okC & S.dC_k(end, :) > 0)); %#ok<AGROW>
    Rs(end+1) = min(S.Rstar_chi(okX & S.dchi_k(end, :) > 0)); %#ok<AGROW>
    rC = min(S.sigma_emp_C(okC), S.sigma_theo_C(okC)) ./ S.dC_k(1, okC);
    rX = min(S.sigma_emp_chi(okX), S.sigma_theo_chi(okX)) ./ S.dchi_k(1, okX);
    ord2(end+1) = log10(min([rC(:); rX(:)])); %#ok<AGROW>
    pool_txt{end+1} = sprintf('changes from %s ($k=1$) to %s ($k=%d$) for %s', ...
        sci(max(S.dC_k(1, :) .* w)), sci(max(S.dC_k(end, :) .* w)), S.k_list(end), names{k, 2}); %#ok<AGROW>
    R.(sprintf('k_max_%s', strrep(names{k, 1}, 'ico_', ''))) = sprintf('%d', S.k_list(end));
    lowseed(end+1) = max([0, S.dC_k(1, :) .* ~w]); %#ok<AGROW>
    if k == 1, R.n_seeds = sprintf('%d', S.Ns); R.R_seeds_eff = sprintf('%d', S.Ns * S.R); end
end
if ~isempty(Rs)
    R.Rstar_min = sci_pow(min(Rs));
    R.fig2_orders = orders_txt(ord2);
    R.dC_pool_text = strjoin(pool_txt, ' and ');
    R.seed_lowT = sci(max(lowseed));
    if min(ord2) < 1, warning('summarize_results:margin', 'FP32 deviation not clearly below the stochastic error in the window'); end
end

%% ---------------- N_L study -----------------------------------------------
for key = {'ico_s1', 'ico_s3o2'}
    f = fullfile(study_dir, sprintf('study_lanczos_steps_%s.mat', key{1}));
    if ~isfile(f), continue; end
    S = load(f);
    w = win(S.T_range);
    Cm = max(S.double.C(end, :));
    tag = strrep(key{1}, 'ico_', '');
    trunc = max(S.dC_trunc .* w, [], 2)' / Cm;           % per N_L, inside the window
    fp32 = max(S.dC_fp32 .* w, [], 2)' / Cm;
    [pk, ip] = max(fp32);
    R.(['nl_fp32_peak_' tag]) = sci(pk);
    R.(['nl_fp32_peakNL_' tag]) = sprintf('%d', S.NL(ip));
    if any(S.NL == 100), R.(['nl_fp32_100_' tag]) = sci(fp32(S.NL == 100)); end
    % at the maximum of the transient FP32 deviation: below the truncation error?
    R.(['nl_peak_below_trunc_' tag]) = iff(fp32(ip) < trunc(ip), 'yes', 'no');
    R.(['nl_trunc_at_peak_' tag]) = sci(trunc(ip));
    if strcmp(tag, 's3o2') && any(S.NL == 60) && any(S.NL == 100)
        R.tr60 = sci(trunc(S.NL == 60));
        R.tr100 = sci(trunc(S.NL == 100));
        R.tr_vs_fp32 = iff(trunc(S.NL == 100) < fp32(S.NL == 100), 'below', 'above');
    end
end

%% ---------------- ED decomposition ----------------------------------------
f = fullfile(study_dir, 'study_ed_cube_s3o2.mat');
if isfile(f)
    S = load(f);
    w = win(S.T_range);
    mw = @(x) max(x.C(:)' .* w);
    R.ed_stoch = sci(mw(S.err_stoch));
    R.ed_trunc = sci(mw(S.err_trunc));
    R.ed_fp32 = sci(mw(S.err_fp32));
    R.ed_fp16 = sci(mw(S.err_fp16));
    R.ed_bf16 = sci(mw(S.err_bf16));
    R.ed_impl = sci(mw(S.err_impl));
    R.ed_spmv = sci(S.spmv_err);
    R.ed_R = sprintf('%d', S.R);
    % pointwise: is the FP32 (FP16) deviation below the stochastic error in the window?
    R.ed_fp32_below = iff(all(S.err_fp32.C(w) <= S.err_stoch.C(w)), 'yes', 'no');
    R.ed_fp16_below = iff(all(S.err_fp16.C(w) <= S.err_stoch.C(w)), 'yes', 'no');
end
f = fullfile(study_dir, 'study_exact_icosahedron.mat');
if isfile(f)
    S = load(f);
    R.ico_exact_err = sprintf('%.2f', S.max_err_total);
    R.ico_exact_pct = sprintf('%.1f', 100 * S.max_err_total / S.C_max);
    R.ico_exact_T = sprintf('%.2f', S.T_at_max);
    R.ico_exact_fp32 = sci(S.max_dev_gpu_single);
    R.ico_exact_orders = num2words(floor(log10(S.max_err_total / S.max_dev_gpu_single)));
    R.ico_exact_Tmin = sprintf('%.3g', min(S.T));
    R.ico_exact_Tmax = sprintf('%.3g', max(S.T));
end

%% ---------------- ghosts --------------------------------------------------
f = fullfile(study_dir, 'study_ghosts.mat');
if isfile(f)
    S = load(f);
    R.ng32 = sprintf('%d', nnz(S.fp32.ghost));
    R.ng64 = sprintf('%d', nnz(S.fp64.ghost));
    R.ghost_nclu64 = num2words(numel(S.fp64.clusters));
    R.ghost_nclu32 = num2words(numel(S.fp32.clusters));
    % total weight of each FP32 cluster vs. the FP64 weight at the same energy
    wrel = 0;  extra = [];
    for c = 1 : numel(S.fp32.clusters)
        th = S.fp32.clusters(c).theta;
        w64 = sum(S.fp64.w(abs(S.fp64.theta - th) < 1e-3));
        wrel = max(wrel, abs(S.fp32.clusters(c).w_total - w64) / w64);
        if ~any(abs([S.fp64.clusters.theta] - th) < 1e-3), extra(end+1) = th; end %#ok<AGROW>
    end
    R.ghost_wrel = sci(wrel);
    R.ghost_theta_extra = strjoin(arrayfun(@(x) sprintf('%.2f', x), extra, 'UniformOutput', false), ', ');
    % isolated ghost-flagged values (not in a cluster): maximal weight
    wiso = 0;
    for name = {'fp64', 'fp32'}
        X = S.(name{1});
        for i = find(X.ghost(:))'
            if nnz(X.cluster == X.cluster(i)) == 1, wiso = max(wiso, X.w(i)); end
        end
    end
    R.ghost_wiso = sci(wiso);
    % range of C_tau with the same classification: the ghost test compares
    % ell_k, the clustering compares neighbour distances, both with tau
    us = S.tau / S.Ctau;
    cand = [S.fp32.ell(:); S.fp64.ell(:); diff(sort(S.fp32.theta(:))); diff(sort(S.fp64.theta(:)))] / us;
    R.Ctau_lo = sprintf('%.1f', max(cand(cand < S.Ctau)));
    R.Ctau_hi = sprintf('%.0f', min(cand(cand >= S.Ctau)));
end

fid = fopen(out_json, 'w', 'n', 'UTF-8');
fwrite(fid, jsonencode(R, 'PrettyPrint', true), 'char');
fclose(fid);
fprintf('wrote %s (Tlo = %s)\n', out_json, R.Tlo);
diagnostics(study_dir, keys(:, 1), Tlo);
end

%% ========================================================================
function diagnostics(study_dir, keys, Tlo)
%DIAGNOSTICS  Temperature-resolved deviations (for the discussion), printed
%   and written to STUDY_DIR/diagnostics.txt.
    fid = fopen(fullfile(study_dir, 'diagnostics.txt'), 'w');
    out = @(varargin) outp(fid, varargin{:});
    out('Window: T >= %.4g\n\n', Tlo);
    out('Precision study: max_T |dC| (all T | T >= Tlo | T >= 0.1), T at max, E0 - E0_FP64\n');
    for k = 1 : numel(keys)
        f = fullfile(study_dir, sprintf('study_precision_%s.mat', keys{k}));
        if ~isfile(f), continue; end
        S = load(f);  T = S.T_range(:)';  ref = S.gpu_double;
        for v = {'gpu_single', 'gpu_half', 'gpu_bfloat16', 'cpu_single', 'cpu_double'}
            if ~isfield(S, v{1}), continue; end
            x = S.(v{1});  d = abs(x.C(:)' - ref.C(:)');  [m, i] = max(d);
            out('  %-12s %-13s %9.2e | %9.2e | %9.2e  T=%6.3f  dE0=%+9.2e  (max C %.3f, C(Tmin) %.2e)\n', ...
                keys{k}, v{1}, m, max(d(T >= Tlo)), max(d(T >= 0.1)), T(i), x.E0 - ref.E0, max(ref.C), ref.C(1));
        end
    end
    out('\nLanczos-step study: per N_L, max_T |dC_trunc| and |dC_fp32| (all T | T >= Tlo), relative to max C\n');
    for key = {'ico_s1', 'ico_s3o2'}
        f = fullfile(study_dir, sprintf('study_lanczos_steps_%s.mat', key{1}));
        if ~isfile(f), continue; end
        S = load(f);  T = S.T_range(:)';  Cm = max(S.double.C(end, :));
        for j = 1 : numel(S.NL)
            dt = S.dC_trunc(j, :);  d3 = S.dC_fp32(j, :);
            out('  %-9s N_L=%4d  trunc %9.2e | %9.2e   fp32 %9.2e | %9.2e\n', key{1}, S.NL(j), ...
                max(dt) / Cm, max(dt(T >= Tlo)) / Cm, max(d3) / Cm, max(d3(T >= Tlo)) / Cm);
        end
    end
    for key = {'ico_s1', 'ico_s3o2'}
        f = fullfile(study_dir, sprintf('study_seeds_%s.mat', key{1}));
        if ~isfile(f), continue; end
        S = load(f);  T = S.T_range(:)';
        [m, i] = min(S.Rstar_C);
        out('\nSeeds %s: min R*_C %.2e at T=%.4f; min R*_C (T>=Tlo) %.2e; min R*_chi %.2e\n', key{1}, ...
            m, T(i), min(S.Rstar_C(T >= Tlo)), min(S.Rstar_chi));
    end
    f = fullfile(study_dir, 'study_ed_cube_s3o2.mat');
    if isfile(f)
        S = load(f);  T = S.T_range(:)';
        out('\nED decomposition (cube s=3/2): max |dC| all T | T >= Tlo | T >= 0.1\n');
        for e = {'err_stoch', 'err_trunc', 'err_fp32', 'err_fp16', 'err_bf16', 'err_impl'}
            d = S.(e{1}).C(:)';
            out('  %-10s %9.2e | %9.2e | %9.2e\n', e{1}, max(d), max(d(T >= Tlo)), max(d(T >= 0.1)));
        end
        for e = {'err_fp32', 'err_fp16', 'err_bf16'}
            above = T(S.(e{1}).C(:)' > S.err_stoch.C(:)');
            if isempty(above), out('  %s <= stochastic at all T\n', e{1});
            else, out('  %s > stochastic at %d T points, up to T = %.3f\n', e{1}, numel(above), max(above));
            end
        end
    end
    fclose(fid);
end

function outp(fid, varargin)
    fprintf(1, varargin{:});
    fprintf(fid, varargin{:});
end

function x = pick(res, sys, m, field)
    k = find(strcmp({res.system}, sys) & strcmp({res.method}, m), 1);
    if isempty(k), x = NaN; else, x = res(k).(field); end
end

function s = f3(x)
    if ~isfinite(x), error('summarize_results:nonfinite', 'missing timing'); end
    if x >= 100, s = sprintf('%.0f', x); elseif x >= 10, s = sprintf('%.1f', x); else, s = sprintf('%.2f', x); end
end

function s = rng_txt(x, f)
    a = sprintf(f, min(x));  b = sprintf(f, max(x));
    if strcmp(a, b), s = a; else, s = [a '–' b]; end
end

function s = rng_sci(x)
    a = sci(min(x));  b = sci(max(x));
    if strcmp(a, b), s = a; else, s = [a '–' b]; end
end

function s = orders_txt(o)
    a = floor(min(o));  b = floor(max(o));
    if a == b, s = sprintf('%d', a); else, s = sprintf('%d to %d', a, b); end
end

function s = sci(x)
    if ~isfinite(x), error('summarize_results:nonfinite', 'non-finite value'); end
    if x == 0, s = '0'; return; end
    e = floor(log10(abs(x)));
    m = x / 10^e;
    if round(m, 1) >= 10, m = m / 10; e = e + 1; end
    if e >= -2 && e <= 2
        s = sprintf('%.2g', x);
    else
        s = sprintf('%.1f×10^^%d^^', m, e);
        s = strrep(s, '^^-', '^^−');
    end
end

function s = sci_or_dash(x)
    if isnan(x), s = '–'; else, s = sci(x); end
end

function s = sci_pow(x)
    s = sprintf('10^^%d^^', floor(log10(x)));
end

function y = iff(c, a, b)
    if c, y = a; else, y = b; end
end

function s = num2words(n)
    w = {'no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'};
    if n >= 0 && n <= 10, s = w{n + 1}; else, s = sprintf('%d', n); end
end
