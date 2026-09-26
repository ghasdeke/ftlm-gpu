function R = summarize_results(study_dir, bench_file, out_json)
%SUMMARIZE_RESULTS  Numbers quoted in the paper, from the study and benchmark files.
%   R = SUMMARIZE_RESULTS(STUDY_DIR, BENCH_FILE, OUT_JSON) reads the
%   study_*.mat files and memory_traffic.json (from memory_traffic.py) in
%   STUDY_DIR and the benchmark_table3 output BENCH_FILE and writes the
%   quoted numbers and the rows of Tables 3 and 5 as JSON (strings;
%   '^^...^^' marks superscripts).

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

%% ---------------- Table 3 (benchmark) ------------------------------------
B = load(bench_file);
res = B.results;
t = @(sys, m) pick(res, sys, m, 't_lanczos');
ts = @(sys, m) pick(res, sys, m, 't_sector');
bb = @(sys, m) pick(res, sys, m, 'B');
e0 = @(sys, m) pick(res, sys, m, 'E0');
sysdef = {'ico_s1', 'Icosahedron, $s=1$', '7.4⋅10^^4^^';
          'ico_s3o2', 'Icosahedron, $s=3∕2$', '1.7⋅10^^6^^';
          'ico_s2_M0', 'Icosahedron, $s=2$ ($M=0$)', '2.0⋅10^^7^^';
          'icosid_M0', 'Icosidodecahedron, $s=1∕2$ ($M=0$)', '1.55⋅10^^8^^'};
rows = {{'System', '$𝒟_(M=0)$', 'CPU FP64', 'CPU FP32', 'GPU FP64', 'GPU FP32 (CLT)', ...
         'GPU FP32 (CR)', 'GPU FP16 (CLT)', 'Speedup FP64', 'Speedup FP32'}};
sp64 = []; sp32 = []; cpu_gain = []; gpu_ratio = []; f16_gain = []; overhead = [];
for k = 1 : size(sysdef, 1)
    s = sysdef{k, 1};
    c64 = t(s, 'CPU-CLT-FP64');  c32 = t(s, 'CPU-CLT-FP32');
    g64 = min(t(s, 'GPU-CLT-FP64'), t(s, 'GPU-CR-FP64'));
    g32c = t(s, 'GPU-CLT-FP32');  g32r = t(s, 'GPU-CR-FP32');  g16 = t(s, 'GPU-CLT-FP16');
    s64 = c64 / g64;  s32 = c32 / min(g32c, g32r);
    rows{end+1} = {sysdef{k, 2}, sysdef{k, 3}, f3(c64), f3(c32), f3(g64), f3(g32c), f3(g32r), ...
                   f3(g16), sprintf('%.1f×', s64), sprintf('%.1f×', s32)}; %#ok<AGROW>
    if k >= 2   % sectors >= 1e6
        sp64(end+1) = s64; sp32(end+1) = s32; %#ok<AGROW>
        cpu_gain(end+1) = 1 - c32 / c64; %#ok<AGROW>
        gpu_ratio(end+1) = t(s, 'GPU-CLT-FP64') / g32c; %#ok<AGROW>
        f16_gain(end+1) = 1 - g16 / g32c; %#ok<AGROW>
        overhead(end+1) = ts(s, 'GPU-CLT-FP32') / g32c - 1; %#ok<AGROW>
    end
end
R.table3_rows = rows;
R.sp64_range = rng_txt(sp64, '%.1f');
R.sp32_range = rng_txt(sp32, '%.1f');
R.sp_min = sprintf('%.0f', floor(min([sp64 sp32])));
R.sp_max = sprintf('%.0f', ceil(max([sp64 sp32])));
R.cpu32_gain = [rng_txt(100 * cpu_gain, '%.0f') ' %'];
R.gpu_ratio = rng_txt(gpu_ratio, '%.1f');
R.fp16_gain = [rng_txt(100 * f16_gain, '%.0f') ' % shorter than FP32'];
R.setup_overhead = [rng_txt(100 * overhead, '%.0f') ' % for the GPU FP32 runs of the large systems'];
nr = arrayfun(@(r) numel(r.t_lanczos_runs), res);
R.n_runs_text = sprintf('%d (icosidodecahedron: %d)', max(nr) - 1, min(nr) - 1);
R.B_footnote = sprintf(['$B=%d$ for the $s=1$ icosahedron and $B=%d$ for the other systems ', ...
    '(CLT and CR, all precisions); the CPU kernel uses $B=8$.'], bb('ico_s1', 'GPU-CLT-FP32'), ...
    bb('icosid_M0', 'GPU-CLT-FP32'));
R.ic_clt32 = f3(t('icosid_M0', 'GPU-CLT-FP32'));  R.ic_cr32 = f3(t('icosid_M0', 'GPU-CR-FP32'));
R.ic_clt64 = f3(t('icosid_M0', 'GPU-CLT-FP64'));
R.ic_cpu64 = f3(t('icosid_M0', 'CPU-CLT-FP64'));  R.ic_cpu32 = f3(t('icosid_M0', 'CPU-CLT-FP32'));
R.hist_cpu = sprintf('%.0f', 2.5 * ts('icosid_M0', 'CPU-CLT-FP64') / 60);
R.hist_gpu = sprintf('%.0f', 2.5 * ts('icosid_M0', 'GPU-CLT-FP32') / 60);
R.b1_text = sprintf('%s s vs. %s s for the $s=3∕2$ icosahedron and %s s vs. %s s for the $s=2$ icosahedron', ...
    f3(t('ico_s3o2', 'GPU-CLT-FP32-B1')), f3(t('ico_s3o2', 'GPU-CR-FP32-B1')), ...
    f3(t('ico_s2_M0', 'GPU-CLT-FP32-B1')), f3(t('ico_s2_M0', 'GPU-CR-FP32-B1')));
dE0 = [];
for k = 1 : size(sysdef, 1)
    dE0(end+1) = abs(e0(sysdef{k, 1}, 'GPU-CLT-FP16') - e0(sysdef{k, 1}, 'GPU-CLT-FP64')); %#ok<AGROW>
end
R.fp16_dE0 = sci(max(dE0));
rel32 = [];
for k = 1 : size(sysdef, 1)
    rel32(end+1) = abs(e0(sysdef{k, 1}, 'GPU-CLT-FP32') / e0(sysdef{k, 1}, 'GPU-CLT-FP64') - 1); %#ok<AGROW>
end
R.E0_rel_fp32 = sprintf('%s to %s', sci(min(rel32)), sci(max(rel32)));

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
    R.bw_l2 = sprintf('%.0f', MT.l2_MB);
    R.mt_vec = rng_txt(vec, '%.0f');
    R.mt_spmv_frac = [rng_txt(100 * frac, '%.0f') ' %'];
    R.mt_offdiag = sprintf('%.1f', i32.offdiag_per_row);
    R.mt_noreuse_GB = sprintf('%.0f', i32.spmv_noreuse_GB);
    R.mt_noreuse_GBs = sprintf('%.0f', i32.spmv_noreuse_GBs);
    R.mt_comp_GBs = sprintf('%.0f', i32.spmv_compulsory_GBs);
    R.mt_ratio64 = rng_txt([i64.t_spmv / i32.t_spmv, s64.t_spmv / s32.t_spmv], '%.1f');
    R.mt_fp16_gain = [rng_txt(100 * [1 - i16.t_spmv / i32.t_spmv, 1 - s16.t_spmv / s32.t_spmv], '%.0f') ' %'];
end

%% ---------------- precision study (Figs. 1, 3; Table 5) -------------------
keys = {'ico_s1', 'Icosahedron, $s=1$'; 'ico_s3o2', 'Icosahedron, $s=3∕2$';
        'ring12_s1', 'Ring $N=12$, $s=1$'; 'ring20_s1o2', 'Ring $N=20$, $s=1∕2$';
        'dodeca_s1o2', 'Dodecahedron, $s=1∕2$'; 'icosid_M0', 'Icosidodecahedron, $s=1∕2$ ($M=0$)'};
rows5 = {{'System', '$𝒟_\"max\"$', '$W$', '$u_\"FP32\" W$', 'FP32 $|ΔC|$', 'FP32 $|Δχ|$', ...
          'FP16 $|ΔC|$', 'FP16 $|Δχ|$', 'BF16 $|ΔC|$', 'CPU–GPU FP64 $|ΔC|$'}};
W = []; d32 = []; d16 = []; d16x = []; dbf = []; dcg = []; ord1 = [];
u32 = 2^-24;
for k = 1 : size(keys, 1)
    f = fullfile(study_dir, sprintf('study_precision_%s.mat', keys{k, 1}));
    if ~isfile(f), continue; end
    S = load(f);
    ref = S.gpu_double;
    Wk = S.N_B * S.s^2 - ref.E0;
    mC = @(x) max(abs(x.C - ref.C));
    mX = @(x) max(abs(x.chi - ref.chi));
    has_chi = any(ref.chi ~= 0);
    xs = @(x) iff(has_chi, sci(mX(x)), '–');
    cg = NaN; if isfield(S, 'cpu_double'), cg = max([mC(S.cpu_double), iff(has_chi, mX(S.cpu_double), 0)]); end
    rows5{end+1} = {keys{k, 2}, sci(S.dim_max), sprintf('%.1f', Wk), sci(u32 * Wk), sci(mC(S.gpu_single)), ...
        xs(S.gpu_single), sci(mC(S.gpu_half)), xs(S.gpu_half), sci(mC(S.gpu_bfloat16)), sci(cg)}; %#ok<AGROW>
    W(end+1) = Wk; d32(end+1) = mC(S.gpu_single); d16(end+1) = mC(S.gpu_half); %#ok<AGROW>
    if has_chi, d16x(end+1) = mX(S.gpu_half); end %#ok<AGROW>
    dbf(end+1) = max(mC(S.gpu_bfloat16), iff(has_chi, mX(S.gpu_bfloat16), 0)); %#ok<AGROW>
    dcg(end+1) = cg; %#ok<AGROW>
    if any(strcmp(keys{k, 1}, {'ico_s1', 'ico_s3o2'}))
        ord1(end+1) = log10(max(ref.C) / mC(S.gpu_single)); %#ok<AGROW>
        if has_chi, ord1(end+1) = log10(max(ref.chi) / mX(S.gpu_single)); end %#ok<AGROW>
    end
    if strcmp(keys{k, 1}, 'icosid_M0'), R.W_icosid = sprintf('%.0f', Wk); end
end
R.table5_rows = rows5;
R.W_min = sprintf('%.0f', min(W));  R.W_max = sprintf('%.0f', max(W));
R.uW_min = sci(u32 * min(W));  R.uW_max = sci(u32 * max(W));
R.fp32_dC_min = sci(min(d32));  R.fp32_dC_max = sci(max(d32));
R.fp16_dC_range = sprintf('%s–%s', sci(min(d16)), sci(max(d16)));
R.fp16_dchi_max = sci(max(d16x));
R.bf16_dC_max = sci(max(dbf));
R.cpu_gpu64_max = sci(max(dcg));
R.fig1_orders = sprintf('%d to %d', floor(min(ord1)), floor(max(ord1)));

%% ---------------- seeds (Fig. 2, R*) --------------------------------------
Rs = []; ord2 = []; pool_txt = {};
names = {'ico_s1', '$s=1$'; 'ico_s3o2', '$s=3∕2$'};
for k = 1 : 2
    f = fullfile(study_dir, sprintf('study_seeds_%s.mat', names{k, 1}));
    if ~isfile(f), continue; end
    S = load(f);
    ok = S.dC_k(end, :) > 0;
    Rs(end+1) = min(S.Rstar_C(ok)); %#ok<AGROW>
    okx = S.dchi_k(end, :) > 0;
    Rs(end+1) = min(S.Rstar_chi(okx)); %#ok<AGROW>
    ord2(end+1) = log10(min(S.sigma_emp_C(ok) ./ S.dC_k(1, ok))); %#ok<AGROW>
    pool_txt{end+1} = sprintf('changes from %s ($k=1$) to %s ($k=%d$) for %s', ...
        sci(max(S.dC_k(1, :))), sci(max(S.dC_k(end, :))), S.k_list(end), names{k, 2}); %#ok<AGROW>
    R.(sprintf('k_max_%s', strrep(names{k, 1}, 'ico_', ''))) = sprintf('%d', S.k_list(end));
end
if ~isempty(Rs)
    R.Rstar_min = sci_pow(min(Rs));
    R.fig2_orders = sprintf('%d to %d', floor(min(ord2)), floor(max(ord2)) + 1);
    R.dC_pool_text = strjoin(pool_txt, ' and ');
end

%% ---------------- N_L study -----------------------------------------------
f = fullfile(study_dir, 'study_lanczos_steps_ico_s3o2.mat');
if isfile(f)
    S = load(f);
    Cm = max(S.double.C(end, :));
    R.tr60 = sci(max(S.dC_trunc(S.NL == 60, :)) / Cm);
    R.tr100 = sci(max(S.dC_trunc(S.NL == 100, :)) / Cm);
end

%% ---------------- ED decomposition ----------------------------------------
f = fullfile(study_dir, 'study_ed_cube_s3o2.mat');
if isfile(f)
    S = load(f);
    R.ed_stoch = sci(max(S.err_stoch.C));
    R.ed_trunc = sci(max(S.err_trunc.C));
    R.ed_fp32 = sci(max(S.err_fp32.C));
    R.ed_impl = sci(max(S.err_impl.C));
    R.ed_spmv = sci(S.spmv_err);
end
f = fullfile(study_dir, 'study_exact_icosahedron.mat');
if isfile(f)
    S = load(f);
    R.ico_exact_err = sprintf('%.2f', S.max_err_total);
    R.ico_exact_pct = sprintf('%.1f', 100 * S.max_err_total / S.C_max);
    R.ico_exact_T = sprintf('%.2f', S.T_at_max);
    R.ico_exact_fp32 = sci(S.max_dev_gpu_single);
end

%% ---------------- ghosts --------------------------------------------------
f = fullfile(study_dir, 'study_ghosts.mat');
if isfile(f)
    S = load(f);
    R.ng32 = sprintf('%d', nnz(S.fp32.ghost));
    R.ng64 = sprintf('%d', nnz(S.fp64.ghost));
end

fid = fopen(out_json, 'w', 'n', 'UTF-8');
fwrite(fid, jsonencode(R, 'PrettyPrint', true), 'char');
fclose(fid);
fprintf('wrote %s\n', out_json);
end

function x = pick(res, sys, m, field)
    k = find(strcmp({res.system}, sys) & strcmp({res.method}, m), 1);
    if isempty(k), x = NaN; else, x = res(k).(field); end
end

function s = f3(x)
    if x >= 100, s = sprintf('%.0f', x); elseif x >= 10, s = sprintf('%.1f', x); else, s = sprintf('%.2f', x); end
end

function s = rng_txt(x, f)
    s = sprintf([f '–' f], min(x), max(x));
end

function s = sci(x)
    if ~isfinite(x) || x == 0, s = '0'; return; end
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

function s = sci_pow(x)
    s = sprintf('10^^%d^^', floor(log10(x)));
end

function y = iff(c, a, b)
    if c, y = a; else, y = b; end
end
