function make_figures(varargin)
%MAKE_FIGURES  Figures of the paper from the study_*.mat files.
%   MAKE_FIGURES('DataDir', D, 'OutDir', O, 'Figures', [1 2 3 4 5 6 7])
%   reads the output of study_precision, study_seeds, study_ghosts,
%   study_lanczos_steps and study_ed_decomposition from D and writes
%   figN.pdf (vector) and figN.png (600 dpi) to O.
%
%     Fig. 1  C, chi and |Delta| (FP32 vs FP64, same GPU kernel), icosahedron s=1, 3/2
%     Fig. 2  multi-seed analysis: runs, pooled result, sigma_emp, sigma_theo, |Delta_FP32|
%     Fig. 3  as Fig. 1 for rings, dodecahedron, icosidodecahedron (M=0)
%     Fig. 4  Cullum-Willoughby ghost diagnostic (FP64 | FP32)
%     Fig. 5  cluster weights (FP64 | FP32)
%     Fig. 6  convergence in the number of Lanczos steps N_L
%     Fig. 7  error decomposition against exact diagonalization

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
p.addParameter('DataDir', '.');
p.addParameter('OutDir', '.');
p.addParameter('Figures', 1:7);
p.parse(varargin{:});
o = p.Results;
if ~isfolder(o.OutDir), mkdir(o.OutDir); end

st = style();
for k = o.Figures
    switch k
        case 1, fig = fig_precision(o, st, {'ico_s1', 'ico_s3o2'}, {'s = 1', 's = 3/2'}, [], 1);
        case 2, fig = fig_seeds(o, st);
        case 3, fig = fig_precision(o, st, {'ring12_s1', 'ring20_s1o2', 'dodeca_s1o2', 'icosid_M0'}, ...
                     {'s = 1', 's = 1/2', 's = 1/2', 's = 1/2'}, [], 3);
        case 4, fig = fig_ghosts(o, st, 'ell');
        case 5, fig = fig_ghosts(o, st, 'weights');
        case 6, fig = fig_lanczos_steps(o, st);
        case 7, fig = fig_ed(o, st);
        otherwise, continue;
    end
    save_fig(fig, fullfile(o.OutDir, sprintf('fig%d', k)));
    close(fig);
end
end

%% ========================================================================
function st = style()
    st.c64   = [0.00 0.45 0.74];      % FP64 (blue)
    st.c32   = [0.85 0.10 0.10];      % FP32 (red)
    st.c16   = [0.47 0.67 0.19];      % FP16 (green)
    st.cbf   = [0.49 0.18 0.56];      % BF16 (purple)
    st.cd    = [0 0 0];               % |Delta| (black)
    st.cg    = [0.55 0.55 0.55];      % grey
    st.lw    = 1.3;
    st.lwd   = 1.0;
    st.fs    = 9;                     % tick labels
    st.fl    = 10;                    % axis labels
    st.width = 17;                    % cm, full text width
end

function ax = setup_axes(ax, st)
    ax.FontSize = st.fs;
    ax.TickLabelInterpreter = 'latex';
    ax.Box = 'on';
    ax.LineWidth = 0.6;
    ax.TickDir = 'in';
end

function panel_label(ax, lab, st)
    % label outside the axes (upper left), never overlapping data
    text(ax, -0.22, 1.0, ['(' lab ')'], 'Units', 'normalized', 'FontSize', st.fl, ...
         'FontWeight', 'bold', 'Interpreter', 'none', 'VerticalAlignment', 'bottom', ...
         'HorizontalAlignment', 'left', 'Clipping', 'off');
end

function fig = new_fig(st, h_cm)
    fig = figure('Visible', 'off', 'Color', 'w', 'Units', 'centimeters', ...
                 'Position', [2 2 st.width h_cm], 'PaperUnits', 'centimeters');
    try
        theme(fig, 'light');           % R2025a+: independent of the desktop theme
    catch
    end
end

function save_fig(fig, base)
    exportgraphics(fig, [base '.pdf'], 'ContentType', 'vector', 'BackgroundColor', 'white');
    exportgraphics(fig, [base '.png'], 'Resolution', 600, 'BackgroundColor', 'white');
    fprintf('saved %s.pdf/.png\n', base);
end

function S = load_study(o, name)
    f = fullfile(o.DataDir, name);
    assert(isfile(f), 'missing data file %s', f);
    S = load(f);
end

function d = floor_log(d)
    d(d < 1e-17) = 1e-17;
end

%% ========================================================================
%  Figs. 1 and 3: observables and |Delta| between FP32 and FP64 (GPU kernel)
function fig = fig_precision(o, st, keys, spins, ~, fignum)
    nk = numel(keys);
    fig = new_fig(st, 5.2 * nk);
    tl = tiledlayout(fig, nk, 2, 'TileSpacing', 'compact', 'Padding', 'loose');
    labs = 'abcdefghij';
    ip = 0;
    for q = 1 : nk
        S = load_study(o, sprintf('study_precision_%s.mat', keys{q}));
        T = S.T_range(:);
        ref = S.gpu_double;  x32 = S.gpu_single;
        obs = {'C', 'chi'};
        ylab = {'$C$', '$\chi$'};
        dlab = {'$|\Delta C|$', '$|\Delta\chi|$'};
        for io = 1 : 2
            if strcmp(obs{io}, 'chi') && all(ref.chi == 0)   % M = 0 only
                nexttile(tl); axis off; continue;
            end
            ax = nexttile(tl);
            ip = ip + 1;
            yyaxis(ax, 'left');
            h1 = plot(ax, T, ref.(obs{io}), '-', 'Color', st.c64, 'LineWidth', st.lw); hold(ax, 'on');
            h2 = plot(ax, T, x32.(obs{io}), '--', 'Color', st.c32, 'LineWidth', st.lw);
            ylabel(ax, ylab{io}, 'Interpreter', 'latex', 'FontSize', st.fl);
            ax.YColor = 'k';
            yl = ylim(ax); ylim(ax, [0, 1.12 * yl(2)]);
            yyaxis(ax, 'right');
            plot(ax, T, floor_log(abs(x32.(obs{io}) - ref.(obs{io}))), '-', 'Color', st.cd, ...
                 'LineWidth', st.lwd);
            set(ax, 'YScale', 'log');
            ylabel(ax, dlab{io}, 'Interpreter', 'latex', 'FontSize', st.fl);
            ax.YColor = 'k';
            xlabel(ax, '$T$', 'Interpreter', 'latex', 'FontSize', st.fl);
            xlim(ax, [0, max(T)]);
            setup_axes(ax, st);
            legend(ax, [h1 h2], {'FP64', 'FP32'}, 'Interpreter', 'latex', 'Box', 'off', ...
                   'Location', 'northeast', 'FontSize', st.fs);
            text(ax, 0.62, 0.12, ['$' strrep(spins{q}, 's = ', 's=') '$'], 'Units', 'normalized', ...
                 'Interpreter', 'latex', 'FontSize', st.fl);
            if io == 1
                inset_cluster(ax, keys{q}, st);
            end
            panel_label(ax, labs(ip), st);
        end
    end
end

function inset_cluster(ax, key, st, box)
    % small rendering of the cluster in the upper middle of the panel
    % BOX = [x y w h] of the inset relative to the panel
    if nargin < 4, box = [0.42 0.45 0.28 0.45]; end
    switch key
        case {'ico_s1', 'ico_s3o2'}, g = {'ico'};
        case 'ring12_s1',   g = {'ring', 12};
        case 'ring20_s1o2', g = {'ring', 20};
        case 'dodeca_s1o2', g = {'dodeca'};
        case 'icosid_M0',   g = {'icosid'};
    end
    [bonds, ~, ~, ~, V] = ftlm.geometry(g{:});
    % axes in a tiled layout cannot be positioned manually: place the inset
    % in the figure, at the pixel position of the panel
    fig = ancestor(ax, 'figure');
    drawnow;
    pos = getpixelposition(ax, true);
    ia = axes(fig, 'Units', 'pixels', ...
              'Position', [pos(1) + box(1) * pos(3), pos(2) + box(2) * pos(4), box(3) * pos(3), box(4) * pos(4)]);
    ia.Units = 'normalized';
    hold(ia, 'on');
    if ~strcmp(g{1}, 'ring')
        K = convhulln(V);
        patch(ia, 'Faces', K, 'Vertices', V, 'FaceColor', [0.45 0.75 0.40], ...
              'FaceAlpha', 0.35, 'EdgeColor', 'none');
    end
    for b = 1 : size(bonds, 1)
        e = V(bonds(b, :), :);
        plot3(ia, e(:, 1), e(:, 2), e(:, 3), '-', 'Color', [0.25 0.25 0.25], 'LineWidth', 0.6);
    end
    scatter3(ia, V(:, 1), V(:, 2), V(:, 3), 10, [0.55 0.15 0.60], 'filled');
    axis(ia, 'equal'); axis(ia, 'off');
    if ~strcmp(g{1}, 'ring'), view(ia, [25 18]); else, view(ia, 2); end
    ia.Clipping = 'off';
    set(ia, 'HitTest', 'off');
    %#ok<*NASGU>
    st; %#ok<VUNUS>
end

%% ========================================================================
%  Fig. 2: multi-seed analysis
function fig = fig_seeds(o, st)
    keys = {'ico_s1', 'ico_s3o2'};
    spins = {'s=1', 's=3/2'};
    fig = new_fig(st, 20);
    tl = tiledlayout(fig, 4, 2, 'TileSpacing', 'compact', 'Padding', 'loose');
    labs = 'abcdefgh';
    ip = 0;
    for q = 1 : 2
        S = load_study(o, sprintf('study_seeds_%s.mat', keys{q}));
        T = S.T_range(:)';
        obs = {'C', 'chi'};  ylab = {'$C$', '$\chi$'};  slab = {'$\sigma(C)$', '$\sigma(\chi)$'};
        for io = 1 : 2
            runs = S.([obs{io} '_runs']);
            pool = S.([obs{io} '_pool']);
            ax = nexttile(tl); ip = ip + 1;
            h1 = plot(ax, T, runs', '-', 'Color', [st.c32 0.25], 'LineWidth', 0.4); hold(ax, 'on');
            h2 = plot(ax, T, pool, '-', 'Color', st.c64, 'LineWidth', st.lw);
            ylabel(ax, ylab{io}, 'Interpreter', 'latex', 'FontSize', st.fl);
            xlabel(ax, '$T$', 'Interpreter', 'latex', 'FontSize', st.fl);
            xlim(ax, [0 max(T)]); yl = ylim(ax); ylim(ax, [0 1.1 * yl(2)]);
            setup_axes(ax, st);
            hl = [h1(1) h2];
            text(ax, 0.80, 0.62, ['$' spins{q} '$'], 'Units', 'normalized', 'Interpreter', 'latex', ...
                 'FontSize', st.fl);
            if io == 1, inset_cluster(ax, keys{q}, st, [0.50 0.28 0.24 0.40]); end
            panel_label(ax, labs(ip), st);

            ax = nexttile(tl); ip = ip + 1;
            if io == 1
                se = S.sigma_emp_C; sth = S.sigma_theo_C; d32 = S.dC_k(1, :); dpool = S.dC_k(end, :);
            else
                se = S.sigma_emp_chi; sth = S.sigma_theo_chi; d32 = S.dchi_k(1, :); dpool = S.dchi_k(end, :);
            end
            hr = semilogy(ax, T, se, '-', 'Color', st.c64, 'LineWidth', st.lw); hold(ax, 'on');
            hr(2) = semilogy(ax, T, sth, '--', 'Color', st.c64, 'LineWidth', st.lw);
            hr(3) = semilogy(ax, T, floor_log(d32), '-', 'Color', st.cd, 'LineWidth', st.lwd);
            hr(4) = semilogy(ax, T, floor_log(dpool), ':', 'Color', st.cd, 'LineWidth', st.lwd);
            ylabel(ax, slab{io}, 'Interpreter', 'latex', 'FontSize', st.fl);
            xlabel(ax, '$T$', 'Interpreter', 'latex', 'FontSize', st.fl);
            xlim(ax, [0 max(T)]);
            vals = [se(T >= 0.1), d32(T >= 0.1)];  vals = vals(vals > 0);
            ylim(ax, [10^(floor(log10(min(vals))) - 1), 10^ceil(log10(1.5 * max([se sth])))]);
            setup_axes(ax, st);
            panel_label(ax, labs(ip), st);
        end
    end
    % one legend for all panels (the pooled sizes are those of the s = 1 study)
    lg = legend(ax, [hl hr], {sprintf('%d runs ($R=%d$)', S.Ns, S.R), ...
                sprintf('pooled ($R_\\mathrm{eff}=%d$)', S.Ns * S.R), ...
                sprintf('$\\sigma_\\mathrm{emp}$ ($R=%d$)', S.R), ...
                sprintf('$\\sigma_\\mathrm{theo}$ ($R=%d$)', S.R), ...
                sprintf('$|\\Delta_\\mathrm{FP32}|$ ($R=%d$)', S.R), ...
                '$|\Delta_\mathrm{FP32}|$ (pooled)'}, ...
                'Interpreter', 'latex', 'Box', 'off', 'FontSize', st.fs, 'NumColumns', 3);
    lg.Layout.Tile = 'north';
end

%% ========================================================================
%  Figs. 4 and 5: ghost diagnostic and cluster weights
function fig = fig_ghosts(o, st, what)
    S = load_study(o, 'study_ghosts.mat');
    fig = new_fig(st, 6.2);
    tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'loose');
    names = {'fp64', 'fp32'};  titles = {'FP64', 'FP32'};
    labs = 'ab';
    th = [S.fp64.theta(:); S.fp32.theta(:)];
    thlim = [floor(min(th)) - 2, ceil(max(th)) + 2];      % same energy range in both panels
    for q = 1 : 2
        X = S.(names{q});
        ax = nexttile(tl);
        k = (1 : numel(X.theta))';
        reg = ~X.ghost;
        switch what
            case 'ell'
                semilogy(ax, k(reg), floor_log(X.ell(reg)), 'o', 'MarkerSize', 3, ...
                         'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k'); hold(ax, 'on');
                semilogy(ax, k(~reg), floor_log(X.ell(~reg)), 'o', 'MarkerSize', 3.5, ...
                         'MarkerFaceColor', st.c32, 'MarkerEdgeColor', st.c32);
                yline(ax, S.tau, '--', 'Color', st.cg, 'LineWidth', 0.8);
                xlabel(ax, '$k$', 'Interpreter', 'latex', 'FontSize', st.fl);
                ylabel(ax, '$\ell_k=\min_j|\theta_k-\theta_j^{(1)}|$', 'Interpreter', 'latex', 'FontSize', st.fl);
                xlim(ax, [0, numel(k) + 1]); ylim(ax, [1e-16 1e1]);
                legend(ax, {'regular', 'ghost', '$\tau$'}, 'Interpreter', 'latex', 'Box', 'off', ...
                       'Location', 'southwest', 'FontSize', st.fs);
            case 'weights'
                incl = arrayfun(@(i) nnz(X.cluster == X.cluster(i)) > 1 && any(X.ghost(X.cluster == X.cluster(i))), k);
                semilogy(ax, X.theta(~incl), floor_log(X.w(~incl)), 'o', 'MarkerSize', 3, ...
                         'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k'); hold(ax, 'on');
                semilogy(ax, X.theta(incl), floor_log(X.w(incl)), 'o', 'MarkerSize', 3.5, ...
                         'MarkerFaceColor', st.c32, 'MarkerEdgeColor', st.c32);
                if ~isempty(X.clusters)
                    semilogy(ax, [X.clusters.theta], [X.clusters.w_total], 'o', 'MarkerSize', 7, ...
                             'MarkerEdgeColor', st.c64, 'LineWidth', 1.1);
                end
                xlabel(ax, '$\theta_k$', 'Interpreter', 'latex', 'FontSize', st.fl);
                ylabel(ax, '$w_k$', 'Interpreter', 'latex', 'FontSize', st.fl);
                ylim(ax, [1e-20 1]);  xlim(ax, thlim);
                legend(ax, {'singleton', 'cluster member', 'cluster total'}, 'Interpreter', 'latex', ...
                       'Box', 'off', 'Location', 'southwest', 'FontSize', st.fs);
        end
        title(ax, titles{q}, 'Interpreter', 'latex', 'FontSize', st.fl);
        setup_axes(ax, st);
        panel_label(ax, labs(q), st);
    end
end

%% ========================================================================
%  Fig. 6: convergence with N_L
function fig = fig_lanczos_steps(o, st)
    keys = {'ico_s1', 'ico_s3o2'};  spins = {'s=1', 's=3/2'};  mk = {'o', 's'};
    fig = new_fig(st, 6.5);
    tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'loose');
    ax1 = nexttile(tl); ax2 = nexttile(tl);
    leg1 = {}; h1 = [];
    Tlo = precision_window(o.DataDir);
    for q = 1 : 2
        S = load_study(o, sprintf('study_lanczos_steps_%s.mat', keys{q}));
        w = S.T_range(:)' >= Tlo;                         % same window as in the text
        NL = S.NL(1:end-1);
        eC = max(S.dC_trunc(1:end-1, :) .* w, [], 2) / max(S.double.C(end, :));
        eX = max(S.dchi_trunc(1:end-1, :) .* w, [], 2) / max(S.double.chi(end, :));
        fC = max(S.dC_fp32 .* w, [], 2) / max(S.double.C(end, :));
        h1(end+1) = semilogy(ax1, NL, floor_log(eC), ['-' mk{q}], 'Color', st.c64, 'MarkerSize', 4, ...
                             'MarkerFaceColor', st.c64, 'LineWidth', st.lwd); hold(ax1, 'on'); %#ok<AGROW>
        h1(end+1) = semilogy(ax1, NL, floor_log(eX), ['--' mk{q}], 'Color', st.c64, 'MarkerSize', 4, ...
                             'LineWidth', st.lwd); %#ok<AGROW>
        h1(end+1) = semilogy(ax1, S.NL, floor_log(fC), [':' mk{q}], 'Color', st.c32, 'MarkerSize', 4, ...
                             'LineWidth', st.lwd); %#ok<AGROW>
        leg1 = [leg1, {sprintf('$C$, $%s$', spins{q}), sprintf('$\\chi$, $%s$', spins{q}), ...
                       sprintf('FP32, $C$, $%s$', spins{q})}]; %#ok<AGROW>
        NLt = S.NL;  if isfield(S, 'NL_time'), NLt = S.NL_time; end
        plot(ax2, NLt, S.double.t_lanczos, ['-' mk{q}], 'Color', st.c64, 'MarkerSize', 4, ...
             'MarkerFaceColor', st.c64, 'LineWidth', st.lwd); hold(ax2, 'on');
        plot(ax2, NLt, S.single.t_lanczos, ['--' mk{q}], 'Color', st.c32, 'MarkerSize', 4, ...
             'LineWidth', st.lwd);
    end
    xlabel(ax1, '$N_L$', 'Interpreter', 'latex', 'FontSize', st.fl);
    ylabel(ax1, 'max. deviation (rel.)', 'Interpreter', 'latex', 'FontSize', st.fl);
    lg = legend(ax1, h1, leg1, 'Interpreter', 'latex', 'Box', 'off', 'FontSize', st.fs, 'NumColumns', 3);
    lg.Layout.Tile = 'south';
    setup_axes(ax1, st); panel_label(ax1, 'a', st);
    xlabel(ax2, '$N_L$', 'Interpreter', 'latex', 'FontSize', st.fl);
    ylabel(ax2, 'Lanczos time (s)', 'Interpreter', 'latex', 'FontSize', st.fl);
    legend(ax2, {'FP64, $s=1$', 'FP32, $s=1$', 'FP64, $s=3/2$', 'FP32, $s=3/2$'}, ...
           'Interpreter', 'latex', 'Box', 'off', 'Location', 'northwest', 'FontSize', st.fs);
    setup_axes(ax2, st); panel_label(ax2, 'b', st);
end

%% ========================================================================
%  Fig. 7: error decomposition against exact diagonalization
function fig = fig_ed(o, st)
    S = load_study(o, 'study_ed_cube_s3o2.mat');
    T = S.T_range(:)';
    fig = new_fig(st, 11.5);
    tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'loose');
    obs = {'C', 'chi'};  ylab = {'$C$', '$\chi$'};  elab = {'$|\delta C|$', '$|\delta\chi|$'};
    labs = 'abcd';
    for io = 1 : 2
        ax = nexttile(tl, io);
        ed = S.([obs{io} '_ed']);
        plot(ax, T, ed, '-', 'Color', 'k', 'LineWidth', st.lw); hold(ax, 'on');
        plot(ax, T, S.gpu_single.(obs{io}), '--', 'Color', st.c32, 'LineWidth', st.lw);
        xlabel(ax, '$T$', 'Interpreter', 'latex', 'FontSize', st.fl);
        ylabel(ax, ylab{io}, 'Interpreter', 'latex', 'FontSize', st.fl);
        xlim(ax, [0 max(T)]); yl = ylim(ax); ylim(ax, [0 1.5 * yl(2)]);
        legend(ax, {'ED', sprintf('FTLM FP32 ($R=%d$, $N_L=%d$)', S.R, S.NL)}, 'Interpreter', 'latex', ...
               'Box', 'off', 'Location', 'northeast', 'FontSize', st.fs);
        setup_axes(ax, st);
        if io == 1, inset_ed(ax); end
        panel_label(ax, labs(io), st);

        ax = nexttile(tl, io + 2);
        f = obs{io};
        semilogy(ax, T, floor_log(S.err_stoch.(f)), '-', 'Color', st.cg, 'LineWidth', st.lw); hold(ax, 'on');
        semilogy(ax, T, floor_log(S.err_trunc.(f)), '-', 'Color', st.c64, 'LineWidth', st.lwd);
        semilogy(ax, T, floor_log(S.err_bf16.(f)), '-', 'Color', st.cbf, 'LineWidth', st.lwd);
        semilogy(ax, T, floor_log(S.err_fp16.(f)), '-', 'Color', st.c16, 'LineWidth', st.lwd);
        semilogy(ax, T, floor_log(S.err_fp32.(f)), '-', 'Color', st.c32, 'LineWidth', st.lwd);
        semilogy(ax, T, floor_log(S.err_impl.(f)), '-', 'Color', 'k', 'LineWidth', st.lwd);
        xlabel(ax, '$T$', 'Interpreter', 'latex', 'FontSize', st.fl);
        ylabel(ax, elab{io}, 'Interpreter', 'latex', 'FontSize', st.fl);
        emax = max([S.err_stoch.(f)(:); S.err_bf16.(f)(:); S.err_fp16.(f)(:)]);
        xlim(ax, [0 max(T)]); ylim(ax, [1e-17 max(1, 10^ceil(log10(emax)))]);
        if io == 1
            lg = legend(ax, {'stochastic', 'Lanczos truncation', 'BF16', 'FP16', 'FP32', ...
                        'CPU vs GPU (FP64)'}, 'Interpreter', 'latex', 'Box', 'off', ...
                        'FontSize', st.fs, 'NumColumns', 3);
            lg.Layout.Tile = 'south';
        end
        setup_axes(ax, st);
        panel_label(ax, labs(io + 2), st);
    end
end

function inset_ed(ax)
    [bonds, ~, ~, ~, V] = ftlm.geometry('cube');
    fig = ancestor(ax, 'figure');
    drawnow;
    pos = getpixelposition(ax, true);
    ia = axes(fig, 'Units', 'pixels', ...
              'Position', [pos(1) + 0.60 * pos(3), pos(2) + 0.28 * pos(4), 0.22 * pos(3), 0.38 * pos(4)]);
    ia.Units = 'normalized';
    hold(ia, 'on');
    K = convhulln(V);
    patch(ia, 'Faces', K, 'Vertices', V, 'FaceColor', [0.45 0.75 0.40], 'FaceAlpha', 0.35, ...
          'EdgeColor', 'none');
    for b = 1 : size(bonds, 1)
        e = V(bonds(b, :), :);
        plot3(ia, e(:, 1), e(:, 2), e(:, 3), '-', 'Color', [0.25 0.25 0.25], 'LineWidth', 0.6);
    end
    scatter3(ia, V(:, 1), V(:, 2), V(:, 3), 10, [0.55 0.15 0.60], 'filled');
    axis(ia, 'equal'); axis(ia, 'off'); view(ia, [30 20]);
end
