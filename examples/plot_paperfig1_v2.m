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

function [fig, info] = plot_paperfig1_v2(varargin)
%PLOT_PAPERFIG1_V2  Two-panel precision figure (FP64/FP32 + |Delta|).
%
%   Produces a 1x2 figure from an ftlm_<geom>_s<s>.mat file generated
%   by ftlm_observables.m (with use_cpu_reference = true so that the
%   CPU FP64 reference is available). Works for any system supported
%   by ftlm_observables.m: s = 1/2, 1, 3/2, ... and icosahedron,
%   cuboctahedron, cube, dodecahedron, icosidodecahedron, ring, ...
%
%     left panel   : C(T)
%                       left  y-axis (linear) -- FP64 (solid blue)
%                                                FP32 (dashed red)
%                       right y-axis (log)    -- |C_FP32 - C_FP64| (black)
%
%     right panel  : chi(T) analogous with |chi_FP32 - chi_FP64| on the
%                           right logarithmic y-axis.
%
%   Default data source: ftlm_ico_s1.mat
%                        required variables: T_range,
%                                            C_T_cpu,   C_T,
%                                            chi_T_cpu, chi_T
%   (C_T / chi_T are the GPU FP32 results, *_cpu the CPU FP64 reference.
%   The .mat must have been produced with use_cpu_reference = true,
%   otherwise the *_cpu fields are empty and this script will error out.)
%
%   Default output location: Bilder/<tag>_fig1_paper_v2.fig + .png
%                            where <tag> is derived from the .mat name,
%                            e.g. ftlm_ico_s1.mat -> ico_s1,
%                                 ftlm_ring_s1o2.mat -> ring_s1o2.
%
% NAME-VALUE OPTIONS
%   'MatFile'   Path to source .mat (default ftlm_ico_s1.mat)
%   'SaveFig'   Path for FIG export    ('' = do not save,
%                                       [] = derive automatically from MatFile)
%   'SavePng'   Path for PNG export    ('' = do not save,
%                                       [] = derive automatically from MatFile)
%   'Visible'   'on' (default) | 'off'
%   'Width'     Figure width  in pixels (default 1100)
%   'Height'    Figure height in pixels (default 420)
%   'TMax'      Upper bound of the plotted T range (default Inf, i.e.
%               the full range from the .mat). Data above TMax are
%               discarded before scale computation so that both the
%               x-axis and the two y-axes only auto-scale on the
%               visible region.
%
% RETURN
%   fig  : handle of the new figure
%   info : struct with sanity statistics (dC_max, dchi_max, T_range,
%          MatFile, Tag, ...)
%
% EXAMPLES
%   plot_paperfig1_v2                                          % s=1 ico
%   plot_paperfig1_v2('MatFile','ftlm_ico_s3o2.mat','TMax',10)
%   plot_paperfig1_v2('MatFile','ftlm_ring_s1o2.mat')
%   plot_paperfig1_v2('Visible','off','SaveFig','','SavePng','')   % display only

% ---------------------------------------------------------------------
% 0. Options
% ---------------------------------------------------------------------
p = inputParser;
p.addParameter('MatFile', 'ftlm_ico_s1.mat', @(x)ischar(x)||isstring(x));
% SaveFig / SavePng: [] (default) = derive automatically from MatFile name.
% '' = do not save. Otherwise: explicit path.
p.addParameter('SaveFig', [], @(x)isempty(x)||ischar(x)||isstring(x));
p.addParameter('SavePng', [], @(x)isempty(x)||ischar(x)||isstring(x));
p.addParameter('Visible', 'on', @(x)ischar(x)||isstring(x));
p.addParameter('Width',   1100, @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('Height',  420,  @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('TMax',    Inf,  @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.parse(varargin{:});
opt = p.Results;

% Derive system tag from the .mat file name (e.g. ftlm_ico_s3o2.mat
% -> 'ico_s3o2') and build default output paths from it, unless
% SaveFig/SavePng were set explicitly ([]). An empty string '' means
% "do not save". Distinction [] (auto) vs. '' (do-not-save) via type:
%   []  is isempty AND isnumeric  -> auto-derive
%   ''  is isempty but NOT numeric -> keep "do not save"
tag = derive_tag_from_matfile(opt.MatFile);
if isempty(opt.SaveFig) && isnumeric(opt.SaveFig)
    opt.SaveFig = fullfile('Bilder', sprintf('%s_fig1_paper_v2.fig', tag));
end
if isempty(opt.SavePng) && isnumeric(opt.SavePng)
    opt.SavePng = fullfile('Bilder', sprintf('%s_fig1_paper_v2.png', tag));
end

% ---------------------------------------------------------------------
% 1. Load data
% ---------------------------------------------------------------------
assert(isfile(opt.MatFile), 'Mat file not found: %s', opt.MatFile);
S = load(opt.MatFile, 'T_range', ...
                      'C_T_cpu',   'C_T', ...
                      'chi_T_cpu', 'chi_T');

required = {'T_range','C_T_cpu','C_T','chi_T_cpu','chi_T'};
for k = 1:numel(required)
    assert(isfield(S, required{k}), ...
        'Variable "%s" missing in %s', required{k}, opt.MatFile);
end

% The CPU FP64 reference fields are empty unless the .mat was generated
% with use_cpu_reference = true. Without them there is no FP64 baseline
% to compare against, so this script cannot do its job.
assert(~isempty(S.C_T_cpu) && ~isempty(S.chi_T_cpu), ...
    ['CPU FP64 reference data not present in %s. ', ...
     'Re-run ftlm_observables.m with use_cpu_reference = true.'], opt.MatFile);

T   = S.T_range(:);
C64 = S.C_T_cpu(:);
C32 = S.C_T(:);
X64 = S.chi_T_cpu(:);
X32 = S.chi_T(:);

% Optional truncation to T <= TMax. Points above TMax are dropped here
% so that both the x-axis and the y-axes auto-scale only on the visible
% range (in particular the log axis on the right).
if isfinite(opt.TMax)
    mask = T <= opt.TMax;
    if ~any(mask)
        error('plot_paperfig1_v2:EmptyTRange', ...
              'No data points with T <= %.6g (TMax too small?).', opt.TMax);
    end
    T   = T(mask);
    C64 = C64(mask);
    C32 = C32(mask);
    X64 = X64(mask);
    X32 = X32(mask);
end

% Upper x-limit: with finite TMax pin it there (even if the T-grid
% does not reach TMax), otherwise use max(T) from the data.
if isfinite(opt.TMax)
    x_hi = opt.TMax;
else
    x_hi = max(T);
end
x_lo = min(T);

% Pointwise (absolute) precision error
dC   = abs(C32 - C64);
dchi = abs(X32 - X64);

% Floor for the log scale so that exact zeros do not map to -Inf
eps_floor = 1e-16;
dC(  dC   < eps_floor) = eps_floor;
dchi(dchi < eps_floor) = eps_floor;

% ---------------------------------------------------------------------
% 2. Colour and style conventions
% ---------------------------------------------------------------------
col_fp64  = [0.00 0.45 0.74];   % MATLAB blue
col_fp32  = [0.85 0.10 0.10];   % red
col_delta = [0.00 0.00 0.00];   % black (right axis, |Delta|)
col_axis  = [0.00 0.00 0.00];   % axis/tick colour (both sides neutral)
col_bg    = [1.00 1.00 1.00];   % white (figure and axes background)

lw_main    = 1.6;    % line width for FP64/FP32
lw_delta   = 1.2;    % line width for |Delta|
font_size  = 16;     % tick labels
label_size = 19;     % axis labels (xlabel/ylabel)
legend_size= 16;     % legend font

% ---------------------------------------------------------------------
% 3. Build the figure
% ---------------------------------------------------------------------
fig = figure('Visible', opt.Visible, ...
             'Color', col_bg, ...
             'InvertHardcopy', 'off', ...   % keep white background on export
             'Units', 'pixels', ...
             'Position', [100 100 opt.Width opt.Height]);
% 'loose' gives more horizontal spacing between the panels so that the
% right y-axis label of the left plot does not crowd the left axis
% label of the right plot.
tl = tiledlayout(fig, 1, 2, 'TileSpacing','loose', 'Padding','compact');
% Set TiledChartLayout background (if the property exists) to white
try, tl.BackgroundColor = col_bg; end %#ok<TRYNC>

% -- left panel: C(T) -----------------------------------------------------
ax1 = nexttile(tl);

yyaxis(ax1, 'left');
h_C_64 = plot(ax1, T, C64, '-',  'Color', col_fp64, 'LineWidth', lw_main); hold(ax1, 'on');
h_C_32 = plot(ax1, T, C32, '--', 'Color', col_fp32, 'LineWidth', lw_main);
ylabel(ax1, '$C$', 'Interpreter','latex', 'FontSize', label_size);
ax1.YColor = col_axis;

yyaxis(ax1, 'right');
h_C_d  = plot(ax1, T, dC, '-',  'Color', col_delta, 'LineWidth', lw_delta);
set(ax1, 'YScale', 'log');
ylabel(ax1, '$|\Delta C|$', 'Interpreter','latex', 'FontSize', label_size);
ax1.YColor = col_axis;

xlabel(ax1, '$T$', 'Interpreter','latex', 'FontSize', label_size);
xlim(ax1, [x_lo x_hi]);
ax1.FontSize = font_size;
ax1.Box      = 'on';
ax1.Color    = col_bg;          % axes background white
ax1.XColor   = col_axis;        % x tick labels and x axis black
ax1.TickLabelInterpreter = 'latex';
% Set YColor explicitly per side again (safety: yyaxis may set the
% colour to the line colour when a side is first created)
yyaxis(ax1, 'left');  ax1.YColor = col_axis;
yyaxis(ax1, 'right'); ax1.YColor = col_axis;

% Legend: only the two left-axis curves; the |Delta| curve is identified
% unambiguously by its right-axis label.
legend(ax1, [h_C_64 h_C_32], ...
       {'FP64','FP32'}, ...
       'Interpreter','latex', 'Location','best', 'Box','off', ...
       'FontSize', legend_size, 'TextColor', col_axis);

% -- right panel: chi(T) --------------------------------------------------
ax2 = nexttile(tl);

yyaxis(ax2, 'left');
h_X_64 = plot(ax2, T, X64, '-',  'Color', col_fp64, 'LineWidth', lw_main); hold(ax2, 'on');
h_X_32 = plot(ax2, T, X32, '--', 'Color', col_fp32, 'LineWidth', lw_main);
ylabel(ax2, '$\chi$', 'Interpreter','latex', 'FontSize', label_size);
ax2.YColor = col_axis;

yyaxis(ax2, 'right');
h_X_d  = plot(ax2, T, dchi, '-', 'Color', col_delta, 'LineWidth', lw_delta);
set(ax2, 'YScale', 'log');
ylabel(ax2, '$|\Delta\chi|$', 'Interpreter','latex', 'FontSize', label_size);
ax2.YColor = col_axis;

xlabel(ax2, '$T$', 'Interpreter','latex', 'FontSize', label_size);
xlim(ax2, [x_lo x_hi]);
ax2.FontSize = font_size;
ax2.Box      = 'on';
ax2.Color    = col_bg;          % axes background white
ax2.XColor   = col_axis;        % x tick labels and x axis black
ax2.TickLabelInterpreter = 'latex';
yyaxis(ax2, 'left');  ax2.YColor = col_axis;
yyaxis(ax2, 'right'); ax2.YColor = col_axis;

% Legend: only the two left-axis curves (see comment above).
legend(ax2, [h_X_64 h_X_32], ...
       {'FP64','FP32'}, ...
       'Interpreter','latex', 'Location','best', 'Box','off', ...
       'FontSize', legend_size, 'TextColor', col_axis);

% ---------------------------------------------------------------------
% 4. Save
% ---------------------------------------------------------------------
if ~isempty(opt.SaveFig)
    save_to(fig, opt.SaveFig, 'fig');
end
if ~isempty(opt.SavePng)
    save_to(fig, opt.SavePng, 'png');
end

% ---------------------------------------------------------------------
% 5. Return info
% ---------------------------------------------------------------------
info = struct();
info.MatFile      = opt.MatFile;
info.Tag          = tag;
info.T_range      = [min(T) max(T)];
info.npts         = numel(T);
info.dC_max       = max(dC);
info.dC_min       = min(dC);
info.dchi_max     = max(dchi);
info.dchi_min     = min(dchi);
info.dC_max_rel   = max(dC   ./ max(abs(C64),  eps));
info.dchi_max_rel = max(dchi ./ max(abs(X64),  eps));
info.SaveFig      = opt.SaveFig;
info.SavePng      = opt.SavePng;
end


% ======================================================================
% Helpers
% ======================================================================
function save_to(fig, out_path, kind)
    out_dir = fileparts(out_path);
    if ~isempty(out_dir) && ~isfolder(out_dir)
        mkdir(out_dir);
    end
    switch lower(kind)
        case 'fig'
            savefig(fig, out_path);
            fprintf('Saved FIG: %s\n', out_path);
        case 'png'
            % exportgraphics produces the cleanest PNG bitmap on modern
            % MATLAB versions (no whitespace, vector-path resolution is
            % preserved for embedded lines)
            exportgraphics(fig, out_path, 'Resolution', 300, ...
                           'BackgroundColor', 'white');
            fprintf('Saved PNG: %s\n', out_path);
    end
end


function tag = derive_tag_from_matfile(matfile)
%DERIVE_TAG_FROM_MATFILE  Extract the system tag (e.g. "ico_s3o2") from
% the file name of an ftlm_<geom>_s<s>.mat file produced by
% ftlm_observables.m.
%
%  Recognised patterns:
%    ftlm_<geom>_s<spin>.mat             -> <geom>_s<spin>
%    ftlm_<geom>_s<spin>_<suffix>.mat    -> <geom>_s<spin>
%  Examples:
%    ftlm_ico_s1.mat                  -> ico_s1
%    ftlm_ico_s3o2.mat                -> ico_s3o2
%    ftlm_ring_20_s1o2.mat            -> ring_20_s1o2
%    ftlm_dodeca_s1.mat               -> dodeca_s1
%
%  If the name does not match this scheme, the base name (without
%  the .mat extension and a leading "ftlm_") is used as the tag.

    [~, base, ~] = fileparts(matfile);

    % Primary pattern: ftlm_<geom_token(s)>_s<digits[o digits]>(?:_<suffix>)?
    % The geometry token can itself contain underscores (e.g. ring_20),
    % so we capture greedily up to the trailing _s<spin> group.
    pat = '^ftlm_(.+_s\d+(?:o\d+)?)(?:_[A-Za-z0-9]+)?$';
    tok = regexp(base, pat, 'tokens', 'once');
    if ~isempty(tok)
        tag = tok{1};
        return
    end

    % Fallback: strip a leading "ftlm_" if present, use the rest as tag
    if startsWith(base, 'ftlm_')
        tag = extractAfter(base, 'ftlm_');
    else
        tag = base;
    end
end
