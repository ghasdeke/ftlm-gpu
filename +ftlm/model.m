function model = model(opts)
%FTLM.MODEL  Spin model (local spins, couplings, basis encoding).
%   MODEL = FTLM.MODEL(OPTS) builds the model description for the
%   isotropic spin Hamiltonian
%
%       H = sum_c  J_c  s_{i_c} . s_{j_c}
%
%   from the fields of the options struct OPTS:
%
%     Couplings (one of):
%       geometry   'ico','cubo','cube','dodeca','icosid','ring' with a
%                  uniform nearest-neighbor coupling J (scalar);
%                  'ring' additionally needs N_ring.
%       couplings  K x 3 matrix [i, j, J_ij] of pairwise couplings
%                  (1-based site indices, any pairs i ~= j, not
%                  restricted to nearest neighbors).  If given, it
%                  replaces the preset bonds; geometry may then be
%                  omitted or 'custom'.  Duplicate pairs (in either
%                  orientation) are summed; zero couplings are dropped.
%
%     Local spins (one of):
%       s_val      scalar local spin applied to all sites
%       spins      1 x N vector of local spins s_i (mixed spins), each a
%                  positive integer or half-integer <= 15/2
%
%     N_sites      optional number of sites for custom couplings
%                  (default: numel(spins), else the largest site index)
%
%   The returned struct contains the site data (spins, two_s, radix),
%   the coupling list (1-based, K x 3), the CLT label weights
%   (power, D_full) and the CR bit layout (shift, bits).

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

MAX_SITES     = 32;
MAX_COUPLINGS = 512;
MAX_TWO_S     = 15;

geometry  = get_opt(opts, 'geometry', '');
couplings = get_opt(opts, 'couplings', []);
spins     = get_opt(opts, 'spins', []);
s_val     = get_opt(opts, 's_val', []);
J         = get_opt(opts, 'J', []);
N_ring    = get_opt(opts, 'N_ring', []);
N_sites   = get_opt(opts, 'N_sites', []);

geometry = char(geometry);
use_custom = ~isempty(couplings);

%% ---- couplings --------------------------------------------------------
if use_custom
    assert(isnumeric(couplings) && ismatrix(couplings) && size(couplings, 2) == 3 ...
           && size(couplings, 1) >= 1 && all(isfinite(couplings(:))), ...
           'ftlm:model', 'couplings must be a K x 3 matrix [i, j, J_ij] of finite numbers.');
    couplings = double(couplings);
    ij = couplings(:, 1:2);
    assert(all(ij(:) == round(ij(:))) && all(ij(:) >= 1), ...
           'ftlm:model', 'coupling site indices must be positive integers (1-based).');
    assert(all(ij(:, 1) ~= ij(:, 2)), ...
           'ftlm:model', 'couplings with i == j are not allowed.');
    if isempty(geometry), geometry = 'custom'; end
    name  = 'Custom';
    short = 'custom';
    N_geo = max(ij(:));
else
    assert(~isempty(geometry), 'ftlm:model', ...
           'Specify either a predefined geometry or a coupling list (couplings).');
    [bonds, N_geo, name, short] = ftlm.geometry(geometry, N_ring);
    assert(~isempty(J) && isnumeric(J) && isscalar(J) && isfinite(J), 'ftlm:model', ...
           'A predefined geometry requires a finite scalar coupling J.');
    couplings = [bonds, double(J) * ones(size(bonds, 1), 1)];
end

%% ---- sites and spins -------------------------------------------------
if ~isempty(spins)
    spins = double(spins(:)');
    N = numel(spins);
    if ~use_custom
        assert(N == N_geo, 'ftlm:model', ...
               'numel(spins) = %d does not match the %d sites of geometry ''%s''.', ...
               N, N_geo, geometry);
    end
else
    assert(~isempty(s_val) && isnumeric(s_val) && isscalar(s_val), 'ftlm:model', ...
           'Specify the local spin(s) via s_val (scalar) or spins (vector).');
    if use_custom && ~isempty(N_sites)
        N = N_sites;
    else
        N = N_geo;
    end
    spins = double(s_val) * ones(1, N);
end
if use_custom && ~isempty(N_sites)
    assert(N_sites == N, 'ftlm:model', 'N_sites = %d but %d spins given.', N_sites, N);
end
assert(N >= 1 && N <= MAX_SITES, 'ftlm:model', 'N = %d outside [1, %d].', N, MAX_SITES);
assert(max(couplings(:, 1:2), [], 'all') <= N, 'ftlm:model', ...
       'coupling site index exceeds the number of sites N = %d.', N);
two_s = round(2 * spins);
assert(all(abs(2 * spins - two_s) < 1e-12) && all(two_s >= 1) && all(two_s <= MAX_TWO_S), ...
       'ftlm:model', 'local spins must be integers or half-integers in [1/2, 15/2].');

%% ---- merge duplicate pairs, drop zero couplings ------------------------
% Duplicates are identified by the unordered pair; the orientation of the
% first occurrence is kept (the order matters only for the floating-point
% summation order inside the kernels).
key = min(couplings(:, 1:2), [], 2) * (N + 1) + max(couplings(:, 1:2), [], 2);
[~, first, grp] = unique(key, 'stable');
Jsum = accumarray(grp, couplings(:, 3));
couplings = [couplings(first, 1:2), Jsum];
couplings = couplings(couplings(:, 3) ~= 0, :);
assert(~isempty(couplings), 'ftlm:model', 'all couplings are zero.');
assert(size(couplings, 1) <= MAX_COUPLINGS, 'ftlm:model', ...
       '%d couplings exceed the kernel limit of %d.', size(couplings, 1), MAX_COUPLINGS);

%% ---- encoding ----------------------------------------------------------
radix  = two_s + 1;
power  = cumprod([1, radix(1:end-1)]);        % CLT label weights P_k
D_full = prod(radix);                         % label-space size (double)
bits   = ceil(log2(radix));
shift  = cumsum([0, bits(1:end-1)]);          % CR bit offsets

model = struct();
model.N         = N;
model.spins     = spins;
model.two_s     = two_s;
model.radix     = radix;
model.S_max     = sum(spins);
model.couplings = couplings;
model.power     = power;
model.D_full    = D_full;
model.bits      = bits;
model.shift     = shift;
model.cr_ok     = sum(bits) <= 64;
model.clt_ok    = D_full <= 2^31;
model.geometry  = geometry;
model.name      = name;
model.tag       = make_tag(short, spins);
end

function v = get_opt(opts, name, default)
    if isfield(opts, name) && ~isempty(opts.(name))
        v = opts.(name);
    else
        v = default;
    end
end

function tag = make_tag(short, spins)
%MAKE_TAG  File-name tag, e.g. 'ico_s1', 'ring_20_s1o2', 'custom_mixed'.
    if all(spins == spins(1))
        two_s = round(2 * spins(1));
        if mod(two_s, 2) == 0
            s_str = sprintf('%d', two_s / 2);
        else
            s_str = sprintf('%do2', two_s);
        end
        tag = sprintf('%s_s%s', short, s_str);
    else
        tag = sprintf('%s_mixed', short);
    end
end
