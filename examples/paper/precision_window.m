function Tlo = precision_window(study_dir, cfrac)
%PRECISION_WINDOW  Lowest temperature of the window used for precision statements.
%   TLO = PRECISION_WINDOW(STUDY_DIR) returns the smallest temperature
%   T_lo such that, for every system of the precision study and for the
%   s = 3/2 cube of the ED study, the reference heat capacity satisfies
%   C(T) >= CFRAC * max_T C for all T >= T_lo (CFRAC = 1e-6 by default).
%   Below T_lo the heat capacity of at least one system is exponentially
%   small, and relative deviations are not meaningful.

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

if nargin < 2, cfrac = 1e-6; end
Tlo = 0;
files = dir(fullfile(study_dir, 'study_precision_*.mat'));
for k = 1 : numel(files)
    if endsWith(files(k).name, '_cpu.mat'), continue; end   % CPU-only part files
    P = load(fullfile(files(k).folder, files(k).name), 'T_range', 'gpu_double');
    if ~isfield(P, 'gpu_double'), continue; end
    Tlo = max(Tlo, first_T(P.T_range, P.gpu_double.C, cfrac));
end
f = fullfile(study_dir, 'study_ed_cube_s3o2.mat');
if isfile(f)
    E = load(f, 'T_range', 'C_ed');
    Tlo = max(Tlo, first_T(E.T_range, E.C_ed, cfrac));
end
end

function T0 = first_T(T, C, cfrac)
    % smallest T above which C stays >= cfrac * max(C)
    T = T(:);  C = C(:);
    bad = find(C < cfrac * max(C), 1, 'last');
    if isempty(bad), T0 = T(1); else, T0 = T(min(bad + 1, numel(T))); end
end
