function sec = sectors(model, only_lowest)
%FTLM.SECTORS  Magnetization sectors M >= 0 of a spin model.
%   SEC = FTLM.SECTORS(MODEL) returns a struct array with one entry per
%   sector with total magnetization M >= 0:
%       A     digit sum of the sector, A = S_max + M
%       M     magnetization quantum number (integer or half-integer)
%       mult  2 for M > 0 (the sector -M has the same spectrum), else 1
%       dim   sector dimension
%   ordered by increasing M.  If S_max = sum_i s_i is half-integer, all
%   sectors have half-integer M and multiplicity 2.
%
%   SEC = FTLM.SECTORS(MODEL, true) returns only the lowest sector
%   (M = 0, or M = 1/2 for half-integer S_max).

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

if nargin < 2, only_lowest = false; end

A_tot2 = sum(model.two_s);             % 2 S_max = maximal digit sum
A_list = ceil(A_tot2 / 2) : A_tot2;    % sectors with M >= 0
if only_lowest
    A_list = A_list(1);
end

% Sector dimensions: coefficients of prod_k (1 + x + ... + x^(2 s_k))
poly = 1;
for k = 1 : model.N
    poly = conv(poly, ones(1, model.radix(k)));
end

n = numel(A_list);
sec = struct('A', cell(1, n), 'M', [], 'mult', [], 'dim', []);
for q = 1 : n
    A = A_list(q);
    sec(q).A    = A;
    sec(q).M    = A - A_tot2 / 2;
    sec(q).mult = 1 + (sec(q).M > 0);
    sec(q).dim  = round(poly(A + 1));
end
end
