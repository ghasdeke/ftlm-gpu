function H = hamiltonian(model, basis)
%FTLM.HAMILTONIAN  Explicit sparse Hamiltonian on a sector basis.
%   H = FTLM.HAMILTONIAN(MODEL, BASIS) returns the sparse matrix of
%       H = sum_c J_c [ s^z_i s^z_j + (s^+_i s^-_j + s^-_i s^+_j)/2 ]
%   on the sorted sector basis BASIS (CLT labels, FTLM.ENUMERATE_SECTOR).
%   Used for exact diagonalization of small sectors and as independent
%   reference in the tests.  Generic in the local spins and couplings.

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

labels = double(basis(:));
dim    = numel(labels);
N      = model.N;
s      = model.spins;
P      = model.power;

% digits a(:,k) and magnetic quantum numbers m = a - s
a   = zeros(dim, N);
tmp = labels;
for k = 1 : N
    a(:, k) = mod(tmp, model.radix(k));
    tmp     = (tmp - a(:, k)) / model.radix(k);
end
m = a - s;

C = model.couplings;
diag_vals = zeros(dim, 1);
for c = 1 : size(C, 1)
    diag_vals = diag_vals + C(c, 3) * m(:, C(c, 1)) .* m(:, C(c, 2));
end

rows = {(1:dim)'};  cols = {(1:dim)'};  vals = {diag_vals};
for c = 1 : size(C, 1)
    i = C(c, 1);  j = C(c, 2);  hJ = 0.5 * C(c, 3);
    ri = s(i) * (s(i) + 1);  rj = s(j) * (s(j) + 1);

    % <out| s^+_i s^-_j |src>: src = out with (m_i - 1, m_j + 1)
    sel = find(a(:, i) > 0 & a(:, j) < model.two_s(j));
    if ~isempty(sel)
        mi = m(sel, i);  mj = m(sel, j);
        coeff = hJ * sqrt(ri - mi .* (mi - 1)) .* sqrt(rj - mj .* (mj + 1));
        [ok, col] = ismember(labels(sel) - P(i) + P(j), labels);
        rows{end+1} = sel(ok);  cols{end+1} = col(ok);  vals{end+1} = coeff(ok); %#ok<AGROW>
    end

    % <out| s^-_i s^+_j |src>: src = out with (m_i + 1, m_j - 1)
    sel = find(a(:, i) < model.two_s(i) & a(:, j) > 0);
    if ~isempty(sel)
        mi = m(sel, i);  mj = m(sel, j);
        coeff = hJ * sqrt(ri - mi .* (mi + 1)) .* sqrt(rj - mj .* (mj - 1));
        [ok, col] = ismember(labels(sel) + P(i) - P(j), labels);
        rows{end+1} = sel(ok);  cols{end+1} = col(ok);  vals{end+1} = coeff(ok); %#ok<AGROW>
    end
end

H = sparse(vertcat(rows{:}), vertcat(cols{:}), vertcat(vals{:}), dim, dim);
end
