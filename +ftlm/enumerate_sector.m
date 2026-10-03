function basis = enumerate_sector(model, A)
%FTLM.ENUMERATE_SECTOR  Sorted CLT labels of the sector with digit sum A.
%   BASIS = FTLM.ENUMERATE_SECTOR(MODEL, A) returns the int32 column
%   vector of all labels n = sum_k a_k P_k (a_k = m_k + s_k) with
%   sum_k a_k = A, in increasing order.  This is the basis order used by
%   all kernels (CLT and CR).
%
%   The basis is built site by site: the sorted label lists of the first
%   k sites, grouped by their partial digit sum, are extended by the
%   digit of site k+1, which is the most significant one so far.  Only
%   partial sums that can still reach A are kept, so the work and memory
%   are O(N * dim).  Requires D_full = prod_k (2 s_k + 1) <= 2^31.

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

assert(model.clt_ok, 'ftlm:enumerate', ...
    ['prod(2 s_k + 1) = %.3g exceeds 2^31: labels do not fit into int32. ', ...
     'The basis array (CLT, CPU backend) requires prod(2 s_k + 1) <= 2^31; ', ...
     'on the GPU, use lookup = ''cr'' (no basis array needed).'], model.D_full);

N     = model.N;
two_s = model.two_s;
P     = int32(model.power);
cap_after = [cumsum(two_s, 'reverse'), 0];   % cap_after(k+1) = sum(two_s(k+1:N))
cap_after = cap_after(2:end);                % max digit sum of sites k+1..N

% lists{q} holds the sorted labels of the first k sites with partial sum
% sums(q)
sums  = 0;
lists = {int32(0)};
for k = 1 : N
    lo = max(0, A - cap_after(k));
    hi = min(A, sum(two_s(1:k)));
    new_sums  = lo : hi;
    new_lists = cell(1, numel(new_sums));
    for q = 1 : numel(new_sums)
        s_new = new_sums(q);
        parts = {};
        for a = 0 : two_s(k)
            idx = find(sums == s_new - a, 1);
            if ~isempty(idx) && ~isempty(lists{idx})
                parts{end+1} = lists{idx} + int32(a) * P(k); %#ok<AGROW>
            end
        end
        if isempty(parts)
            new_lists{q} = zeros(0, 1, 'int32');
        else
            new_lists{q} = vertcat(parts{:});
        end
    end
    sums  = new_sums;
    lists = new_lists;
end

idx = find(sums == A, 1);
if isempty(idx)
    basis = zeros(0, 1, 'int32');
else
    basis = lists{idx};
end
end
