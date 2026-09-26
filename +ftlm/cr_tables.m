function [dcum, pstride, astride, dim] = cr_tables(model, A)
%FTLM.CR_TABLES  Cumulative dimension table for combinatorial ranking.
%   [DCUM, PSTRIDE, ASTRIDE, DIM] = FTLM.CR_TABLES(MODEL, A) returns the
%   table
%       D_c(p, A', a) = sum_{q=0}^{a-1} D(p, A' - q),
%   p = 0..N-1, A' = 0..A, a = 0..2 s_p, where D(p, A') is the number of
%   digit strings of the first p sites (digit k in {0..2 s_k}) with digit
%   sum A'.  DCUM is an int32 vector with element (p, A', a) at the
%   0-based position p*PSTRIDE + A'*ASTRIDE + a.  DIM = D(N, A) is the
%   sector dimension.
%
%   The rank of a packed state x with digits a_p is
%       rank(x) = sum_{p=N-1..0} D_c(p, A_p, a_p),
%   A_{N-1} = A, A_{p-1} = A_p - a_p, which reproduces the increasing
%   CLT label order of FTLM.ENUMERATE_SECTOR.  Mixed spins enter only
%   through the site-dependent digit ranges.

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

N     = model.N;
two_s = model.two_s;

% D(p+1, A'+1): number of strings of sites 0..p-1 with digit sum A'
D = zeros(N + 1, A + 1);
D(1, 1) = 1;
for p = 1 : N
    for Ap = 0 : A
        q = 0 : min(two_s(p), Ap);
        D(p + 1, Ap + 1) = sum(D(p, Ap - q + 1));
    end
end
dim = D(N + 1, A + 1);
assert(dim < 2^31, 'ftlm:cr_tables', ...
       'sector dimension %.3g exceeds the int32 rank range.', dim);

astride = max(two_s) + 1;
pstride = (A + 1) * astride;
dcum    = zeros(N * pstride, 1);
for p = 0 : N - 1
    for Ap = 0 : A
        base = p * pstride + Ap * astride;
        acc  = 0;
        for a = 0 : two_s(p + 1)
            dcum(base + a + 1) = acc;
            if Ap - a >= 0
                acc = acc + D(p + 1, Ap - a + 1);
            end
        end
    end
end
dcum = int32(dcum);
end
