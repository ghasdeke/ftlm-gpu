function [E, w, M] = collect_ritz(res)
%FTLM.COLLECT_RITZ  Concatenated spectral data of a run with save_ritz.
%   [E, W, M] = FTLM.COLLECT_RITZ(RES) returns the Ritz values (FTLM
%   sectors) or eigenvalues (ED sectors) E, the weights W including the
%   M-multiplicity, and the magnetization M of every entry, i.e. the
%   input of FTLM.OBSERVABLES.  RES is the output of FTLM.RUN with
%   save_ritz = true.
%
%   Pooling of K independent runs (e.g. different seeds):
%       [E, W, M] from each run, W divided by K, concatenated.

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

assert(isfield(res, 'ritz'), 'ftlm:collect_ritz', 'run with save_ritz = true');
n = numel(res.ritz);
E = cell(n, 1);  w = cell(n, 1);  M = cell(n, 1);
for q = 1 : n
    r = res.ritz{q};
    E{q} = r.E(:);
    w{q} = res.sector_mult(q) * r.w(:);
    M{q} = res.sector_M(q) * ones(numel(r.E), 1);
end
E = vertcat(E{:});  w = vertcat(w{:});  M = vertcat(M{:});
end
