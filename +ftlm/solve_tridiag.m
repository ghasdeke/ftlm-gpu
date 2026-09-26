function [theta, w] = solve_tridiag(alpha, beta)
%FTLM.SOLVE_TRIDIAG  Ritz values and FTLM weights of a Lanczos run.
%   [THETA, W] = FTLM.SOLVE_TRIDIAG(ALPHA, BETA) diagonalizes the
%   symmetric tridiagonal matrix with diagonal ALPHA (n entries) and
%   off-diagonal BETA(1:n-1) in double precision and returns the Ritz
%   values THETA and the weights W = |s_{k,1}|^2 (squared first
%   components of the normalized eigenvectors, sum(W) = 1).

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

alpha = double(alpha(:));
beta  = double(beta(:));
n     = numel(alpha);
T     = diag(alpha);
if n > 1
    T = T + diag(beta(1:n-1), 1) + diag(beta(1:n-1), -1);
end
[Q, D] = eig(T, 'vector');
theta  = D;
w      = abs(Q(1, :)').^2;
end
