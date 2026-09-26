function [C_T, chi_T, Z_eff] = observables(all_E, all_w, all_M, T_range)
%FTLM.OBSERVABLES  C(T), chi(T), Z_eff(T) from sector-resolved spectra.
%
%  Inputs:
%     all_E   - Ritz values (FTLM) or eigenvalues (ED), column vector
%     all_w   - weights: mult * (dim_M / R_eff) * |s_k1|^2 (FTLM) or
%               mult (ED)
%     all_M   - magnetization quantum number (M >= 0) per entry
%     T_range - temperatures (units of J/k_B)
%
%  Outputs (row vectors):
%     C_T   - heat capacity        C = beta^2 Var(E)        (k_B = 1)
%     chi_T - zero-field susceptibility chi = beta <M^2>    (per g^2 mu_B^2)
%     Z_eff - Z(T) * exp(beta * E_0)
%
%  Var(E) is evaluated in the centred form <(E - <E>)^2> (sum of
%  non-negative terms), free from cancellation at low T.

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

all_E    = double(all_E(:));
all_w    = double(all_w(:));
all_M    = double(all_M(:));
T_range  = double(T_range(:)');
n_T      = numel(T_range);
beta_arr = 1.0 ./ T_range;

E_min = min(all_E);
dE    = all_E - E_min;
M2    = all_M .^ 2;

C_T   = zeros(1, n_T);
chi_T = zeros(1, n_T);
Z_eff = zeros(1, n_T);

for iT = 1 : n_T
    bet   = beta_arr(iT);
    boltz = all_w .* exp(-bet * dE);

    Z = sum(boltz);
    if Z < 1e-250
        continue;   % protect against underflow at very large beta
    end

    dE_avg = sum(dE .* boltz) / Z;
    E_var  = sum((dE - dE_avg).^2 .* boltz) / Z;
    M2_avg = sum(M2 .* boltz) / Z;

    C_T(iT)   = bet^2 * E_var;
    chi_T(iT) = bet * M2_avg;
    Z_eff(iT) = Z;
end
end
