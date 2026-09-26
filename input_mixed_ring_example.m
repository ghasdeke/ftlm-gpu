%% input_mixed_ring_example.m
%  ================================================================
%  Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
%  and Helmholtz-Zentrum Dresden-Rossendorf e.V.
%
%  Licensed under the Apache License, Version 2.0 (the "License");
%  you may not use this file except in compliance with the License.
%  You may obtain a copy of the License at
%
%      http://www.apache.org/licenses/LICENSE-2.0
%
%  Unless required by applicable law or agreed to in writing, software
%  distributed under the License is distributed on an "AS IS" BASIS,
%  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
%  See the License for the specific language governing permissions and
%  limitations under the License.
%  ================================================================
%  Example: user-defined couplings and mixed local spins.
%
%  Invoke as
%      ftlm_observables('input_mixed_ring_example.m')
%
%  A ring of N = 12 sites with alternating local spins s = 1 and
%  s = 3/2, nearest-neighbor coupling J1 and next-nearest-neighbor
%  coupling J2:
%
%      H = J1 sum_i s_i . s_{i+1}  +  J2 sum_i s_i . s_{i+2}
%
%  Any list of pairwise couplings [i, j, J_ij] (1-based site indices,
%  arbitrary pairs, not restricted to nearest neighbors) and any vector
%  of local spins (integers or half-integers up to 15/2) can be given
%  in the same way.  Duplicate pairs are summed.
%  ================================================================

%% Model
N  = 12;
spins = repmat([1, 1.5], 1, N/2);        % local spin of each site
J1 = 1.0;                                % nearest neighbors
J2 = 0.3;                                % next-nearest neighbors
i  = (1:N)';
couplings = [i, mod(i, N) + 1,     J1 * ones(N, 1);    % [i, j, J_ij]
             i, mod(i + 1, N) + 1, J2 * ones(N, 1)];

%% FTLM
R       = 50;                            % random vectors per sector
M_lz    = 100;                           % Lanczos steps per random vector
T_range = logspace(-2, 1, 100);          % temperatures (units of J1/k_B)

%% Optional (see help ftlm.defaults)
precision  = 'single';                   % 'single' | 'double' | 'half' | 'bfloat16'
lookup     = 'clt';                      % 'clt' | 'cr'
ed_thresh  = 1000;                       % exact diagonalization for dim <= 1000
output_name = 'ftlm_mixed_ring.mat';
