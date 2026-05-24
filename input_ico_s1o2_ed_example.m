%% input_ico_s1o2_ed_example.m
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
%  Example: s = 1/2 icosahedron with FULL EXACT DIAGONALIZATION.
%
%  Invoke as
%      ftlm_observables('input_ico_s1o2_ed_example.m')
%
%  Demonstrates the ed_thresh option (cf. input_ico_s1_example.m for
%  the verbose template).  For the s = 1/2 icosahedron (N=12) the
%  sector dimensions are
%
%      M = 0 : dim = 924    M = 4 : dim = 66
%      M = 1 : dim = 792    M = 5 : dim = 12
%      M = 2 : dim = 495    M = 6 : dim =  1
%      M = 3 : dim = 220
%
%  Setting ed_thresh = 1000 (any value >= 924 works) therefore routes
%  every sector through the exact-diagonalization branch: each sector
%  Hamiltonian is built sparsely and fully diagonalized via
%  eig(full(H)), yielding the exact thermodynamic observables to
%  machine precision.  The FTLM parameters R and M_lz are still
%  required by the input schema but are unused on the ED path.
%
%  With use_cpu_reference = true, the "GPU FP32 vs CPU FP64"
%  cross-check shows a max relative deviation at the level of
%  machine epsilon -- a direct illustration that both code paths
%  reduce to the same exact computation when ed_thresh covers all
%  sectors.
%  ================================================================

%% Required inputs
geometry = 'ico';
s_val    = 0.5;
J        = 1.0;
R        = 50;                       % unused under ed_thresh = 1000
M_lz     = 100;                      % unused under ed_thresh = 1000
T_range  = logspace(-2, 1, 100);

%% Optional inputs
use_cpu_reference = true;            % cross-check (will be ~eps with ED only)
ed_thresh         = 1000;            % >= largest sector dim (924); routes all sectors to ED
