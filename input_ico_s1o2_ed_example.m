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
%  Example: s = 1/2 icosahedron, FTLM vs. exact diagonalization.
%
%  Invoke as
%      ftlm_observables('input_ico_s1o2_ed_example.m')
%
%  For the s = 1/2 icosahedron (N = 12) the sector dimensions are
%
%      M = 0 : dim = 924    M = 4 : dim = 66
%      M = 1 : dim = 792    M = 5 : dim = 12
%      M = 2 : dim = 495    M = 6 : dim =  1
%      M = 3 : dim = 220
%
%  With ed_thresh = 1000 (the default; any value >= 924 works) every
%  sector is diagonalized exactly, i.e. the observables are exact to
%  machine precision; R and M_lz are then unused.  Set ed_thresh = 0 to
%  run FTLM in all sectors and compare with the exact result.
%  ================================================================

%% Required inputs
geometry = 'ico';
s_val    = 0.5;
J        = 1.0;
R        = 50;
M_lz     = 100;
T_range  = logspace(-2, 1, 100);

%% Optional inputs
ed_thresh   = 1000;          % >= largest sector dim (924): all sectors exact
output_name = 'ftlm_ico_s1o2_ed.mat';
