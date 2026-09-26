%% input_ico_s1_example.m
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
%  Example input file for ftlm_observables (commented template).
%
%  Invoke as
%      ftlm_observables('input_ico_s1_example.m')
%
%  This file is a plain MATLAB script: each line is an ordinary
%  variable assignment.  Required inputs come first; optional inputs
%  (with documented defaults) at the end may be omitted entirely.
%  For user-defined couplings and mixed spins see
%  input_mixed_ring_example.m.
%  ================================================================

%% ----------------------------------------------------------------
%  Required inputs
%  ----------------------------------------------------------------

% Model.  Either a predefined geometry with a uniform nearest-neighbor
% coupling J ...
%   geometry: 'ico', 'cubo', 'cube', 'dodeca', 'icosid', 'ring'
%   (for 'ring' also set N_ring)
geometry = 'ico';
J        = 1.0;          % H = +J sum_<i,j> s_i . s_j  (J > 0: antiferromagnetic)
% ... or a list of pairwise couplings [i, j, J_ij] (1-based, any pairs):
%   couplings = [1 2 1.0; 2 3 1.0; 1 3 0.5];

% Local spin: a scalar for all sites, or a vector for mixed spins.
s_val    = 1.0;
%   spins = [0.5 1 0.5 1 ...];

% Number of FTLM random vectors per S^z sector.  The quick-start demo
% uses R = 50 to keep the runtime short; the precision analysis in the
% paper (Figs. 1-3) uses R = 100.
R        = 50;

% Lanczos steps per random vector (N_L).
M_lz     = 100;

% Temperature grid (in units of J/k_B), any MATLAB expression, e.g.
T_range  = logspace(-2, 1, 100);
%   T_range = linspace(0.01, 10, 100);
%   T_range = load('T_grid.dat');       % one T value per line

%% ----------------------------------------------------------------
%  Optional inputs (defaults shown; the lines below may be deleted)
%  ----------------------------------------------------------------

% Backend and arithmetic.
%   backend   'gpu' | 'cpu'
%   precision GPU: 'single' (FP32) | 'double' (FP64) | 'half' (FP16
%             storage, FP32 arithmetic) | 'bfloat16' (BF16 storage);
%             CPU: 'double' | 'single'
%   lookup    'clt' (compressed lookup table) | 'cr' (combinatorial
%             ranking, no per-state tables; GPU only)
backend   = 'gpu';
precision = 'single';
lookup    = 'clt';

% Additionally run the CPU reference (same start vectors) and report
% max |C_T - C_T_cpu| / max|C_T_cpu|.
use_cpu_reference = true;
cpu_precision     = 'double';

% Exact diagonalization of small sectors: sectors with dim <= ed_thresh
% are diagonalized densely (8 * dim^2 bytes) instead of FTLM, giving
% their exact contribution.  0 = FTLM in every sector.  Independently,
% N_L and R are capped at the sector dimension, and each Lanczos chain
% stops early if its Krylov space is exhausted.
ed_thresh = 1000;

% Start vectors: seed = 0 reproduces v1 (rng(dim) per sector);
% seed = k > 0 gives an independent set k (multi-seed error analysis).
seed = 0;

% If true, restrict the calculation to the lowest sector (M = 0).
only_M0 = false;

% Block sizes (number of Lanczos chains processed together).
% B_gpu = 0: adaptive (8 if three blocks of 8 vectors fit into
% L2_cache_bytes, else 4), otherwise an integer in [1, 16].
B_gpu = 0;
B_cpu = 8;
L2_cache_bytes = 48e6;     % 48 MB = NVIDIA RTX 4000 Ada; adjust to your GPU

% Keep Ritz values/weights and Lanczos coefficients per sector.
save_ritz = false;

% Output
output_dir  = '.';
output_name = '';          % default: ftlm_<tag>.mat, e.g. ftlm_ico_s1.mat
