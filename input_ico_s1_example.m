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
%  Example input file for ftlm_observables.
%
%  Invoke as
%      ftlm_observables('input_ico_s1_example.m')
%
%  This file is a plain MATLAB script: each line is an ordinary
%  variable assignment. Comments use the usual '%' syntax. Required
%  inputs come first; optional inputs (with documented defaults) at
%  the end may be omitted entirely.
%  ================================================================

%% ----------------------------------------------------------------
%  Required inputs
%  ----------------------------------------------------------------

% System geometry.  Allowed values: 'ico', 'cubo', 'cube', 'dodeca',
% 'icosid', 'ring'.  For 'ring', also set N_ring further down.
geometry = 'ico';

% Local spin quantum number (0.5, 1.0, 1.5, 2.0, ...).
s_val    = 1.0;

% Heisenberg exchange coupling, applied uniformly to every nearest-neighbor
% bond.  Sign convention: H = +J * sum_{<i,j>} S_i . S_j.  J > 0 yields
% antiferromagnetic exchange, J < 0 ferromagnetic.  Any finite real value.
J        = 1.0;

% Number of FTLM random vectors per S^z sector.  The quick-start demo
% uses R = 50 to keep the runtime short; the precision analysis in the
% paper (Figs. 1-3) uses R = 100.  Set R = 100 to match the paper.
R        = 50;

% Lanczos iterations per random vector.
M_lz     = 100;

% Temperature grid (in units of J/k_B).  Use any MATLAB expression
% that produces a vector of positive numbers.  Two common idioms:
%
%   Inline expression (logspace or linspace):
T_range  = logspace(-2, 1, 100);
%   T_range = linspace(0.01, 10, 100);
%
%   Read from an external file (one T value per line):
%   T_range = load('T_grid.dat');

%% ----------------------------------------------------------------
%  Optional inputs (defaults shown; the lines below may be deleted)
%  ----------------------------------------------------------------

% Number of sites for geometry='ring' (ignored otherwise).
% N_ring = 20;

% If true, restrict the calculation to the M=0 sector
% (chi(T) will then be 0; useful for quick smoke tests).
only_M0 = false;

% If true, additionally run a CPU FP64 reference and report the
% maximum relative deviation of C(T) between GPU FP32 and CPU FP64.
use_cpu_reference = true;

% Output directory for the .mat file.
output_dir = '.';

% Block-Lanczos block size on the CPU reference path.  Integer in
% [1, 32].  The default 8 corresponds to one x86 cache line of FP64
% and the natural AVX2/AVX-512 register width.
B_cpu = 8;

% Block-Lanczos block size on the GPU path.  Set to 0 (default) to
% use the adaptive L2 heuristic (8 if three FP32 blocks of the
% sector dimension fit into L2_cache_bytes, else 4).  Setting B_gpu
% to a fixed integer in [1, 32] overrides the heuristic.
B_gpu = 0;

% L2 cache size used by the B_gpu=0 heuristic, in bytes.  Default
% 48 MB (NVIDIA RTX 4000 Ada).  Adjust to match your GPU.
L2_cache_bytes = 48e6;

% Exact-diagonalization threshold for small sectors.  Sectors with
% dim_sector <= ed_thresh are handled by dense diagonalization
% (sparse H -> eig(full(H))) instead of FTLM, giving the exact
% partition-function contribution for that sector at essentially
% zero cost.  Default 0 disables the feature (every sector uses
% FTLM).  Sensible values range from 100 (only the tiniest sectors)
% to ~2000 (anything that still fits comfortably in memory as a
% dense matrix: 8 * dim^2 bytes).  Larger values increase ED memory
% pressure quadratically.
ed_thresh = 0;
