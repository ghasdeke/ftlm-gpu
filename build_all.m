%% build_all.m
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
%  Master build script for the FTLM release kernels.
%
%  Compiles the three kernels of the OSS release:
%
%    1. cuda_lanczos_clut_block.cu        (GPU, FP32, CLT block Lanczos)
%    2. cuda_lanczos_crank_Sr_general.cu  (GPU, FP32, CRank block Lanczos)
%    3. cpu_lanczos_omp.c                 (CPU, FP64, OpenMP, B=8)
%
%  This script performs NO numerical verification.  After successful
%  compilation only a harmless cpu_lanczos_omp('info') call is issued,
%  which returns the number of OpenMP threads -- without MEX
%  initialization, without any GPU call.  To verify that the kernels
%  work end-to-end on a real physics problem, run ftlm_observables.m
%  afterwards.
%
%  Prerequisites:
%    - MATLAB R2022a or newer
%    - Parallel Computing Toolbox (for mexcuda)
%    - NVIDIA CUDA Toolkit (compatible with the MATLAB CUDA version)
%    - Windows: MSVC with OpenMP support, configured via "mex -setup C++"
%    - Linux:   GCC with OpenMP support
%
%  After a successful run:  launch ftlm_observables to compute the
%  finite-temperature observables (and optionally cross-check against a
%  CPU FP64 reference).  ftlm_observables takes the path to an input
%  file as its only argument, e.g.
%      ftlm_observables('input_ico_s1_example.m')
%  ================================================================

clear functions; clear mex;

fprintf('\n=== Compilation of release kernels ===\n\n');

%% 1.  CUDA: cuda_lanczos_clut_block.cu
fprintf('Compiling cuda_lanczos_clut_block.cu ...\n');
try
    mexcuda('cuda_lanczos_clut_block.cu');
    fprintf('  Compiled successfully.\n\n');
catch ME
    fprintf('  ERROR: %s\n', ME.message);
    fprintf('  Hint: check "mex -setup C++" and the CUDA installation.\n');
    rethrow(ME);
end

%% 2.  CUDA: cuda_lanczos_crank_Sr_general.cu
fprintf('Compiling cuda_lanczos_crank_Sr_general.cu ...\n');
try
    mexcuda('cuda_lanczos_crank_Sr_general.cu');
    fprintf('  Compiled successfully.\n\n');
catch ME
    fprintf('  ERROR: %s\n', ME.message);
    rethrow(ME);
end

%% 3.  CPU: cpu_lanczos_omp.c  (OpenMP, platform-specific)
fprintf('Compiling cpu_lanczos_omp.c (with OpenMP) ...\n');
try
    if ispc
        % Windows / MSVC
        mex('cpu_lanczos_omp.c', 'COMPFLAGS=$COMPFLAGS /openmp');
    elseif isunix
        % Linux / macOS / GCC or Clang
        mex('cpu_lanczos_omp.c', ...
            'CFLAGS=$CFLAGS -fopenmp', ...
            'LDFLAGS=$LDFLAGS -fopenmp');
    else
        error('Unsupported platform.');
    end
    fprintf('  Compiled successfully.\n\n');
catch ME
    fprintf('  ERROR: %s\n', ME.message);
    fprintf('  Hint: check the OpenMP support of your compiler.\n');
    fprintf('  Note: if cpu_lanczos_omp was already used in this MATLAB\n');
    fprintf('  session, the MEX file is locked in memory (deliberately, see\n');
    fprintf('  the source header). Restart MATLAB and re-run build_all.\n');
    rethrow(ME);
end

%% 4.  Harmless sanity check  (no init, no GPU call)
fprintf('=== Sanity check ===\n');
try
    n_omp = cpu_lanczos_omp('info');
    fprintf('  cpu_lanczos_omp reports %d OpenMP threads.\n', n_omp);
catch ME
    fprintf('  WARNING: cpu_lanczos_omp(''info'') failed: %s\n', ME.message);
end

fprintf('\n=== Build complete. ===\n');
fprintf('Next step:  >> ftlm_observables(''input_ico_s1_example.m'')\n\n');
