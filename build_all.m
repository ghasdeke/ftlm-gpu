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
%  Build script for the MATLAB MEX kernels.
%
%    1. ftlm_gpu_mex.cu   GPU block Lanczos (CLT and CR lookup;
%                         FP64, FP32, FP16, BF16 storage).  Device code
%                         in cuda/ftlm_kernels.cuh (shared with Python).
%    2. ftlm_cpu_mex.cpp  CPU block Lanczos (CLT lookup; FP64, FP32;
%                         OpenMP)
%
%  The GPU kernel is skipped (with a warning) if the Parallel Computing
%  Toolbox / mexcuda is not available; the CPU backend then still works
%  (backend = 'cpu').
%
%  Prerequisites:
%    - MATLAB R2022a or newer
%    - GPU: Parallel Computing Toolbox (mexcuda), NVIDIA GPU with compute
%      capability >= 7.0
%    - CPU: C++ compiler with OpenMP (Windows: MSVC via "mex -setup C++";
%      Linux: GCC >= 9 or Clang >= 12)
%
%  After building, run the tests (tests/run_tests.m) and an example:
%      ftlm_observables('input_ico_s1_example.m')
%  ================================================================

clear functions; clear mex;
root = fileparts(mfilename('fullpath'));
old_dir = cd(root);
cleanup_dir = onCleanup(@() cd(old_dir));

fprintf('\n=== Compilation of the FTLM kernels ===\n\n');

%% 1.  GPU kernel
fprintf('Compiling ftlm_gpu_mex.cu ...\n');
if exist('mexcuda', 'file') == 2
    try
        mexcuda('ftlm_gpu_mex.cu');
        fprintf('  Compiled successfully.\n\n');
    catch ME
        fprintf('  ERROR: %s\n', ME.message);
        fprintf('  Hint: check "mex -setup C++" and the CUDA installation.\n');
        rethrow(ME);
    end
else
    warning('build_all:NoMexcuda', ...
        'mexcuda not found (Parallel Computing Toolbox missing): GPU kernel skipped.');
end

%% 2.  CPU kernel (OpenMP, platform-specific flags)
fprintf('Compiling ftlm_cpu_mex.cpp (with OpenMP) ...\n');
try
    if ispc
        mex('ftlm_cpu_mex.cpp', 'COMPFLAGS=$COMPFLAGS /openmp');
    elseif ismac
        % Apple clang needs libomp (e.g. Homebrew) for OpenMP support
        mex('ftlm_cpu_mex.cpp', 'CXXFLAGS=$CXXFLAGS -Xpreprocessor -fopenmp', ...
            'LDFLAGS=$LDFLAGS -lomp');
    else
        mex('ftlm_cpu_mex.cpp', 'CXXFLAGS=$CXXFLAGS -fopenmp', ...
            'LDFLAGS=$LDFLAGS -fopenmp');
    end
    fprintf('  Compiled successfully.\n\n');
catch ME
    fprintf('  ERROR: %s\n', ME.message);
    fprintf('  Hint: check the OpenMP support of your compiler.\n');
    fprintf('  Note: if ftlm_cpu_mex was already used in this MATLAB session,\n');
    fprintf('  the MEX file is locked in memory (deliberately, see the source\n');
    fprintf('  header). Restart MATLAB and re-run build_all.\n');
    rethrow(ME);
end

%% 3.  Sanity check (no initialization, no GPU call)
fprintf('=== Sanity check ===\n');
try
    n_omp = ftlm_cpu_mex('info');
    fprintf('  ftlm_cpu_mex reports %d OpenMP threads.\n', n_omp);
catch ME
    fprintf('  WARNING: ftlm_cpu_mex(''info'') failed: %s\n', ME.message);
end

fprintf('\n=== Build complete. ===\n');
fprintf('Next steps:  >> cd tests; run_tests\n');
fprintf('             >> ftlm_observables(''input_ico_s1_example.m'')\n\n');
