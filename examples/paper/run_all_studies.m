function run_all_studies(out_dir)
%RUN_ALL_STUDIES  Run all studies of the revised paper and make the figures.
%   RUN_ALL_STUDIES(OUT_DIR) runs, in this order,
%       study_ghosts, study_ed_decomposition, study_lanczos_steps,
%       study_precision, study_exact_icosahedron, study_seeds
%   writing the study_*.mat files and the figures fig1..fig7 to OUT_DIR.
%   On Windows, the CPU variants of study_precision run in a second MATLAB
%   session in parallel with the GPU studies (run_cpu_reference); on other
%   systems they run in the same session.  The results are the same in
%   both cases.
%   Total run time on the RTX 4000 SFF Ada workstation: about 2.7 hours
%   (dominated by study_seeds; the CPU references run in parallel).
%   The timings of Table 3 are produced separately by
%   examples/benchmark_table3.m (run it on an otherwise idle machine).

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

if nargin < 1, out_dir = '.'; end
if ~isfolder(out_dir), mkdir(out_dir); end
here = fileparts(mfilename('fullpath'));
addpath(here);
t0 = tic;
step = @(name) fprintf('\n##### %s (%.0f min elapsed) #####\n', name, toc(t0) / 60);

% the MATLAB launcher returns immediately only on Windows; elsewhere the CPU
% variants of study_precision run in this session
par  = ispc;
vars = {'gpu_double', 'gpu_single', 'gpu_half', 'gpu_bfloat16'};
if ~par, vars = [vars, {'cpu_double', 'cpu_single'}]; end

steps = {'study_ghosts',            @() study_ghosts('OutDir', out_dir);
         'study_ed_decomposition',  @() study_ed_decomposition('OutDir', out_dir);
         'study_lanczos_steps',     @() study_lanczos_steps('OutDir', out_dir);
         'cpu_reference (start)',   @() cpu_reference(par, out_dir, 'start');
         'study_precision',         @() study_precision('OutDir', out_dir, 'Variants', vars);
         'study_exact_icosahedron', @() study_exact_icosahedron('DataDir', out_dir);
         'study_seeds',             @() study_seeds('OutDir', out_dir);
         'cpu_reference (merge)',   @() cpu_reference(par, out_dir, 'merge');
         'make_figures',            @() make_figures('DataDir', out_dir, 'OutDir', out_dir)};
for k = 1 : size(steps, 1)
    step(steps{k, 1});
    try
        steps{k, 2}();
    catch ME          % one failing study must not stop the others
        fprintf(2, '##### %s FAILED:\n%s\n', steps{k, 1}, getReport(ME));
    end
end
fprintf('\nAll studies done in %.1f h.\n', toc(t0) / 3600);
end

function cpu_reference(par, out_dir, action)
% second MATLAB session for the CPU variants (Windows only, see above)
if par
    run_cpu_reference(out_dir, action);
else
    fprintf('CPU variants run within study_precision in this session.\n');
end
end
