function run_all_studies(out_dir)
%RUN_ALL_STUDIES  Run all studies of the revised paper and make the figures.
%   RUN_ALL_STUDIES(OUT_DIR) runs, in this order,
%       study_ghosts, study_ed_decomposition, study_lanczos_steps,
%       study_precision, study_exact_icosahedron, study_seeds
%   writing the study_*.mat files and the figures fig1..fig7 to OUT_DIR.
%   Total run time on the RTX 4000 Ada workstation: roughly 4 hours
%   (dominated by study_seeds and the CPU references in study_precision).
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

step('study_ghosts');           study_ghosts('OutDir', out_dir);
step('study_ed_decomposition'); study_ed_decomposition('OutDir', out_dir);
step('study_lanczos_steps');    study_lanczos_steps('OutDir', out_dir);
step('study_precision');        study_precision('OutDir', out_dir);
step('study_exact_icosahedron'); study_exact_icosahedron('DataDir', out_dir);
step('study_seeds');            study_seeds('OutDir', out_dir);
step('make_figures');           make_figures('DataDir', out_dir, 'OutDir', out_dir);
fprintf('\nAll studies done in %.1f h.\n', toc(t0) / 3600);
end
