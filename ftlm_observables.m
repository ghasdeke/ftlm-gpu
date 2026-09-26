function res = ftlm_observables(input)
%FTLM_OBSERVABLES  Sector-FTLM thermodynamics of isotropic spin clusters.
%   FTLM_OBSERVABLES(INPUT_FILE) computes the heat capacity C(T), the
%   zero-field magnetic susceptibility chi(T) and the effective partition
%   function Z_eff(T) = Z(T) * exp(beta * E0) of the spin Hamiltonian
%
%       H = sum_c  J_c  s_{i_c} . s_{j_c}
%
%   (arbitrary pairwise couplings, arbitrary local spins s_i) with the
%   finite-temperature Lanczos method (FTLM) and sector decomposition by
%   total S^z.  The Hamiltonian action is evaluated matrix-free on the
%   GPU (or CPU) with a compressed lookup table (CLT) or combinatorial
%   ranking (CR) as state-to-index map.
%
%   INPUT_FILE is a plain MATLAB script (.m) with variable assignments;
%   see input_ico_s1_example.m (uniform preset geometry) and
%   input_custom_mixed_example.m (coupling list, mixed spins).
%   Alternatively, pass a struct with the same fields.
%
%   Model (either a preset geometry or a coupling list):
%       geometry   'ico','cubo','cube','dodeca','icosid','ring' (+ N_ring)
%       J          uniform nearest-neighbor coupling (presets)
%       couplings  K x 3 matrix [i, j, J_ij], 1-based, arbitrary pairs
%       s_val      uniform local spin, or
%       spins      1 x N vector of local spins (mixed spins)
%   FTLM (required):
%       R, M_lz, T_range
%   Optional inputs and defaults: see help ftlm.defaults
%       (backend, precision, lookup, use_cpu_reference, cpu_precision,
%        only_M0, ed_thresh, seed, B_gpu, B_cpu, L2_cache_bytes,
%        save_ritz, output_dir, output_name, verbose)
%
%   Output:
%       A .mat file 'ftlm_<tag>.mat' (or output_name) in output_dir with
%       T_range, C_T, chi_T, Z_eff, optional *_cpu reference results,
%       the configuration, per-sector diagnostics and timings.  The same
%       data are returned as a struct if an output argument is requested.
%
%   Requirements:
%       MATLAB R2022a or newer; Parallel Computing Toolbox for the GPU
%       backend.  Build the MEX files with build_all.m.
%
%   Citation:
%       S. Ghassemi Tabrizi and T. D. Kuehne, "GPU-accelerated
%       finite-temperature Lanczos method for spin Hamiltonians"
%       (see CITATION.cff for the up-to-date reference).

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

if nargin < 1 || isempty(input)
    error('ftlm_observables:NoInput', ...
          ['ftlm_observables requires an input file or struct. Usage:\n', ...
           '    ftlm_observables(''input.m'')\n', ...
           'See input_ico_s1_example.m for a commented template.']);
end

if isstruct(input)
    opts = input;
    src  = '(struct)';
else
    src = char(input);
    if exist(src, 'file') ~= 2
        error('ftlm_observables:InputNotFound', 'Input file not found: %s', src);
    end
    opts = read_input_file(src);
end

fprintf('=== ftlm_observables: sector-FTLM thermodynamics ===\n');
fprintf('Input: %s\n\n', src);

res = ftlm.run(opts);

% ---- save ---------------------------------------------------------------
o = res.opts;
if isempty(o.output_name)
    mat_name = sprintf('ftlm_%s.mat', res.model.tag);
else
    mat_name = o.output_name;
end
mat_path = fullfile(o.output_dir, mat_name);

S = res;
% flat copies of the most frequently used configuration fields (v1 names)
S.geometry  = res.model.geometry;
S.spins     = res.model.spins;
S.couplings = res.model.couplings;
S.N         = res.model.N;
S.n_total_save = res.model.D_full;
S.M_max     = max(res.sector_M);
for f = {'R', 'M_lz', 'J', 's_val', 'precision', 'lookup', 'backend', ...
         'cpu_precision', 'only_M0', 'use_cpu_reference', 'B_cpu', 'B_gpu', ...
         'ed_thresh', 'seed'}
    S.(f{1}) = o.(f{1});
end
save(mat_path, '-struct', 'S', '-v7.3');
fprintf('\nResults saved to: %s\n', mat_path);

if nargout == 0
    clear res
end
end

function opts = read_input_file(input_file)
%READ_INPUT_FILE  Run the input script and collect its variables.
    run(input_file);
    clear input_file
    vars = who;
    opts = struct();
    for k = 1 : numel(vars)
        if ~strcmp(vars{k}, 'opts')
            opts.(vars{k}) = eval(vars{k});
        end
    end
end
