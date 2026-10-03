function run_cpu_reference(out_dir, action)
%RUN_CPU_REFERENCE  CPU variants of the precision study in a separate session.
%   The CPU variants of study_precision (OpenMP kernel, FP64 and FP32) do
%   not use the GPU, and no timing of them is reported.  RUN_ALL_STUDIES
%   therefore starts them in a second MATLAB session in parallel with the
%   GPU studies (on Windows, where the MATLAB launcher returns at once; on
%   other systems RUN_ALL_STUDIES runs them in its own session):
%
%   RUN_CPU_REFERENCE(OUT_DIR, 'start')  starts the second session
%   RUN_CPU_REFERENCE(OUT_DIR, 'run')    (in that session) computes the CPU
%                                        variants -> study_precision_<key>_cpu.mat
%                                        and writes cpu_reference_done.txt
%   RUN_CPU_REFERENCE(OUT_DIR, 'merge')  waits for the marker file and copies
%                                        the CPU variants into
%                                        study_precision_<key>.mat
%   The results are identical to a sequential run of study_precision.

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

here = fileparts(mfilename('fullpath'));
out_dir = char(out_dir);
is_abs = startsWith(out_dir, {'/', '\'}) || (numel(out_dir) >= 2 && out_dir(2) == ':');
if ~is_abs, out_dir = fullfile(pwd, out_dir); end
marker = fullfile(out_dir, 'cpu_reference_done.txt');
switch action
    case 'start'
        if isfile(marker), delete(marker); end
        % the launcher returns immediately; with -r (not -batch) the session
        % has its own command window, whose output goes to the log file
        cmd = sprintf(['matlab -nosplash -nodesktop -minimize -logfile "%s" ', ...
                       '-r "addpath(''%s''); run_cpu_reference(''%s'', ''run''); exit"'], ...
                      fullfile(out_dir, 'cpu_reference.log'), here, out_dir);
        status = system(cmd);
        assert(status == 0, 'run_cpu_reference:start', 'could not start the CPU session');
        fprintf('CPU reference session started (log: %s)\n', fullfile(out_dir, 'cpu_reference.log'));
    case 'run'
        addpath(here);
        try
            study_precision('Variants', {'cpu_double', 'cpu_single'}, 'OutDir', out_dir, 'Suffix', '_cpu');
            msg = 'ok';
        catch ME
            msg = getReport(ME);
        end
        fid = fopen(marker, 'w');  fprintf(fid, '%s\n', msg);  fclose(fid);
    case 'merge'
        t0 = tic;
        while ~isfile(marker)
            assert(toc(t0) < 1.5 * 3600, 'run_cpu_reference:timeout', 'CPU session did not finish');
            pause(30);
        end
        msg = strtrim(fileread(marker));
        if ~strcmp(msg, 'ok'), warning('run_cpu_reference:failed', 'CPU session: %s', msg); end
        files = dir(fullfile(out_dir, 'study_precision_*_cpu.mat'));
        for k = 1 : numel(files)
            fc = fullfile(files(k).folder, files(k).name);
            fm = strrep(fc, '_cpu.mat', '.mat');
            C = load(fc);
            if isfile(fm), S = load(fm); else, S = C; end
            for v = {'cpu_double', 'cpu_single'}
                if isfield(C, v{1}), S.(v{1}) = C.(v{1}); end
            end
            save(fm, '-struct', 'S');
            fprintf('merged %s into %s\n', files(k).name, fm);
        end
    otherwise
        error('run_cpu_reference:action', 'unknown action %s', action);
end
end
