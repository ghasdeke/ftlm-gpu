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

function s = format_num(n)
%FORMAT_NUM  Format an integer count compactly, e.g. "19.61M", "531.4k", "4096".
%
%   Small utility used by the benchmark scripts (benchmark_ico_v1.m,
%   benchmark_icosid_v1.m) to print Hilbert-space and sector dimensions
%   in a human-readable form.
    if n >= 1e6
        s = sprintf('%.2fM', n/1e6);
    elseif n >= 1e3
        s = sprintf('%.1fk', n/1e3);
    else
        s = sprintf('%d', n);
    end
end
