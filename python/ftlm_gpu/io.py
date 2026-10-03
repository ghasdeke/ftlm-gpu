# Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
# and Helmholtz-Zentrum Dresden-Rossendorf e.V.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Input files (TOML / JSON) and result files (.npz / .mat).

Input files use the variable names of the MATLAB input files.  Coupling
site indices are 1-based by default (``index_base = 1``), so that the
same coupling lists can be used in MATLAB and Python.  ``T_range`` is a
list of temperatures or a table with one of the keys ``linspace``,
``logspace`` (``[start, stop, num]``, as in numpy) or ``file``.

Example (TOML)::

    spins     = [0.5, 1.0, 0.5, 1.0]
    couplings = [[1, 2, 1.0], [2, 3, 1.0], [3, 4, 1.0], [4, 1, 1.0], [1, 3, -0.2]]
    R    = 100
    M_lz = 100
    T_range = { logspace = [-2, 1, 100] }
    precision = "single"
"""

import json
from pathlib import Path

import numpy as np

try:
    import tomllib
except ImportError:  # Python < 3.11
    tomllib = None


def load_input(path):
    path = Path(path)
    text = path.read_bytes()
    if path.suffix.lower() == ".json":
        opts = json.loads(text.decode("utf-8"))
    else:
        if tomllib is None:
            raise ImportError("TOML input requires Python >= 3.11 (or use JSON)")
        opts = tomllib.loads(text.decode("utf-8"))
    T = opts.get("T_range")
    if isinstance(T, dict):
        if "linspace" in T:
            a, b, n = T["linspace"]
            opts["T_range"] = np.linspace(a, b, int(n))
        elif "logspace" in T:
            a, b, n = T["logspace"]
            opts["T_range"] = np.logspace(a, b, int(n))
        elif "file" in T:
            opts["T_range"] = np.loadtxt(path.parent / T["file"], ndmin=1)
        else:
            raise ValueError("T_range table needs 'linspace', 'logspace' or 'file'")
    return opts


def result_arrays(res):
    """Flatten a result dict into arrays suitable for .npz / .mat files."""
    m = res["model"]
    o = res["opts"]
    out = {k: v for k, v in res.items()
           if k not in ("model", "opts", "ritz", "sector_method") and v is not None}
    out.update(spins=m.spins, couplings=np.column_stack((m.couplings[:, :2] + 1,
                                                          m.couplings[:, 2])),
               N=m.N, n_total_save=float(m.D_full), geometry=m.geometry,
               sector_method=np.array(res["sector_method"], dtype=object))
    if res.get("ritz") is not None:
        ritz = np.empty(len(res["ritz"]), dtype=object)
        for i, r in enumerate(res["ritz"]):
            ritz[i] = r
        out["ritz"] = ritz
    for k in ("R", "M_lz", "precision", "lookup", "backend", "cpu_precision",
              "ed_thresh", "seed", "B_gpu", "B_cpu", "only_M0", "use_cpu_reference"):
        out[k] = o[k]
    return out


def save_results(res, path):
    """Save to .mat (scipy.io, MATLAB names; couplings 1-based) or .npz.

    In .npz files, sector_method is stored as a string array; the optional
    per-sector Ritz data ('ritz', with save_ritz) is an object array and
    requires np.load(..., allow_pickle=True).
    """
    path = Path(path)
    arrays = result_arrays(res)
    if path.suffix.lower() == ".mat":
        from scipy.io import savemat
        savemat(path, {k: v for k, v in arrays.items()
                       if not (isinstance(v, float) and np.isnan(v))})
    else:
        np.savez(path, **{k: (np.asarray(v, dtype=str) if k == "sector_method" else np.asarray(v))
                          for k, v in arrays.items()})
    return path
