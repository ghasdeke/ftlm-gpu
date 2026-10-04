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
"""GPU-accelerated finite-temperature Lanczos method for spin Hamiltonians.

Python front end of ftlm-gpu (MATLAB-free).  The matrix-free block Lanczos
kernels (cuda/ftlm_kernels.cuh) are compiled at run time with NVRTC via
CuPy; the same device code is used by the MATLAB MEX gateway.

Quick start::

    import numpy as np
    from ftlm_gpu import Model, run

    model = Model.preset("ico", s=1.0, J=1.0)
    res = run(model=model, R=100, M_lz=100, T_range=np.logspace(-2, 1, 100))
    res["C_T"], res["chi_T"]
"""

from .model import Model, geometry, GEOMETRIES
from .basis import Sector, sectors, enumerate_sector, build_clt, cr_tables, hamiltonian
from .ftlm import run, sector_ftlm, solve_tridiag, observables, CpuLanczos, DEFAULTS
from .io import load_input, save_results

__version__ = "2.0.1"

__all__ = ["Model", "geometry", "GEOMETRIES", "Sector", "sectors", "enumerate_sector",
           "build_clt", "cr_tables", "hamiltonian", "run", "sector_ftlm", "solve_tridiag",
           "observables", "CpuLanczos", "DEFAULTS", "load_input", "save_results",
           "__version__"]


def GpuLanczos(*args, **kwargs):
    """Matrix-free GPU block Lanczos for one sector (see ftlm_gpu.gpu)."""
    from .gpu import GpuLanczos as _G
    return _G(*args, **kwargs)
