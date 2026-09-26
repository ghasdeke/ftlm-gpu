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
"""Python API example: mixed-spin ring with next-nearest-neighbor coupling.

    python example_api.py
"""

import numpy as np

from ftlm_gpu import Model, run, save_results

N = 12
spins = [1.0, 1.5] * (N // 2)
couplings = ([(i, (i + 1) % N, 1.0) for i in range(N)]        # J1, 0-based sites
             + [(i, (i + 2) % N, 0.3) for i in range(N)])     # J2
model = Model(spins, couplings)

T = np.logspace(-2, 1, 100)
res = run(model=model, R=50, M_lz=100, T_range=T, precision="single")
k = np.argmax(res["C_T"])
print(f"\nmax C = {res['C_T'][k]:.4f} at T = {T[k]:.3f}")
save_results(res, "ftlm_mixed_ring.npz")
