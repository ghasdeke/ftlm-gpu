# ftlm-gpu: GPU-accelerated finite-temperature Lanczos method for spin Hamiltonians

Matrix-free implementation of the finite-temperature Lanczos method (FTLM)
for isotropic spin Hamiltonians

```
H = sum_c  J_c  s_{i_c} . s_{j_c}
```

with an **arbitrary list of pairwise couplings** (i, j, J_ij) — not
restricted to nearest neighbors or to a uniform coupling — and **arbitrary
local spins** s_i (mixed spins, integers or half-integers up to 15/2). The
Hilbert space is decomposed into sectors of total S^z, and each sector is
treated by a block-Lanczos sweep over random start vectors. The code
computes the heat capacity C(T), the zero-field magnetic susceptibility
chi(T) and the effective partition function Z_eff(T) = Z(T) exp(beta E_0).

Two front ends share the same CUDA kernels (`cuda/ftlm_kernels.cuh`):

- **MATLAB** (`ftlm_observables.m`, package `+ftlm`, MEX gateways), and
- **Python** (`python/ftlm_gpu`, kernels compiled at run time with NVRTC
  through CuPy) — no MATLAB license needed.

For identical start vectors both front ends produce identical Lanczos
coefficients.

## Citation

If you use this code in academic work, please cite **both** the software
and the accompanying paper. A machine-readable [`CITATION.cff`](CITATION.cff)
is included.

> S. Ghassemi Tabrizi and T. D. Kühne,
> *GPU-accelerated finite-temperature Lanczos method for spin
> Hamiltonians*, [arXiv:2605.26261](https://arxiv.org/abs/2605.26261) (2026).

## Overview

The Hamiltonian action H v is evaluated matrix-free in a row-wise gather
formulation: each GPU thread generates the basis states connected to its
output state and accumulates one output element for B Lanczos chains at
once. The position of a generated state in the sector basis is obtained
with one of two state-to-index maps:

| Lookup | Memory | Notes |
|---|---|---|
| `clt` — compressed lookup table | basis array (4 B/state) + prod(2s_i+1)/4 bytes | fixed, branch-light lookup; default |
| `cr` — combinatorial ranking | a few kB (cumulative dimension table in shared memory) | no per-state tables; for label spaces beyond 2^31 |

Precision (`precision`):

| Value | Storage | Arithmetic | Backends |
|---|---|---|---|
| `double` | FP64 | FP64 | GPU, CPU |
| `single` | FP32 | FP32 | GPU (default), CPU |
| `half` | FP16 | FP32 | GPU |
| `bfloat16` | BF16 | FP32 | GPU |

The 16-bit formats store the Lanczos vectors in 16 bit (halving the vector
memory relative to FP32); the vectors are kept at norm sqrt(dim) so that
their entries stay in the normal FP16 range. The small tridiagonal
eigenproblems are always solved in FP64 on the host.

Accuracy: FP32 is the recommended default; its deviations from FP64 are
far below the stochastic FTLM error (paper, Sections 3.1-3.5). With 16-bit
storage the energy resolution is of order u W (u = 2^-11 for FP16, 2^-8
for BF16; W the spectral width), so FP16 results are reliable only for
temperatures well above u W, and BF16 is not recommended for
thermodynamics. For `half`, the couplings should be of order 1 (FP16
overflows above 65504): rescale J and T together, e.g. use units of the
largest |J|. A non-finite Lanczos coefficient stops the run with an error.

Kernels:

| File | Backend | Lookup | Precision |
|---|---|---|---|
| `cuda/ftlm_kernels.cuh` | CUDA device code (shared) | CLT, CR | FP64, FP32, FP16, BF16 |
| `ftlm_gpu_mex.cu` | MATLAB MEX gateway (GPU) | CLT, CR | FP64, FP32, FP16, BF16 |
| `ftlm_cpu_mex.cpp` | MATLAB MEX, OpenMP (CPU) | CLT | FP64, FP32 |
| `python/ftlm_gpu/gpu.py` | Python/CuPy driver (GPU) | CLT, CR | FP64, FP32, FP16, BF16 |

**Small sectors.** The number of Lanczos steps and of random vectors are
capped at the sector dimension, N_L,eff = min(N_L, dim) and R_eff =
min(R, dim). Each Lanczos chain stops individually when its Krylov space
is exhausted (beta_j <= c u ||T_j||_1 with the unit roundoff u of the
storage precision; c = 64 for FP64/FP32 and 8 for FP16/BF16), which
happens when the start vector overlaps with fewer than N_L distinct
eigenvalues. Sectors with dim <= `ed_thresh` (default 1000) are
diagonalized exactly instead of FTLM.

**Size limits.** N <= 32 sites, at most 512 couplings, s_i <= 15/2, sector
dimension < 2^31. CLT: prod_i(2 s_i + 1) <= 2^31. CR: the packed state
needs sum_i ceil(log2(2 s_i + 1)) <= 64 bits and the cumulative dimension
table at most 48 kB. The code checks all limits and reports clear errors.

## Requirements

**MATLAB front end**
- MATLAB R2022a or newer; for the GPU: Parallel Computing Toolbox
  (`mexcuda`) and an NVIDIA GPU with compute capability >= 7.0.
- A C++ compiler with OpenMP for the CPU kernel (Windows: MSVC via
  `mex -setup C++`; Linux: GCC >= 9 or Clang >= 12).

**Python front end**
- Python >= 3.9 (TOML input files: >= 3.11), NumPy, SciPy.
- For the GPU: CuPy for your CUDA version (e.g. `cupy-cuda12x`) and an
  NVIDIA GPU with compute capability >= 7.0. Without CuPy, the CPU
  reference backend (explicit sparse matrix, small systems) is available.

The kernels use CUDA; AMD GPUs are not supported at present (the device
code uses no CUDA libraries and could be ported with HIP).

## Installation

**MATLAB**, in the repository root:

```matlab
>> build_all
>> cd tests; run_tests
```

**Python**, in the repository root:

```bash
pip install ".[cuda12]"
python -m unittest discover python/tests
```

(or, without installing, add `python/` to `PYTHONPATH`).

## Quick start

MATLAB:

```matlab
>> ftlm_observables('input_ico_s1_example.m')         % s = 1 icosahedron
>> ftlm_observables('input_mixed_ring_example.m')     % mixed spins, J1-J2 ring
```

Python (command line or API):

```bash
python -m ftlm_gpu examples/python/ico_s1.toml
python -m ftlm_gpu examples/python/mixed_ring.toml -o mixed_ring.mat
```

```python
import numpy as np
from ftlm_gpu import Model, run

N = 12
model = Model(spins=[1.0, 1.5] * (N // 2),
              couplings=[(i, (i + 1) % N, 1.0) for i in range(N)]      # 0-based
                      + [(i, (i + 2) % N, 0.3) for i in range(N)])
res = run(model=model, R=50, M_lz=100, T_range=np.logspace(-2, 1, 100))
res["C_T"], res["chi_T"]
```

## Input

MATLAB input files are plain scripts with variable assignments; Python
input files (TOML or JSON) use the same names. In both, coupling site
indices are **1-based** (the Python API uses 0-based indices by default,
`index_base=1` accepts 1-based lists).

**Model** — a predefined geometry or a coupling list:

| Variable | Meaning |
|---|---|
| `geometry` | `'ico'`, `'cubo'`, `'cube'`, `'dodeca'`, `'icosid'`, `'ring'` (with `N_ring`): nearest-neighbor bonds of the cluster |
| `J` | uniform coupling on all bonds of a predefined geometry (`J > 0` antiferromagnetic) |
| `couplings` | K x 3 matrix `[i, j, J_ij]` of pairwise couplings (any pairs; duplicate pairs are summed); replaces `geometry`/`J` |
| `s_val` | local spin of all sites |
| `spins` | vector of local spins s_i (mixed spins); replaces `s_val` |
| `N_sites` | number of sites for a coupling list with uncoupled sites (optional) |

**FTLM (required):** `R` (random vectors per sector), `M_lz` (Lanczos
steps N_L), `T_range` (temperatures in units of J/k_B; in TOML a list or
`{ logspace = [a, b, n] }`, `{ linspace = [a, b, n] }`, `{ file = "T.dat" }`).

**Optional (default):**

| Variable | Default | Meaning |
|---|---|---|
| `backend` | `'gpu'` | `'gpu'` or `'cpu'` |
| `precision` | `'single'` | see table above |
| `lookup` | `'clt'` | `'clt'` or `'cr'` (GPU) |
| `use_cpu_reference` | `false` | repeat the run on the CPU with `cpu_precision` (same start vectors) |
| `cpu_precision` | `'double'` | `'double'` or `'single'` |
| `ed_thresh` | `1000` | sectors with dim <= ed_thresh: exact diagonalization |
| `only_M0` | `false` | lowest magnetization sector only |
| `seed` | `0` | 0: v1-compatible start vectors; k > 0: independent set k |
| `B_gpu` | `0` | GPU block size (0: adaptive, 8 if three blocks of 8 vectors fit into `L2_cache_bytes`, else 4) |
| `B_cpu` | `8` | CPU block size |
| `L2_cache_bytes` | `48e6` | threshold for the adaptive `B_gpu` (value used for all results of the paper; the RTX 4000 SFF Ada has a 40 MB L2 cache) |
| `save_ritz` | `false` | store Ritz values, weights and Lanczos coefficients per sector |
| `output_dir`, `output_name` | `'.'`, `''` | output file (default `ftlm_<tag>.mat`) |

See `help ftlm.defaults` (MATLAB) or `ftlm_gpu.DEFAULTS` (Python).

## Output

`ftlm_observables` writes a `.mat` file (Python: `.npz` or `.mat`) with
`T_range`, `C_T`, `chi_T`, `Z_eff`, the optional CPU reference
`C_T_cpu`, `chi_T_cpu`, `Z_eff_cpu`, the model (`spins`, `couplings`),
the configuration, per-sector data (`sector_M`, `sector_dims`,
`sector_method`, ...) and timings. `chi_T` is given per g^2 mu_B^2 / k_B.

## Tests

`tests/run_tests.m` (MATLAB) and `python/tests/test_ftlm.py` (Python)
check the SpMV of every kernel variant against an explicit sparse
Hamiltonian with random couplings between all pairs and mixed spins, the
identical basis order of CLT and CR, the exact Gauss quadrature for full
Krylov spaces, the weight sum rule (including half-integer total spin),
the Lanczos breakdown handling, and FTLM against exact diagonalization.

## Reproducing the paper

All numbers, figures and tables of the paper (revised version) were
produced with v2.0.0 by the following scripts:

| Script | Content |
|---|---|
| `examples/benchmark_table3.m` | Table 3: timings of all kernel variants (CPU/GPU x FP64/FP32/FP16/BF16 x CLT/CR, single-vector runs) for the icosahedron and icosidodecahedron workloads; run it on an otherwise idle machine |
| `examples/paper/memory_traffic.py` | memory-traffic analysis of Section 3.4 (SpMV and vector operations timed separately; Python front end) |
| `examples/paper/run_all_studies.m` | runs the studies below and `make_figures` (about 4.5 h on an RTX 4000 SFF Ada) |
| `examples/paper/study_precision.m` | Figs. 1, 3, Table 5: FP64/FP32/FP16/BF16 GPU and FP64/FP32 CPU runs with identical start vectors |
| `examples/paper/study_seeds.m` | Fig. 2: 50 independent FTLM runs, empirical and theoretical stochastic error, FP32 deviation of single and pooled runs |
| `examples/paper/study_ghosts.m` | Figs. 4, 5: ghost diagnostic and cluster weights |
| `examples/paper/study_lanczos_steps.m` | Fig. 6: convergence in the number of Lanczos steps (one run with N_L = 300 per precision; smaller N_L by truncating the recorded Lanczos coefficients, which is identical to separate runs) |
| `examples/paper/run_cpu_reference.m` | CPU variants of the precision study in a second MATLAB session, in parallel with the GPU studies (started by `run_all_studies`) |
| `examples/paper/study_ed_decomposition.m` | Fig. 7: error decomposition against exact diagonalization |
| `examples/paper/study_exact_icosahedron.m` | comparison with the exact heat capacity of the s = 3/2 icosahedron (`examples/paper/data`) |
| `examples/paper/make_figures.m` | Figs. 1-7 (PDF and 600 dpi PNG) from the study files |
| `examples/paper/summarize_results.m` | the numbers quoted in the text and the rows of Tables 3 and 5 (JSON) |

The precision studies use `R = 100`, `M_lz = 100`, `ed_thresh = 0` and
`seed = 0` (start vectors drawn in FP64 from `rng(dim, 'twister')` and
rounded to the storage precision). `plot_paperfig1_v2.m` plots C(T) and
chi(T) of a single `ftlm_observables` result with `use_cpu_reference`.

Because the dot products are now reduced by a tree (GPU) or a cascade
(CPU) summation, v2.0.0 reproduces the v1 results only up to rounding
(relative deviations of order 10^-7 in FP32), not bit for bit.

## Hardware-specific tuning

- **GPU block size.** Set `L2_cache_bytes` to the L2 size of your GPU or
  fix `B_gpu`. The block size is reduced automatically if a sector does
  not fit into free device memory.
- **CPU.** `B_cpu = 8` corresponds to one cache line of FP64 values;
  the thread count follows `OMP_NUM_THREADS` / `maxNumCompThreads`
  (`ftlm_cpu_mex('set_threads', n)`).

## Files

| File | Purpose |
|---|---|
| `ftlm_observables.m` | MATLAB entry point |
| `+ftlm/` | MATLAB package: model, sectors, basis, CLT/CR tables, sparse H, FTLM driver, observables |
| `ftlm_gpu_mex.cu`, `ftlm_cpu_mex.cpp` | MEX gateways (GPU, CPU) |
| `cuda/ftlm_kernels.cuh` | CUDA device code (shared by MATLAB and Python) |
| `build_all.m` | builds the MEX files |
| `input_*.m` | MATLAB example inputs |
| `python/ftlm_gpu/` | Python package; `python -m ftlm_gpu input.toml` |
| `pyproject.toml` | Python packaging |
| `tests/`, `python/tests/` | test suites |
| `examples/` | benchmarks, figure scripts, Python examples |
| `CHANGELOG.md`, `CITATION.cff`, `LICENSE`, `NOTICE` | release history, citation, license |

## License

Apache License, Version 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

## Authors

**Shadan Ghassemi Tabrizi** — Technische Universität Dresden /
Helmholtz-Zentrum Dresden-Rossendorf

Scientific co-author of the accompanying paper: **Thomas D. Kühne**.
