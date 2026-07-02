# GPU-Accelerated FTLM for Heisenberg Spin Clusters

Matrix-free implementation of the Finite-Temperature Lanczos Method (FTLM)
for Heisenberg spin clusters with uniform nearest-neighbor exchange
coupling, with sector decomposition by total S^z and a block-Lanczos sweep
over random vectors. The production path runs in single precision on the
GPU; a double-precision CPU path is available as a reference.

## Citation

If you use this code in academic work, please cite **both** the software
and the accompanying paper. A machine-readable [`CITATION.cff`](CITATION.cff)
file is included; most citation managers (GitHub, Zenodo, Zotero, ...) can
read it directly.

The accompanying paper is:

> S. Ghassemi Tabrizi and T. D. Kühne,
> *GPU-accelerated finite-temperature Lanczos method for spin
> Hamiltonians*, [arXiv:2605.26261](https://arxiv.org/abs/2605.26261) (2026).

## Overview

`ftlm_observables.m` is the single entry point. It computes the three
finite-temperature observables

- specific heat `C(T)`,
- magnetic susceptibility `chi(T)`,
- effective partition function `Z_eff(T) = Z(T) * exp(beta * E_0)`,

on a user-supplied temperature grid, for a Heisenberg Hamiltonian of the
form

```
H = J * sum_{<i,j>} S_i . S_j
```

on one of six predefined cluster geometries (icosahedron, cuboctahedron,
cube, dodecahedron, icosidodecahedron, ring) for any local spin
`s = 1/2, 1, 3/2, 2, ...`. The coupling `J` is a single scalar that
applies uniformly to every nearest-neighbor bond. `J > 0` corresponds to
antiferromagnetic, `J < 0` to ferromagnetic exchange. Extending the model
to bond-dependent couplings `J_{ij}` is straightforward: it only requires
passing a per-bond coupling vector instead of a scalar through the kernel
init arguments and replacing the scalar `J` in the diagonal and S+S- /
S-S+ terms by the corresponding entry of that vector.

The matrix-vector product `H * v` is matrix-free: bonds and on-site
quantum numbers reconstruct the Hamiltonian action on the fly, and basis
lookup uses a compressed table (CLT). The Lanczos iteration is a block
sweep over `R` random vectors per sector. Three MEX kernels are provided:

| Kernel | Backend | Precision | Notes |
|---|---|---|---|
| `cuda_lanczos_clut_block.cu` | CUDA | FP32 | Production GPU path |
| `cuda_lanczos_crank_Sr_general.cu` | CUDA | FP32 | Alternative GPU path (combinatorial ranking) |
| `cpu_lanczos_omp.c` | OpenMP | FP64 | Reference CPU path |

Single-vector Lanczos corresponds to running a block kernel with block
size `B = 1`; the benchmark scripts use this to time the single-vector
variants.

**Size limits:** the kernels encode basis states as 32-bit integers, so
`(2*s_val + 1)^N` must not exceed `2^31` (about 2.1e9), and `N <= 32`
sites are supported. `ftlm_observables` checks both limits and reports
a clear error for out-of-range systems.

## Requirements

- **MATLAB** R2022a or newer, with the **Parallel Computing Toolbox**
  (provides `gpuArray` and `mexcuda`).
- **NVIDIA CUDA Toolkit** compatible with the MATLAB CUDA version. The
  GPU must support compute capability 7.0 or newer (Volta, Turing,
  Ampere, Ada, Hopper, Blackwell).
- A **C compiler with OpenMP support** for the CPU kernel:
  - Windows: MSVC (configure via `mex -setup C++`).
  - Linux:   GCC 9+ or Clang 12+.

## Installation

From the MATLAB command prompt, in the repository root:

```matlab
>> build_all
```

This compiles the three MEX kernels in place. The script reports each
kernel as it builds and ends with a harmless sanity check
(`cpu_lanczos_omp('info')`) that returns the number of OpenMP threads.
No numerical computation is performed during the build.

## Quick start

After a successful build:

```matlab
>> ftlm_observables('input_ico_s1_example.m')
```

The script prints a per-sector progress line, then saves the results to a
single `.mat` file (`ftlm_ico_s1.mat` for the example input). A typical
run of the example takes well under a minute on a contemporary GPU.

A second example, `input_ico_s1o2_ed_example.m`, demonstrates the
`ed_thresh` option: for the s = 1/2 icosahedron all sectors are routed
through exact diagonalization, yielding observables to machine
precision in a few seconds.

## Input file

All run parameters live in a plain MATLAB script. See
`input_ico_s1_example.m` for a commented template. The required and
optional input variables are:

**Required:**

| Variable | Type | Meaning |
|---|---|---|
| `geometry` | char | One of `'ico'`, `'cubo'`, `'cube'`, `'dodeca'`, `'icosid'`, `'ring'`. |
| `s_val` | scalar | Local spin: `0.5`, `1.0`, `1.5`, `2.0`, ... |
| `J` | scalar | Heisenberg exchange coupling, applied uniformly to every nearest-neighbor bond (any finite real; `J > 0` antiferromagnetic, `J < 0` ferromagnetic). |
| `R` | integer | FTLM random vectors per sector. |
| `M_lz` | integer | Lanczos iterations per random vector. |
| `T_range` | vector | Temperature grid (positive, units of `J/k_B`). |

`T_range` is a regular MATLAB expression, so both inline forms and loads
from external files work natively:

```matlab
T_range = logspace(-2, 1, 100);     % 100 points, log-spaced
T_range = linspace(0.01, 10, 100);  % 100 points, linearly spaced
T_range = load('T_grid.dat');       % one T value per line
```

For `geometry='ring'`, the additional required input `N_ring` (integer
>= 3) gives the number of sites.

**Optional (with defaults):**

| Variable | Default | Meaning |
|---|---|---|
| `only_M0` | `false` | If `true`, restrict to the M=0 sector (chi(T) will then be zero; useful for quick smoke tests). |
| `use_cpu_reference` | `false` | If `true`, additionally run the CPU FP64 reference and report `max |C_T_gpu - C_T_cpu|`. |
| `output_dir` | `'.'` | Directory for the output `.mat` file. |
| `B_cpu` | `8` | Block-Lanczos block size on the CPU path (integer in `[1, 32]`). |
| `B_gpu` | `0` | Block size on the GPU path. `0` = adaptive L2 heuristic; integer in `[1, 32]` overrides. |
| `L2_cache_bytes` | `48e6` | L2 cache size used by the `B_gpu=0` heuristic (48 MB matches RTX 4000 Ada). |
| `ed_thresh` | `0` | Sectors with `dim_sector <= ed_thresh` are handled by exact diagonalization (sparse `H`, then `eig(full(H))`) instead of FTLM, yielding the exact partition-function contribution at essentially zero cost. `0` disables the feature; sensible values are `100`–`2000` depending on memory budget (dense `H` uses `8 * dim^2` bytes). |

## Output

`ftlm_observables` writes one `.mat` file per run, named
`ftlm_<geom>_s<s_str>.mat` (for example `ftlm_ico_s1.mat`). It contains:

- **Observables (GPU FP32):** `T_range`, `C_T`, `chi_T`, `Z_eff`.
- **Optional CPU FP64 reference:** `C_T_cpu`, `chi_T_cpu`, `Z_eff_cpu`
  (empty if `use_cpu_reference` was `false`).
- **Configuration:** `geometry`, `s_val`, `J`, `R`, `M_lz`, `N`,
  `n_total_save`, `M_max`, `only_M0`, `use_cpu_reference`, `B_cpu`,
  `B_gpu`.
- **Per-sector diagnostics:** `sector_M`, `sector_dims`.
- **Timing:** `t_wall_gpu`, `t_wall_cpu`.

## Hardware-specific tuning

The default values reproduce the timings reported in the paper on an
NVIDIA RTX 4000 Ada (48 MB L2) and a typical AVX2/AVX-512 x86 CPU.

- **GPU block size.** The default `B_gpu=0` picks `B=8` if three FP32
  vectors of the largest sector fit into the L2 cache, otherwise `B=4`.
  On GPUs with a different L2 size, set `L2_cache_bytes` accordingly,
  or set `B_gpu` to a fixed value in `[1, 32]` to bypass the heuristic.
- **CPU block size.** `B_cpu=8` corresponds to exactly one x86 cache
  line of FP64 (64 bytes) and the natural AVX2/AVX-512 register width.
  Tuning is rarely necessary; values 4 or 16 may help on non-x86
  architectures or in unusually cache-pressured runs.
- **Threads.** The CPU kernel honours the `OMP_NUM_THREADS` environment
  variable (or whatever MATLAB exposes via `maxNumCompThreads`).

## Reproducing the paper figures

The `examples/` directory contains the additional drivers used in the
paper:

- `benchmark_ico_v1.m`  — runtime benchmark on the icosahedron
  (Paper Table 3, methods M1...M5). The header documents which
  configuration reproduces which Table 3 row.
- `benchmark_icosid_v1.m` — same benchmark on the icosidodecahedron
  (M = 0 sector, dim = 1.55e8; requires a GPU with roughly 16 GB or
  more of memory).
- `plot_paperfig1_v2.m` — generates Figure 1 (`C(T)` and `chi(T)`,
  GPU FP32 vs.\ CPU FP64) from a `.mat` file produced by
  `ftlm_observables`.

In the benchmarks, the single-vector methods (GPU-CLT-single,
GPU-CRank-single) run the corresponding block kernel with `B = 1`,
which is exactly the single-vector Lanczos recursion.

These scripts depend on the small helper `format_num.m`.

## Files in this release

| File | Purpose |
|---|---|
| `README.md` | This file. |
| `CHANGELOG.md` | Release history. |
| `LICENSE` | Full Apache License 2.0 text. |
| `NOTICE` | Copyright statement. |
| `CITATION.cff` | Machine-readable citation metadata. |
| `build_all.m` | Master build script (compiles all three MEX kernels). |
| `ftlm_observables.m` | Main entry point — see *Quick start*. |
| `input_ico_s1_example.m` | Commented input file template (FTLM default workflow). |
| `input_ico_s1o2_ed_example.m` | Second example: full exact diagonalization (s = 1/2 icosahedron). |
| `cuda_lanczos_clut_block.cu` | GPU block-Lanczos kernel (CLT, FP32). |
| `cuda_lanczos_crank_Sr_general.cu` | GPU block-Lanczos kernel (CRank, FP32). |
| `cpu_lanczos_omp.c` | CPU block-Lanczos kernel (FP64, OpenMP). |
| `examples/benchmark_ico_v1.m` | Paper Table 3 reproduction, icosahedron. |
| `examples/benchmark_icosid_v1.m` | Paper Table 3 reproduction, icosidodecahedron. |
| `examples/plot_paperfig1_v2.m` | Paper Figure 1 plotting script. |
| `examples/format_num.m` | Small formatting helper used by the benchmark scripts. |

## License

This project is licensed under the **Apache License, Version 2.0** — see the
[`LICENSE`](LICENSE) file for the full text and [`NOTICE`](NOTICE) for the
copyright statement.

## Authors

**Shadan Ghassemi Tabrizi** — Technische Universität Dresden /
Helmholtz-Zentrum Dresden-Rossendorf

Scientific co-author of the accompanying paper: **Thomas D. Kühne** (see
*Citation* above).
