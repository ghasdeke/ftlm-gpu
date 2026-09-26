# Changelog

## v2.0.0 (unreleased)

Feature release accompanying the revised paper. The FP32 GPU results of
v1 are reproduced bit for bit (CLT kernel, uniform model, `seed = 0`,
`ed_thresh = 0`).

### Added
- **General isotropic spin Hamiltonians**
  `H = sum_c J_c s_{i_c} . s_{j_c}` with an arbitrary list of pairwise
  couplings `couplings = [i, j, J_ij]` (any pairs, not restricted to
  nearest neighbors, any signs; up to 512 couplings). The predefined
  geometries with a uniform nearest-neighbor `J` remain available.
- **Mixed spins:** a vector `spins` of local spins s_i (integers or
  half-integers up to 15/2). Mixed-radix basis encoding for the CLT and
  site-dependent digit ranges in the combinatorial ranking; sectors with
  half-integer M are handled.
- **Selectable precision:** GPU `precision = 'double' | 'single' |
  'half' | 'bfloat16'` (the 16-bit formats are storage formats with FP32
  arithmetic; vectors are stored with norm sqrt(dim) to stay in the
  normal FP16 range) and CPU `precision`/`cpu_precision = 'double' |
  'single'`.
- **Python front end** (`python/ftlm_gpu`, `pip install .[cuda12]`): no
  MATLAB license needed. The CUDA kernels are compiled at run time with
  NVRTC through CuPy; the same device code (`cuda/ftlm_kernels.cuh`) is
  used by the MATLAB MEX gateway, and both front ends produce identical
  Lanczos coefficients for identical start vectors. Command line:
  `python -m ftlm_gpu input.toml`.
- `backend = 'cpu'` runs the whole calculation with the OpenMP kernel
  (no GPU required); `lookup = 'cr'` selects the combinatorial-ranking
  GPU kernel in `ftlm_observables`.
- `seed` option for statistically independent start-vector sets
  (multi-seed error analysis); `save_ritz` stores Ritz values, weights and
  Lanczos coefficients per sector.
- Test suites `tests/run_tests.m` and `python/tests` (SpMV of every
  kernel variant vs. an explicit sparse Hamiltonian with random
  couplings and mixed spins, CLT/CR order, exact Gauss quadrature for
  full Krylov spaces, weight sum rule, Lanczos breakdown, FTLM vs. ED).
- `examples/benchmark_table3.m`: CPU/GPU x FP64/FP32/FP16/BF16 x CLT/CR
  timings.

### Changed
- One GPU MEX file `ftlm_gpu_mex` (replaces `cuda_lanczos_clut_block`
  and `cuda_lanczos_crank_Sr_general`) and one CPU MEX file
  `ftlm_cpu_mex` (replaces `cpu_lanczos_omp`); MATLAB helper functions
  in the package `+ftlm`. `build_all` builds both.
- **Lanczos termination per chain:** a chain stops when its Krylov
  space is exhausted (`beta_j <= c u ||T_j||_1`, c = 64 for FP64/FP32,
  8 for FP16/BF16), independently of the other chains of the block
  (v1: GPU stopped only when all chains had `beta < 1e-6`, the CPU
  stopped all chains as soon as one had `beta < 1e-14`). N_L and R are
  capped at the sector dimension as before.
- Default `ed_thresh = 1000` (v1: 0): small sectors are diagonalized
  exactly.
- The combinatorial ranking uses the digit convention a = m + s of the
  CLT, so both lookup strategies act on identically ordered vectors.
- Basis enumeration by a site-by-site construction, O(N dim) instead of
  a scan over the full label space.
- CPU kernel: pointer swap instead of vector copies, deterministic
  reductions (thread partials summed in thread order).
- 64-bit vector indexing in all kernels (v1 used 32-bit indices, which
  overflow for dim x B > 2^31; not reached with the v1 default block
  sizes); CUDA errors, including allocation failures, are
  reported; the GPU block size is reduced automatically if the sector
  does not fit into free device memory.

### Removed
- `cuda_lanczos_clut_block.cu`, `cuda_lanczos_crank_Sr_general.cu`,
  `cpu_lanczos_omp.c`, `examples/benchmark_ico_v1.m`,
  `examples/benchmark_icosid_v1.m` (superseded; available in v1.1.1).

## v1.1.1 (2026-07-11)

Documentation/usability release. No changes to the physics, the
algorithms, or the numerical results.

### Fixed
- Benchmark scripts are now location-independent: they add the
  repository root (MEX binaries) and `examples/` (helpers) to the
  MATLAB path themselves. Previously, starting `benchmark_ico_v1` /
  `benchmark_icosid_v1` from inside `examples/` failed with
  "Required MEX file missing: cpu_lanczos_omp".

### Added
- README: explicit invocation line for the benchmarks and a note that
  the paper's precision analysis (Figs. 1-3) uses `R = 100` random
  vectors per sector, whereas the quick-start example uses `R = 50`
  for speed (same note in `input_ico_s1_example.m`).
- This release is tagged so that the archived version includes the
  arXiv reference in README/CITATION.cff (the v1.1.0 tag predates
  that commit).

## v1.1.0 (2026-07-03)

Code-quality release. No changes to the physics, the algorithms, or the
numerical results: the GPU FP32 results are identical to v1.0.0, the CPU
FP64 reference agrees to machine precision.

### Fixed
- **MATLAB crashed with an access violation on exit (and on `clear
  mex`) after any run that used the CPU FP64 reference kernel.** Cause:
  unloading a MEX DLL after MSVC OpenMP worker threads have been
  started is unsafe. `cpu_lanczos_omp` now stays locked in memory for
  the rest of the session ('cleanup' still frees all buffers); to
  rebuild it after use, restart MATLAB first.
- **Benchmark scripts referenced a kernel that is not part of the
  release** (`cuda_lanczos_clut`, single-vector CLT). The single-vector
  method M2 (GPU-CLT-single) now runs `cuda_lanczos_clut_block` with
  block size `B = 1`, which is exactly the single-vector Lanczos
  recursion — the same convention already used for M4 (GPU-CRank-single).
  Both benchmark scripts now run out of the box after `build_all`.
- `benchmark_ico_v1.m`: with `n_runs = 1` the aggregation discarded the
  only run and reported `NaN` medians; a single run is now kept.
- Benchmarks: an off-by-one `nargin` check caused the configured
  `L2_cache_bytes` to be silently replaced by the 48 MB default inside
  `benchmark_cuda_clut_block_seq` (no effect on the RTX 4000 Ada used in
  the paper, wrong on GPUs with a different L2 size).
- `cpu_lanczos_omp.c`: loop variables used inside OpenMP parallel
  regions of `block_dot_omp` / `block_nrm2_omp` were shared between
  threads (formally a data race; benign under the tested optimizing
  compilers, now correct by construction).
- `ftlm_observables.m`: new validity checks `N <= 32` and
  `(2*s_val+1)^N <= 2^31`; out-of-range systems (e.g. rings with more
  than 31 spin-1/2 sites) previously risked silent int32 overflow.
- MEX kernels: `mexLock`/`mexUnlock` are now balanced under repeated
  `init`/`cleanup` calls, and an unknown mode string raises an error in
  all three kernels.

### Changed
- **cuBLAS dependency removed.** `cuda_lanczos_clut_block.cu` created a
  cuBLAS handle but never called cuBLAS (all reductions use custom fused
  kernels); the kernel now builds without `-lcublas`.
- `cpu_lanczos_omp.c` reduced to the code paths actually used by the
  release (`init_clut`, `block_lanczos_clut`, `cleanup`, `info`,
  `set_threads`); the legacy full-lookup-table modes (`init`, `lanczos`,
  `block_lanczos`) and their helpers were removed (~500 lines).
- Removed dead code from the GPU kernels (unused `warp_reduce_sum`,
  unused debug transpose kernel, duplicate cleanup function) and unused
  helper functions from the benchmark scripts.
- Quieter output: the CPU kernel and the CRank kernel no longer print
  per-init/per-call diagnostics (`cpu_lanczos_omp('info')` still reports
  the thread count).
- `benchmark_ico_v1.m` defaults now reproduce the s = 1 icosahedron row
  of paper Table 3 (`s_val = 1.0`, full FTLM, `n_runs = 3`); the header
  documents the configuration for every Table 3 row.
- Documentation: README size limits and single-vector (`B = 1`)
  convention documented; stale references to internal development files
  removed; CITATION.cff carries the software DOI and the paper title.

## v1.0.0

Initial release accompanying the paper (archived on Zenodo,
DOI 10.5281/zenodo.20378647).
