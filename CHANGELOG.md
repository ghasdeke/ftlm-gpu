# Changelog

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
