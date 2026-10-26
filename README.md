# rust-pw-dft

An educational, fail-closed plane-wave DFT prototype aimed at Apple Silicon.
It is dual-licensed under MIT or Apache-2.0 and contains no code copied from
GPL DFT implementations.

The path from the validated silicon prototype to a research-ready Mac package
is tracked in [`ROADMAP.md`](ROADMAP.md). Claims are tied to committed,
machine-readable evidence rather than feature presence alone.

## Implemented

- strict fixed-cell, spinless, Gamma-only LDA-PZ silicon input contract;
- direct/reciprocal lattice construction and cutoff-safe G-vector enumeration;
- FP64 multidimensional CPU FFT with Parseval and round-trip tests;
- LDA-PZ exchange-correlation energy and potential;
- strict parser for QE's legacy `Si.pz-vbc.UPF` and scalar/non-relativistic
  norm-conserving UPF v2 PZ files, including local potential, radial mesh,
  separable projectors, and `Dij`;
- explicit provenance and precision policy in JSON results;
- required version identifiers for input, result, comparison, benchmark, and
  checkpoint contracts;
- checksummed binary SCF checkpoints with atomic replacement, strict
  compatibility validation, cumulative timing/counter restoration, compact
  inspection, and deterministic CPU and Metal interruption/resume tests;
- a long-lived MPSGraph complex-FP32 context with persistent input, potential,
  kinetic, and projector buffers behind the macOS-only `metal` feature;
- compact separable `B D B-dagger` nonlocal application, avoiding the former
  plane-wave-squared dense validation matrix in production SCF;
- on-device real/imaginary gather of FFT output into the compact plane-wave
  basis, so a Hamiltonian application returns only compact `H|psi>` blocks;
- rectangular, chunked complex-FP32 cross-Gram partials on Metal with final
  accumulation in CPU complex FP64, used for every Davidson Rayleigh-Ritz
  projection;
- direct complex-FP32 residual-norm partials on Metal with CPU-FP64 final
  accumulation and fail-closed backend result validation;
- resident complex-FP32 Ritz-vector, applied-Ritz, residual, and guarded
  kinetic-preconditioner graphs; the fixed-density preconditioner agrees with
  the CPU formula to `3.07e-8` relative L2 error and masks converged roots
  exactly;
- two-pass resident correction orthogonalization with CPU-FP64 accumulation of
  compact overlap and norm partials, device-local normalization/copies, and
  FP32-aware rank rejection; fixed-density corrections agree with CPU to
  `7.08e-7` relative L2 and the combined-basis orthonormality error is
  `5.35e-8`;
- a custom Metal compact-to-FFT-grid scatter feeding the persistent
  Hamiltonian directly from accepted resident corrections, followed by
  device-local basis/H-basis journal append; the appended Hamiltonian images
  agree with CPU FP64 to `2.22e-7` relative L2 with zero full-vector transfer;
- device-local Davidson subspace restart that replaces the journal with
  resident Ritz/applied-Ritz blocks, preserves raw corrections for
  reorthogonalization, and shows zero observed vector error and zero
  full-vector transfer in the fixed-density gate;
- a persistent compact-FP32 Davidson journal: Hamiltonian inputs and outputs
  append to caller-owned MTLBuffers, normal projections require no host-vector
  upload, and subspace restarts replace the journal device-locally;
- complete CPU-FP64 silicon SCF with atomic-density initialization, local and
  separable nonlocal pseudopotentials, Hartree and LDA-PZ potentials, Ewald
  energy, linear density mixing, and dual convergence checks;
- matrix-free block Davidson with kinetic preconditioning, restarts, prior-SCF
  orbital reuse, batched Hamiltonian application, and CPU-FP64 `faer`
  Rayleigh-Ritz solves;
- a complete mixed-precision resident Metal SCF: wavefunctions, Hamiltonian
  images, Ritz vectors, residuals, preconditioned corrections, orthogonalized
  corrections, and subspace restarts remain in complex FP32 GPU buffers during
  each Davidson solve; only compact projected data and reduction partials cross
  per iteration, while projected solves, density construction, convergence,
  and energy assembly remain CPU FP64;
- `run`, `compare`, and `bench` command surfaces that refuse scientific or
  performance claims for incomplete/non-converged results, plus
  `bench-compare`, which applies one recorded convergence policy to CPU and
  Metal and accepts a speed claim only when provenance, accuracy, and the
  end-to-end threshold all pass together.

## Accepted CPU result

For the pinned two-atom silicon LDA case, the CPU solver converges to
`-14.5294683423 Ry`. Against the clean QE 7.6 XML reference it differs by
`0.000083 meV/atom`; the maximum occupied-eigenvalue error is
`0.000253 meV`, and the integrated charge error is `4.3e-14` electrons.
The result is in `results/cpu-scf.json` and the normalized QE reference is in
`results/qe-reference.json`.

## Metal result and boundary

The latest two-atom resident Metal SCF converges to `-14.5294700206 Ry`.
Relative to the accepted CPU result, its errors are `0.01142 meV/atom`,
`0.00165 meV` for the worst occupied eigenvalue, and `5.1e-14` electrons. Its
117 Davidson iterations perform zero full-vector transfers between the one
initial-basis upload and one final-Ritz download per SCF eigensolve. The
measured Metal FFT
round-trip error is `1.75e-7`; the complete fixed-density silicon Hamiltonian
through the persistent compact-projector context differs from CPU by
`8.57e-8` in relative L2 norm. The fixed-density occupied-eigenvalue error is
`0.00342 meV`.

On the pinned 16-atom workload, `bench-compare` applied the same
`matched-metal-compatible-v1` convergence policy to both backends. Five warmed
runs measured CPU min/median/max times of `8.050/8.072/8.097 s` and resident
Metal times of `6.220/6.252/6.256 s`, a `1.291x` median speedup. The paired run
also passed scientific comparison at `0.00976 meV/atom`, `0.00853 meV` maximum
occupied-eigenvalue error, and `8.1e-13` electrons. This accepts the performance
claim only for this pinned workload and Apple M5 Pro; it is not a general DFT
performance claim. The complete paired report is
[`results/si16-matched-benchmark.json`](results/si16-matched-benchmark.json).

There is no silent CPU Hamiltonian or reduction fallback; the actual device,
effective mixed-precision convergence tolerances, resident/resynchronization
counts, memory measurements, and precision policy are recorded in every result.

## Commands

First download and verify the external QE 7.6 fixture as described in
[`pseudopotentials/README.md`](pseudopotentials/README.md). Relative
pseudopotential paths are resolved from the input file, so the examples work
from any current directory once that file is present.

```sh
cargo test
cargo test --features metal
cargo run -- run examples/si2.toml --backend cpu --output result.json
cargo run --release -- compare results/cpu-scf.json results/qe-reference.json
cargo run --release --features metal -- run examples/si2.toml --backend metal --output result-metal.json
cargo run --release --features metal -- bench-compare examples/si16.toml --repetitions 5 --output benchmark-comparison.json
cargo run --release --features metal -- run examples/si16.toml --backend metal --output paused.json --checkpoint si16.chk --stop-after 10
cargo run --release -- inspect-checkpoint si16.chk
cargo run --release --features metal -- resume examples/si16.toml si16.chk --backend metal --output resumed.json
```

Checkpoint format, atomicity, and compatibility rules are documented in
[`docs/checkpoints.md`](docs/checkpoints.md). A checkpoint is intentionally
bound to its input, pseudopotentials, backend, precision policy, run profile,
source revision, FFT grid, and basis dimensions; incompatible state is rejected
rather than approximately resumed.

The CPU scientific milestone and resident Metal Si2/Si16 accuracy, transfer,
and matched-performance milestones are accepted. The stateful driver is
integrated into production SCF, and its normal-iteration
zero-full-vector-transfer contract is checked at runtime and in portable and
Apple-GPU tests. Phase 2 now focuses on versioned schemas, checkpoints,
reproducible fixture manifests, and stronger validation infrastructure before
the physics scope expands. The resident buffer contract, dataflow, transfer
accounting, and fail-closed gates are specified in
[`docs/resident-davidson.md`](docs/resident-davidson.md).
