# rust-pw-dft

An educational, fail-closed plane-wave DFT prototype aimed at Apple Silicon.
It is dual-licensed under MIT or Apache-2.0 and contains no code copied from
GPL DFT implementations.

## Implemented

- strict fixed-cell, spinless, Gamma-only LDA-PZ silicon input contract;
- direct/reciprocal lattice construction and cutoff-safe G-vector enumeration;
- FP64 multidimensional CPU FFT with Parseval and round-trip tests;
- LDA-PZ exchange-correlation energy and potential;
- strict parser for QE's legacy `Si.pz-vbc.UPF` and scalar/non-relativistic
  norm-conserving UPF v2 PZ files, including local potential, radial mesh,
  separable projectors, and `Dij`;
- explicit provenance and precision policy in JSON results;
- a long-lived MPSGraph complex-FP32 context with persistent input, potential,
  kinetic, and projector buffers behind the macOS-only `metal` feature;
- compact separable `B D B-dagger` nonlocal application, avoiding the former
  plane-wave-squared dense validation matrix in production SCF;
- on-device real/imaginary gather of FFT output into the compact plane-wave
  basis, so a Hamiltonian application returns only compact `H|psi>` blocks;
- rectangular, chunked complex-FP32 cross-Gram partials on Metal with final
  accumulation in CPU complex FP64, used for every Davidson Rayleigh-Ritz
  projection;
- a persistent compact-FP32 Davidson journal: Hamiltonian inputs and outputs
  append to caller-owned MTLBuffers, normal projections require no host-vector
  upload, and subspace restarts resynchronize explicitly;
- complete CPU-FP64 silicon SCF with atomic-density initialization, local and
  separable nonlocal pseudopotentials, Hartree and LDA-PZ potentials, Ewald
  energy, linear density mixing, and dual convergence checks;
- matrix-free block Davidson with kinetic preconditioning, restarts, prior-SCF
  orbital reuse, batched Hamiltonian application, and CPU-FP64 `faer`
  Rayleigh-Ritz solves;
- a complete mixed-precision Metal validation SCF: Hamiltonian applications
  run in complex FP32 on the Apple GPU while reductions, projected solves,
  density construction, convergence, and energy assembly remain CPU FP64;
- `run`, `compare`, and `bench` command surfaces that refuse scientific or
  performance claims for incomplete/non-converged results.

## Accepted CPU result

For the pinned two-atom silicon LDA case, the CPU solver converges to
`-14.5294683423 Ry`. Against the clean QE 7.6 XML reference it differs by
`0.000083 meV/atom`; the maximum occupied-eigenvalue error is
`0.000253 meV`, and the integrated charge error is `4.3e-14` electrons.
The result is in `results/cpu-scf.json` and the normalized QE reference is in
`results/qe-reference.json`.

## Metal result and boundary

The two-atom Metal SCF converges to `-14.5294698064 Ry`. Relative to the
accepted CPU result, its errors are `0.00996 meV/atom`, `0.00252 meV` for the
worst occupied eigenvalue, and `5.8e-14` electrons. The measured Metal FFT
round-trip error is `1.75e-7`; the complete fixed-density silicon Hamiltonian
through the persistent compact-projector context differs from CPU by
`1.34e-7` in relative L2 norm. With GPU cross-Gram reductions enabled, the
fixed-density occupied-eigenvalue error is `0.00342 meV`.

On the pinned 16-atom silicon supercell, five warmed runs give median wall
times of `22.249 s` CPU and `3.178 s` Metal, an observed `7.000x` speedup.
Persistent Metal buffers occupy 71.0 MB and measured peak device allocation is
122.7 MB on this workload, well below the 12 GiB project cap. The run executes
165 resident GPU cross-Gram calls covering 2,404,854 vector pairs; only one
subspace restart requires a host resynchronization.

Metal remains marked ineligible for a production performance claim because
the canonical Davidson basis, orthogonalization, residual construction, and
correction vectors still live in host FP64. The compact FP32 projection journal
is resident, but correction blocks are still produced and staged by the CPU.
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
cargo run --release --features metal -- bench examples/si16.toml --backend metal --repetitions 5 --output benchmark-metal.json
```

The CPU scientific milestone, persistent Metal operator accuracy milestone,
integrated Rayleigh-Ritz reduction milestone, and larger-workload numeric speed
threshold are accepted. The next bounded milestone is a GPU-resident Davidson
basis with GPU orthogonalization, residual/norm evaluation, and correction
updates; only that path may turn the observed speedup into a production
performance acceptance claim.
