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
- chunked complex-FP32 overlap partials on Metal with final accumulation in
  CPU complex FP64 (validated as a primitive, not yet wired into Davidson);
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

The two-atom Metal SCF converges to `-14.5294701731 Ry`. Relative to the
accepted CPU result, its errors are `0.01245 meV/atom`, `0.00106 meV` for the
worst occupied eigenvalue, and `3.4e-14` electrons. The measured Metal FFT
round-trip error is `1.75e-7`; the complete fixed-density silicon Hamiltonian
through the persistent compact-projector context differs from CPU by
`1.34e-7` in relative L2 norm.

On the pinned 16-atom silicon supercell, five warmed runs give median wall
times of `22.361 s` CPU and `5.124 s` Metal, an observed `4.364x` speedup.
Persistent Metal buffers occupy 9.82 MB and measured peak device allocation is
71.0 MB on this workload, well below the 12 GiB project cap.

Metal remains marked ineligible for a production performance claim because
the Davidson basis and correction vectors still live in host FP64 and are
staged into the persistent buffers by block. There is no silent CPU
Hamiltonian fallback; the actual device, effective mixed-precision convergence
tolerances, memory measurements, and precision policy are recorded in every
result.

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
and larger-workload numeric speed threshold are accepted. The next bounded
milestone is a GPU-resident Davidson basis with blockwise GPU overlap/norm
partial reductions and CPU-FP64 final accumulation; only that path may turn
the observed speedup into a production performance acceptance claim.
