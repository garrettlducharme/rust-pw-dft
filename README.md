# rust-pw-dft

An educational, fail-closed plane-wave DFT prototype aimed at Apple Silicon.
It is dual-licensed under MIT or Apache-2.0 and contains no code copied from
GPL DFT implementations.

## Implemented

- strict fixed-cell, spinless, Gamma-only LDA-PZ silicon input contract;
- direct/reciprocal lattice construction and cutoff-safe G-vector enumeration;
- FP64 multidimensional CPU FFT with Parseval and round-trip tests;
- LDA-PZ exchange-correlation energy and potential;
- strict parser for QE's legacy norm-conserving `Si.pz-vbc.UPF`, including
  local potential, radial mesh, separable projectors, and `Dij`;
- explicit provenance and precision policy in JSON results;
- cached MPSGraph complex-FP32 FFT, local/kinetic, and dense nonlocal operator
  graphs behind the macOS-only `metal` feature;
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

The two-atom Metal SCF converges to `-14.5294700166 Ry`. Relative to the
accepted CPU result, its errors are `0.01139 meV/atom`, `0.00157 meV` for the
worst occupied eigenvalue, and `5.8e-14` electrons. The measured Metal FFT
round-trip error is `1.75e-7`; the complete fixed-density silicon Hamiltonian
differs from CPU by `1.03e-7` in relative L2 norm.

On the pinned 16-atom silicon supercell, five warmed runs give median wall
times of `27.274 s` CPU and `11.565 s` Metal, an observed `2.358x` speedup.
Metal remains marked ineligible for a production performance claim because
wavefunctions are repacked from host FP64 into shared Metal buffers for each
batched Davidson application and the nonlocal operator is currently a dense
validation matrix. There is no silent CPU Hamiltonian fallback; the actual
device and transitional precision policy are recorded in every result.

## Commands

```sh
cargo test
cargo test --features metal
cargo run -- run examples/si2.toml --backend cpu --output result.json
cargo run --release -- compare results/cpu-scf.json results/qe-reference.json
cargo run --release --features metal -- run examples/si2.toml --backend metal --output result-metal.json
cargo run --release --features metal -- bench examples/si16.toml --backend metal --repetitions 5 --output benchmark-metal.json
```

The CPU scientific milestone and transitional Metal accuracy milestone are
accepted. The observed larger-workload speed threshold is also met. The next
bounded milestone is a genuinely resident Metal backend with compact
projector kernels and blockwise GPU reductions; only that path may turn the
observed speedup into a production performance acceptance claim.
