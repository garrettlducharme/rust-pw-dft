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
- cached MPSGraph complex-FP32 FFT bridge behind the `metal` feature;
- complete CPU-FP64 silicon SCF with atomic-density initialization, local and
  separable nonlocal pseudopotentials, Hartree and LDA-PZ potentials, Ewald
  energy, linear density mixing, and dual convergence checks;
- a dense `faer` eigensolve used as the independently validated small-system
  reference boundary (block Davidson remains the next scaling milestone);
- `run`, `compare`, and `bench` command surfaces that refuse scientific or
  performance claims for incomplete/non-converged results.

## Accepted CPU result

For the pinned two-atom silicon LDA case, the CPU solver converges to
`-14.5294683423 Ry`. Against the clean QE 7.6 XML reference it differs by
`0.000083 meV/atom`; the maximum occupied-eigenvalue error is
`0.000253 meV`, and the integrated charge error is `4.3e-14` electrons.
The result is in `results/cpu-scf.json` and the normalized QE reference is in
`results/qe-reference.json`.

## Current hard stop

The Metal backend has passed only its native MPSGraph complex-FP32 FFT gate.
The GPU-resident local/nonlocal Hamiltonian, mixed-precision reductions, and
complete SCF orchestration are not integrated. A Metal `run` therefore writes
`converged: false` and exits nonzero; it never silently falls back to CPU.

## Commands

```sh
cargo test
cargo test --features metal metal_fft_roundtrip_reports_error -- --nocapture
cargo run -- run examples/si2.toml --backend cpu --output result.json
cargo run --release -- compare results/cpu-scf.json results/qe-reference.json
cargo run --features metal -- run examples/si2.toml --backend metal --output result-metal.json
```

The CPU scientific milestone is accepted. The next bounded milestone is to
replace the dense reference eigensolve with block Davidson and extend the
Metal bridge to a resident Hamiltonian batch, while preserving these gates.
