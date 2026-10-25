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
- `run`, `compare`, and `bench` command surfaces that refuse scientific or
  performance claims for incomplete/non-converged results.

## Current hard stop

The Hamiltonian application, reciprocal nonlocal projectors, Davidson loop,
Hartree/Ewald terms, density mixing, and full SCF energy accounting have not
yet been integrated.  `run` performs all validated setup, writes a result with
`converged: false`, and returns an error.  This is intentional: there is no
placeholder energy and no silent CPU fallback.

## Commands

```sh
cargo test
cargo test --features metal metal_fft_roundtrip_reports_error -- --nocapture
cargo run -- run examples/si2.toml --backend cpu --output result.json
cargo run --features metal -- run examples/si2.toml --backend metal --output result-metal.json
```

The first scientific milestone is complete only when the CPU result matches a
clean QE 7.6 reference within the acceptance gates in
`../dft studies/studies/apple-gpu-dft/README.md`.
