# rust-pw-dft

An educational, fail-closed plane-wave DFT prototype aimed at Apple Silicon.
It is dual-licensed under MIT or Apache-2.0 and contains no code copied from
GPL DFT implementations.

Version 0.7 adds an independently validated CPU-FP64 metallic aluminum SCF,
including global Fermi-Dirac occupations and free-energy terms. Version 0.6
added Monkhorst-Pack and explicit k-point sampling, time-reversal reduction,
and an independently validated 2x2x2 silicon PBE SCF. Version 0.5 added true
multi-species norm-conserving operators and validated CPU and Metal PBE MgO
SCFs. Earlier releases added native variational PBE, complete SCF convergence
histories, a CI-enforced scientific-fixture provenance manifest, and atomic
checksummed CPU/Metal restart checkpoints.

The path from the validated silicon prototype to a research-ready Mac package
is tracked in [`ROADMAP.md`](ROADMAP.md). Claims are tied to committed,
machine-readable evidence rather than feature presence alone.

## Implemented

- strict fixed-cell, spinless LDA-PZ or PBE multi-species input contract;
- versioned Gamma, Monkhorst-Pack, and explicit weighted k-point inputs with
  deterministic reciprocal-cell canonicalization and optional time-reversal
  reduction; CPU FP64 supports complete multi-k-point SCFs, while Metal rejects
  these inputs until its resident batched path is validated;
- fixed closed-shell and Fermi-Dirac occupations with a globally solved chemical
  potential, charge-conserving Brillouin-zone weights, explicit entropy/free-
  energy terms, and configurable extra bands;
- direct/reciprocal lattice construction and cutoff-safe G-vector enumeration;
- FP64 multidimensional CPU FFT with Parseval and round-trip tests;
- LDA-PZ exchange-correlation energy and potential;
- a typed, fail-closed grid-level exchange-correlation boundary with analytic
  potential-versus-energy-derivative tests, ready for semilocal gradient terms;
- a native unpolarized PBE functional whose point energy and derivatives match
  a pinned Libxc 7.1.2 oracle and whose spectral grid potential passes a
  whole-functional finite-difference test; a pinned ONCV UPF and independent
  QE 7.6 full-SCF comparison now accept the Si2 PBE workload;
- strict parser for QE's legacy `Si.pz-vbc.UPF` and scalar/non-relativistic
  norm-conserving UPF v2 PZ/PBE files, including local potential, radial mesh,
  separable projectors, and `Dij`;
- species-resolved local potentials, nonlocal projector couplings, atomic
  starting densities, valence counts, and mixed-charge Ewald sums;
- QE-compatible numerical UPF radial transforms that discard cancellation-noisy
  tails beyond 10 bohr and use an odd Simpson mesh;
- explicit provenance and precision policy in JSON results;
- required version identifiers for input, result, comparison, benchmark, and
  checkpoint contracts;
- checksummed binary SCF checkpoints with atomic replacement, strict
  compatibility validation, cumulative timing/counter restoration, compact
  inspection, and deterministic CPU and Metal interruption/resume tests;
- complete per-iteration SCF histories with explicit continue/converged/
  eigensolver-failure states, energy and density changes, charge, orbital
  residuals, and per-step Davidson/Hamiltonian work; histories survive resume;
- a versioned fixture manifest covering inputs, upstream provenance, licenses,
  redistribution policy, reference commands, and hashes, enforced locally and
  in CI by `verify-fixtures`;
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
runs measured CPU min/median/max times of `8.218/8.252/8.255 s` and resident
Metal times of `6.088/6.106/6.256 s`, a `1.351x` median speedup. The paired run
also passed scientific comparison at `0.00976 meV/atom`, `0.00853 meV` maximum
occupied-eigenvalue error, and `8.1e-13` electrons. This accepts the performance
claim only for this pinned workload and Apple M5 Pro; it is not a general DFT
performance claim. The complete paired report is
[`results/si16-matched-benchmark.json`](results/si16-matched-benchmark.json).

There is no silent CPU Hamiltonian or reduction fallback; the actual device,
effective mixed-precision convergence tolerances, resident/resynchronization
counts, memory measurements, and precision policy are recorded in every result.

## Accepted PBE result

For the pinned two-atom silicon PBE/ONCV case, the CPU solver converges to
`-14.4679818597 Ry`. Against full-precision QE 7.6 XML it differs by
`0.01001 meV/atom`; the maximum occupied-eigenvalue error is `0.003352 meV`,
and the integrated charge error is `1.1e-14` electrons. The resident Metal SCF
differs from CPU by `0.00823 meV/atom`, `0.000668 meV` for occupied
eigenvalues, and `8.9e-16` electrons, with zero projection resynchronizations.
These results admit PBE only for this exact silicon workload; they do not
establish chemical transferability. The exact-source records are
[`results/si2-pbe-cpu-scf.json`](results/si2-pbe-cpu-scf.json),
[`results/si2-pbe-metal-scf.json`](results/si2-pbe-metal-scf.json), and their
committed CPU/QE and Metal/CPU comparison reports.

## Accepted multi-species MgO result

The pinned two-atom rock-salt MgO workload uses 100 Ry PBE with ten-electron
Mg and six-electron O ONCV pseudopotentials. CPU FP64 differs from the matched
QE 7.6 result by `0.0426 meV/atom`, `0.0103 meV` for the worst occupied
eigenvalue, and `4.2e-13` electrons. The resident Metal result differs from CPU
by `0.0203 meV/atom`, `0.00734 meV` for occupied eigenvalues, and `3.0e-13`
electrons, with zero projection resynchronizations.

This closes one binary-insulator transfer test. It does not establish broad
oxide chemistry, forces, stress, or pseudopotential transferability. The
matched five-warm-run performance gate is intentionally
retained as a negative result: CPU min/median/max were
`2.821/2.843/2.865 s`, Metal was `3.592/3.619/3.699 s`, and the `0.786x`
speedup fails the required `1.20x`. MgO therefore extends scientific scope but
does not add a performance claim. Exact-source evidence is in
[`results/mgo2-pbe-cpu-scf.json`](results/mgo2-pbe-cpu-scf.json),
[`results/mgo2-pbe-metal-scf.json`](results/mgo2-pbe-metal-scf.json), and
[`results/mgo2-pbe-matched-benchmark.json`](results/mgo2-pbe-matched-benchmark.json).

## Accepted CPU k-point result

The shifted 2x2x2 silicon PBE fixture reduces eight points to four exact
time-reversal representatives. The CPU solver converges in 50 SCF iterations
to `-15.7186708554 Ry`. Against full-precision QE 7.6 XML it differs by
`0.00226 meV/atom`; the maximum occupied per-k-point eigenvalue error is
`0.06568 meV`, the occupation error is zero, and the charge error is
`1.9e-14` electrons. The release-mode CI test compares coordinates, weights,
occupied eigenvalues, occupations, charge, and total energy.

This accepts only the CPU-FP64 silicon fixture with fixed occupations.
Multi-k-point Metal execution and checkpoint/restart both fail closed. Exact-
source records are
[`results/si2-pbe-2x2x2-cpu-scf.json`](results/si2-pbe-2x2x2-cpu-scf.json),
[`results/qe-si2-pbe-2x2x2-reference.json`](results/qe-si2-pbe-2x2x2-reference.json),
and
[`results/si2-pbe-2x2x2-cpu-vs-qe.json`](results/si2-pbe-2x2x2-cpu-vs-qe.json).
The input schema and conventions are documented in
[`docs/kpoints.md`](docs/kpoints.md).

## Accepted metallic aluminum result

The one-atom fcc aluminum fixture uses a shifted 4x4x4 mesh, 0.02 Ry
Fermi-Dirac smearing, six bands, and the official legacy QE `Al.pz-vbc.UPF`
LDA-PZ norm-conserving pseudopotential. The full 64-point mesh reduces to 32
time-reversal representatives. CPU FP64 converges in 74 iterations to a Mermin
free energy of `-4.1855171980 Ry`, with chemical potential `0.6135522910 Ry`
and dimensionless entropy `0.4209874149`.

Against clean QE 7.6 XML, the free-energy error is `0.00210 meV/atom`, the
maximum occupied eigenvalue error is `0.00174 meV`, the maximum absolute
occupation error is `7.13e-7`, and the charge error is `4.4e-16` electrons.
The release-mode CI gate also requires a nonzero entropy, a negative `-TS`
correction, a reported chemical potential, and at least one fractional
occupation. Exact-source evidence is
[`results/al1-lda-4x4x4-cpu-scf.json`](results/al1-lda-4x4x4-cpu-scf.json),
[`results/qe-al1-lda-4x4x4-reference.json`](results/qe-al1-lda-4x4x4-reference.json),
and
[`results/al1-lda-4x4x4-cpu-vs-qe.json`](results/al1-lda-4x4x4-cpu-vs-qe.json).

The pseudopotential comes from QE's original legacy table, which QE itself
keeps for reference rather than recommends for new research. This result
validates the solver's metallic numerical path for the exact fixture; it is not
a recommendation of that pseudopotential or a converged materials prediction.

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
cargo run --release -- run examples/si2-pbe.toml --backend cpu --output result-pbe.json
cargo run --release -- compare result-pbe.json results/qe-pbe-reference.json
cargo run --release -- run examples/si2-pbe-2x2x2.toml --backend cpu --output result-kmesh.json
cargo run --release -- compare result-kmesh.json results/qe-si2-pbe-2x2x2-reference.json
cargo run --release -- run examples/al1-lda-4x4x4.toml --backend cpu --output result-al.json
cargo run --release -- compare result-al.json results/qe-al1-lda-4x4x4-reference.json
cargo run --release -- run examples/mgo2-pbe.toml --backend cpu --output result-mgo.json
cargo run --release -- compare result-mgo.json results/qe-mgo2-pbe-reference.json
cargo run --release --features metal -- run examples/mgo2-pbe.toml --backend metal --output result-mgo-metal.json
cargo run --release --features metal -- run examples/si2.toml --backend metal --output result-metal.json
cargo run --release --features metal -- bench-compare examples/si16.toml --repetitions 5 --output benchmark-comparison.json
cargo run --release -- verify-fixtures
cargo run --release --features metal -- run examples/si16.toml --backend metal --output paused.json --checkpoint si16.chk --stop-after 10
cargo run --release -- inspect-checkpoint si16.chk
cargo run --release --features metal -- resume examples/si16.toml si16.chk --backend metal --output resumed.json
```

Checkpoint format, atomicity, and compatibility rules are documented in
[`docs/checkpoints.md`](docs/checkpoints.md). A checkpoint is intentionally
bound to its input, pseudopotentials, backend, precision policy, run profile,
source revision, FFT grid, and basis dimensions; incompatible state is rejected
rather than approximately resumed.
Fixture provenance and reproduction commands are documented in
[`docs/fixtures.md`](docs/fixtures.md); [`fixtures/manifest.toml`](fixtures/manifest.toml)
is the machine-readable authority.

The LDA and PBE Si2, PBE MgO, PBE Si2 2x2x2, and metallic Al CPU scientific
milestones and the Gamma-point resident Metal accuracy milestones are accepted.
The LDA Si16 transfer and matched-performance milestones are also accepted. The stateful driver is
integrated into production SCF, and its normal-iteration
zero-full-vector-transfer contract is checked at runtime and in portable and
Apple-GPU tests. The next physics frontier is resident multi-k-point Metal
execution, followed by denser-mesh convergence studies and analytic forces,
with the same CPU-reference-first acceptance policy. The
resident buffer contract, dataflow, transfer
accounting, and fail-closed gates are specified in
[`docs/resident-davidson.md`](docs/resident-davidson.md).
