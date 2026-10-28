# Roadmap

`rust-pw-dft` is intended to become a dependable plane-wave DFT package for
Apple Silicon, not a benchmark that happens to run one silicon input. This
roadmap deliberately separates numerical architecture, scientific scope, and
product readiness. A later phase does not begin merely because an earlier one
has code; its acceptance evidence must also be committed.

## Release labels

- **Experimental**: useful for development and learning; no scientific claim.
- **Validated workload**: agrees with independent references for the exact
  published fixtures and tolerances.
- **Research preview**: multiple materials and independent test sets pass;
  interfaces and result provenance are stable enough for outside replication.
- **Research release**: documented physical scope, convergence behavior,
  restart compatibility, and release-specific validation suite all pass.

No Metal result is promoted solely from agreement with another mixed-precision
path. CPU FP64 and an independent established code remain required references.

## Phase 1 — resident Apple-GPU eigensolver

Status: complete for the pinned validated silicon workloads.

The current code has a persistent Metal Hamiltonian and a production resident
Davidson driver. Basis and Hamiltonian journals, Ritz and residual formation,
preconditioning, two-pass correction orthogonalization, correction Hamiltonian
application, journal append, and subspace restart all remain on the GPU.
Runtime transfer counters fail closed if a normal iteration uploads or
downloads a full vector. Si2 accuracy and transfer gates pass, and a matched
five-warm-run Si16 comparison passes scientific parity and the `1.20x`
end-to-end gate at `1.351x` on the Apple M5 Pro. The complete implementation
contract is in
[`docs/resident-davidson.md`](docs/resident-davidson.md).

Exit criteria:

- normal Davidson iterations transfer no full wavefunction between CPU and GPU;
- subspace restart is device-local;
- transfer counters and graph-operation counters are part of every Metal result;
- Si2 and Si16 scientific gates still pass;
- five warmed Si16 runs retain at least `1.20x` median end-to-end speedup over
  the CPU FP64 path;
- a public macOS CI run builds and tests the committed revision.

## Phase 2 — validation infrastructure and restartability

Build the machinery required to add physics without weakening evidence.

- version the input, result, comparison, benchmark, checkpoint, and
  validation-manifest schemas; **implemented at the format level**
- add binary checkpoints with hashes, compatibility checks, atomic writes,
  compact inspection, and CPU/Metal interruption equivalence tests;
  **implemented for the current SCF state boundary**
- provide a fixture manifest that records licenses, origins, download hashes,
  redistribution policy, and independent reference commands; **implemented
  and enforced in CI**
- separate smoke, invariant, scientific, and performance suites;
- publish machine-readable accuracy and timing histories by hardware model;
- add deterministic replay tests and interruption/restart equivalence tests;
  **CPU and Metal interruption equivalence implemented**
- add structured diagnostics for SCF, eigensolver, memory, and transfer
  behavior; **per-iteration SCF/eigensolver histories and aggregate Metal
  memory/transfer counters implemented**

Exit criteria include restart equivalence within the active precision policy,
fixture reproducibility from a clean checkout, and no unversioned result fields.

## Phase 3 — useful norm-conserving ground-state scope

Add physical capabilities one at a time, each behind fail-closed input
validation and independent fixtures.

1. XC abstraction followed by LDA and PBE reference tables and finite-difference
   potential checks. **Typed grid-level abstraction and variational LDA-PZ
   checks implemented; the native PBE point energy and derivatives match a
   pinned Libxc 7.1.2 table, and the spectral grid potential passes a
   whole-functional finite-difference test. A pinned ONCV PBE UPF now passes
   strict parsing, and the Si2 CPU and Metal full-SCF comparisons pass their
   independent QE and mixed-precision gates. PBE is accepted only for that
   explicitly validated silicon workload. Multi-species local/nonlocal
   operators, starting densities, electron counts, and Ewald sums now pass a
   matched 100 Ry MgO CPU/QE and Metal/CPU gate; this admits one binary oxide,
   not broad chemical transferability.**
2. Monkhorst-Pack and explicit k-point lists, weights, time-reversal handling,
   and band/density accumulation. **CPU FP64 is implemented and accepted for a
   shifted 2x2x2 Si2 PBE fixture against QE 7.6. Per-k-point coordinates,
   weights, occupied eigenvalues, occupations, charge, and total energy pass a
   release-mode CI gate. Resident mixed-precision Metal is accepted against the
   CPU result for the same fixture, with one persistent GPU context per
   irreducible point, zero projection resynchronizations, and explicit transfer
   accounting. The contexts currently run sequentially and do not establish a
   performance claim. Multi-k-point checkpoint/restart remains fail closed.**
3. Fractional occupations and finite-temperature smearing for metals.
   **Global Fermi-Dirac chemical-potential solving, entropy/free-energy terms,
   extra-band allocation, and charge-conservation tests are implemented on CPU.
   A shifted 4x4x4 fcc Al LDA-PZ free-energy calculation passes independent QE
   7.6 total-energy, occupied-eigenvalue, occupation, and charge gates. This
   accepts only that exact CPU fixture. The corresponding resident Metal SCF
   passes free-energy, occupied-eigenvalue, occupation, charge, and transfer-
   audit gates against CPU. Denser-mesh physical convergence and a modern
   production pseudopotential remain pending.**
4. Collinear spin polarization.
5. Analytic norm-conserving forces, finite-difference verification, and fixed-
   cell ionic relaxation.
6. Stress and variable-cell relaxation only after force validation is mature.

Each feature must pass CPU FP64 first. Metal acceptance then requires a
fixed-density comparison, a full-SCF comparison, determinism, and a transfer
audit. Candidate validation materials should cover semiconductors, insulators,
metals, magnetic systems, molecules in periodic cells, and chemically distinct
norm-conserving pseudopotentials.

## Phase 4 — usability for independent researchers

- stable CLI with `run`, `resume`, `inspect`, `compare`, and `bench` workflows;
- actionable convergence diagnostics and conservative presets;
- documented units, coordinate conventions, FFT/cutoff behavior, and precision
  implications;
- pseudopotential discovery and verification without redistributing files whose
  licenses do not permit it;
- self-contained provenance bundles suitable for archiving with a publication;
- installation through signed release artifacts plus source builds;
- examples that state both what is supported and what scientific conclusions
  they do not establish;
- a compatibility table by macOS version and Apple GPU family.

The project remains fail closed: an unavailable GPU, unsupported
pseudopotential, out-of-scope functional, incompatible checkpoint, or failed
convergence gate produces an explicit error rather than a hidden fallback.

## Phase 5 — broad pseudopotential and performance work

Ultrasoft and PAW support require generalized-overlap eigensolvers, augmentation
charges, additional energy terms, forces, and substantially broader validation.
They are separate scientific projects and will not be represented as parser
extensions. They begin only after the norm-conserving research preview is
credible.

Performance work in every phase follows the same order:

1. preserve a correct CPU FP64 reference;
2. measure end-to-end profiles and transfer volume;
3. move a complete residency boundary, not an isolated kernel;
4. validate fixed-density and SCF results;
5. publish cold and warmed timing distributions with exact hardware/software
   provenance.

Potential later work includes custom Metal kernels where MPSGraph leaves large
gaps, batched k-point execution, memory-aware scheduling across Apple GPU
families, and multi-process CPU work. Distributed GPU execution is not a near-
term goal for a Mac-focused package.

## Release-blocking quality requirements

Every research release must provide:

- a clean source revision and reproducible dependency lockfile;
- CPU and Metal test results on supported macOS versions;
- independent total-energy, eigenvalue, force, and charge comparisons for the
  advertised scope;
- invariant tests for Hermiticity, orthonormality, Parseval, charge, and
  deterministic initialization;
- min/median/max end-to-end benchmarks with identical physics and convergence
  gates, produced by `bench-compare` rather than inferred from independent
  single-backend reports;
- documented known limitations and unsupported input behavior;
- license and provenance records for code, dependencies, and fixtures;
- no silent numerical, backend, or physics fallback.

The repository README may summarize accepted results, but
`results/validation.json` and its source fixtures are the authoritative
machine-readable release evidence.
