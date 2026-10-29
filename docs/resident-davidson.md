# Resident Davidson design

This document defines the first production Metal eigensolver milestone. It is an
implementation contract, not a claim that the full path already exists.

## Goal

Keep the Davidson basis, its Hamiltonian images, Ritz vectors, residuals, and
preconditioned correction blocks in complex FP32 Metal buffers for the lifetime
of one eigensolve. Transfer only small projected matrices, residual norms,
eigenvalues, convergence decisions, and final converged orbitals across the
CPU/GPU boundary. Rayleigh-Ritz diagonalization and final reductions remain
CPU FP64.

The production-performance gate stays closed until profiling proves that a
normal Davidson iteration performs no full-wavefunction CPU/GPU transfer.

## Current boundary

The implementation already provides:

- persistent Metal Hamiltonian buffers and compact plane-wave output;
- a compact FP32 journal of basis vectors and Hamiltonian images;
- resident cross-Gram partial reductions with CPU complex-FP64 final sums;
- direct residual-norm partial reductions with CPU FP64 final sums;
- resident Ritz/applied-Ritz/residual formation and guarded kinetic
  preconditioning with active-root masking;
- resident two-pass block modified Gram-Schmidt correction orthogonalization,
  normalization, device-local row copies, and rank rejection;
- custom Metal compact-to-grid scatter, resident correction Hamiltonian
  application, and device-local basis/H-basis journal append;
- device-local journal replacement from resident Ritz/applied-Ritz blocks that
  preserves raw corrections for post-restart reorthogonalization;
- per-iteration transfer accounting and fail-closed error handling.

The production SCF driver now uses this resident path. CPU FP64 receives the
small projected Hamiltonian, finishes reduction partials, solves the projected
Rayleigh-Ritz problem, and makes convergence decisions. A normal Davidson
iteration transfers no full wavefunction. The initial basis is uploaded once,
and final Ritz vectors are downloaded once and reorthonormalized in CPU FP64
for density and energy assembly. Fixed-density kernel tests and a complete Si2
SCF test cover the integrated path; matched Si16 evidence remains pending.

## Buffer ownership

One `HamiltonianContext` owns all Metal allocations for one SCF calculation.
Davidson state is reset at the start of each eigensolve and has capacity
`max_subspace`.

| Buffer | Shape | Precision | Lifetime |
| --- | --- | --- | --- |
| `basis` | `[max_subspace, npw_padded]` | complex FP32 | eigensolve |
| `applied_basis` | `[max_subspace, npw_padded]` | complex FP32 | eigensolve |
| `ritz` | `[roots, npw_padded]` | complex FP32 | iteration |
| `applied_ritz` | `[roots, npw_padded]` | complex FP32 | iteration |
| `residual` | `[roots, npw_padded]` | complex FP32 | iteration |
| `correction` | `[roots, npw_padded]` | complex FP32 | iteration |
| `correction_work` | `[2, npw_padded]` | complex FP32 | active correction |
| `accepted_corrections` | `[roots, npw_padded]` | complex FP32 | iteration |
| `projected_partials` | `[subspace, subspace, chunks]` | complex FP32 | iteration |
| `norm_partials` | `[roots, chunks]` | complex FP32 | iteration |

Padding is always zeroed before a buffer becomes visible to a reduction. All
capacity arithmetic is overflow-checked and counted against the 12 GiB project
limit before allocation.

## Iteration dataflow

1. Metal computes chunked partials for `<B|HB>` from resident `basis` and
   `applied_basis`.
2. CPU accumulates the compact partials in complex FP64, Hermitianizes the
   projected matrix, and diagonalizes it with `faer`.
3. CPU uploads only the `subspace × roots` FP64-to-FP32 Ritz coefficient matrix
   and `roots` FP64-to-FP32 eigenvalues.
4. Metal matrix multiplication forms `ritz = C^T B` and
   `applied_ritz = C^T HB`, then forms
   `residual = applied_ritz - eigenvalue * ritz`.
5. Metal computes chunked direct residual-norm partials; CPU accumulates and
   decides convergence in FP64.
6. For unconverged roots, Metal applies the guarded kinetic preconditioner
   `-r / safe(T - eigenvalue)`. The `0.25 Ry` denominator floor must exactly
   match the CPU reference path.
7. Metal orthogonalizes corrections against the resident basis and earlier
   accepted corrections using two-pass block modified Gram-Schmidt.
   Dot-product partials finish in CPU FP64; only compact overlap coefficients
   and norms cross the boundary. A correction is rejected when its final norm
   is at most `max(1e-11, 1e-6 * initial_norm)`, preventing normalized FP32
   roundoff from creating a spurious direction.
8. Metal normalizes accepted corrections, applies the Hamiltonian, and appends
   both blocks to the resident journals. A custom compact-to-grid scatter
   kernel supplies FFT input without staging a full vector on the host.
9. On subspace restart, Metal replaces the journal with resident Ritz vectors
   and their Hamiltonian images. A host resynchronization is an error in the
   final resident mode, not a normal restart mechanism.

The first implementation may split steps 4–8 into several cached MPSGraph
executions. Correct residency and explicit synchronization take priority over
graph fusion.

## Rust interface boundary

The general CPU Davidson implementation remains independently testable. Metal
uses a stateful driver rather than adding more loosely coupled closures to the
generic solver:

```text
ResidentDavidsonBackend
  begin(initial_basis) -> ResidentState
  projected_hamiltonian(state) -> Complex64 matrix
  update_ritz(state, coefficients_f64, eigenvalues_f64)
  residual_norms(state) -> f64 vector
  append_corrections(state, unconverged_roots) -> accepted count
  restart_from_ritz(state)
  download_converged(state) -> Complex64 vectors
  metrics(state) -> ResidentDavidsonMetrics
```

Every method validates state generation, dimensions, capacity, finite scalar
inputs, and the expected resident row count. Unsupported transitions fail with
an error; none may trigger a hidden CPU implementation.

## Correctness gates

Each kernel is first tested against the CPU implementation at fixed density.
The integrated path must then meet all existing Si2 and Si16 gates.

- Ritz and applied-Ritz relative L2 error: at most `2e-5`.
- Residual-vector relative L2 error: at most `2e-5`.
- Direct residual-norm relative error: at most `2e-5`, with measured values
  printed by the test.
- Preconditioned-correction relative L2 error: at most `2e-5`.
- Orthonormality: maximum `|<psi_i|psi_j> - delta_ij|` at most `2e-5` on Metal.
- Hamiltonian application, energy, occupied-eigenvalue, charge, and
  determinism gates remain unchanged.
- Malformed dimensions, NaN/Inf coefficients, capacity overflow, stale state,
  or missing Metal support must fail closed.

A fixed-density test must distinguish operator/eigensolver error from the SCF
trajectory. Full SCF comparisons alone are insufficient.

## Residency and performance evidence

The result schema must record:

- full-vector host-to-device and device-to-host bytes;
- compact reduction bytes in each direction;
- graph executions by operation;
- normal iterations, restarts, and unexpected resynchronizations;
- persistent and peak device allocation;
- CPU and Metal Davidson iterations and SCF iterations.

Acceptance requires zero full-vector transfers during a normal post-
initialization iteration. A restart may not download or re-upload the basis.
Only the final converged orbitals may be downloaded for CPU-FP64 density and
energy assembly in this milestone.

The existing five-warm-run Si16 protocol remains the performance gate. The
resident path must preserve scientific acceptance and at least `1.20x` median
end-to-end speedup over CPU FP64. Results must report min/median/max and may not
reuse timings from an earlier source revision.

The `bench-compare` command enforces this as a single operation. Both backends
use the recorded `matched-metal-compatible-v1` convergence profile; the report
fails closed if source/input/pseudopotential provenance or effective
tolerances differ, or if either scientific accuracy or speed misses its gate.

## Delivery sequence

1. Add resident Ritz/residual construction and fixed-density tests. **Done.**
2. Add resident preconditioning and two-pass orthogonalization. **Done.**
3. Add resident Hamiltonian application and correction append. **Done.**
4. Replace host restart resynchronization with a device-local restart. **Done.**
5. Add transfer accounting and assert the normal-iteration residency invariant.
   **Done.**
6. Integrate the stateful driver into SCF behind the Metal backend. **Done.**
7. Regenerate Si2 and Si16 scientific and benchmark evidence at the committed
   revision. **Done.**

PBE, MgO multi-species operators, silicon k-point sampling, and metallic
aluminum occupations now reuse this resident boundary and pass their published
accuracy gates. K-point contexts are persistent but scheduled sequentially, so
the tiny-cell `0.212x` silicon matched benchmark is a documented negative
performance result rather than a speed claim. The compute-bound sixteen-atom,
24 Ry shifted-mesh fixture passes the same five-run gate at `1.856x`, defining
the current multi-k-point crossover on the Apple M5 Pro. Forces, spin, broader
chemical validation, ultrasoft pseudopotentials, and PAW remain outside the
accepted scope.
