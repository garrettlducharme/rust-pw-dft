# Analytic forces

Schema-4 converged results contain `forces_ry_bohr`, one Cartesian vector per
input atom in input order. The unit is Rydberg per bohr. Unconverged results
must omit the field, and schema-2/3 records cannot contain it. This prevents a
partially converged calculation from being consumed as a scientific force
result.

## Implemented terms

The fixed-cell norm-conserving force is assembled from:

- the explicit derivative of each species-resolved reciprocal-space local
  pseudopotential at the final orbital density;
- the explicit derivative of every separable nonlocal projector, using
  `G + k` for its phase derivative and the final band occupations;
- the analytic real- and reciprocal-space Ewald ion-ion derivative.

For a weighted k-point calculation, each nonlocal band contribution is
multiplied by its irreducible Brillouin-zone weight. The local-density and
Ewald terms are included once. A fixed plane-wave basis has no ionic-position
Pulay term. Stress, variable-cell Pulay terms, ultrasoft augmentation, PAW,
spin, and constrained-coordinate forces are not implemented.

Metal SCFs retain mixed-precision resident eigensolves. Once a converged state
has crossed the existing final-output boundary, force reductions and assembly
run in CPU FP64 from the final density, orbitals, and occupations. This is
recorded as `forces_seconds` and does not introduce a hidden Hamiltonian
fallback.

## Validation boundary

Three independent layers guard the implementation:

1. Local, nonlocal, and Ewald operator derivatives each match central finite
   differences of their corresponding energy term.
2. A displaced-Si2 fixture runs five independently converged CPU SCFs and
   compares both nonzero atomic forces with central total-energy derivatives.
   The maximum observed error is `9.23e-9 Ry/bohr`.
3. The same displaced structure is compared with full-precision forces from
   Quantum ESPRESSO 7.6 commit
   `9f93ddec427d2b9a45bb72d828c6d324f62fcabd`. The maximum component error is
   `5.17e-6 Ry/bohr`.

CPU/Metal parity tests cover Gamma-point LDA and PBE silicon, shifted 2x2x2 PBE
silicon, and Fermi-Dirac metallic aluminum. Checkpoint/resume tests also compare
the final forces. On the nonzero displaced-Si2 fixture, Metal's maximum force
component error relative to CPU is `1.38e-7 Ry/bohr`.

The code reports raw forces and does not impose an acoustic-sum-rule
correction. At the displaced low-cutoff fixture, the raw net Cartesian-x force
is about `1.03e-5 Ry/bohr`; both individual components still reproduce the
solver's total-energy derivatives and remain within the independent QE gate.
Users should converge cutoff, k mesh, and SCF tolerances for their own force
workloads rather than interpreting the fixture threshold as universal.

The CLI `compare` command performs a force comparison only when both inputs
contain schema-4 forces. Its report states `force_comparison_performed` and the
measured maximum component error. The current acceptance tolerances are
`1e-4 Ry/bohr` for CPU-FP64 results and `5e-4 Ry/bohr` for Metal mixed
precision.
