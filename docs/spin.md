# Collinear spin-polarization development

Version 0.13 introduced the independently validated LSDA kernel. Version 0.14
admits a deliberately bounded CPU-FP64 collinear-spin SCF while keeping every
unsupported path fail-closed. Version 0.15 adds two persistent resident Metal
spin-channel Davidson contexts, precision-aware charge/magnetization mixing,
schema-5 CLI comparison, and fixed-density plus complete-SCF accelerator
gates. `spin_components = 2` therefore enters only the Gamma-point LDA-PZ
solver whose orbitals, occupations, density mixing, results, and force assembly
are explicitly spin resolved.

## Implemented foundation

The XC layer now has an explicit two-density boundary:

```text
XcFunctional::evaluate_collinear(fft, lattice, [rho_up, rho_down])
  -> SpinXcEvaluation {
       potential_ry: [v_up, v_down],
       energy_ry,
       integral_rho_vxc_ry
     }
```

LDA-PZ implements this boundary. Exchange uses exact spin scaling and
correlation uses the Perdew-Zunger interpolation between the unpolarized and
fully polarized Ceperley-Alder parameterizations. Point inputs are electron
densities in `electron/bohr^3`; point energies and derivatives are Hartree.
The grid boundary converts energy and both potentials to Rydberg.

PBE deliberately returns an unsupported error at this boundary. A polarized
PBE implementation requires the full spin-gradient invariant set and its own
independent point and whole-functional derivative validation; it is not
approximated by applying the unpolarized functional independently to each spin.

## Independent validation

The tracked
[`libxc-7.1.2-lda-pz-polarized.json`](../fixtures/xc/libxc-7.1.2-lda-pz-polarized.json)
fixture records seven independent Libxc 7.1.2 `LDA_X + LDA_C_PZ` evaluations:

- equal-spin densities at low density;
- three partially polarized points;
- a high-density partially polarized point;
- fully up-polarized and fully down-polarized endpoints.

Native energy per electron and both spin potentials agree with the table to
the test's `2e-13` relative tolerance. Additional invariants require:

- exact reduction to the existing unpolarized LDA-PZ energy and potential when
  `rho_up == rho_down`;
- agreement of `v_up` and `v_down` with independent central finite differences
  of the two-density energy;
- correct grid units, integrated energy, and spin-channel potentials;
- finite zero values in vacuum and explicit rejection of materially negative
  or non-finite densities;
- fail-closed behavior for polarized PBE.

The oracle generator is
[`tools/libxc_lsda_reference.c`](../tools/libxc_lsda_reference.c), and the
fixture hash, source version, license, and reproduction command are enforced by
the fixture manifest.

## Accepted CPU and Metal vertical slice

Input schema 3 admits `spin_components = 2` only when all of these conditions
hold:

- backend `cpu-f64` or `metal-mixed` on macOS;
- fixed cell and Gamma point;
- LDA-PZ;
- fixed occupations with positive integer up/down electron counts;
- a finite `magnetization_electrons` consistent with the pseudopotential
  valence count.

For example, the accepted silicon fixture requests eight electrons split into
five up and three down:

```toml
schema_version = 3
xc = "lda-pz"
k_points = "gamma"
spin_components = 2
magnetization_electrons = 2.0
```

Each channel owns its Davidson solve, occupied orbital block, output density,
and mixing state. The Hartree potential is formed once from
`rho_up + rho_down`; LSDA supplies separate channel potentials. On CPU, small
projected eigensolves, reductions, total-energy assembly, convergence decisions,
and all storage remain FP64.

On Metal, each spin channel has a separate persistent MPSGraph Hamiltonian and
resident Davidson journal. Complex-FP32 wavefunctions, Hamiltonian images,
Ritz vectors, residuals, preconditioned corrections, orthogonalization, and
restarts remain device resident within every solve. CPU-FP64 finishes compact
reductions and projected eigensolves, assembles energies, mixes densities, and
makes convergence decisions. There is no silent CPU Hamiltonian fallback.

The symmetric open-shell silicon fixture exposed a mixed-precision SCF branch
instability when both densities used the requested `0.3` mixing coefficient.
The accepted policy therefore mixes total charge at the requested rate while
capping the Metal magnetization-channel coefficient at `0.1`. CPU retains the
requested coefficient for both channels, which is algebraically identical to
the original independent channel mixer. The Metal density gate is explicitly
`2e-4` for this workload, reflecting the observed FP32 spin-density floor; the
measured residual and both effective coefficients are recorded in
`energy_terms`. Scientific acceptance still requires independent energy,
eigenvalue, charge, magnetization, force, and determinism gates.

Schema-5 results leave the spin-degenerate top-level band arrays empty and
instead report `spin.up` and `spin.down`, each with its electron count,
k-points, eigenvalues, occupations, and residuals. Validation integrates the
occupations independently, reconstructs total charge and magnetization, and
rejects ambiguous or inconsistent records. Analytic local, nonlocal, and Ewald
forces use the total local density and both occupied spin-channel orbital sets.

## Independent full-SCF validation

[`examples/si2-spin.toml`](../examples/si2-spin.toml) and
[`fixtures/qe/si2-spin-reference.in`](../fixtures/qe/si2-spin-reference.in)
define the same two-atom silicon cell, 12 Ry cutoff, LDA-PZ functional,
norm-conserving `Si.pz-vbc.UPF`, fixed magnetization of two electrons, fixed
occupations, and Gamma point. The reference was run with QE 7.6 source commit
`9f93ddec427d2b9a45bb72d828c6d324f62fcabd` and normalized from its
full-precision XML into result schema 5.

The accepted energies are `-14.426337119974042 Ry` for Rust and
`-14.426337128492126 Ry` for QE, an error of about `0.000058 meV/atom`.
Occupied spin-channel eigenvalues are required to agree within 5 meV, total
energy within 2 meV/atom, charge and magnetization within `1e-8` electrons,
and Cartesian forces within `1e-4 Ry/bohr`. The release-mode ignored test
enforces all of those gates, while the fixture manifest pins every relevant
hash and provenance field.

The accepted resident Metal run converges in 239 SCF iterations. Relative to
CPU it differs by `0.00876 meV/atom` in total energy, `0.00988 meV` at most for
occupied eigenvalues, about `3.91e-5 Ry/bohr` in the largest force component,
and less than `3e-14` electrons in charge or magnetization. Relative to QE the
energy error is `0.00870 meV/atom`. A second complete Metal run reproduces the
iteration count, SCF history, energy, eigenvalues, forces, and magnetization
exactly on the M5 Pro, exceeding the `0.1 meV/atom` repeatability gate.

The fixed-density polarized Hamiltonian is checked separately from the SCF
trajectory. The measured relative L2 errors are `8.26e-8` for spin up and
`8.28e-8` for spin down. The complete run uses exactly two persistent contexts,
reports zero projection resynchronizations, performs one transfer-invariant
check per Davidson iteration, and peaks below 10 MB of device allocation for
this small fixture. This is an accuracy and residency result, not a performance
claim; the tiny open-shell cell is substantially slower on GPU.

## Remaining fail-closed boundaries

PBE, weighted or generated k-point meshes, fractional occupations,
checkpoint/restart, matched performance benchmarks, and ionic relaxation
reject spin-polarized input. Fully polarized fixed-occupation SCFs with an
empty channel are also deferred, even though the point LSDA kernel validates
its endpoint values.

The next admission projects are spin checkpoint/restart, a compute-bound
magnetic fixture that can support an honest performance decision, and broader
magnetic chemistry. Noncollinear magnetism, spin-orbit coupling, constrained
magnetization, and finite magnetic fields are separate later projects.
