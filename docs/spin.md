# Collinear spin-polarization development

Version 0.13 begins collinear spin, the next ground-state physics expansion
after the validated spin-unpolarized force and relaxation boundary. It is being
added in layers so
that `spin_components = 2` cannot silently enter a solver whose orbitals,
occupations, density mixing, results, checkpoints, or Metal buffers still
assume spin degeneracy.

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

## Remaining admission gates

This foundation does not yet admit spin-polarized input. Before that happens,
one coherent CPU-FP64 vertical slice must provide:

1. a versioned spin input contract with total electron and magnetization
   consistency checks;
2. separate up/down band counts, occupations, orbitals, densities, and mixing;
3. a common Hartree potential from total density plus channel-specific XC
   potentials;
4. spin-resolved eigenvalues, occupations, charges, magnetic moment, SCF
   history, result validation, comparison, and checkpoint state;
5. an independently reproduced magnetic reference calculation and fixed-
   density operator checks.

Only after the CPU result passes those gates should the Metal path add two
resident spin-channel eigensolver contexts and prove CPU/Metal parity,
deterministic restart, and transfer accounting. Noncollinear magnetism,
spin-orbit coupling, constrained magnetization, and finite magnetic fields are
separate later projects.
