# Scientific fixtures

[`fixtures/manifest.toml`](../fixtures/manifest.toml) is the machine-readable
authority for every input, pseudopotential, and independent reference result
used in an acceptance claim. Run:

```sh
rust-pw-dft verify-fixtures
```

The verifier rejects unknown manifest fields, unsupported schema versions,
duplicate identifiers, missing provenance, malformed hashes, absolute paths,
parent-directory traversal, missing files, and checksum mismatches. CI runs it
after downloading the five intentionally untracked fixtures.

## Silicon pseudopotential

`Si.pz-vbc.UPF` is fetched from the Quantum ESPRESSO `qe-7.6` tree and checked
against SHA-256
`d75dd6b0be0aa10587fc95900cfd6ba7314d461a8276a81df34f009d0bfc075d`.
The project does not redistribute the file. Its upstream data license is not
separately declared, so the manifest records SPDX-style `NOASSERTION` instead
of inventing a license conclusion.

The PBE workload uses `Si_ONCV_PBE-1.0.upf` from Wannier90 commit
`1d6b187374a2d50b509e5e79e2cab01a79ff7ce1`, the exact submodule revision
pinned by QE 7.6. Its SHA-256 is
`481a5fae1167fd712d4e8e2e4df33f8bfc1e7d2abbc0565290d39fbba9dccb41`.
The UPF declares CC-BY-SA-4.0 and identifies the Schlipf-Gygi
norm-conserving library and ONCVPSP generator. The project records that license
and source but still avoids redistributing the file.

The LDA silicon performance fixtures use the same pinned legacy file. The
Gamma-point `si16.toml` input measures the original resident-Hamiltonian
crossover. `si16-lda-2x2x2.toml` expands that cell to a shifted mesh and a
24 Ry cutoff so each of its four irreducible-point contexts is compute-bound.
Its five-run report is a performance fixture, not an independent physical
reference: scientific parity is measured against the CPU-FP64 backend, while
the smaller Si2 k-point workload remains the independently QE-validated case.

## Independent QE result

The accepted Si2 reference was produced by Quantum ESPRESSO 7.6 source commit
`9f93ddec427d2b9a45bb72d828c6d324f62fcabd`. The original machine-local input
hash is recorded in the manifest and result provenance. A portable equivalent
is tracked at [`fixtures/qe/si2-reference.in`](../fixtures/qe/si2-reference.in)
and can be run from the repository root with:

```sh
mkdir -p qe-si2-reference-tmp
pw.x -in fixtures/qe/si2-reference.in
```

The portable file changes only `pseudo_dir` and `outdir`; lattice, atoms,
cutoffs, functional, occupations, symmetry settings, mixing, diagonalization,
and convergence threshold match the accepted input.

The normalized JSON contains the independent total energy, occupied
eigenvalues, electron count, and energy decomposition. Its SCF history is
marked incomplete because QE's iteration log was not embedded in the original
normalized artifact; native `rust-pw-dft` results require a complete structured
history.

### Displaced-silicon force reference

[`fixtures/qe/si2-displaced-force-reference.in`](../fixtures/qe/si2-displaced-force-reference.in)
moves the second silicon atom by `0.05 bohr` along Cartesian x and requests QE
forces. This prevents a symmetry-zero force from satisfying the validation.
The pinned QE 7.6 XML records forces in Hartree/bohr; the normalizer converts
them to the schema-4 Ry/bohr convention and requires exactly three finite
components per atom. The resulting independent artifact is
[`results/qe-si2-displaced-force-reference.json`](../results/qe-si2-displaced-force-reference.json).

Regenerate it with:

```sh
python3 tools/normalize_qe_xml.py \
  qe-si2-displaced-force-reference-tmp/si2-displaced-force-reference.save/data-file-schema.xml \
  qe-si2-displaced-force-reference.out \
  fixtures/qe/si2-displaced-force-reference.in \
  pseudopotentials/Si.pz-vbc.UPF \
  results/qe-si2-displaced-force-reference.json \
  --qe-revision 9f93ddec427d2b9a45bb72d828c6d324f62fcabd
```

The accepted CPU result differs from QE by `5.17e-6 Ry/bohr` in its largest
force component. A separate five-SCF central-difference gate checks both atoms
at the displaced geometry; its maximum analytic-versus-energy-derivative error
is `9.23e-9 Ry/bohr`.

The accepted PBE reference uses the tracked, byte-identical
[`fixtures/qe/si2-pbe-reference.in`](../fixtures/qe/si2-pbe-reference.in), QE
7.6 source commit `9f93ddec427d2b9a45bb72d828c6d324f62fcabd`, and the pinned ONCV PBE
pseudopotential. Full-precision energies and eigenvalues were read from QE's
`data-file-schema.xml`, not its rounded text summary, and normalized into
[`results/qe-pbe-reference.json`](../results/qe-pbe-reference.json).
Regenerate that JSON after running QE with:

```sh
python3 tools/normalize_qe_xml.py \
  qe-si2-pbe-reference-tmp/si2-pbe-gamma-reference.save/data-file-schema.xml \
  qe-si2-pbe-reference.out fixtures/qe/si2-pbe-reference.in \
  pseudopotentials/Si_ONCV_PBE-1.0.upf results/qe-pbe-reference.json \
  --qe-revision 9f93ddec427d2b9a45bb72d828c6d324f62fcabd
```

The accepted multi-k-point PBE reference uses the shifted 2x2x2 mesh in
[`fixtures/qe/si2-pbe-2x2x2-reference.in`](../fixtures/qe/si2-pbe-2x2x2-reference.in).
The normalizer converts QE Cartesian reciprocal coordinates to reduced crystal
coordinates, normalizes weights and occupations to the project convention, and
combines only numerically verified time-reversal pairs:

```sh
python3 tools/normalize_qe_xml.py \
  qe-si2-pbe-2x2x2-reference-tmp/si2-pbe-2x2x2-reference.save/data-file-schema.xml \
  qe-si2-pbe-2x2x2-reference.out fixtures/qe/si2-pbe-2x2x2-reference.in \
  pseudopotentials/Si_ONCV_PBE-1.0.upf \
  results/qe-si2-pbe-2x2x2-reference.json \
  --qe-revision 9f93ddec427d2b9a45bb72d828c6d324f62fcabd \
  --reduce-time-reversal
```

## Independent Libxc PBE point table

`fixtures/xc/libxc-7.1.2-pbe-unpolarized.json` records unpolarized PBE
exchange plus correlation values from Libxc 7.1.2: energy per particle,
the density derivative, and the squared-gradient derivative. It covers the
uniform-density limit and two finite-gradient points. The fixture is an
independent numerical oracle for the native PBE point kernel. The spectral
grid-level gradient/divergence implementation is separately checked through a
whole-functional finite difference. The compatible ONCV fixture and full CPU
SCF comparison now pass, so PBE is admitted for the pinned silicon workload.

The small generator in `tools/libxc_xc_reference.c` uses Libxc functional IDs
101 and 130. With Homebrew Libxc 7.1.2 installed, regenerate it with:

```sh
clang tools/libxc_xc_reference.c -I/opt/homebrew/include \
  -L/opt/homebrew/lib -lxc -o /tmp/libxc_xc_reference
/tmp/libxc_xc_reference
```

The generator is project code under MIT/Apache-2.0. The recorded Libxc-derived
reference data is identified as MPL-2.0 in the manifest.

## Independent Libxc collinear LDA-PZ point table

`fixtures/xc/libxc-7.1.2-lda-pz-polarized.json` records spin-polarized Slater
exchange plus Perdew-Zunger correlation from Libxc 7.1.2. Its seven points
cover the unpolarized limit, several partial polarizations, and both fully
polarized endpoints. Each point records energy per electron and independent
up/down density derivatives.

The native LSDA kernel is required to match this table, reduce to the existing
unpolarized LDA-PZ implementation at equal spin densities, and pass separate
finite differences in both spin channels. The grid boundary is also checked
for unit conversion and variational potentials. This validates the XC
foundation only: spin-polarized SCF input remains rejected until the orbital,
occupation, density, result, checkpoint, and Metal paths all support it.

Regenerate the independent table with:

```sh
clang tools/libxc_lsda_reference.c -I/opt/homebrew/include \
  -L/opt/homebrew/lib -lxc -o /tmp/libxc_lsda_reference
/tmp/libxc_lsda_reference
```

## Multi-species MgO reference

The first chemical-transfer fixture uses the ten-electron Mg and six-electron
O ONCV PBE files from `d3q` commit
`6e5f97816af439702196c248ec2ad0092c561b89`, the revision pinned by QE 7.6.
The upstream files do not state a separate data license, so the project records
`NOASSERTION` and does not redistribute them.

[`fixtures/qe/mgo2-pbe-reference.in`](../fixtures/qe/mgo2-pbe-reference.in)
and [`examples/mgo2-pbe.toml`](../examples/mgo2-pbe.toml) use the same
two-atom rock-salt primitive cell, 100 Ry wavefunction cutoff, PBE functional,
fixed occupations, Gamma point, disabled symmetry, and `0.2` linear-mixing
factor. The reference converged with QE 7.6 source commit
`9f93ddec427d2b9a45bb72d828c6d324f62fcabd`.

Normalize its full-precision XML with both pseudopotential identities:

```sh
python3 tools/normalize_qe_xml.py \
  qe-mgo2-pbe-reference-tmp/mgo2-pbe-gamma-reference.save/data-file-schema.xml \
  qe-mgo2-pbe-reference.out fixtures/qe/mgo2-pbe-reference.in \
  results/qe-mgo2-pbe-reference.json \
  --qe-revision 9f93ddec427d2b9a45bb72d828c6d324f62fcabd \
  --pseudo Mg=pseudopotentials/Mg_ONCV_PBE-1.0.UPF \
  --pseudo O=pseudopotentials/O_ONCV_PBE-1.0.UPF
```

The UPF reciprocal transform follows the standard QE convention: truncate
large-radius cancellation noise beyond 10 bohr and force an odd Simpson mesh.
This convention is covered by the independent total-energy and eigenvalue
comparison rather than hidden as an empirical energy correction.

## Metallic aluminum reference

The first fractional-occupation fixture uses the non-relativistic three-electron
`Al.pz-vbc.UPF` from QE's original legacy pseudopotential table. The upstream
table labels this collection as not recommended and kept for reference. The
project therefore uses it only to validate numerics, records the upstream data
license as `NOASSERTION`, does not redistribute it, and makes no production
pseudopotential claim.

[`fixtures/qe/al1-lda-4x4x4-reference.in`](../fixtures/qe/al1-lda-4x4x4-reference.in)
and [`examples/al1-lda-4x4x4.toml`](../examples/al1-lda-4x4x4.toml) use the
same one-atom fcc cell, 12 Ry cutoff, shifted 4x4x4 mesh, LDA-PZ functional,
0.02 Ry Fermi-Dirac smearing, and 0.2 linear mixing. Normalize the clean QE
7.6 XML with verified time-reversal averaging:

```sh
python3 tools/normalize_qe_xml.py \
  qe-al1-lda-4x4x4-reference-tmp/al1-lda-4x4x4-reference.save/data-file-schema.xml \
  qe-al1-lda-4x4x4-reference.out fixtures/qe/al1-lda-4x4x4-reference.in \
  pseudopotentials/Al.pz-vbc.UPF \
  results/qe-al1-lda-4x4x4-reference.json \
  --qe-revision 9f93ddec427d2b9a45bb72d828c6d324f62fcabd \
  --element Al --reduce-time-reversal
```

The normalizer rejects missing time-reversal partners, band-count mismatches,
pair eigenvalue differences above `1e-5 Ry`, or occupation differences above
`1e-8`. It records the observed maxima and uses weight-aware pair averages.
