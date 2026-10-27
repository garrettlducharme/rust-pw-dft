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
after downloading the four intentionally untracked fixtures.

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
