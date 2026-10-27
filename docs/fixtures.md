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
after downloading the one intentionally untracked fixture.

## Silicon pseudopotential

`Si.pz-vbc.UPF` is fetched from the Quantum ESPRESSO `qe-7.6` tree and checked
against SHA-256
`d75dd6b0be0aa10587fc95900cfd6ba7314d461a8276a81df34f009d0bfc075d`.
The project does not redistribute the file. Its upstream data license is not
separately declared, so the manifest records SPDX-style `NOASSERTION` instead
of inventing a license conclusion.

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

## Independent Libxc PBE point table

`fixtures/xc/libxc-7.1.2-pbe-unpolarized.json` records unpolarized PBE
exchange plus correlation values from Libxc 7.1.2: energy per particle,
the density derivative, and the squared-gradient derivative. It covers the
uniform-density limit and two finite-gradient points. The fixture is an
independent numerical oracle for the native PBE point kernel. PBE remains
unavailable as an input functional until the grid-level gradient/divergence
implementation and full-SCF comparisons pass.

The small generator in `tools/libxc_xc_reference.c` uses Libxc functional IDs
101 and 130. With Homebrew Libxc 7.1.2 installed, regenerate it with:

```sh
clang tools/libxc_xc_reference.c -I/opt/homebrew/include \
  -L/opt/homebrew/lib -lxc -o /tmp/libxc_xc_reference
/tmp/libxc_xc_reference
```

The generator is project code under MIT/Apache-2.0. The recorded Libxc-derived
reference data is identified as MPL-2.0 in the manifest.
