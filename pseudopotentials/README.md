# Pseudopotential fixture

The silicon examples use two pinned norm-conserving pseudopotentials. The files
are intentionally not redistributed by this project. Download them from the
repository root:

```sh
curl --fail --location \
  https://gitlab.com/QEF/q-e/-/raw/qe-7.6/EPW/examples/sic/pp/Si.pz-vbc.UPF \
  --output pseudopotentials/Si.pz-vbc.UPF
echo 'd75dd6b0be0aa10587fc95900cfd6ba7314d461a8276a81df34f009d0bfc075d  pseudopotentials/Si.pz-vbc.UPF' \
  | shasum -a 256 --check

curl --fail --location \
  https://raw.githubusercontent.com/wannier-developers/wannier90/1d6b187374a2d50b509e5e79e2cab01a79ff7ce1/pseudo/Si_ONCV_PBE-1.0.upf \
  --output pseudopotentials/Si_ONCV_PBE-1.0.upf
echo '481a5fae1167fd712d4e8e2e4df33f8bfc1e7d2abbc0565290d39fbba9dccb41  pseudopotentials/Si_ONCV_PBE-1.0.upf' \
  | shasum -a 256 --check
```

`Si.pz-vbc.UPF` supplies the LDA-PZ cases. `Si_ONCV_PBE-1.0.upf` is the
CC-BY-SA-4.0 Schlipf-Gygi/ONCV PBE fixture at the exact Wannier90 revision
pinned by Quantum ESPRESSO 7.6.

The solver records this SHA-256 in every result and rejects unsupported
pseudopotential types and functionals rather than substituting another model.
The source URL, upstream revision, redistribution policy, and checksum are also
recorded in [`../fixtures/manifest.toml`](../fixtures/manifest.toml) and checked
by `rust-pw-dft verify-fixtures` in CI.
