# Pseudopotential fixture

The validation examples use five pinned norm-conserving pseudopotentials. The files
are intentionally not redistributed by this project. Download them from the
repository root:

```sh
curl --fail --location \
  https://gitlab.com/QEF/q-e/-/raw/qe-7.6/EPW/examples/sic/pp/Si.pz-vbc.UPF \
  --output pseudopotentials/Si.pz-vbc.UPF
echo 'd75dd6b0be0aa10587fc95900cfd6ba7314d461a8276a81df34f009d0bfc075d  pseudopotentials/Si.pz-vbc.UPF' \
  | shasum -a 256 --check

curl --fail --location \
  https://pseudopotentials.quantumespresso.org/upf_files/Al.pz-vbc.UPF \
  --output pseudopotentials/Al.pz-vbc.UPF
echo '4eab06b63f87f07ede2d5a193e6d993a09107167fd6b8647afa807342501d6e5  pseudopotentials/Al.pz-vbc.UPF' \
  | shasum -a 256 --check

curl --fail --location \
  https://raw.githubusercontent.com/wannier-developers/wannier90/1d6b187374a2d50b509e5e79e2cab01a79ff7ce1/pseudo/Si_ONCV_PBE-1.0.upf \
  --output pseudopotentials/Si_ONCV_PBE-1.0.upf
echo '481a5fae1167fd712d4e8e2e4df33f8bfc1e7d2abbc0565290d39fbba9dccb41  pseudopotentials/Si_ONCV_PBE-1.0.upf' \
  | shasum -a 256 --check

curl --fail --location \
  https://raw.githubusercontent.com/anharmonic/d3q/6e5f97816af439702196c248ec2ad0092c561b89/Examples/MgO/UPF/Mg_ONCV_PBE-1.0.UPF \
  --output pseudopotentials/Mg_ONCV_PBE-1.0.UPF
echo '7d19fe11bd630b5e4092c8fa884d11d76731386fd36ea0cdd999d723a6330e80  pseudopotentials/Mg_ONCV_PBE-1.0.UPF' \
  | shasum -a 256 --check

curl --fail --location \
  https://raw.githubusercontent.com/anharmonic/d3q/6e5f97816af439702196c248ec2ad0092c561b89/Examples/MgO/UPF/O_ONCV_PBE-1.0.UPF \
  --output pseudopotentials/O_ONCV_PBE-1.0.UPF
echo '374452e85db23c51d33c3f69a541d6be6a74c6eed080a0545b06ac697c8b7cad  pseudopotentials/O_ONCV_PBE-1.0.UPF' \
  | shasum -a 256 --check
```

`Si.pz-vbc.UPF` supplies the LDA-PZ cases. `Si_ONCV_PBE-1.0.upf` is the
CC-BY-SA-4.0 Schlipf-Gygi/ONCV PBE fixture at the exact Wannier90 revision
pinned by Quantum ESPRESSO 7.6.

`Al.pz-vbc.UPF` is a legacy QE LDA-PZ fixture used only to validate the
fractional-occupation path. QE's upstream table keeps this collection for
reference and does not recommend it for new production work. Its data license
is recorded as `NOASSERTION`, and the file is not redistributed.

The MgO workload uses the ten-electron Mg and six-electron O ONCV PBE files
from the `d3q` revision pinned by QE 7.6. Their embedded metadata does not
declare a separate data license, so they remain unredistributed and are
recorded as `NOASSERTION` rather than assigning an unsupported license.

The solver records this SHA-256 in every result and rejects unsupported
pseudopotential types and functionals rather than substituting another model.
The source URL, upstream revision, redistribution policy, and checksum are also
recorded in [`../fixtures/manifest.toml`](../fixtures/manifest.toml) and checked
by `rust-pw-dft verify-fixtures` in CI.
