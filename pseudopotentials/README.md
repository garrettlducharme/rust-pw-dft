# Pseudopotential fixture

The silicon examples use Quantum ESPRESSO 7.6's legacy norm-conserving
`Si.pz-vbc.UPF`. The file is intentionally not redistributed by this project.
Download the pinned upstream fixture from the repository root:

```sh
curl --fail --location \
  https://gitlab.com/QEF/q-e/-/raw/qe-7.6/EPW/examples/sic/pp/Si.pz-vbc.UPF \
  --output pseudopotentials/Si.pz-vbc.UPF
echo 'd75dd6b0be0aa10587fc95900cfd6ba7314d461a8276a81df34f009d0bfc075d  pseudopotentials/Si.pz-vbc.UPF' \
  | shasum -a 256 --check
```

The solver records this SHA-256 in every result and rejects unsupported
pseudopotential types and functionals rather than substituting another model.
The source URL, upstream revision, redistribution policy, and checksum are also
recorded in [`../fixtures/manifest.toml`](../fixtures/manifest.toml) and checked
by `rust-pw-dft verify-fixtures` in CI.
