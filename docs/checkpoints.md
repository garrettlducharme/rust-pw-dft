# SCF checkpoints

`rust-pw-dft` 0.2 writes backend-neutral SCF checkpoints at the boundary after
an eigensolve, density construction, energy evaluation, and density mixing.
At this point both CPU and Metal calculations have a CPU-FP64 density and
orbital set suitable for deterministic continuation. No live Metal object or
opaque framework serialization is stored.

## Commands

Pause deliberately after ten SCF iterations while atomically maintaining a
checkpoint:

```sh
rust-pw-dft run input.toml --backend metal --output paused.json \
  --checkpoint calculation.chk --checkpoint-every 2 --stop-after 10
```

The command writes `paused.json` and exits unsuccessfully because no converged
scientific result exists yet. Inspect the checkpoint without loading its large
arrays:

```sh
rust-pw-dft inspect-checkpoint calculation.chk
```

Continue it, replacing the checkpoint periodically until convergence:

```sh
rust-pw-dft resume input.toml calculation.chk --backend metal \
  --output result.json --checkpoint-every 2
```

`--checkpoint-output` may be used to preserve the input checkpoint and write
continuation state elsewhere.

## Stored state

The version-1 payload contains:

- exact input and pseudopotential SHA-256 hashes;
- writer Git revision, requested backend, precision policy, and run profile;
- completed SCF iteration, FFT shape, plane-wave count, and occupied-state
  count;
- mixed CPU-FP64 density, previous total energy, and CPU complex-FP64 orbitals;
- cumulative Davidson/Hamiltonian counts, phase timings, elapsed time, and
  Metal transfer/operation counters.

The file starts with the `RPDFTCK1` magic, a little-endian envelope version and
payload length, and a SHA-256 digest of a bincode/Serde payload. Files larger
than 16 GiB are rejected before decoding.

## Atomicity and validation

Checkpoint output is written to a unique file in the destination directory,
flushed with `sync_all`, and renamed over the destination. Keeping the
temporary file on the same filesystem makes replacement atomic on supported
macOS and Linux filesystems. A failed write removes only that temporary file.

Resume rejects a checkpoint if any of these differ or are invalid:

- envelope or payload schema, magic, length, checksum, or trailing bytes;
- source revision, input hash, or pseudopotential hashes;
- backend, precision policy, or scientific/benchmark run profile;
- FFT grid, plane-wave basis, occupied-state count, array shapes, or numeric
  finiteness;
- iteration bounds or unsupported checkpoint size.

Source revision matching is deliberately strict in version 1. A future schema
may permit migrations, but silently carrying numerical state across code
changes would weaken reproducibility.

Local acceptance tests interrupt and resume the pinned Si2 calculation on both
CPU FP64 and resident Metal. They require the same final SCF iteration and
energies/eigenvalues within the active precision policy. The Metal test also
requires cumulative full-vector transfer accounting to match an uninterrupted
run.
