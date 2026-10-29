# SCF checkpoints

`rust-pw-dft` 0.9 writes backend-neutral SCF checkpoints at the boundary after
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

The version-3 payload contains:

- exact input and pseudopotential SHA-256 hashes;
- writer Git revision, requested backend, precision policy, and run profile;
- completed SCF iteration and FFT shape;
- an ordered record for every irreducible k point: exact reduced coordinates
  and weight, plane-wave and band counts, and CPU complex-FP64 orbital blocks;
- mixed CPU-FP64 density and previous total energy;
- cumulative Davidson/Hamiltonian counts, phase timings, elapsed time, and
  Metal transfer/operation counters;
- the complete structured SCF convergence history through the checkpoint
  iteration.

The file starts with the `RPDFTCK3` magic, a little-endian envelope version and
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
- FFT grid, k-point count/order/coordinates/weights, per-point plane-wave and
  band counts, array shapes, or numeric finiteness;
- iteration bounds or unsupported checkpoint size.

Source revision matching is deliberately strict in version 3. Schema-2 files
are rejected rather than guessed or migrated. A future schema may permit
migrations, but silently carrying numerical state across code changes would
weaken reproducibility.

Local and CI acceptance tests interrupt and resume both the pinned Gamma Si2
and shifted 2x2x2 Si2 calculations on CPU FP64 and resident Metal. They require
the same final SCF iteration, complete history, per-k-point eigenvalues, and
energies within the active precision policy. The Metal tests also require
cumulative full-vector transfer and projection-resynchronization accounting to
match an uninterrupted run. A tampered k-point coordinate must fail closed.
