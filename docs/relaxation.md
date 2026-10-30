# Fixed-cell ionic relaxation

Version 0.11 added a fail-closed `relax` command that repeatedly evaluates the
schema-4 SCF energy and analytic forces while holding the lattice fixed. The
first accepted workload is the published displaced two-atom silicon LDA
fixture; this is not yet a general geometry-optimization claim.

Version 0.12 adds atomic relaxation checkpoint/restart with exact optimizer-
state restoration and CPU/Metal interruption-equivalence gates.

```sh
cargo run --release -- relax examples/si2-displaced.toml \
  --backend cpu \
  --output relaxation.json \
  --max-steps 12 \
  --force-tolerance-ry-bohr 1e-4 \
  --initial-step-bohr 0.05 \
  --maximum-step-bohr 0.15
```

Use `--features metal --backend metal` for the Apple-GPU path. The command
writes its complete result even when ionic convergence fails, then returns a
nonzero exit status. An unavailable Metal device or an unconverged initial SCF
is an error; there is no CPU fallback.

## Checkpoint and resume

Write a checkpoint after every accepted ionic step and deliberately pause after
one step with:

```sh
cargo run --release -- relax examples/si2-displaced.toml \
  --backend cpu \
  --output paused-relaxation.json \
  --checkpoint relaxation.chk \
  --checkpoint-every 1 \
  --stop-after 1 \
  --max-steps 12 \
  --force-tolerance-ry-bohr 1e-4 \
  --initial-step-bohr 0.01 \
  --maximum-step-bohr 0.01
```

The paused result is written before the command returns a nonzero status. Read
the compact checkpoint metadata without launching an SCF:

```sh
cargo run --release -- inspect-relaxation-checkpoint relaxation.chk
```

Resume with the same source revision, input, pseudopotentials, backend, and
optimizer options recorded in the checkpoint:

```sh
cargo run --release -- relax-resume examples/si2-displaced.toml relaxation.chk \
  --backend cpu \
  --output resumed-relaxation.json
```

`relax-resume` obtains the optimizer options from the checkpoint and replaces
the same checkpoint after each subsequent accepted step unless
`--checkpoint-output` selects another path. `--checkpoint-every` and
`--stop-after` have the same meaning as on `relax`.

The checkpoint is a schema-1 checksummed MessagePack envelope written through
an fsync-and-rename sequence. It stores the complete accepted history, current
input and SCF result, projected force, conjugate-gradient direction, proposed
step, cumulative elapsed time, and SCF evaluation count. A private binary wire
representation preserves every floating-point value exactly while supporting
the human-readable tagged input enums.

Resume fails closed before another SCF if the checksum, schema, source git
revision, initial input hash, pseudopotential hashes, requested backend, or
optimizer options differ. It also validates the restored history, force,
geometry, SCF provenance, counters, and immutable input fields. Checkpoints are
not promised to be portable across source revisions.

## Algorithm and coordinates

The optimizer uses a Polak-Ribiere nonlinear conjugate-gradient direction with
an Armijo backtracking line search. Proposed steps are limited by the largest
Cartesian atomic displacement in bohr. Cartesian displacements are converted
through the reciprocal lattice, so skew cells do not receive component-wise
fractional-coordinate approximations. Fractional positions are wrapped into
the periodic cell after each proposal.

A uniform Cartesian translation is projected out of the search direction. The
reported forces are never modified, and convergence uses the raw maximum
per-atom force norm—not the projected search force. This means a large net
force cannot be hidden by the translation projection. All atoms currently move;
there are no frozen-atom, collective-variable, symmetry, or other geometric
constraints.

Each line-search geometry receives a fresh, fully converged electronic SCF.
Rejected trials count toward `total_scf_evaluations`; if the terminal line
search fails, its unaccepted evaluations are recorded separately. Orbital and
density extrapolation between geometries and quasi-Newton Hessian persistence
are not implemented yet.

## Result contract

Relaxation schema 2 records:

- requested backend and exact relaxation options;
- every accepted geometry, total energy, raw and translation-projected force
  maxima, full Cartesian force array, SCF iterations, SCF input hash, line-
  search evaluation count, and accepted maximum displacement;
- total and terminal-unaccepted SCF evaluation counts;
- the complete final schema-4 `ScfResult`, including backend/device, precision,
  input and pseudopotential hashes, source revision, timings, eigenvalues,
  convergence history, energy terms, and forces;
- an explicit stop reason for every unconverged relaxation.
- cumulative elapsed time and, after a resume, the source checkpoint hash,
  accepted step, and writer revision.

Schema-1 relaxation JSON remains readable. It has no elapsed-time or restart-
provenance claim.

Validation rejects nonsequential histories, invalid numeric values, inconsistent
evaluation counts, missing final forces, backend mismatches, or convergence
claims that disagree with the raw final force.

## Interruption-equivalence gate

The release-mode acceptance test uses a `0.01 bohr` displacement cap so the
published silicon fixture requires three accepted ionic steps. It compares an
uninterrupted run with a run paused after step one and resumed from disk.

For both CPU FP64 and resident mixed-precision Metal, the complete accepted-step
history, final total energy, final forces, SCF input hashes, and cumulative SCF
evaluation count are bit-exact. The same test verifies that changed input atoms
and changed optimizer options are rejected. CI runs these gates explicitly;
the ordinary test suite keeps them ignored because each invokes multiple full
SCFs.

## Accepted displaced-silicon gate

The fixture begins with the second silicon atom displaced by `0.05 bohr` along
Cartesian x. With the command above, both CPU and Metal require three SCF
evaluations: the initial geometry, one rejected Armijo trial, and one accepted
`0.025 bohr` maximum-displacement step.

- CPU lowers the energy by `7.42759e-4 Ry` and finishes at a raw maximum force
  of `6.11e-6 Ry/bohr`.
- Metal lowers the energy by `7.42713e-4 Ry` and finishes at a raw maximum
  force of `6.17e-6 Ry/bohr`.
- Both recover the ideal relative fractional coordinate `[0.25, 0.25, 0.25]`;
  corresponding CPU/Metal final fractional components differ by less than
  `1.3e-8`.

The source-linked artifacts are
[`results/si2-displaced-cpu-relaxation.json`](../results/si2-displaced-cpu-relaxation.json)
and
[`results/si2-displaced-metal-relaxation.json`](../results/si2-displaced-metal-relaxation.json).
They were produced from source commit
`30940ac018202722cfdb7c5ef52356231fa4531f` on the documented Apple M5 Pro
host.

## Scientific boundary

Passing this fixture establishes only a fixed-cell norm-conserving Si2
relaxation. Before using relaxation for independent research, converge the
plane-wave cutoff, k mesh, SCF tolerances, and force threshold for the target
system and compare a representative displaced geometry with an independent
code. Stress, lattice relaxation, molecular dynamics, spin, constraints,
ultrasoft pseudopotentials, and PAW remain unsupported.
