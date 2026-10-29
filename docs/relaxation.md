# Fixed-cell ionic relaxation

Version 0.11 adds a fail-closed `relax` command that repeatedly evaluates the
schema-4 SCF energy and analytic forces while holding the lattice fixed. The
first accepted workload is the published displaced two-atom silicon LDA
fixture; this is not yet a general geometry-optimization claim.

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
density extrapolation between geometries, ionic checkpoint/restart, and
quasi-Newton Hessian persistence are not implemented yet.

## Result contract

Relaxation schema 1 records:

- requested backend and exact relaxation options;
- every accepted geometry, total energy, raw and translation-projected force
  maxima, full Cartesian force array, SCF iterations, SCF input hash, line-
  search evaluation count, and accepted maximum displacement;
- total and terminal-unaccepted SCF evaluation counts;
- the complete final schema-4 `ScfResult`, including backend/device, precision,
  input and pseudopotential hashes, source revision, timings, eigenvalues,
  convergence history, energy terms, and forces;
- an explicit stop reason for every unconverged relaxation.

Validation rejects nonsequential histories, invalid numeric values, inconsistent
evaluation counts, missing final forces, backend mismatches, or convergence
claims that disagree with the raw final force.

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
