# K-point sampling and occupations

Input schema 2 adds periodic Brillouin-zone sampling to the CPU-FP64 and
resident mixed-precision Metal solvers.
The legacy schema-1 `k_points = "gamma"` form remains accepted unchanged.

## Monkhorst-Pack meshes

```toml
schema_version = 2

[k_points]
kind = "monkhorst-pack"
grid = [2, 2, 2]
shift = [0.5, 0.5, 0.5]
reduce_time_reversal = true

[occupations]
kind = "fixed"
```

For axis size `N`, integer index `i`, and shift `s`, the reduced coordinate is
`(i + s) / N`, wrapped into the half-open interval `[-0.5, 0.5)`. Shifts must
be finite and lie in `[0, 1)`. This convention maps the example above to the
same eight-point mesh as QE's `K_POINTS automatic` line `2 2 2 1 1 1`.

Every unreduced point starts with weight `1 / (N1 N2 N3)`. Optional time-
reversal reduction joins exact `k`/`-k` partners and adds their weights;
self-inverse points such as Gamma and reciprocal-zone boundaries retain their
own weight. Resolved points are sorted deterministically and weights must sum
to one.

## Explicit points

```toml
[k_points]
kind = "explicit"
reduce_time_reversal = false
points = [
  { reduced = [0.0, 0.0, 0.0], weight = 0.5 },
  { reduced = [0.25, 0.25, 0.25], weight = 0.5 },
]
```

Coordinates are reduced reciprocal-lattice coefficients, not Cartesian
vectors. Weights must be positive, finite, unique after reciprocal wrapping,
and sum to one. Invalid or duplicate inputs are rejected.

## Occupations

Fixed occupations require an even closed-shell electron count. Every k point
uses the same number of doubly occupied bands, and the weighted charge is
checked after density construction.

Fermi-Dirac occupations are configured as:

```toml
[occupations]
kind = "fermi-dirac"
smearing_ry = 0.01
extra_bands = 8
```

`smearing_ry` is `k_B T` in Rydberg. A single chemical potential is solved
across all points and bands so the weighted electron count matches the ionic
valence charge. Results report the chemical potential, dimensionless spin-
degenerate entropy, and Mermin free-energy correction `-smearing_ry * entropy`.
At least one extra band is required. Users must increase `extra_bands` until
the highest bands have negligible occupation; automatic band-convergence is
not yet implemented.

## Current acceptance boundary

The CPU-FP64 shifted 2x2x2 silicon PBE fixed-occupation fixture and shifted
4x4x4 fcc aluminum LDA-PZ Fermi-Dirac fixture are independently accepted
against QE 7.6. The corresponding Metal calculations are accepted against the
CPU results at the documented mixed-precision thresholds. Metal creates one
persistent Hamiltonian and resident Davidson context per irreducible point;
the contexts are currently scheduled sequentially. These establish only the
exact published workloads; users must still perform cutoff, band-count,
smearing, and k-mesh convergence studies. Multi-k-point checkpoint/resume is
rejected explicitly, and the program never substitutes a Gamma calculation or
silently falls back to CPU.
