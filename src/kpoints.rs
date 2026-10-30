use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

const KPOINT_TOLERANCE: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KPoint {
    /// Crystal momentum in reduced reciprocal coordinates, canonicalized to
    /// the half-open interval `[-0.5, 0.5)` on every axis.
    pub reduced: [f64; 3],
    /// Brillouin-zone integration weight. Resolved sets always sum to one.
    pub weight: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum KPointSamplingInput {
    Legacy(String),
    Detailed(KPointSpecification),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum KPointSpecification {
    Gamma,
    MonkhorstPack {
        grid: [usize; 3],
        #[serde(default)]
        shift: [f64; 3],
        #[serde(default = "default_true")]
        reduce_time_reversal: bool,
    },
    Explicit {
        points: Vec<KPoint>,
        #[serde(default)]
        reduce_time_reversal: bool,
    },
}

impl Default for KPointSamplingInput {
    fn default() -> Self {
        Self::Legacy("gamma".into())
    }
}

impl KPointSamplingInput {
    pub fn is_legacy_gamma(&self) -> bool {
        matches!(self, Self::Legacy(value) if value.eq_ignore_ascii_case("gamma"))
    }

    pub fn resolve(&self) -> Result<Vec<KPoint>> {
        match self {
            Self::Legacy(value) => {
                ensure!(
                    value.eq_ignore_ascii_case("gamma"),
                    "legacy k_points accepts only `gamma`; use the schema-2 table form for meshes or explicit points"
                );
                Ok(vec![KPoint {
                    reduced: [0.0; 3],
                    weight: 1.0,
                }])
            }
            Self::Detailed(KPointSpecification::Gamma) => Ok(vec![KPoint {
                reduced: [0.0; 3],
                weight: 1.0,
            }]),
            Self::Detailed(KPointSpecification::MonkhorstPack {
                grid,
                shift,
                reduce_time_reversal,
            }) => monkhorst_pack(*grid, *shift, *reduce_time_reversal),
            Self::Detailed(KPointSpecification::Explicit {
                points,
                reduce_time_reversal,
            }) => resolve_explicit(points, *reduce_time_reversal),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum OccupationScheme {
    #[default]
    Fixed,
    FermiDirac {
        smearing_ry: f64,
        #[serde(default = "default_extra_bands")]
        extra_bands: usize,
    },
}

impl OccupationScheme {
    pub fn validate(self) -> Result<()> {
        if let Self::FermiDirac {
            smearing_ry,
            extra_bands,
        } = self
        {
            ensure!(
                smearing_ry.is_finite() && smearing_ry > 0.0,
                "Fermi-Dirac smearing_ry must be positive and finite"
            );
            ensure!(
                extra_bands > 0,
                "Fermi-Dirac occupations require at least one extra band"
            );
        }
        Ok(())
    }

    pub fn band_count(self, electron_count: f64) -> Result<usize> {
        ensure!(
            electron_count.is_finite() && electron_count > 0.0,
            "electron count must be positive and finite"
        );
        match self {
            Self::Fixed => {
                let occupied = electron_count / 2.0;
                ensure!(
                    (occupied - occupied.round()).abs() <= KPOINT_TOLERANCE,
                    "fixed occupations require a closed-shell electron count"
                );
                Ok(occupied.round() as usize)
            }
            Self::FermiDirac { extra_bands, .. } => {
                Ok((electron_count / 2.0).ceil() as usize + extra_bands)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OccupationResult {
    pub occupations: Vec<Vec<f64>>,
    pub chemical_potential_ry: Option<f64>,
    /// Dimensionless entropy in units of `k_B`, including spin degeneracy and
    /// Brillouin-zone weights. The Mermin correction is `-smearing_ry * entropy`.
    pub entropy: f64,
    pub electron_count: f64,
}

pub fn assign_occupations(
    k_points: &[KPoint],
    eigenvalues_ry: &[Vec<f64>],
    electron_count: f64,
    scheme: OccupationScheme,
) -> Result<OccupationResult> {
    ensure!(!k_points.is_empty(), "k-point set is empty");
    ensure!(
        k_points.len() == eigenvalues_ry.len(),
        "k-point and eigenvalue set counts differ"
    );
    validate_resolved(k_points)?;
    scheme.validate()?;
    ensure!(
        eigenvalues_ry
            .iter()
            .all(|values| !values.is_empty() && values.iter().all(|value| value.is_finite())),
        "eigenvalue sets must be nonempty and finite"
    );

    match scheme {
        OccupationScheme::Fixed => {
            let occupied = scheme.band_count(electron_count)?;
            ensure!(
                eigenvalues_ry.iter().all(|values| values.len() >= occupied),
                "a k-point has too few bands for the fixed occupations"
            );
            let occupations: Vec<_> = eigenvalues_ry
                .iter()
                .map(|values| {
                    (0..values.len())
                        .map(|band| if band < occupied { 2.0 } else { 0.0 })
                        .collect::<Vec<_>>()
                })
                .collect();
            let integrated = integrated_electron_count(k_points, &occupations);
            ensure!(
                (integrated - electron_count).abs() <= 1.0e-10,
                "fixed occupations do not reproduce the requested electron count"
            );
            Ok(OccupationResult {
                occupations,
                chemical_potential_ry: None,
                entropy: 0.0,
                electron_count: integrated,
            })
        }
        OccupationScheme::FermiDirac { smearing_ry, .. } => {
            let capacity = 2.0
                * k_points
                    .iter()
                    .zip(eigenvalues_ry)
                    .map(|(point, values)| point.weight * values.len() as f64)
                    .sum::<f64>();
            ensure!(
                electron_count > 0.0 && electron_count < capacity - 1.0e-12,
                "Fermi-Dirac band capacity must exceed the requested electron count"
            );
            let minimum = eigenvalues_ry
                .iter()
                .flatten()
                .copied()
                .fold(f64::INFINITY, f64::min);
            let maximum = eigenvalues_ry
                .iter()
                .flatten()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let margin = (64.0 * smearing_ry).max(1.0);
            let mut low = minimum - margin;
            let mut high = maximum + margin;
            for _ in 0..200 {
                let midpoint = 0.5 * (low + high);
                let count = fermi_count(k_points, eigenvalues_ry, midpoint, smearing_ry);
                if count < electron_count {
                    low = midpoint;
                } else {
                    high = midpoint;
                }
            }
            let chemical_potential = 0.5 * (low + high);
            let occupations: Vec<Vec<f64>> = eigenvalues_ry
                .iter()
                .map(|values| {
                    values
                        .iter()
                        .map(|&energy| {
                            2.0 * fermi_probability(energy, chemical_potential, smearing_ry)
                        })
                        .collect()
                })
                .collect();
            let integrated = integrated_electron_count(k_points, &occupations);
            ensure!(
                (integrated - electron_count).abs() <= 1.0e-10,
                "Fermi-Dirac occupations missed the electron-count target"
            );
            let entropy = k_points
                .iter()
                .zip(&occupations)
                .map(|(point, values)| {
                    point.weight
                        * values
                            .iter()
                            .map(|&occupation| {
                                let probability = 0.5 * occupation;
                                if probability <= f64::EPSILON || probability >= 1.0 - f64::EPSILON
                                {
                                    0.0
                                } else {
                                    -2.0 * (probability * probability.ln()
                                        + (1.0 - probability) * (1.0 - probability).ln())
                                }
                            })
                            .sum::<f64>()
                })
                .sum();
            Ok(OccupationResult {
                occupations,
                chemical_potential_ry: Some(chemical_potential),
                entropy,
                electron_count: integrated,
            })
        }
    }
}

fn default_true() -> bool {
    true
}

fn default_extra_bands() -> usize {
    4
}

fn wrap_reduced(value: f64) -> f64 {
    let wrapped = value - (value + 0.5).floor();
    if wrapped.abs() <= KPOINT_TOLERANCE {
        0.0
    } else if (wrapped + 0.5).abs() <= KPOINT_TOLERANCE {
        -0.5
    } else {
        wrapped
    }
}

fn canonicalize(reduced: [f64; 3]) -> [f64; 3] {
    reduced.map(wrap_reduced)
}

fn same_reduced(left: [f64; 3], right: [f64; 3]) -> bool {
    left.iter()
        .zip(right)
        .all(|(&a, b)| wrap_reduced(a - b).abs() <= KPOINT_TOLERANCE)
}

fn validate_resolved(points: &[KPoint]) -> Result<()> {
    ensure!(!points.is_empty(), "k-point set is empty");
    ensure!(
        points.iter().all(|point| {
            point.reduced.iter().all(|value| value.is_finite())
                && point.weight.is_finite()
                && point.weight > 0.0
        }),
        "k-point coordinates and weights must be finite and weights must be positive"
    );
    let weight_sum: f64 = points.iter().map(|point| point.weight).sum();
    ensure!(
        (weight_sum - 1.0).abs() <= 1.0e-12,
        "k-point weights must sum to one (got {weight_sum:.16})"
    );
    for (index, point) in points.iter().enumerate() {
        ensure!(
            points[..index]
                .iter()
                .all(|other| !same_reduced(other.reduced, point.reduced)),
            "k-point set contains duplicate reduced coordinates"
        );
    }
    Ok(())
}

fn monkhorst_pack(
    grid: [usize; 3],
    shift: [f64; 3],
    reduce_time_reversal: bool,
) -> Result<Vec<KPoint>> {
    ensure!(
        grid.iter().all(|&size| size > 0),
        "Monkhorst-Pack grid dimensions must be positive"
    );
    ensure!(
        shift
            .iter()
            .all(|value| value.is_finite() && (0.0..1.0).contains(value)),
        "Monkhorst-Pack shifts must be finite and in [0, 1)"
    );
    let count = grid
        .iter()
        .try_fold(1_usize, |product, &size| product.checked_mul(size))
        .ok_or_else(|| anyhow::anyhow!("Monkhorst-Pack grid is too large"))?;
    let weight = 1.0 / count as f64;
    let mut points = Vec::with_capacity(count);
    for i in 0..grid[0] {
        for j in 0..grid[1] {
            for k in 0..grid[2] {
                let indices = [i, j, k];
                let reduced = std::array::from_fn(|axis| {
                    wrap_reduced((indices[axis] as f64 + shift[axis]) / grid[axis] as f64)
                });
                points.push(KPoint { reduced, weight });
            }
        }
    }
    finalize_points(points, reduce_time_reversal)
}

fn resolve_explicit(points: &[KPoint], reduce_time_reversal: bool) -> Result<Vec<KPoint>> {
    let canonical: Vec<_> = points
        .iter()
        .map(|point| KPoint {
            reduced: canonicalize(point.reduced),
            weight: point.weight,
        })
        .collect();
    validate_resolved(&canonical)?;
    finalize_points(canonical, reduce_time_reversal)
}

fn finalize_points(mut points: Vec<KPoint>, reduce_time_reversal: bool) -> Result<Vec<KPoint>> {
    if reduce_time_reversal {
        let mut used = vec![false; points.len()];
        let mut reduced = Vec::new();
        for index in 0..points.len() {
            if used[index] {
                continue;
            }
            used[index] = true;
            let opposite = canonicalize(points[index].reduced.map(|value| -value));
            let pair = ((index + 1)..points.len()).find(|&candidate| {
                !used[candidate] && same_reduced(points[candidate].reduced, opposite)
            });
            let mut point = points[index];
            if let Some(pair) = pair {
                used[pair] = true;
                point.weight += points[pair].weight;
            }
            reduced.push(point);
        }
        points = reduced;
    }
    let weight_sum: f64 = points.iter().map(|point| point.weight).sum();
    for point in &mut points {
        point.weight /= weight_sum;
    }
    points.sort_by(|left, right| {
        left.reduced
            .iter()
            .zip(right.reduced)
            .find_map(|(&a, b)| {
                let ordering = a.total_cmp(&b);
                ordering.ne(&std::cmp::Ordering::Equal).then_some(ordering)
            })
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    validate_resolved(&points)?;
    Ok(points)
}

fn fermi_probability(energy: f64, chemical_potential: f64, smearing: f64) -> f64 {
    let argument = (energy - chemical_potential) / smearing;
    if argument >= 40.0 {
        (-argument).exp()
    } else if argument <= -40.0 {
        1.0 - argument.exp()
    } else {
        1.0 / (1.0 + argument.exp())
    }
}

fn fermi_count(
    k_points: &[KPoint],
    eigenvalues: &[Vec<f64>],
    chemical_potential: f64,
    smearing: f64,
) -> f64 {
    k_points
        .iter()
        .zip(eigenvalues)
        .map(|(point, values)| {
            2.0 * point.weight
                * values
                    .iter()
                    .map(|&energy| fermi_probability(energy, chemical_potential, smearing))
                    .sum::<f64>()
        })
        .sum()
}

fn integrated_electron_count(k_points: &[KPoint], occupations: &[Vec<f64>]) -> f64 {
    k_points
        .iter()
        .zip(occupations)
        .map(|(point, values)| point.weight * values.iter().sum::<f64>())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monkhorst_pack_weights_and_time_reversal_are_exact() {
        let full = monkhorst_pack([2, 2, 2], [0.5; 3], false).unwrap();
        assert_eq!(full.len(), 8);
        assert!((full.iter().map(|point| point.weight).sum::<f64>() - 1.0).abs() < 1.0e-15);
        assert!(
            full.iter()
                .all(|point| (point.weight - 0.125).abs() < 1.0e-15)
        );

        let reduced = monkhorst_pack([2, 2, 2], [0.5; 3], true).unwrap();
        assert_eq!(reduced.len(), 4);
        assert!(
            reduced
                .iter()
                .all(|point| (point.weight - 0.25).abs() < 1.0e-15)
        );
    }

    #[test]
    fn zone_boundary_points_are_self_time_reversal_partners() {
        let points = monkhorst_pack([2, 1, 1], [0.0; 3], true).unwrap();
        assert_eq!(points.len(), 2);
        assert_eq!(points[0].reduced, [-0.5, 0.0, 0.0]);
        assert_eq!(points[1].reduced, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn explicit_weights_fail_closed() {
        let invalid = KPointSamplingInput::Detailed(KPointSpecification::Explicit {
            points: vec![KPoint {
                reduced: [0.0; 3],
                weight: 0.75,
            }],
            reduce_time_reversal: false,
        });
        assert!(invalid.resolve().is_err());
    }

    #[test]
    fn fixed_and_fermi_dirac_occupations_conserve_charge() {
        let points = monkhorst_pack([2, 1, 1], [0.5, 0.0, 0.0], false).unwrap();
        let eigenvalues = vec![vec![-1.0, 0.2, 0.8], vec![-0.8, 0.1, 0.9]];
        let fixed =
            assign_occupations(&points, &eigenvalues, 2.0, OccupationScheme::Fixed).unwrap();
        assert_eq!(fixed.occupations, vec![vec![2.0, 0.0, 0.0]; 2]);
        assert!((fixed.electron_count - 2.0).abs() < 1.0e-12);

        let smeared = assign_occupations(
            &points,
            &eigenvalues,
            3.0,
            OccupationScheme::FermiDirac {
                smearing_ry: 0.05,
                extra_bands: 2,
            },
        )
        .unwrap();
        assert!((smeared.electron_count - 3.0).abs() < 1.0e-10);
        assert!(smeared.chemical_potential_ry.is_some());
        assert!(smeared.entropy > 0.0);
        assert!(
            smeared
                .occupations
                .iter()
                .flatten()
                .all(|&value| (0.0..=2.0).contains(&value))
        );
    }
}
