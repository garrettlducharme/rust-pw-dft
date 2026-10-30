use crate::backend::{BackendKind, PrecisionPolicy};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const RESULT_SCHEMA_VERSION: u32 = 5;
const MINIMUM_RESULT_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KPointResult {
    pub reduced: [f64; 3],
    pub weight: f64,
    pub eigenvalues_ry: Vec<f64>,
    pub occupations: Vec<f64>,
    pub residuals: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpinChannelResult {
    pub electron_count: f64,
    pub k_points: Vec<KPointResult>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CollinearSpinResult {
    pub magnetization_electrons: f64,
    pub up: SpinChannelResult,
    pub down: SpinChannelResult,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ScfIterationStatus {
    Continue,
    Converged,
    EigensolverNotConverged,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScfIterationRecord {
    pub iteration: usize,
    pub status: ScfIterationStatus,
    pub total_energy_ry: Option<f64>,
    pub energy_change_ry: Option<f64>,
    pub density_residual: Option<f64>,
    pub maximum_orbital_residual_ry: f64,
    pub electron_count: Option<f64>,
    pub highest_occupied_eigenvalue_ry: Option<f64>,
    pub davidson_iterations: usize,
    pub hamiltonian_applications: usize,
}

pub(crate) fn validate_scf_history(
    history: &[ScfIterationRecord],
    expected_length: Option<usize>,
) -> Result<()> {
    if let Some(expected) = expected_length {
        ensure!(
            history.len() == expected,
            "complete SCF history length differs from iterations"
        );
    }
    for (index, record) in history.iter().enumerate() {
        ensure!(
            record.iteration == index + 1,
            "SCF history is not sequential"
        );
        ensure!(
            record.total_energy_ry.is_none_or(|value| value.is_finite())
                && record
                    .energy_change_ry
                    .is_none_or(|value| value.is_finite() && value >= 0.0)
                && record
                    .density_residual
                    .is_none_or(|value| value.is_finite() && value >= 0.0)
                && record.maximum_orbital_residual_ry.is_finite()
                && record.maximum_orbital_residual_ry >= 0.0
                && record.electron_count.is_none_or(|value| value.is_finite())
                && record
                    .highest_occupied_eigenvalue_ry
                    .is_none_or(|value| value.is_finite()),
            "SCF history contains invalid numeric values"
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScfResult {
    pub schema_version: u32,
    pub converged: bool,
    pub iterations: usize,
    #[serde(default)]
    pub atom_count: usize,
    pub total_energy_ry: Option<f64>,
    pub energy_terms: BTreeMap<String, f64>,
    pub eigenvalues_ry: Vec<f64>,
    pub electron_count: Option<f64>,
    pub residuals: Vec<f64>,
    /// Cartesian Hellmann-Feynman forces in Rydberg per bohr, ordered exactly
    /// like the input atoms. Schema-2/3 records predate analytic forces.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forces_ry_bohr: Option<Vec<[f64; 3]>>,
    /// Per-k-point band data. Schema-2 Gamma-only records predate this field
    /// and deserialize to an empty vector.
    #[serde(default)]
    pub k_points: Vec<KPointResult>,
    /// Spin-resolved scientific data. Schema-5 results use this field instead
    /// of the ambiguous spin-degenerate top-level eigenvalue and k-point arrays.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spin: Option<CollinearSpinResult>,
    pub history: Vec<ScfIterationRecord>,
    pub history_complete: bool,
    pub timings: BTreeMap<String, f64>,
    pub provenance: Provenance,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
}

fn validate_spin_channel(channel: &SpinChannelResult) -> Result<()> {
    ensure!(
        channel.electron_count.is_finite() && channel.electron_count >= 0.0,
        "spin-channel electron count is invalid"
    );
    ensure!(
        !channel.k_points.is_empty(),
        "spin-channel k-point data is empty"
    );
    let weight_sum: f64 = channel.k_points.iter().map(|point| point.weight).sum();
    ensure!(
        (weight_sum - 1.0).abs() <= 1.0e-12,
        "spin-channel k-point weights do not sum to one"
    );
    for point in &channel.k_points {
        ensure!(
            point.reduced.iter().all(|value| value.is_finite())
                && point.weight.is_finite()
                && point.weight > 0.0
                && point.eigenvalues_ry.len() == point.occupations.len()
                && (point.residuals.is_empty()
                    || point.eigenvalues_ry.len() == point.residuals.len())
                && point.eigenvalues_ry.iter().all(|value| value.is_finite())
                && point
                    .occupations
                    .iter()
                    .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
                && point
                    .residuals
                    .iter()
                    .all(|value| value.is_finite() && *value >= 0.0),
            "spin-channel result contains invalid per-k-point data"
        );
    }
    let integrated: f64 = channel
        .k_points
        .iter()
        .map(|point| point.weight * point.occupations.iter().sum::<f64>())
        .sum();
    ensure!(
        (integrated - channel.electron_count).abs() <= 1.0e-8,
        "spin-channel occupations disagree with its electron count"
    );
    Ok(())
}

fn validate_spin_result(spin: &CollinearSpinResult, total_electrons: Option<f64>) -> Result<()> {
    validate_spin_channel(&spin.up)?;
    validate_spin_channel(&spin.down)?;
    ensure!(
        spin.up.k_points.len() == spin.down.k_points.len()
            && spin
                .up
                .k_points
                .iter()
                .zip(&spin.down.k_points)
                .all(|(up, down)| up.reduced == down.reduced
                    && (up.weight - down.weight).abs() <= 1.0e-15),
        "spin channels use different k-point sets"
    );
    ensure!(
        spin.magnetization_electrons.is_finite()
            && (spin.magnetization_electrons - (spin.up.electron_count - spin.down.electron_count))
                .abs()
                <= 1.0e-8,
        "spin magnetization disagrees with channel electron counts"
    );
    ensure!(
        total_electrons.is_some_and(|total| {
            total.is_finite()
                && (total - spin.up.electron_count - spin.down.electron_count).abs() <= 1.0e-8
        }),
        "total electron count disagrees with spin channels"
    );
    Ok(())
}

impl ScfResult {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (MINIMUM_RESULT_SCHEMA_VERSION..=RESULT_SCHEMA_VERSION).contains(&self.schema_version),
            "unsupported result schema_version {}; expected {} through {}",
            self.schema_version,
            MINIMUM_RESULT_SCHEMA_VERSION,
            RESULT_SCHEMA_VERSION
        );
        if self.schema_version < 5 {
            ensure!(
                self.spin.is_none(),
                "result schemas before 5 cannot contain spin-resolved data"
            );
        }
        if let Some(spin) = &self.spin {
            ensure!(
                self.schema_version == 5
                    && self.k_points.is_empty()
                    && self.eigenvalues_ry.is_empty()
                    && self.residuals.is_empty(),
                "spin-resolved results require schema 5 and unambiguous channel arrays"
            );
            validate_spin_result(spin, self.electron_count)?;
        } else if self.schema_version == 2 {
            ensure!(
                self.k_points.is_empty(),
                "schema-2 results cannot contain per-k-point data"
            );
        } else {
            ensure!(
                !self.k_points.is_empty(),
                "schema-3 and newer results require per-k-point data"
            );
            let weight_sum: f64 = self.k_points.iter().map(|point| point.weight).sum();
            ensure!(
                (weight_sum - 1.0).abs() <= 1.0e-12,
                "result k-point weights do not sum to one"
            );
            for point in &self.k_points {
                ensure!(
                    point.reduced.iter().all(|value| value.is_finite())
                        && point.weight.is_finite()
                        && point.weight > 0.0
                        && point.eigenvalues_ry.len() == point.occupations.len()
                        && (point.residuals.is_empty()
                            || point.eigenvalues_ry.len() == point.residuals.len())
                        && point.eigenvalues_ry.iter().all(|value| value.is_finite())
                        && point
                            .occupations
                            .iter()
                            .all(|value| value.is_finite() && (0.0..=2.0).contains(value))
                        && point
                            .residuals
                            .iter()
                            .all(|value| value.is_finite() && *value >= 0.0),
                    "result contains invalid per-k-point data"
                );
            }
        }
        ensure!(self.atom_count > 0, "result atom_count must be positive");
        if self.schema_version < 4 {
            ensure!(
                self.forces_ry_bohr.is_none(),
                "result schemas before 4 cannot contain forces"
            );
        } else if self.converged {
            let forces = self
                .forces_ry_bohr
                .as_ref()
                .context("converged schema-4 result requires forces")?;
            ensure!(
                forces.len() == self.atom_count
                    && forces.iter().flatten().all(|value| value.is_finite()),
                "result forces are invalid"
            );
        } else {
            ensure!(
                self.forces_ry_bohr.is_none(),
                "unconverged result cannot contain scientific forces"
            );
        }
        ensure!(
            self.timings
                .values()
                .all(|value| value.is_finite() && *value >= 0.0),
            "result timings are invalid"
        );
        ensure!(
            self.energy_terms.values().all(|value| value.is_finite()),
            "result energy terms are invalid"
        );
        ensure!(
            self.eigenvalues_ry.iter().all(|value| value.is_finite())
                && self
                    .residuals
                    .iter()
                    .all(|value| value.is_finite() && *value >= 0.0)
                && self.electron_count.is_none_or(|value| value.is_finite()),
            "result scientific arrays contain invalid values"
        );
        ensure!(
            self.total_energy_ry.is_none_or(|value| value.is_finite()),
            "result total energy is invalid"
        );
        validate_scf_history(
            &self.history,
            self.history_complete.then_some(self.iterations),
        )?;
        if self.history_complete {
            ensure!(!self.history.is_empty(), "complete SCF history is empty");
            let final_record = self.history.last().unwrap();
            ensure!(
                self.converged == (final_record.status == ScfIterationStatus::Converged),
                "result convergence disagrees with SCF history"
            );
            if final_record.status != ScfIterationStatus::EigensolverNotConverged {
                ensure!(
                    final_record.total_energy_ry == self.total_energy_ry
                        && final_record.electron_count == self.electron_count,
                    "final scientific values disagree with SCF history"
                );
            }
            if let Some(value) = self.energy_terms.get("davidson_iterations") {
                let recorded: usize = self
                    .history
                    .iter()
                    .map(|entry| entry.davidson_iterations)
                    .sum();
                ensure!(*value == recorded as f64, "Davidson history total differs");
            }
            if let Some(value) = self.energy_terms.get("hamiltonian_applications") {
                let recorded: usize = self
                    .history
                    .iter()
                    .map(|entry| entry.hamiltonian_applications)
                    .sum();
                ensure!(
                    *value == recorded as f64,
                    "Hamiltonian history total differs"
                );
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RestartProvenance {
    pub checkpoint_sha256: String,
    pub checkpoint_iteration: usize,
    pub writer_git_revision: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    pub requested_backend: BackendKind,
    pub actual_backend: BackendKind,
    pub actual_device: String,
    pub precision_policy: PrecisionPolicy,
    pub input_sha256: String,
    pub pseudopotential_sha256: BTreeMap<String, String>,
    pub git_revision: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub restart: Option<RestartProvenance>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn failed_result() -> ScfResult {
        ScfResult {
            schema_version: RESULT_SCHEMA_VERSION,
            converged: false,
            iterations: 1,
            atom_count: 1,
            total_energy_ry: Some(-1.0),
            energy_terms: BTreeMap::from([
                ("davidson_iterations".into(), 3.0),
                ("hamiltonian_applications".into(), 5.0),
            ]),
            eigenvalues_ry: vec![0.25],
            electron_count: Some(2.0),
            residuals: vec![0.1],
            forces_ry_bohr: None,
            k_points: vec![KPointResult {
                reduced: [0.0; 3],
                weight: 1.0,
                eigenvalues_ry: vec![0.25],
                occupations: vec![2.0],
                residuals: vec![0.1],
            }],
            spin: None,
            history: vec![ScfIterationRecord {
                iteration: 1,
                status: ScfIterationStatus::EigensolverNotConverged,
                total_energy_ry: None,
                energy_change_ry: None,
                density_residual: None,
                maximum_orbital_residual_ry: 0.1,
                electron_count: None,
                highest_occupied_eigenvalue_ry: Some(0.25),
                davidson_iterations: 3,
                hamiltonian_applications: 5,
            }],
            history_complete: true,
            timings: BTreeMap::from([("total_seconds".into(), 0.1)]),
            provenance: Provenance {
                requested_backend: BackendKind::CpuF64,
                actual_backend: BackendKind::CpuF64,
                actual_device: "test".into(),
                precision_policy: PrecisionPolicy::cpu_f64(),
                input_sha256: "input".into(),
                pseudopotential_sha256: BTreeMap::new(),
                git_revision: "test".into(),
                restart: None,
            },
            stop_reason: Some("eigensolver did not converge".into()),
        }
    }

    #[test]
    fn validates_failed_iteration_and_rejects_counter_mismatch() {
        let mut result = failed_result();
        result.validate().unwrap();
        result
            .energy_terms
            .insert("davidson_iterations".into(), 4.0);
        assert!(
            result
                .validate()
                .unwrap_err()
                .to_string()
                .contains("Davidson history total differs")
        );
    }

    #[test]
    fn schema_four_requires_forces_only_for_converged_results() {
        let mut result = failed_result();
        result.forces_ry_bohr = Some(vec![[0.0; 3]]);
        assert!(
            result
                .validate()
                .unwrap_err()
                .to_string()
                .contains("unconverged result")
        );

        result.forces_ry_bohr = None;
        result.converged = true;
        result.stop_reason = None;
        let final_record = result.history.last_mut().unwrap();
        final_record.status = ScfIterationStatus::Converged;
        final_record.total_energy_ry = result.total_energy_ry;
        final_record.electron_count = result.electron_count;
        assert!(
            result
                .validate()
                .unwrap_err()
                .to_string()
                .contains("requires forces")
        );
        result.forces_ry_bohr = Some(vec![[0.1, -0.2, 0.3]]);
        result.validate().unwrap();
    }

    #[test]
    fn schema_five_spin_channels_are_unambiguous_and_charge_conserving() {
        let mut result = failed_result();
        result.eigenvalues_ry.clear();
        result.residuals.clear();
        result.k_points.clear();
        result.electron_count = Some(3.0);
        result.spin = Some(CollinearSpinResult {
            magnetization_electrons: 1.0,
            up: SpinChannelResult {
                electron_count: 2.0,
                k_points: vec![KPointResult {
                    reduced: [0.0; 3],
                    weight: 1.0,
                    eigenvalues_ry: vec![-0.5, -0.1],
                    occupations: vec![1.0, 1.0],
                    residuals: vec![1.0e-10, 2.0e-10],
                }],
            },
            down: SpinChannelResult {
                electron_count: 1.0,
                k_points: vec![KPointResult {
                    reduced: [0.0; 3],
                    weight: 1.0,
                    eigenvalues_ry: vec![-0.4],
                    occupations: vec![1.0],
                    residuals: vec![3.0e-10],
                }],
            },
        });
        result.validate().unwrap();

        result.spin.as_mut().unwrap().magnetization_electrons = 0.5;
        assert!(
            result
                .validate()
                .unwrap_err()
                .to_string()
                .contains("spin magnetization disagrees")
        );
    }
}
