use crate::backend::{BackendKind, PrecisionPolicy};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const RESULT_SCHEMA_VERSION: u32 = 2;

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
    pub history: Vec<ScfIterationRecord>,
    pub history_complete: bool,
    pub timings: BTreeMap<String, f64>,
    pub provenance: Provenance,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
}

impl ScfResult {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == RESULT_SCHEMA_VERSION,
            "unsupported result schema_version {}; expected {}",
            self.schema_version,
            RESULT_SCHEMA_VERSION
        );
        ensure!(self.atom_count > 0, "result atom_count must be positive");
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
}
