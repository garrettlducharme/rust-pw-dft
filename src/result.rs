use crate::backend::{BackendKind, PrecisionPolicy};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const RESULT_SCHEMA_VERSION: u32 = 1;

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
    pub timings: BTreeMap<String, f64>,
    pub provenance: Provenance,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
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
