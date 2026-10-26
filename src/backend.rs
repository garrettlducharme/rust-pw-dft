use anyhow::Result;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BackendKind {
    CpuF64,
    MetalMixed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrecisionPolicy {
    pub storage: String,
    pub reductions: String,
    pub eigensolve: String,
    pub energy: String,
}

impl PrecisionPolicy {
    pub fn cpu_f64() -> Self {
        Self {
            storage: "complex-f64".into(),
            reductions: "f64".into(),
            eigensolve: "f64".into(),
            energy: "f64".into(),
        }
    }
    pub fn metal_mixed() -> Self {
        Self {
            storage: "complex-f32-gpu".into(),
            reductions: "f32-partials+f64-final".into(),
            eigensolve: "f64-cpu".into(),
            energy: "f64-cpu".into(),
        }
    }

    /// Transitional persistent-buffer policy: Hamiltonian blocks and Davidson
    /// cross-Gram partials run in Metal complex-FP32. The basis remains host
    /// FP64, partial sums finish on the CPU in complex-FP64, and Rayleigh-Ritz
    /// remains CPU-FP64. This is distinct from fully resident eigensolver state.
    pub fn metal_validation() -> Self {
        Self {
            storage: "complex-f64-host-canonical+persistent-complex-f32-metal-hamiltonian+journal"
                .into(),
            reductions: "complex-f32-metal-partials+complex-f64-cpu-final".into(),
            eigensolve: "f64-cpu".into(),
            energy: "f64-cpu".into(),
        }
    }
}

/// Backend boundary for the complete Hamiltonian/eigensolver chain.
/// Implementations must report their actual device and may not fall back.
pub trait KohnShamBackend {
    fn kind(&self) -> BackendKind;
    fn actual_device(&self) -> &str;
    fn precision_policy(&self) -> PrecisionPolicy;
    fn initialize_orbitals(&mut self, states: usize, coefficients: usize, seed: u64) -> Result<()>;
    fn apply_hamiltonian(&mut self, local_potential_ry: &[f64], kinetic_ry: &[f64]) -> Result<()>;
    fn orthonormalize(&mut self) -> Result<()>;
    fn build_density(&mut self, occupations: &[f64], density: &mut [f64]) -> Result<()>;
    fn synchronize(&mut self) -> Result<()>;
    fn orbitals_for_validation(&self) -> Result<Vec<Vec<Complex64>>>;
}
