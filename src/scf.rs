use crate::backend::{BackendKind, PrecisionPolicy};
use crate::input::SimulationInput;
use crate::lattice::Lattice;
use crate::provenance::{git_revision, sha256};
use crate::pseudo::NormConservingPseudo;
use crate::result::{Provenance, ScfResult};
use anyhow::{Context, Result, ensure};
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

pub fn preflight(input_path: &Path, requested_backend: BackendKind) -> Result<ScfResult> {
    let started = Instant::now();
    let input = SimulationInput::from_path(input_path)?;
    let lattice = Lattice::new(input.cell.lattice_bohr)?;
    let g_vectors = lattice.enumerate_g_vectors(input.cutoff_ry);
    let grid = Lattice::density_fft_grid(&g_vectors);
    ensure!(grid.iter().product::<usize>() > 0, "empty FFT grid");
    let mut pseudo_hashes = BTreeMap::new();
    let mut electrons = 0.0;
    for (element, path) in &input.pseudopotentials {
        let pseudo = NormConservingPseudo::from_legacy_upf(path)?;
        ensure!(
            &pseudo.element == element,
            "pseudopotential element mismatch"
        );
        electrons += pseudo.valence_charge
            * input
                .atoms
                .iter()
                .filter(|atom| &atom.element == element)
                .count() as f64;
        pseudo_hashes.insert(element.clone(), sha256(path)?);
    }
    let (actual_backend, actual_device, policy) = match requested_backend {
        BackendKind::CpuF64 => (
            BackendKind::CpuF64,
            "Apple CPU (rustfft/f64)".into(),
            PrecisionPolicy::cpu_f64(),
        ),
        BackendKind::MetalMixed => {
            let device = crate::metal::device_name()
                .context("Metal backend unavailable (no CPU fallback)")?;
            (
                BackendKind::MetalMixed,
                device,
                PrecisionPolicy::metal_mixed(),
            )
        }
    };
    let mut timings = BTreeMap::new();
    timings.insert("preflight_seconds".into(), started.elapsed().as_secs_f64());
    let stop_reason = format!(
        "scientific SCF is fail-closed: Hamiltonian, nonlocal projector application, Davidson, Hartree/Ewald, and density mixing are not yet integrated ({} plane waves, grid {}x{}x{})",
        g_vectors.len(),
        grid[0],
        grid[1],
        grid[2]
    );
    Ok(ScfResult {
        converged: false,
        iterations: 0,
        total_energy_ry: None,
        energy_terms: BTreeMap::new(),
        eigenvalues_ry: Vec::new(),
        electron_count: Some(electrons),
        residuals: Vec::new(),
        timings,
        provenance: Provenance {
            requested_backend,
            actual_backend,
            actual_device,
            precision_policy: policy,
            input_sha256: sha256(input_path)?,
            pseudopotential_sha256: pseudo_hashes,
            git_revision: git_revision(),
        },
        stop_reason: Some(stop_reason),
    })
}
