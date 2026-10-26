use crate::backend::{BackendKind, PrecisionPolicy};
use crate::checkpoint::{self, CHECKPOINT_SCHEMA_VERSION, ScfCheckpoint};
use crate::eigensolver::{
    DavidsonOptions, ResidentDavidsonBackend, block_davidson_batched, block_davidson_resident,
};
use crate::input::SimulationInput;
use crate::lattice::Lattice;
use crate::operator::{
    CpuHamiltonian, atomic_starting_density, density_from_orbitals, ewald_energy_ry,
    hartree_xc_potential,
};
use crate::provenance::{git_revision, sha256};
use crate::pseudo::NormConservingPseudo;
use crate::result::{Provenance, RESULT_SCHEMA_VERSION, RestartProvenance, ScfResult};
use anyhow::{Context, Result, ensure};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

const RY_TO_EV: f64 = 13.605_693_122_994;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScfRunProfile {
    Scientific,
    MatchedBenchmark,
}

impl ScfRunProfile {
    fn identifier(self) -> &'static str {
        match self {
            Self::Scientific => "scientific-v1",
            Self::MatchedBenchmark => "matched-metal-compatible-v1",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScfRunControl {
    pub resume_from: Option<PathBuf>,
    pub checkpoint_output: Option<PathBuf>,
    pub checkpoint_every: usize,
    /// Deliberately pause after this many SCF iterations in the current
    /// invocation. Primarily useful for schedulers and restart verification.
    pub stop_after_scf_iterations: Option<usize>,
}

impl Default for ScfRunControl {
    fn default() -> Self {
        Self {
            resume_from: None,
            checkpoint_output: None,
            checkpoint_every: 1,
            stop_after_scf_iterations: None,
        }
    }
}

struct MetalResidentDavidson<'a> {
    hamiltonian: &'a CpuHamiltonian,
    context: &'a mut crate::metal::HamiltonianContext,
    potential_ry: &'a [f64],
}

impl ResidentDavidsonBackend for MetalResidentDavidson<'_> {
    fn initialize(&mut self, basis: &[Vec<num_complex::Complex64>]) -> Result<()> {
        self.hamiltonian
            .initialize_metal_resident_davidson(self.context, basis, self.potential_ry)
    }

    fn resident_count(&self) -> usize {
        self.context.resident_count()
    }

    fn projected_hamiltonian(&mut self, subspace: usize) -> Result<Vec<num_complex::Complex64>> {
        self.context.resident_projected_hamiltonian(subspace)
    }

    fn form_ritz_residuals(
        &mut self,
        coefficients: &[Vec<num_complex::Complex64>],
        eigenvalues: &[f64],
    ) -> Result<()> {
        self.context
            .form_resident_ritz_residuals(coefficients, eigenvalues)
    }

    fn residual_norms(&mut self, roots: usize) -> Result<Vec<f64>> {
        self.context.resident_residual_norms(roots)
    }

    fn precondition(&mut self, active_roots: &[bool]) -> Result<()> {
        self.context.precondition_resident_residuals(active_roots)
    }

    fn restart_from_ritz(&mut self, roots: usize) -> Result<()> {
        self.context.restart_resident_from_ritz(roots)
    }

    fn orthonormalize_corrections(&mut self, active_roots: &[bool]) -> Result<usize> {
        self.context
            .orthonormalize_resident_corrections(active_roots)
    }

    fn apply_accepted_corrections(&mut self, accepted_count: usize) -> Result<()> {
        self.context
            .apply_resident_accepted_corrections(accepted_count)
    }

    fn download_ritz(&mut self, roots: usize) -> Result<Vec<Vec<num_complex::Complex64>>> {
        self.context.download_resident_ritz(roots)
    }

    fn full_vector_transfer_bytes(&self) -> (usize, usize) {
        (
            self.context.full_vector_upload_bytes(),
            self.context.full_vector_download_bytes(),
        )
    }

    fn verify_iteration_residency(
        &mut self,
        expected_upload_bytes: usize,
        expected_download_bytes: usize,
    ) -> Result<()> {
        self.context
            .verify_resident_iteration_transfers(expected_upload_bytes, expected_download_bytes)
    }
}

fn add_timing(timings: &mut BTreeMap<String, f64>, name: &str, elapsed: f64) {
    *timings.entry(name.into()).or_default() += elapsed;
}

fn relative_density_residual(input: &[f64], output: &[f64]) -> f64 {
    let difference = input
        .iter()
        .zip(output)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>();
    let reference = output.iter().map(|value| value * value).sum::<f64>();
    (difference / reference.max(f64::MIN_POSITIVE)).sqrt()
}

fn metal_metrics(context: &crate::metal::HamiltonianContext) -> BTreeMap<String, f64> {
    BTreeMap::from([
        (
            "metal_projector_count".into(),
            context.projector_count() as f64,
        ),
        (
            "metal_persistent_buffer_bytes".into(),
            context.persistent_buffer_bytes() as f64,
        ),
        (
            "metal_peak_device_allocated_bytes".into(),
            context.peak_device_allocated_bytes() as f64,
        ),
        (
            "metal_gram_reduction_calls".into(),
            context.gram_reduction_calls() as f64,
        ),
        (
            "metal_gram_reduction_pairs".into(),
            context.gram_reduction_pairs() as f64,
        ),
        (
            "metal_resident_projection_calls".into(),
            context.resident_projection_calls() as f64,
        ),
        (
            "metal_resident_projection_resyncs".into(),
            context.resident_projection_resyncs() as f64,
        ),
        (
            "metal_norm_reduction_calls".into(),
            context.norm_reduction_calls() as f64,
        ),
        (
            "metal_norm_reduction_vectors".into(),
            context.norm_reduction_vectors() as f64,
        ),
        (
            "metal_resident_ritz_updates".into(),
            context.resident_ritz_updates() as f64,
        ),
        (
            "metal_resident_preconditioner_updates".into(),
            context.resident_preconditioner_updates() as f64,
        ),
        (
            "metal_resident_orthogonalization_calls".into(),
            context.resident_orthogonalization_calls() as f64,
        ),
        (
            "metal_resident_orthogonalization_passes".into(),
            context.resident_orthogonalization_passes() as f64,
        ),
        (
            "metal_resident_corrections_accepted".into(),
            context.resident_corrections_accepted() as f64,
        ),
        (
            "metal_resident_corrections_rejected".into(),
            context.resident_corrections_rejected() as f64,
        ),
        (
            "metal_resident_correction_hamiltonian_applications".into(),
            context.resident_correction_hamiltonian_applications() as f64,
        ),
        (
            "metal_resident_corrections_appended".into(),
            context.resident_corrections_appended() as f64,
        ),
        (
            "metal_resident_restarts".into(),
            context.resident_restarts() as f64,
        ),
        (
            "metal_resident_driver_iterations".into(),
            context.resident_driver_iterations() as f64,
        ),
        (
            "metal_resident_transfer_invariant_checks".into(),
            context.resident_transfer_invariant_checks() as f64,
        ),
        (
            "metal_resident_compact_upload_bytes".into(),
            context.resident_coefficient_upload_bytes() as f64,
        ),
        (
            "metal_resident_compact_download_bytes".into(),
            context.resident_reduction_download_bytes() as f64,
        ),
        (
            "metal_resident_vector_download_bytes".into(),
            context.resident_vector_download_bytes() as f64,
        ),
        (
            "metal_full_vector_upload_bytes".into(),
            context.full_vector_upload_bytes() as f64,
        ),
        (
            "metal_full_vector_download_bytes".into(),
            context.full_vector_download_bytes() as f64,
        ),
    ])
}

fn accumulated_metrics(
    baseline: &BTreeMap<String, f64>,
    current: Option<&crate::metal::HamiltonianContext>,
) -> BTreeMap<String, f64> {
    let mut accumulated = baseline.clone();
    if let Some(context) = current {
        for (name, value) in metal_metrics(context) {
            if name == "metal_projector_count" || name == "metal_persistent_buffer_bytes" {
                accumulated.insert(name, value);
            } else if name == "metal_peak_device_allocated_bytes" {
                accumulated
                    .entry(name)
                    .and_modify(|prior| *prior = prior.max(value))
                    .or_insert(value);
            } else {
                *accumulated.entry(name).or_default() += value;
            }
        }
    }
    accumulated
}

pub fn run(input_path: &Path, requested_backend: BackendKind) -> Result<ScfResult> {
    run_with_profile(
        input_path,
        requested_backend,
        ScfRunProfile::Scientific,
        &ScfRunControl::default(),
    )
}

pub fn run_with_control(
    input_path: &Path,
    requested_backend: BackendKind,
    control: &ScfRunControl,
) -> Result<ScfResult> {
    run_with_profile(
        input_path,
        requested_backend,
        ScfRunProfile::Scientific,
        control,
    )
}

/// Run either backend with the same mixed-precision-compatible convergence
/// policy. This is intentionally separate from the stricter CPU scientific
/// path so benchmark reports cannot accidentally compare different gates.
pub fn run_matched_benchmark(
    input_path: &Path,
    requested_backend: BackendKind,
) -> Result<ScfResult> {
    run_with_profile(
        input_path,
        requested_backend,
        ScfRunProfile::MatchedBenchmark,
        &ScfRunControl::default(),
    )
}

fn run_with_profile(
    input_path: &Path,
    requested_backend: BackendKind,
    profile: ScfRunProfile,
    control: &ScfRunControl,
) -> Result<ScfResult> {
    ensure!(
        control.checkpoint_every > 0,
        "checkpoint_every must be positive"
    );
    if let Some(stop_after) = control.stop_after_scf_iterations {
        ensure!(stop_after > 0, "stop_after_scf_iterations must be positive");
        ensure!(
            control.checkpoint_output.is_some(),
            "a deliberate pause requires checkpoint_output"
        );
    }
    let total_started = Instant::now();
    let input = SimulationInput::from_path(input_path)?;
    let input_hash = sha256(input_path)?;
    let current_git_revision = git_revision();
    let lattice = Lattice::new(input.cell.lattice_bohr)?;
    let g_vectors = lattice.enumerate_g_vectors(input.cutoff_ry);
    let mut pseudo_hashes = BTreeMap::new();
    ensure!(
        input.pseudopotentials.len() == 1,
        "the CPU scientific slice supports one species"
    );
    let (element, pseudo_path) = input.pseudopotentials.first_key_value().unwrap();
    let pseudo = NormConservingPseudo::from_upf(pseudo_path)?;
    ensure!(
        &pseudo.element == element,
        "pseudopotential element mismatch"
    );
    pseudo_hashes.insert(element.clone(), sha256(pseudo_path)?);
    let electron_count = pseudo.valence_charge * input.atoms.len() as f64;
    let occupied_states_f64 = electron_count / 2.0;
    ensure!(
        (occupied_states_f64 - occupied_states_f64.round()).abs() < 1.0e-12,
        "closed-shell electron count is required"
    );
    let occupied_states = occupied_states_f64.round() as usize;
    ensure!(
        occupied_states > 0 && occupied_states <= g_vectors.len(),
        "invalid occupied-state count"
    );
    let davidson_max_subspace = (occupied_states * 8)
        .max(occupied_states + 4)
        .min(g_vectors.len());

    let setup_started = Instant::now();
    let hamiltonian = CpuHamiltonian::new(&lattice, &g_vectors, &input.atoms, &pseudo)?;
    let (
        actual_device,
        precision_policy,
        mut eigensolver_tolerance,
        mut scf_orbital_tolerance,
        mut scf_energy_tolerance,
        mut scf_density_tolerance,
    ) = match requested_backend {
        BackendKind::CpuF64 => (
            "Apple CPU (rustfft/f64 + block-Davidson/faer-f64)".into(),
            PrecisionPolicy::cpu_f64(),
            input.scf_options.orbital_residual_tolerance.min(1.0e-10),
            input.scf_options.orbital_residual_tolerance,
            input.scf_options.energy_tolerance_ry,
            input.scf_options.density_tolerance,
        ),
        BackendKind::MetalMixed => {
            let device = crate::metal::device_name()
                .context("Metal backend unavailable (no CPU Hamiltonian fallback)")?;
            (
                format!(
                    "{device} (persistent MPSGraph Hamiltonian; fully resident GPU Davidson iterations)"
                ),
                PrecisionPolicy::metal_mixed(),
                1.0e-5,
                1.0e-5,
                input.scf_options.energy_tolerance_ry.max(5.0e-7),
                input.scf_options.density_tolerance.max(5.0e-8),
            )
        }
    };
    if profile == ScfRunProfile::MatchedBenchmark {
        eigensolver_tolerance = 1.0e-5;
        scf_orbital_tolerance = 1.0e-5;
        scf_energy_tolerance = input.scf_options.energy_tolerance_ry.max(5.0e-7);
        scf_density_tolerance = input.scf_options.density_tolerance.max(5.0e-8);
    }
    let mut metal_context = match requested_backend {
        BackendKind::CpuF64 => None,
        BackendKind::MetalMixed => Some(
            hamiltonian
                .create_metal_hamiltonian_context(davidson_max_subspace)
                .context("failed to initialize persistent Metal Hamiltonian")?,
        ),
    };
    let initial_density = atomic_starting_density(
        hamiltonian.fft(),
        &lattice,
        &input.atoms,
        &pseudo,
        electron_count,
    )?;
    let ewald_ry = ewald_energy_ry(&lattice, &input.atoms, pseudo.valence_charge, 0.7)?;
    let occupations = vec![2.0; occupied_states];
    let mut density = initial_density;
    let mut timings = BTreeMap::new();
    let mut previous_energy: Option<f64> = None;
    let mut final_energy = None;
    let mut final_terms = BTreeMap::new();
    let mut final_eigenvalues = Vec::new();
    let mut final_residuals = Vec::new();
    let mut final_charge = electron_count;
    let mut converged = false;
    let mut completed_iterations = 0;
    let mut final_density_residual = f64::INFINITY;
    let mut final_energy_change = f64::INFINITY;
    let mut previous_orbitals: Option<Vec<Vec<num_complex::Complex64>>> = None;
    let mut davidson_iterations = 0_usize;
    let mut hamiltonian_applications = 0_usize;
    let mut eigensolver_stop_reason = None;
    let mut paused = false;
    let mut prior_elapsed_seconds = 0.0_f64;
    let mut baseline_backend_metrics = BTreeMap::new();
    let mut restart_provenance = None;

    if let Some(path) = &control.resume_from {
        let loaded = checkpoint::read(path)?;
        let checkpoint = loaded.checkpoint;
        ensure!(
            checkpoint.writer_git_revision == current_git_revision,
            "checkpoint source revision differs ({} != {})",
            checkpoint.writer_git_revision,
            current_git_revision
        );
        ensure!(
            checkpoint.input_sha256 == input_hash,
            "checkpoint input hash differs"
        );
        ensure!(
            checkpoint.pseudopotential_sha256 == pseudo_hashes,
            "checkpoint pseudopotential hashes differ"
        );
        ensure!(
            checkpoint.requested_backend == requested_backend,
            "checkpoint backend differs"
        );
        ensure!(
            checkpoint.precision_policy == precision_policy,
            "checkpoint precision policy differs"
        );
        ensure!(
            checkpoint.run_profile == profile.identifier(),
            "checkpoint run profile differs"
        );
        ensure!(
            checkpoint.fft_shape == hamiltonian.fft().shape()
                && checkpoint.g_vector_count == g_vectors.len()
                && checkpoint.occupied_states == occupied_states,
            "checkpoint basis or FFT dimensions differ"
        );
        ensure!(
            checkpoint.iteration_completed > 0
                && checkpoint.iteration_completed < input.scf_options.max_iterations,
            "checkpoint iteration is outside the resumable range"
        );
        ensure!(
            checkpoint.density.len() == hamiltonian.fft().len()
                && checkpoint.density.iter().all(|value| value.is_finite()),
            "checkpoint density is invalid"
        );
        ensure!(
            checkpoint.orbitals.len() == occupied_states
                && checkpoint.orbitals.iter().all(|orbital| {
                    orbital.len() == g_vectors.len()
                        && orbital
                            .iter()
                            .all(|value| value.re.is_finite() && value.im.is_finite())
                }),
            "checkpoint orbitals are invalid"
        );
        ensure!(
            checkpoint.previous_energy_ry.is_finite()
                && checkpoint.elapsed_seconds.is_finite()
                && checkpoint.elapsed_seconds >= 0.0
                && checkpoint
                    .timings
                    .values()
                    .all(|value| value.is_finite() && *value >= 0.0)
                && checkpoint
                    .backend_metrics
                    .values()
                    .all(|value| value.is_finite() && *value >= 0.0),
            "checkpoint scalar state is invalid"
        );
        completed_iterations = checkpoint.iteration_completed;
        density = checkpoint.density;
        previous_energy = Some(checkpoint.previous_energy_ry);
        previous_orbitals = Some(checkpoint.orbitals);
        davidson_iterations = checkpoint.davidson_iterations;
        hamiltonian_applications = checkpoint.hamiltonian_applications;
        timings = checkpoint.timings;
        prior_elapsed_seconds = checkpoint.elapsed_seconds;
        baseline_backend_metrics = checkpoint.backend_metrics;
        restart_provenance = Some(RestartProvenance {
            checkpoint_sha256: loaded.file_sha256,
            checkpoint_iteration: checkpoint.iteration_completed,
            writer_git_revision: checkpoint.writer_git_revision,
        });
    }
    add_timing(
        &mut timings,
        "setup_seconds",
        setup_started.elapsed().as_secs_f64(),
    );

    for (iterations_this_run, iteration) in
        (completed_iterations + 1..=input.scf_options.max_iterations).enumerate()
    {
        let iterations_this_run = iterations_this_run + 1;
        completed_iterations = iteration;
        let potential_started = Instant::now();
        let potential = hartree_xc_potential(hamiltonian.fft(), &lattice, &density)?;
        add_timing(
            &mut timings,
            "potential_seconds",
            potential_started.elapsed().as_secs_f64(),
        );

        let eigensolve_started = Instant::now();
        let davidson_options = DavidsonOptions {
            roots: occupied_states,
            residual_tolerance: eigensolver_tolerance,
            max_iterations: 80,
            max_subspace: davidson_max_subspace,
        };
        let eigenpairs = match requested_backend {
            BackendKind::CpuF64 => block_davidson_batched(
                hamiltonian.kinetic_ry(),
                previous_orbitals.as_deref(),
                &davidson_options,
                |orbitals| {
                    orbitals
                        .iter()
                        .map(|orbital| hamiltonian.apply_fft(orbital, &potential.real_ry))
                        .collect()
                },
            ),
            BackendKind::MetalMixed => {
                let context = metal_context
                    .as_mut()
                    .context("persistent Metal Hamiltonian context is missing")?;
                let mut backend = MetalResidentDavidson {
                    hamiltonian: &hamiltonian,
                    context,
                    potential_ry: &potential.real_ry,
                };
                block_davidson_resident(
                    hamiltonian.kinetic_ry(),
                    previous_orbitals.as_deref(),
                    &davidson_options,
                    &mut backend,
                )
            }
        }?;
        add_timing(
            &mut timings,
            "eigensolve_seconds",
            eigensolve_started.elapsed().as_secs_f64(),
        );
        davidson_iterations += eigenpairs.iterations;
        hamiltonian_applications += eigenpairs.hamiltonian_applications;
        if !eigenpairs.converged {
            final_eigenvalues = eigenpairs.eigenvalues;
            final_residuals = eigenpairs.residual_norms;
            eigensolver_stop_reason = Some(format!(
                "block Davidson did not converge in {} iterations (maximum residual {:.3e} Ry)",
                eigenpairs.iterations,
                final_residuals.iter().copied().fold(0.0_f64, f64::max)
            ));
            break;
        }
        let eigenvalues = eigenpairs.eigenvalues;
        let orbitals = eigenpairs.eigenvectors;
        let orbital_residuals = eigenpairs.residual_norms;

        let density_started = Instant::now();
        let output_density = density_from_orbitals(
            hamiltonian.fft(),
            &g_vectors,
            &orbitals,
            &occupations,
            lattice.volume,
        )?;
        add_timing(
            &mut timings,
            "density_seconds",
            density_started.elapsed().as_secs_f64(),
        );
        final_density_residual = relative_density_residual(&density, &output_density);
        final_charge =
            output_density.iter().sum::<f64>() * lattice.volume / output_density.len() as f64;

        let band_energy_ry = eigenvalues.iter().sum::<f64>() * 2.0;
        let total_energy_ry = band_energy_ry - potential.hartree_energy_ry + potential.xc_energy_ry
            - potential.integral_rho_vxc_ry
            + ewald_ry;
        final_energy_change = previous_energy
            .map(|previous| (total_energy_ry - previous).abs())
            .unwrap_or(f64::INFINITY);
        previous_energy = Some(total_energy_ry);
        final_energy = Some(total_energy_ry);
        final_eigenvalues = eigenvalues;
        final_residuals = orbital_residuals;
        previous_orbitals = Some(orbitals);
        final_terms = BTreeMap::from([
            ("band_energy_ry".into(), band_energy_ry),
            ("hartree_ry".into(), potential.hartree_energy_ry),
            ("xc_ry".into(), potential.xc_energy_ry),
            ("integral_rho_vxc_ry".into(), potential.integral_rho_vxc_ry),
            ("ewald_ry".into(), ewald_ry),
            ("density_residual".into(), final_density_residual),
            ("density_tolerance".into(), scf_density_tolerance),
            ("energy_change_ry".into(), final_energy_change),
            ("energy_tolerance_ry".into(), scf_energy_tolerance),
            ("eigensolver_tolerance_ry".into(), eigensolver_tolerance),
            (
                "orbital_residual_tolerance_ry".into(),
                scf_orbital_tolerance,
            ),
        ]);

        let maximum_orbital_residual = final_residuals.iter().copied().fold(0.0_f64, f64::max);
        if final_density_residual <= scf_density_tolerance
            && final_energy_change <= scf_energy_tolerance
            && maximum_orbital_residual <= scf_orbital_tolerance
        {
            converged = true;
            break;
        }
        let beta = input.scf_options.mixing_beta;
        for (current, output) in density.iter_mut().zip(output_density) {
            *current = (1.0 - beta) * *current + beta * output;
        }
        let pause_due = control
            .stop_after_scf_iterations
            .is_some_and(|limit| iterations_this_run >= limit);
        let checkpoint_due = control.checkpoint_output.is_some()
            && (iteration % control.checkpoint_every == 0 || pause_due);
        if checkpoint_due {
            let checkpoint = ScfCheckpoint {
                schema_version: CHECKPOINT_SCHEMA_VERSION,
                writer_git_revision: current_git_revision.clone(),
                input_sha256: input_hash.clone(),
                pseudopotential_sha256: pseudo_hashes.clone(),
                requested_backend,
                precision_policy: precision_policy.clone(),
                run_profile: profile.identifier().into(),
                iteration_completed: iteration,
                fft_shape: hamiltonian.fft().shape(),
                g_vector_count: g_vectors.len(),
                occupied_states,
                density: density.clone(),
                previous_energy_ry: previous_energy
                    .context("checkpoint requested before a complete SCF iteration")?,
                orbitals: previous_orbitals
                    .clone()
                    .context("checkpoint requested without orbitals")?,
                davidson_iterations,
                hamiltonian_applications,
                timings: timings.clone(),
                elapsed_seconds: prior_elapsed_seconds + total_started.elapsed().as_secs_f64(),
                backend_metrics: accumulated_metrics(
                    &baseline_backend_metrics,
                    metal_context.as_ref(),
                ),
            };
            checkpoint::write_atomic(control.checkpoint_output.as_ref().unwrap(), &checkpoint)?;
        }
        if pause_due {
            paused = true;
            break;
        }
    }
    timings.insert(
        "total_seconds".into(),
        prior_elapsed_seconds + total_started.elapsed().as_secs_f64(),
    );
    let stop_reason = if converged {
        None
    } else if paused {
        Some(format!(
            "SCF paused after iteration {completed_iterations}; resume from the written checkpoint"
        ))
    } else if let Some(reason) = eigensolver_stop_reason {
        Some(format!(
            "{reason}; no scientific acceptance claim is allowed"
        ))
    } else {
        Some(format!(
            "{:?} SCF did not converge in {} iterations (density residual {:.3e}, energy change {:.3e} Ry); no scientific acceptance claim is allowed",
            requested_backend,
            input.scf_options.max_iterations,
            final_density_residual,
            final_energy_change
        ))
    };
    final_terms.insert("davidson_iterations".into(), davidson_iterations as f64);
    final_terms.insert(
        "hamiltonian_applications".into(),
        hamiltonian_applications as f64,
    );
    final_terms.insert(
        "maximum_orbital_residual_ry".into(),
        final_residuals.iter().copied().fold(0.0_f64, f64::max),
    );
    final_terms.insert(
        "highest_occupied_eigenvalue_ev".into(),
        final_eigenvalues.last().copied().unwrap_or_default() * RY_TO_EV,
    );
    for (name, value) in accumulated_metrics(&baseline_backend_metrics, metal_context.as_ref()) {
        final_terms.insert(name, value);
    }
    Ok(ScfResult {
        schema_version: RESULT_SCHEMA_VERSION,
        converged,
        iterations: completed_iterations,
        atom_count: input.atoms.len(),
        total_energy_ry: final_energy,
        energy_terms: final_terms,
        eigenvalues_ry: final_eigenvalues,
        electron_count: Some(final_charge),
        residuals: final_residuals,
        timings,
        provenance: Provenance {
            requested_backend,
            actual_backend: requested_backend,
            actual_device,
            precision_policy,
            input_sha256: input_hash,
            pseudopotential_sha256: pseudo_hashes,
            git_revision: current_git_revision,
            restart: restart_provenance,
        },
        stop_reason,
    })
}

#[cfg(test)]
mod checkpoint_tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static SEQUENCE: AtomicU64 = AtomicU64::new(0);

    fn temporary_path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "rust-pw-dft-{label}-{}-{}.bin",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn interrupted_cpu_scf_resumes_equivalently_and_rejects_mismatch() {
        let input_path = Path::new("examples/si2.toml");
        if !Path::new("pseudopotentials/Si.pz-vbc.UPF").exists() {
            return;
        }
        let checkpoint_path = temporary_path("resume");
        let bad_checkpoint_path = temporary_path("bad-resume");
        let uninterrupted = run(input_path, BackendKind::CpuF64).unwrap();
        let paused = run_with_control(
            input_path,
            BackendKind::CpuF64,
            &ScfRunControl {
                checkpoint_output: Some(checkpoint_path.clone()),
                checkpoint_every: 2,
                stop_after_scf_iterations: Some(5),
                ..ScfRunControl::default()
            },
        )
        .unwrap();
        assert!(!paused.converged);
        assert_eq!(paused.iterations, 5);
        assert!(paused.stop_reason.unwrap().contains("paused"));

        let loaded = checkpoint::read(&checkpoint_path).unwrap();
        assert_eq!(loaded.checkpoint.iteration_completed, 5);
        let resumed = run_with_control(
            input_path,
            BackendKind::CpuF64,
            &ScfRunControl {
                resume_from: Some(checkpoint_path.clone()),
                ..ScfRunControl::default()
            },
        )
        .unwrap();
        assert!(resumed.converged);
        assert_eq!(resumed.iterations, uninterrupted.iterations);
        assert!(
            (resumed.total_energy_ry.unwrap() - uninterrupted.total_energy_ry.unwrap()).abs()
                < 1.0e-13
        );
        assert!(
            resumed
                .eigenvalues_ry
                .iter()
                .zip(&uninterrupted.eigenvalues_ry)
                .all(|(left, right)| (left - right).abs() < 1.0e-13)
        );
        let restart = resumed.provenance.restart.as_ref().unwrap();
        assert_eq!(restart.checkpoint_iteration, 5);
        assert_eq!(restart.checkpoint_sha256, loaded.file_sha256);

        let mut mismatched = loaded.checkpoint;
        mismatched.input_sha256 = "not-the-input-hash".into();
        checkpoint::write_atomic(&bad_checkpoint_path, &mismatched).unwrap();
        let error = run_with_control(
            input_path,
            BackendKind::CpuF64,
            &ScfRunControl {
                resume_from: Some(bad_checkpoint_path.clone()),
                ..ScfRunControl::default()
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("input hash differs"));

        std::fs::remove_file(checkpoint_path).unwrap();
        std::fs::remove_file(bad_checkpoint_path).unwrap();
    }
}

#[cfg(all(test, feature = "metal", target_os = "macos"))]
mod tests {
    use super::*;

    #[test]
    fn resident_metal_scf_matches_cpu_and_transfers_only_at_boundaries() {
        let input_path = Path::new("examples/si2.toml");
        let input = SimulationInput::from_path(input_path).unwrap();
        let lattice = Lattice::new(input.cell.lattice_bohr).unwrap();
        let g_vectors = lattice.enumerate_g_vectors(input.cutoff_ry);
        let grid_points = Lattice::density_fft_grid(&g_vectors)
            .iter()
            .product::<usize>();
        let pseudo = NormConservingPseudo::from_upf(&input.pseudopotentials["Si"]).unwrap();
        let roots = (pseudo.valence_charge * input.atoms.len() as f64 / 2.0) as usize;

        let cpu = run(input_path, BackendKind::CpuF64).unwrap();
        let metal = run(input_path, BackendKind::MetalMixed).unwrap();
        assert!(cpu.converged);
        assert!(metal.converged);
        let energy_error_mev_atom = (metal.total_energy_ry.unwrap() - cpu.total_energy_ry.unwrap())
            .abs()
            * 13_605.693_122_994
            / input.atoms.len() as f64;
        let eigenvalue_error_mev = metal
            .eigenvalues_ry
            .iter()
            .zip(&cpu.eigenvalues_ry)
            .map(|(left, right)| (left - right).abs() * 13_605.693_122_994)
            .fold(0.0_f64, f64::max);
        assert!(energy_error_mev_atom <= 1.0, "{energy_error_mev_atom:e}");
        assert!(eigenvalue_error_mev <= 5.0, "{eigenvalue_error_mev:e}");
        assert!((metal.electron_count.unwrap() - cpu.electron_count.unwrap()).abs() <= 1.0e-5);

        let terms = &metal.energy_terms;
        assert_eq!(terms["metal_resident_projection_resyncs"], 0.0);
        assert_eq!(
            terms["metal_resident_driver_iterations"],
            terms["davidson_iterations"]
        );
        assert_eq!(
            terms["metal_resident_transfer_invariant_checks"],
            terms["davidson_iterations"]
        );
        assert!(terms["metal_resident_corrections_appended"] > 0.0);
        assert!(terms["metal_resident_restarts"] > 0.0);

        let complex_bytes = std::mem::size_of::<num_complex::Complex32>();
        let expected_upload_bytes =
            metal.iterations * roots * (grid_points + g_vectors.len()) * complex_bytes;
        let expected_download_bytes = metal.iterations * roots * g_vectors.len() * complex_bytes;
        assert_eq!(
            terms["metal_full_vector_upload_bytes"] as usize,
            expected_upload_bytes
        );
        assert_eq!(
            terms["metal_full_vector_download_bytes"] as usize,
            expected_download_bytes
        );
        assert_eq!(
            terms["metal_resident_vector_download_bytes"] as usize,
            expected_download_bytes
        );

        let checkpoint_path = std::env::temp_dir().join(format!(
            "rust-pw-dft-metal-resume-{}.bin",
            std::process::id()
        ));
        let paused = run_with_control(
            input_path,
            BackendKind::MetalMixed,
            &ScfRunControl {
                checkpoint_output: Some(checkpoint_path.clone()),
                checkpoint_every: 5,
                stop_after_scf_iterations: Some(5),
                ..ScfRunControl::default()
            },
        )
        .unwrap();
        assert!(!paused.converged);
        let resumed = run_with_control(
            input_path,
            BackendKind::MetalMixed,
            &ScfRunControl {
                resume_from: Some(checkpoint_path.clone()),
                ..ScfRunControl::default()
            },
        )
        .unwrap();
        assert!(resumed.converged);
        assert_eq!(resumed.iterations, metal.iterations);
        assert!(
            (resumed.total_energy_ry.unwrap() - metal.total_energy_ry.unwrap()).abs() < 1.0e-10
        );
        assert!(
            resumed
                .eigenvalues_ry
                .iter()
                .zip(&metal.eigenvalues_ry)
                .all(|(left, right)| (left - right).abs() < 1.0e-9)
        );
        assert_eq!(
            resumed.energy_terms["metal_full_vector_upload_bytes"],
            metal.energy_terms["metal_full_vector_upload_bytes"]
        );
        assert_eq!(
            resumed.energy_terms["metal_full_vector_download_bytes"],
            metal.energy_terms["metal_full_vector_download_bytes"]
        );
        assert_eq!(
            resumed
                .provenance
                .restart
                .as_ref()
                .unwrap()
                .checkpoint_iteration,
            5
        );
        std::fs::remove_file(checkpoint_path).unwrap();
    }
}
