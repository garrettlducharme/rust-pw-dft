use crate::backend::{BackendKind, PrecisionPolicy};
use crate::eigensolver::{DavidsonOptions, block_davidson};
use crate::input::SimulationInput;
use crate::lattice::Lattice;
use crate::operator::{
    CpuHamiltonian, atomic_starting_density, density_from_orbitals, ewald_energy_ry,
    hartree_xc_potential,
};
use crate::provenance::{git_revision, sha256};
use crate::pseudo::NormConservingPseudo;
use crate::result::{Provenance, ScfResult};
use anyhow::{Context, Result, ensure};
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

const RY_TO_EV: f64 = 13.605_693_122_994;

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
        "Metal scientific SCF is fail-closed: only the native MPSGraph complex-FP32 FFT has passed; the resident Hamiltonian, mixed-precision reductions, and SCF orchestration are not yet integrated ({} plane waves, grid {}x{}x{})",
        g_vectors.len(),
        grid[0],
        grid[1],
        grid[2]
    );
    Ok(ScfResult {
        converged: false,
        iterations: 0,
        atom_count: input.atoms.len(),
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

pub fn run(input_path: &Path, requested_backend: BackendKind) -> Result<ScfResult> {
    if requested_backend == BackendKind::MetalMixed {
        return preflight(input_path, requested_backend);
    }
    let total_started = Instant::now();
    let input = SimulationInput::from_path(input_path)?;
    let lattice = Lattice::new(input.cell.lattice_bohr)?;
    let g_vectors = lattice.enumerate_g_vectors(input.cutoff_ry);
    let mut pseudo_hashes = BTreeMap::new();
    ensure!(
        input.pseudopotentials.len() == 1,
        "the CPU scientific slice supports one species"
    );
    let (element, pseudo_path) = input.pseudopotentials.first_key_value().unwrap();
    let pseudo = NormConservingPseudo::from_legacy_upf(pseudo_path)?;
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

    let setup_started = Instant::now();
    let hamiltonian = CpuHamiltonian::new(&lattice, &g_vectors, &input.atoms, &pseudo)?;
    let mut density = atomic_starting_density(
        hamiltonian.fft(),
        &lattice,
        &input.atoms,
        &pseudo,
        electron_count,
    )?;
    let ewald_ry = ewald_energy_ry(&lattice, &input.atoms, pseudo.valence_charge, 0.7)?;
    let mut timings = BTreeMap::new();
    add_timing(
        &mut timings,
        "setup_seconds",
        setup_started.elapsed().as_secs_f64(),
    );

    let occupations = vec![2.0; occupied_states];
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

    for iteration in 1..=input.scf_options.max_iterations {
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
            residual_tolerance: input.scf_options.orbital_residual_tolerance.min(1.0e-10),
            max_iterations: 80,
            max_subspace: (occupied_states * 8)
                .max(occupied_states + 4)
                .min(hamiltonian.dimension()),
        };
        let eigenpairs = block_davidson(
            hamiltonian.kinetic_ry(),
            previous_orbitals.as_deref(),
            &davidson_options,
            |orbital| hamiltonian.apply_fft(orbital, &potential.real_ry),
        )?;
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
            ("energy_change_ry".into(), final_energy_change),
        ]);

        let maximum_orbital_residual = final_residuals.iter().copied().fold(0.0_f64, f64::max);
        if final_density_residual <= input.scf_options.density_tolerance
            && final_energy_change <= input.scf_options.energy_tolerance_ry
            && maximum_orbital_residual <= input.scf_options.orbital_residual_tolerance
        {
            converged = true;
            break;
        }
        let beta = input.scf_options.mixing_beta;
        for (current, output) in density.iter_mut().zip(output_density) {
            *current = (1.0 - beta) * *current + beta * output;
        }
    }
    timings.insert(
        "total_seconds".into(),
        total_started.elapsed().as_secs_f64(),
    );
    let stop_reason = if converged {
        None
    } else if let Some(reason) = eigensolver_stop_reason {
        Some(format!(
            "{reason}; no scientific acceptance claim is allowed"
        ))
    } else {
        Some(format!(
            "CPU SCF did not converge in {} iterations (density residual {:.3e}, energy change {:.3e} Ry); no scientific acceptance claim is allowed",
            input.scf_options.max_iterations, final_density_residual, final_energy_change
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
    Ok(ScfResult {
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
            actual_backend: BackendKind::CpuF64,
            actual_device: "Apple CPU (rustfft/f64 + block-Davidson/faer-f64)".into(),
            precision_policy: PrecisionPolicy::cpu_f64(),
            input_sha256: sha256(input_path)?,
            pseudopotential_sha256: pseudo_hashes,
            git_revision: git_revision(),
        },
        stop_reason,
    })
}
