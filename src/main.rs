use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, Subcommand, ValueEnum};
use rust_pw_dft::backend::BackendKind;
use rust_pw_dft::checkpoint;
use rust_pw_dft::fixtures;
use rust_pw_dft::result::{Provenance, ScfResult};
use rust_pw_dft::scf::{ScfRunControl, run_matched_benchmark, run_with_control};
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

const COMPARISON_SCHEMA_VERSION: u32 = 4;
const BENCHMARK_SCHEMA_VERSION: u32 = 1;

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Run {
        input: PathBuf,
        #[arg(long, value_enum)]
        backend: BackendArg,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        #[arg(long, default_value_t = 1)]
        checkpoint_every: usize,
        #[arg(long)]
        stop_after: Option<usize>,
    },
    Resume {
        input: PathBuf,
        checkpoint: PathBuf,
        #[arg(long, value_enum)]
        backend: BackendArg,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        checkpoint_output: Option<PathBuf>,
        #[arg(long, default_value_t = 1)]
        checkpoint_every: usize,
        #[arg(long)]
        stop_after: Option<usize>,
    },
    InspectCheckpoint {
        checkpoint: PathBuf,
    },
    VerifyFixtures {
        #[arg(long, default_value = "fixtures/manifest.toml")]
        manifest: PathBuf,
        #[arg(long, default_value = ".")]
        repository_root: PathBuf,
    },
    Compare {
        result: PathBuf,
        reference: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    Bench {
        input: PathBuf,
        #[arg(long, value_enum)]
        backend: BackendArg,
        #[arg(long, default_value_t = 5)]
        repetitions: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    BenchCompare {
        input: PathBuf,
        #[arg(long, default_value_t = 5)]
        repetitions: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum BackendArg {
    Cpu,
    Metal,
}
impl From<BackendArg> for BackendKind {
    fn from(value: BackendArg) -> Self {
        match value {
            BackendArg::Cpu => Self::CpuF64,
            BackendArg::Metal => Self::MetalMixed,
        }
    }
}

#[derive(Serialize)]
struct Comparison {
    schema_version: u32,
    accepted: bool,
    total_energy_error_mev_atom: f64,
    total_energy_tolerance_mev_atom: f64,
    maximum_occupied_eigenvalue_error_mev: f64,
    occupied_eigenvalue_tolerance_mev: f64,
    maximum_occupation_error: f64,
    occupation_tolerance: f64,
    electron_count_error: f64,
    electron_count_tolerance: f64,
    force_comparison_performed: bool,
    maximum_force_error_ry_bohr: Option<f64>,
    force_tolerance_ry_bohr: Option<f64>,
    k_point_count: usize,
}

#[derive(Serialize)]
struct Benchmark {
    schema_version: u32,
    accepted_for_performance_claim: bool,
    requested_backend: BackendKind,
    actual_backend: BackendKind,
    converged: bool,
    iterations: usize,
    repetitions: usize,
    min_seconds: f64,
    median_seconds: f64,
    max_seconds: f64,
    provenance: Provenance,
    stop_reason: Option<String>,
}

#[derive(Serialize)]
struct MatchedBenchmarkComparison {
    schema_version: u32,
    accepted_for_performance_claim: bool,
    convergence_profile: &'static str,
    repetitions: usize,
    cpu: Benchmark,
    metal: Benchmark,
    scientific_comparison: Comparison,
    median_speedup: f64,
    stop_reason: Option<String>,
}

#[derive(Serialize)]
struct CheckpointSummary {
    schema_version: u32,
    file_sha256: String,
    writer_git_revision: String,
    input_sha256: String,
    pseudopotential_sha256: std::collections::BTreeMap<String, String>,
    requested_backend: BackendKind,
    precision_policy: rust_pw_dft::PrecisionPolicy,
    run_profile: String,
    iteration_completed: usize,
    fft_shape: [usize; 3],
    k_point_count: usize,
    g_vector_counts: Vec<usize>,
    band_counts: Vec<usize>,
    density_values: usize,
    orbital_counts: Vec<usize>,
    elapsed_seconds: f64,
}

fn read_result(path: &Path) -> Result<ScfResult> {
    let result: ScfResult = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("invalid result {}", path.display()))?;
    result.validate()?;
    Ok(result)
}

fn write_result(path: &Path, result: &ScfResult) -> Result<()> {
    result.validate()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(result)?)?;
    println!(
        "actual backend: {:?}; device: {}",
        result.provenance.actual_backend, result.provenance.actual_device
    );
    if !result.converged {
        bail!(
            "SCF did not converge: {}",
            result.stop_reason.as_deref().unwrap_or("unknown reason")
        );
    }
    Ok(())
}

fn compare(result_path: &Path, reference_path: &Path) -> Result<Comparison> {
    let result = read_result(result_path)?;
    let reference = read_result(reference_path)?;
    compare_results(&result, &reference)
}

fn compare_results(result: &ScfResult, reference: &ScfResult) -> Result<Comparison> {
    ensure!(
        result.converged && reference.converged,
        "comparison requires two converged SCF results"
    );
    let energy = result
        .total_energy_ry
        .context("result has no total energy")?;
    let reference_energy = reference
        .total_energy_ry
        .context("reference has no total energy")?;
    ensure!(
        result.atom_count > 0 && result.atom_count == reference.atom_count,
        "comparison requires matching nonzero atom counts"
    );
    let (max_eigen_ry, maximum_occupation_error, k_point_count) =
        if !result.k_points.is_empty() && !reference.k_points.is_empty() {
            ensure!(
                result.k_points.len() == reference.k_points.len(),
                "k-point counts differ"
            );
            let mut maximum_eigenvalue_error = 0.0_f64;
            let mut maximum_occupation_error = 0.0_f64;
            for (actual, expected) in result.k_points.iter().zip(&reference.k_points) {
                ensure!(
                    actual
                        .reduced
                        .iter()
                        .zip(expected.reduced)
                        .all(|(&a, b)| (a - b).abs() <= 1.0e-10)
                        && (actual.weight - expected.weight).abs() <= 1.0e-12,
                    "k-point coordinates or weights differ"
                );
                ensure!(
                    actual.eigenvalues_ry.len() == expected.eigenvalues_ry.len()
                        && actual.occupations.len() == expected.occupations.len(),
                    "per-k-point band counts differ"
                );
                for (
                    ((&actual_eigenvalue, &expected_eigenvalue), &actual_occupation),
                    &expected_occupation,
                ) in actual
                    .eigenvalues_ry
                    .iter()
                    .zip(&expected.eigenvalues_ry)
                    .zip(&actual.occupations)
                    .zip(&expected.occupations)
                {
                    maximum_occupation_error = maximum_occupation_error
                        .max((actual_occupation - expected_occupation).abs());
                    if actual_occupation > 1.0e-8 || expected_occupation > 1.0e-8 {
                        maximum_eigenvalue_error = maximum_eigenvalue_error
                            .max((actual_eigenvalue - expected_eigenvalue).abs());
                    }
                }
            }
            (
                maximum_eigenvalue_error,
                maximum_occupation_error,
                result.k_points.len(),
            )
        } else {
            ensure!(
                result.eigenvalues_ry.len() == reference.eigenvalues_ry.len(),
                "eigenvalue counts differ"
            );
            (
                result
                    .eigenvalues_ry
                    .iter()
                    .zip(&reference.eigenvalues_ry)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max),
                0.0,
                1,
            )
        };
    let electron_error = (result
        .electron_count
        .context("result has no electron count")?
        - reference
            .electron_count
            .context("reference has no electron count")?)
    .abs();
    let maximum_force_error_ry_bohr = match (&result.forces_ry_bohr, &reference.forces_ry_bohr) {
        (Some(actual), Some(expected)) => {
            ensure!(
                actual.len() == expected.len() && actual.len() == result.atom_count,
                "force counts differ"
            );
            Some(
                actual
                    .iter()
                    .flatten()
                    .zip(expected.iter().flatten())
                    .map(|(&left, &right)| (left - right).abs())
                    .fold(0.0_f64, f64::max),
            )
        }
        _ => None,
    };
    // 1 Ry = 13.605693122994 eV.
    let energy_mev =
        (energy - reference_energy).abs() * 13_605.693_122_994 / result.atom_count as f64;
    let eigen_mev = max_eigen_ry * 13_605.693_122_994;
    let (energy_tolerance_mev_atom, occupation_tolerance, electron_tolerance) =
        if result.provenance.actual_backend == BackendKind::MetalMixed {
            (1.0, 1.0e-5, 1.0e-5)
        } else {
            (2.0, 1.0e-6, 1.0e-8)
        };
    let occupied_eigenvalue_tolerance_mev = 5.0;
    let force_tolerance_ry_bohr = maximum_force_error_ry_bohr.map(|_| {
        if result.provenance.actual_backend == BackendKind::MetalMixed {
            5.0e-4
        } else {
            1.0e-4
        }
    });
    Ok(Comparison {
        schema_version: COMPARISON_SCHEMA_VERSION,
        accepted: energy_mev <= energy_tolerance_mev_atom
            && eigen_mev <= occupied_eigenvalue_tolerance_mev
            && maximum_occupation_error <= occupation_tolerance
            && electron_error <= electron_tolerance
            && maximum_force_error_ry_bohr
                .zip(force_tolerance_ry_bohr)
                .is_none_or(|(error, tolerance)| error <= tolerance),
        total_energy_error_mev_atom: energy_mev,
        total_energy_tolerance_mev_atom: energy_tolerance_mev_atom,
        maximum_occupied_eigenvalue_error_mev: eigen_mev,
        occupied_eigenvalue_tolerance_mev,
        maximum_occupation_error,
        occupation_tolerance,
        electron_count_error: electron_error,
        electron_count_tolerance: electron_tolerance,
        force_comparison_performed: maximum_force_error_ry_bohr.is_some(),
        maximum_force_error_ry_bohr,
        force_tolerance_ry_bohr,
        k_point_count,
    })
}

fn benchmark(
    input: &Path,
    kind: BackendKind,
    repetitions: usize,
) -> Result<(Benchmark, ScfResult)> {
    ensure!(repetitions >= 3, "at least three repetitions are required");
    let _cold = run_matched_benchmark(input, kind)?;
    let mut timings = Vec::with_capacity(repetitions);
    let mut last = None;
    for _ in 0..repetitions {
        let started = Instant::now();
        last = Some(run_matched_benchmark(input, kind)?);
        timings.push(started.elapsed().as_secs_f64());
    }
    timings.sort_by(f64::total_cmp);
    let last = last.unwrap();
    let report = Benchmark {
        schema_version: BENCHMARK_SCHEMA_VERSION,
        accepted_for_performance_claim: false,
        requested_backend: kind,
        actual_backend: last.provenance.actual_backend,
        converged: last.converged,
        iterations: last.iterations,
        repetitions,
        min_seconds: timings[0],
        median_seconds: timings[timings.len() / 2],
        max_seconds: *timings.last().unwrap(),
        provenance: last.provenance.clone(),
        stop_reason: Some(
            "A single-backend timing cannot support a performance claim; use bench-compare".into(),
        ),
    };
    Ok((report, last))
}

fn matched_tolerances(result: &ScfResult) -> Result<[f64; 4]> {
    let term = |name: &str| {
        result
            .energy_terms
            .get(name)
            .copied()
            .with_context(|| format!("benchmark result is missing {name}"))
    };
    Ok([
        term("eigensolver_tolerance_ry")?,
        term("orbital_residual_tolerance_ry")?,
        term("energy_tolerance_ry")?,
        term("density_tolerance")?,
    ])
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Run {
            input,
            backend,
            output,
            checkpoint,
            checkpoint_every,
            stop_after,
        } => {
            let control = ScfRunControl {
                checkpoint_output: checkpoint,
                checkpoint_every,
                stop_after_scf_iterations: stop_after,
                ..ScfRunControl::default()
            };
            let result = run_with_control(&input, backend.into(), &control)?;
            write_result(&output, &result)?;
        }
        Command::Resume {
            input,
            checkpoint,
            backend,
            output,
            checkpoint_output,
            checkpoint_every,
            stop_after,
        } => {
            let control = ScfRunControl {
                resume_from: Some(checkpoint.clone()),
                checkpoint_output: Some(checkpoint_output.unwrap_or(checkpoint)),
                checkpoint_every,
                stop_after_scf_iterations: stop_after,
            };
            let result = run_with_control(&input, backend.into(), &control)?;
            write_result(&output, &result)?;
        }
        Command::InspectCheckpoint { checkpoint: path } => {
            let loaded = checkpoint::read(&path)?;
            let value = loaded.checkpoint;
            let summary = CheckpointSummary {
                schema_version: value.schema_version,
                file_sha256: loaded.file_sha256,
                writer_git_revision: value.writer_git_revision,
                input_sha256: value.input_sha256,
                pseudopotential_sha256: value.pseudopotential_sha256,
                requested_backend: value.requested_backend,
                precision_policy: value.precision_policy,
                run_profile: value.run_profile,
                iteration_completed: value.iteration_completed,
                fft_shape: value.fft_shape,
                k_point_count: value.k_points.len(),
                g_vector_counts: value
                    .k_points
                    .iter()
                    .map(|point| point.g_vector_count)
                    .collect(),
                band_counts: value
                    .k_points
                    .iter()
                    .map(|point| point.band_count)
                    .collect(),
                density_values: value.density.len(),
                orbital_counts: value
                    .k_points
                    .iter()
                    .map(|point| point.orbitals.len())
                    .collect(),
                elapsed_seconds: value.elapsed_seconds,
            };
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        Command::VerifyFixtures {
            manifest,
            repository_root,
        } => println!(
            "{}",
            serde_json::to_string_pretty(&fixtures::verify_manifest(&manifest, &repository_root)?)?
        ),
        Command::Compare {
            result,
            reference,
            output,
        } => {
            let rendered = serde_json::to_string_pretty(&compare(&result, &reference)?)?;
            if let Some(path) = output {
                fs::write(path, rendered + "\n")?;
            } else {
                println!("{rendered}");
            }
        }
        Command::Bench {
            input,
            backend,
            repetitions,
            output,
        } => {
            let kind = BackendKind::from(backend);
            let (report, _) = benchmark(&input, kind, repetitions)?;
            let rendered = serde_json::to_string_pretty(&report)?;
            if let Some(path) = output {
                fs::write(path, rendered.as_bytes())?;
            } else {
                println!("{rendered}");
            }
            bail!("single-backend benchmark cannot support a performance claim; use bench-compare");
        }
        Command::BenchCompare {
            input,
            repetitions,
            output,
        } => {
            let (mut cpu, cpu_result) = benchmark(&input, BackendKind::CpuF64, repetitions)?;
            let (mut metal, metal_result) =
                benchmark(&input, BackendKind::MetalMixed, repetitions)?;
            ensure!(
                cpu.provenance.git_revision == metal.provenance.git_revision
                    && cpu.provenance.input_sha256 == metal.provenance.input_sha256
                    && cpu.provenance.pseudopotential_sha256
                        == metal.provenance.pseudopotential_sha256,
                "matched benchmark provenance differs"
            );
            ensure!(
                matched_tolerances(&cpu_result)? == matched_tolerances(&metal_result)?,
                "matched benchmark convergence tolerances differ"
            );
            let scientific_comparison = compare_results(&metal_result, &cpu_result)?;
            let median_speedup = cpu.median_seconds / metal.median_seconds;
            let accepted = cpu.converged
                && metal.converged
                && scientific_comparison.accepted
                && median_speedup >= 1.20;
            let stop_reason = (!accepted).then(|| {
                format!(
                    "matched benchmark failed: converged CPU/Metal={}/{}, scientific acceptance={}, median speedup={median_speedup:.3}x (required 1.20x)",
                    cpu.converged, metal.converged, scientific_comparison.accepted
                )
            });
            cpu.stop_reason = None;
            metal.stop_reason = None;
            let report = MatchedBenchmarkComparison {
                schema_version: BENCHMARK_SCHEMA_VERSION,
                accepted_for_performance_claim: accepted,
                convergence_profile: "matched-metal-compatible-v1",
                repetitions,
                cpu,
                metal,
                scientific_comparison,
                median_speedup,
                stop_reason,
            };
            let rendered = serde_json::to_string_pretty(&report)?;
            if let Some(path) = output {
                fs::write(path, rendered.as_bytes())?;
            } else {
                println!("{rendered}");
            }
            if !report.accepted_for_performance_claim {
                bail!("matched benchmark cannot support a performance claim");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparison_enforces_schema_four_forces_without_breaking_legacy_references() {
        let reference =
            read_result(Path::new("results/qe-si2-displaced-force-reference.json")).unwrap();
        let identical = compare_results(&reference, &reference).unwrap();
        assert!(identical.accepted);
        assert!(identical.force_comparison_performed);
        assert_eq!(identical.maximum_force_error_ry_bohr, Some(0.0));

        let mut perturbed = reference.clone();
        perturbed.forces_ry_bohr.as_mut().unwrap()[0][0] += 2.0e-4;
        let rejected = compare_results(&perturbed, &reference).unwrap();
        assert!(!rejected.accepted);
        assert!(rejected.force_comparison_performed);

        let mut legacy = reference.clone();
        legacy.schema_version = 3;
        legacy.forces_ry_bohr = None;
        legacy.validate().unwrap();
        let compatible = compare_results(&reference, &legacy).unwrap();
        assert!(compatible.accepted);
        assert!(!compatible.force_comparison_performed);
        assert_eq!(compatible.maximum_force_error_ry_bohr, None);
    }
}
