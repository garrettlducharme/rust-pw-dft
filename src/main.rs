use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, Subcommand, ValueEnum};
use rust_pw_dft::backend::BackendKind;
use rust_pw_dft::result::{Provenance, ScfResult};
use rust_pw_dft::scf::{run, run_matched_benchmark};
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

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
    },
    Compare {
        result: PathBuf,
        reference: PathBuf,
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
    accepted: bool,
    total_energy_error_mev_atom: f64,
    maximum_occupied_eigenvalue_error_mev: f64,
    electron_count_error: f64,
}

#[derive(Serialize)]
struct Benchmark {
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
    accepted_for_performance_claim: bool,
    convergence_profile: &'static str,
    repetitions: usize,
    cpu: Benchmark,
    metal: Benchmark,
    scientific_comparison: Comparison,
    median_speedup: f64,
    stop_reason: Option<String>,
}

fn read_result(path: &Path) -> Result<ScfResult> {
    serde_json::from_slice(
        &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("invalid result {}", path.display()))
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
        result.eigenvalues_ry.len() == reference.eigenvalues_ry.len(),
        "eigenvalue counts differ"
    );
    ensure!(
        result.atom_count > 0 && result.atom_count == reference.atom_count,
        "comparison requires matching nonzero atom counts"
    );
    let max_eigen_ry = result
        .eigenvalues_ry
        .iter()
        .zip(&reference.eigenvalues_ry)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let electron_error = (result
        .electron_count
        .context("result has no electron count")?
        - reference
            .electron_count
            .context("reference has no electron count")?)
    .abs();
    // 1 Ry = 13.605693122994 eV.
    let energy_mev =
        (energy - reference_energy).abs() * 13_605.693_122_994 / result.atom_count as f64;
    let eigen_mev = max_eigen_ry * 13_605.693_122_994;
    let (energy_tolerance_mev_atom, electron_tolerance) =
        if result.provenance.actual_backend == BackendKind::MetalMixed {
            (1.0, 1.0e-5)
        } else {
            (2.0, 1.0e-8)
        };
    Ok(Comparison {
        accepted: energy_mev <= energy_tolerance_mev_atom
            && eigen_mev <= 5.0
            && electron_error <= electron_tolerance,
        total_energy_error_mev_atom: energy_mev,
        maximum_occupied_eigenvalue_error_mev: eigen_mev,
        electron_count_error: electron_error,
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
        } => {
            let result = run(&input, backend.into())?;
            if let Some(parent) = output.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&output, serde_json::to_vec_pretty(&result)?)?;
            println!(
                "actual backend: {:?}; device: {}",
                result.provenance.actual_backend, result.provenance.actual_device
            );
            if !result.converged {
                bail!(
                    "SCF did not run: {}",
                    result.stop_reason.as_deref().unwrap_or("unknown reason")
                );
            }
        }
        Command::Compare { result, reference } => println!(
            "{}",
            serde_json::to_string_pretty(&compare(&result, &reference)?)?
        ),
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
