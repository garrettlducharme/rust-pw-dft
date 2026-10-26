use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, Subcommand, ValueEnum};
use rust_pw_dft::backend::BackendKind;
use rust_pw_dft::result::{Provenance, ScfResult};
use rust_pw_dft::scf::run;
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

fn read_result(path: &Path) -> Result<ScfResult> {
    serde_json::from_slice(
        &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("invalid result {}", path.display()))
}

fn compare(result_path: &Path, reference_path: &Path) -> Result<Comparison> {
    let result = read_result(result_path)?;
    let reference = read_result(reference_path)?;
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
            ensure!(repetitions >= 3, "at least three repetitions are required");
            let kind = BackendKind::from(backend);
            let _cold = run(&input, kind)?;
            let mut timings = Vec::with_capacity(repetitions);
            let mut last = None;
            for _ in 0..repetitions {
                let started = Instant::now();
                last = Some(run(&input, kind)?);
                timings.push(started.elapsed().as_secs_f64());
            }
            timings.sort_by(f64::total_cmp);
            let last = last.unwrap();
            let performance_eligible = last.converged && kind == BackendKind::CpuF64;
            let benchmark_stop_reason = if kind == BackendKind::MetalMixed && last.converged {
                Some(
                    "Metal uses persistent operator buffers and GPU cross-Gram partial reductions, but the Davidson basis and correction vectors remain host FP64 and are staged by block; a fully resident eigensolver is required for a production performance claim"
                        .into(),
                )
            } else {
                last.stop_reason.clone()
            };
            let report = Benchmark {
                accepted_for_performance_claim: performance_eligible,
                requested_backend: kind,
                actual_backend: last.provenance.actual_backend,
                converged: last.converged,
                iterations: last.iterations,
                repetitions,
                min_seconds: timings[0],
                median_seconds: timings[timings.len() / 2],
                max_seconds: *timings.last().unwrap(),
                provenance: last.provenance,
                stop_reason: benchmark_stop_reason,
            };
            let rendered = serde_json::to_string_pretty(&report)?;
            if let Some(path) = output {
                fs::write(path, rendered.as_bytes())?;
            } else {
                println!("{rendered}");
            }
            if !report.accepted_for_performance_claim {
                bail!("benchmark cannot support a performance claim; see its stop_reason");
            }
        }
    }
    Ok(())
}
