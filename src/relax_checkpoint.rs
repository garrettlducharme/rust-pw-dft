use crate::backend::BackendKind;
use crate::input::{AtomInput, CellInput, ScfOptions, SimulationInput};
use crate::kpoints::{KPoint, KPointSamplingInput, KPointSpecification, OccupationScheme};
use crate::relax::{RelaxationOptions, RelaxationStep};
use crate::result::ScfResult;
use crate::xc::XcFunctional;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub const RELAXATION_CHECKPOINT_SCHEMA_VERSION: u32 = 1;
const MAGIC: &[u8; 8] = b"RPDFTRX1";
const HEADER_BYTES: usize = 8 + 4 + 8 + 32;
const MAX_CHECKPOINT_BYTES: u64 = 16 * 1024 * 1024 * 1024;
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RelaxationCheckpoint {
    pub schema_version: u32,
    pub writer_git_revision: String,
    pub initial_input_sha256: String,
    pub pseudopotential_sha256: BTreeMap<String, String>,
    pub requested_backend: BackendKind,
    pub options: RelaxationOptions,
    pub accepted_steps: usize,
    pub current_input: SimulationInput,
    pub current_scf: ScfResult,
    pub direction: Vec<[f64; 3]>,
    pub projected_force: Vec<[f64; 3]>,
    pub proposed_step_bohr: f64,
    pub total_scf_evaluations: usize,
    pub history: Vec<RelaxationStep>,
    pub elapsed_seconds: f64,
}

#[derive(Debug, Clone)]
pub struct LoadedRelaxationCheckpoint {
    pub checkpoint: RelaxationCheckpoint,
    pub file_sha256: String,
}

// `KPointSamplingInput` and `OccupationScheme` use human-readable tagged
// Serde representations that bincode cannot deserialize. This private wire
// format keeps every floating-point value bit-exact across a restart while the
// public checkpoint state continues to use the normal input types.
#[derive(Serialize, Deserialize)]
enum KPointSamplingWire {
    Legacy(String),
    Gamma,
    MonkhorstPack {
        grid: [usize; 3],
        shift: [f64; 3],
        reduce_time_reversal: bool,
    },
    Explicit {
        points: Vec<KPoint>,
        reduce_time_reversal: bool,
    },
}

#[derive(Serialize, Deserialize)]
enum OccupationWire {
    Fixed,
    FermiDirac {
        smearing_ry: f64,
        extra_bands: usize,
    },
}

#[derive(Serialize, Deserialize)]
struct SimulationInputWire {
    schema_version: u32,
    cell: CellInput,
    atoms: Vec<AtomInput>,
    pseudopotentials: BTreeMap<String, PathBuf>,
    cutoff_ry: f64,
    xc: XcFunctional,
    scf_options: ScfOptions,
    k_points: KPointSamplingWire,
    occupations: OccupationWire,
    spin_components: u8,
    magnetization_electrons: Option<f64>,
}

impl From<&SimulationInput> for SimulationInputWire {
    fn from(input: &SimulationInput) -> Self {
        let k_points = match &input.k_points {
            KPointSamplingInput::Legacy(value) => KPointSamplingWire::Legacy(value.clone()),
            KPointSamplingInput::Detailed(KPointSpecification::Gamma) => KPointSamplingWire::Gamma,
            KPointSamplingInput::Detailed(KPointSpecification::MonkhorstPack {
                grid,
                shift,
                reduce_time_reversal,
            }) => KPointSamplingWire::MonkhorstPack {
                grid: *grid,
                shift: *shift,
                reduce_time_reversal: *reduce_time_reversal,
            },
            KPointSamplingInput::Detailed(KPointSpecification::Explicit {
                points,
                reduce_time_reversal,
            }) => KPointSamplingWire::Explicit {
                points: points.clone(),
                reduce_time_reversal: *reduce_time_reversal,
            },
        };
        let occupations = match input.occupations {
            OccupationScheme::Fixed => OccupationWire::Fixed,
            OccupationScheme::FermiDirac {
                smearing_ry,
                extra_bands,
            } => OccupationWire::FermiDirac {
                smearing_ry,
                extra_bands,
            },
        };
        Self {
            schema_version: input.schema_version,
            cell: input.cell.clone(),
            atoms: input.atoms.clone(),
            pseudopotentials: input.pseudopotentials.clone(),
            cutoff_ry: input.cutoff_ry,
            xc: input.xc,
            scf_options: input.scf_options.clone(),
            k_points,
            occupations,
            spin_components: input.spin_components,
            magnetization_electrons: input.magnetization_electrons,
        }
    }
}

impl SimulationInputWire {
    fn into_input(self) -> SimulationInput {
        let k_points = match self.k_points {
            KPointSamplingWire::Legacy(value) => KPointSamplingInput::Legacy(value),
            KPointSamplingWire::Gamma => KPointSamplingInput::Detailed(KPointSpecification::Gamma),
            KPointSamplingWire::MonkhorstPack {
                grid,
                shift,
                reduce_time_reversal,
            } => KPointSamplingInput::Detailed(KPointSpecification::MonkhorstPack {
                grid,
                shift,
                reduce_time_reversal,
            }),
            KPointSamplingWire::Explicit {
                points,
                reduce_time_reversal,
            } => KPointSamplingInput::Detailed(KPointSpecification::Explicit {
                points,
                reduce_time_reversal,
            }),
        };
        let occupations = match self.occupations {
            OccupationWire::Fixed => OccupationScheme::Fixed,
            OccupationWire::FermiDirac {
                smearing_ry,
                extra_bands,
            } => OccupationScheme::FermiDirac {
                smearing_ry,
                extra_bands,
            },
        };
        SimulationInput {
            schema_version: self.schema_version,
            cell: self.cell,
            atoms: self.atoms,
            pseudopotentials: self.pseudopotentials,
            cutoff_ry: self.cutoff_ry,
            xc: self.xc,
            scf_options: self.scf_options,
            k_points,
            occupations,
            spin_components: self.spin_components,
            magnetization_electrons: self.magnetization_electrons,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct RelaxationCheckpointWire {
    schema_version: u32,
    writer_git_revision: String,
    initial_input_sha256: String,
    pseudopotential_sha256: BTreeMap<String, String>,
    requested_backend: BackendKind,
    options: RelaxationOptions,
    accepted_steps: usize,
    current_input: SimulationInputWire,
    current_scf: ScfResult,
    direction: Vec<[f64; 3]>,
    projected_force: Vec<[f64; 3]>,
    proposed_step_bohr: f64,
    total_scf_evaluations: usize,
    history: Vec<RelaxationStep>,
    elapsed_seconds: f64,
}

impl From<&RelaxationCheckpoint> for RelaxationCheckpointWire {
    fn from(checkpoint: &RelaxationCheckpoint) -> Self {
        Self {
            schema_version: checkpoint.schema_version,
            writer_git_revision: checkpoint.writer_git_revision.clone(),
            initial_input_sha256: checkpoint.initial_input_sha256.clone(),
            pseudopotential_sha256: checkpoint.pseudopotential_sha256.clone(),
            requested_backend: checkpoint.requested_backend,
            options: checkpoint.options.clone(),
            accepted_steps: checkpoint.accepted_steps,
            current_input: SimulationInputWire::from(&checkpoint.current_input),
            current_scf: checkpoint.current_scf.clone(),
            direction: checkpoint.direction.clone(),
            projected_force: checkpoint.projected_force.clone(),
            proposed_step_bohr: checkpoint.proposed_step_bohr,
            total_scf_evaluations: checkpoint.total_scf_evaluations,
            history: checkpoint.history.clone(),
            elapsed_seconds: checkpoint.elapsed_seconds,
        }
    }
}

impl RelaxationCheckpointWire {
    fn into_checkpoint(self) -> RelaxationCheckpoint {
        RelaxationCheckpoint {
            schema_version: self.schema_version,
            writer_git_revision: self.writer_git_revision,
            initial_input_sha256: self.initial_input_sha256,
            pseudopotential_sha256: self.pseudopotential_sha256,
            requested_backend: self.requested_backend,
            options: self.options,
            accepted_steps: self.accepted_steps,
            current_input: self.current_input.into_input(),
            current_scf: self.current_scf,
            direction: self.direction,
            projected_force: self.projected_force,
            proposed_step_bohr: self.proposed_step_bohr,
            total_scf_evaluations: self.total_scf_evaluations,
            history: self.history,
            elapsed_seconds: self.elapsed_seconds,
        }
    }
}

fn digest_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn temporary_path(path: &Path) -> Result<PathBuf> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("relaxation checkpoint path must have a UTF-8 file name")?;
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    Ok(parent.join(format!(".{name}.tmp-{}-{sequence}", std::process::id())))
}

pub fn write_atomic(path: &Path, checkpoint: &RelaxationCheckpoint) -> Result<String> {
    ensure!(
        checkpoint.schema_version == RELAXATION_CHECKPOINT_SCHEMA_VERSION,
        "cannot write unsupported relaxation checkpoint schema"
    );
    let wire = RelaxationCheckpointWire::from(checkpoint);
    let payload =
        rmp_serde::to_vec_named(&wire).context("failed to encode relaxation checkpoint")?;
    ensure!(
        payload.len() as u64 <= MAX_CHECKPOINT_BYTES,
        "relaxation checkpoint exceeds the 16 GiB safety limit"
    );
    let payload_digest = Sha256::digest(&payload);
    let mut encoded = Vec::with_capacity(HEADER_BYTES + payload.len());
    encoded.extend_from_slice(MAGIC);
    encoded.extend_from_slice(&RELAXATION_CHECKPOINT_SCHEMA_VERSION.to_le_bytes());
    encoded.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    encoded.extend_from_slice(&payload_digest);
    encoded.extend_from_slice(&payload);

    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).with_context(|| {
        format!(
            "failed to create relaxation checkpoint directory {}",
            parent.display()
        )
    })?;
    let temporary = temporary_path(path)?;
    let write_result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .with_context(|| format!("failed to create {}", temporary.display()))?;
        file.write_all(&encoded)
            .context("failed to write relaxation checkpoint")?;
        file.sync_all()
            .context("failed to sync relaxation checkpoint")?;
        fs::rename(&temporary, path).with_context(|| {
            format!(
                "failed to atomically replace relaxation checkpoint {}",
                path.display()
            )
        })?;
        if let Ok(directory) = OpenOptions::new().read(true).open(parent) {
            let _ = directory.sync_all();
        }
        Ok(())
    })();
    if write_result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    write_result?;
    Ok(digest_hex(&encoded))
}

pub fn read(path: &Path) -> Result<LoadedRelaxationCheckpoint> {
    let metadata = fs::metadata(path)
        .with_context(|| format!("failed to inspect relaxation checkpoint {}", path.display()))?;
    ensure!(
        metadata.is_file(),
        "relaxation checkpoint is not a regular file"
    );
    ensure!(
        metadata.len() >= HEADER_BYTES as u64 && metadata.len() <= MAX_CHECKPOINT_BYTES,
        "relaxation checkpoint size is invalid"
    );
    let mut encoded = Vec::with_capacity(metadata.len() as usize);
    fs::File::open(path)
        .with_context(|| format!("failed to open relaxation checkpoint {}", path.display()))?
        .read_to_end(&mut encoded)
        .context("failed to read relaxation checkpoint")?;
    ensure!(
        encoded.len() >= HEADER_BYTES,
        "relaxation checkpoint was truncated while reading"
    );
    ensure!(
        &encoded[..8] == MAGIC,
        "relaxation checkpoint magic is invalid"
    );
    let envelope_version = u32::from_le_bytes(encoded[8..12].try_into().unwrap());
    ensure!(
        envelope_version == RELAXATION_CHECKPOINT_SCHEMA_VERSION,
        "unsupported relaxation checkpoint envelope version {envelope_version}"
    );
    let payload_len = u64::from_le_bytes(encoded[12..20].try_into().unwrap()) as usize;
    ensure!(
        payload_len == encoded.len() - HEADER_BYTES,
        "relaxation checkpoint payload length is invalid"
    );
    let expected_digest = &encoded[20..52];
    let payload = &encoded[HEADER_BYTES..];
    ensure!(
        Sha256::digest(payload).as_slice() == expected_digest,
        "relaxation checkpoint payload checksum mismatch"
    );
    let wire: RelaxationCheckpointWire =
        rmp_serde::from_slice(payload).context("failed to decode relaxation checkpoint")?;
    let checkpoint = wire.into_checkpoint();
    ensure!(
        checkpoint.schema_version == RELAXATION_CHECKPOINT_SCHEMA_VERSION,
        "unsupported relaxation checkpoint schema_version {}",
        checkpoint.schema_version
    );
    Ok(LoadedRelaxationCheckpoint {
        checkpoint,
        file_sha256: digest_hex(&encoded),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::relax::RelaxationResult;

    #[test]
    fn relaxation_checkpoint_round_trip_and_checksum_fail_closed() {
        let result: RelaxationResult = serde_json::from_str(
            &fs::read_to_string("results/si2-displaced-cpu-relaxation.json").unwrap(),
        )
        .unwrap();
        let input = SimulationInput::from_path(Path::new("examples/si2-displaced.toml")).unwrap();
        let checkpoint = RelaxationCheckpoint {
            schema_version: RELAXATION_CHECKPOINT_SCHEMA_VERSION,
            writer_git_revision: "test-revision".into(),
            initial_input_sha256: result.steps[0].scf_input_sha256.clone(),
            pseudopotential_sha256: result.final_scf.provenance.pseudopotential_sha256.clone(),
            requested_backend: BackendKind::CpuF64,
            options: result.options.clone(),
            accepted_steps: 1,
            current_input: input,
            current_scf: result.final_scf.clone(),
            direction: vec![[0.0; 3]; result.atom_count],
            projected_force: vec![[0.0; 3]; result.atom_count],
            proposed_step_bohr: result.options.initial_step_bohr,
            total_scf_evaluations: result.total_scf_evaluations,
            history: result.steps,
            elapsed_seconds: 0.5,
        };
        let path = std::env::temp_dir().join(format!(
            "rust-pw-dft-relax-checkpoint-{}-{}.bin",
            std::process::id(),
            TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let written_hash = write_atomic(&path, &checkpoint).unwrap();
        let loaded = read(&path).unwrap();
        assert_eq!(loaded.file_sha256, written_hash);
        assert_eq!(loaded.checkpoint.accepted_steps, 1);

        let mut bytes = fs::read(&path).unwrap();
        *bytes.last_mut().unwrap() ^= 1;
        fs::write(&path, bytes).unwrap();
        assert!(
            read(&path)
                .unwrap_err()
                .to_string()
                .contains("checksum mismatch")
        );
        fs::remove_file(path).unwrap();
    }
}
