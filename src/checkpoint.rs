use crate::backend::{BackendKind, PrecisionPolicy};
use crate::result::ScfIterationRecord;
use anyhow::{Context, Result, ensure};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub const CHECKPOINT_SCHEMA_VERSION: u32 = 4;
const MAGIC: &[u8; 8] = b"RPDFTCK4";
const HEADER_BYTES: usize = 8 + 4 + 8 + 32;
const MAX_CHECKPOINT_BYTES: u64 = 16 * 1024 * 1024 * 1024;
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointKPointState {
    pub spin_channel: Option<usize>,
    pub reduced: [f64; 3],
    pub weight: f64,
    pub g_vector_count: usize,
    pub band_count: usize,
    pub orbitals: Vec<Vec<Complex64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScfCheckpoint {
    pub schema_version: u32,
    pub writer_git_revision: String,
    pub input_sha256: String,
    pub pseudopotential_sha256: BTreeMap<String, String>,
    pub requested_backend: BackendKind,
    pub precision_policy: PrecisionPolicy,
    pub run_profile: String,
    pub iteration_completed: usize,
    pub fft_shape: [usize; 3],
    pub density_channels: Vec<Vec<f64>>,
    pub previous_energy_ry: f64,
    pub k_points: Vec<CheckpointKPointState>,
    pub davidson_iterations: usize,
    pub hamiltonian_applications: usize,
    pub history: Vec<ScfIterationRecord>,
    pub timings: BTreeMap<String, f64>,
    pub elapsed_seconds: f64,
    pub backend_metrics: BTreeMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct LoadedCheckpoint {
    pub checkpoint: ScfCheckpoint,
    pub file_sha256: String,
}

fn digest_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn temporary_path(path: &Path) -> Result<PathBuf> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("checkpoint path must have a UTF-8 file name")?;
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    Ok(parent.join(format!(".{name}.tmp-{}-{sequence}", std::process::id())))
}

pub fn write_atomic(path: &Path, checkpoint: &ScfCheckpoint) -> Result<String> {
    ensure!(
        checkpoint.schema_version == CHECKPOINT_SCHEMA_VERSION,
        "cannot write unsupported checkpoint schema"
    );
    let payload = bincode::serde::encode_to_vec(checkpoint, bincode::config::standard())
        .context("failed to encode checkpoint")?;
    ensure!(
        payload.len() as u64 <= MAX_CHECKPOINT_BYTES,
        "checkpoint exceeds the 16 GiB safety limit"
    );
    let payload_digest = Sha256::digest(&payload);
    let mut encoded = Vec::with_capacity(HEADER_BYTES + payload.len());
    encoded.extend_from_slice(MAGIC);
    encoded.extend_from_slice(&CHECKPOINT_SCHEMA_VERSION.to_le_bytes());
    encoded.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    encoded.extend_from_slice(&payload_digest);
    encoded.extend_from_slice(&payload);

    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .with_context(|| format!("failed to create checkpoint directory {}", parent.display()))?;
    let temporary = temporary_path(path)?;
    let write_result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .with_context(|| format!("failed to create {}", temporary.display()))?;
        file.write_all(&encoded)
            .context("failed to write checkpoint")?;
        file.sync_all().context("failed to sync checkpoint")?;
        fs::rename(&temporary, path).with_context(|| {
            format!("failed to atomically replace checkpoint {}", path.display())
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

pub fn read(path: &Path) -> Result<LoadedCheckpoint> {
    let metadata = fs::metadata(path)
        .with_context(|| format!("failed to inspect checkpoint {}", path.display()))?;
    ensure!(metadata.is_file(), "checkpoint is not a regular file");
    ensure!(
        metadata.len() >= HEADER_BYTES as u64 && metadata.len() <= MAX_CHECKPOINT_BYTES,
        "checkpoint size is invalid"
    );
    let mut encoded = Vec::with_capacity(metadata.len() as usize);
    fs::File::open(path)
        .with_context(|| format!("failed to open checkpoint {}", path.display()))?
        .read_to_end(&mut encoded)
        .context("failed to read checkpoint")?;
    ensure!(
        encoded.len() >= HEADER_BYTES,
        "checkpoint was truncated while reading"
    );
    ensure!(&encoded[..8] == MAGIC, "checkpoint magic is invalid");
    let envelope_version = u32::from_le_bytes(encoded[8..12].try_into().unwrap());
    ensure!(
        envelope_version == CHECKPOINT_SCHEMA_VERSION,
        "unsupported checkpoint envelope version {envelope_version}"
    );
    let payload_len = u64::from_le_bytes(encoded[12..20].try_into().unwrap()) as usize;
    ensure!(
        payload_len == encoded.len() - HEADER_BYTES,
        "checkpoint payload length is invalid"
    );
    let expected_digest = &encoded[20..52];
    let payload = &encoded[HEADER_BYTES..];
    ensure!(
        Sha256::digest(payload).as_slice() == expected_digest,
        "checkpoint payload checksum mismatch"
    );
    let (checkpoint, consumed): (ScfCheckpoint, usize) =
        bincode::serde::decode_from_slice(payload, bincode::config::standard())
            .context("failed to decode checkpoint")?;
    ensure!(
        consumed == payload.len(),
        "checkpoint has trailing payload data"
    );
    ensure!(
        checkpoint.schema_version == CHECKPOINT_SCHEMA_VERSION,
        "unsupported checkpoint schema_version {}",
        checkpoint.schema_version
    );
    Ok(LoadedCheckpoint {
        checkpoint,
        file_sha256: digest_hex(&encoded),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> ScfCheckpoint {
        ScfCheckpoint {
            schema_version: CHECKPOINT_SCHEMA_VERSION,
            writer_git_revision: "test-revision".into(),
            input_sha256: "input".into(),
            pseudopotential_sha256: BTreeMap::from([("Si".into(), "pseudo".into())]),
            requested_backend: BackendKind::CpuF64,
            precision_policy: PrecisionPolicy::cpu_f64(),
            run_profile: "scientific-v1".into(),
            iteration_completed: 3,
            fft_shape: [4, 4, 4],
            density_channels: vec![vec![0.25; 64]],
            previous_energy_ry: -1.25,
            k_points: vec![CheckpointKPointState {
                spin_channel: None,
                reduced: [0.0, 0.0, 0.0],
                weight: 1.0,
                g_vector_count: 7,
                band_count: 2,
                orbitals: vec![vec![Complex64::new(1.0, 0.0); 7]; 2],
            }],
            davidson_iterations: 8,
            hamiltonian_applications: 14,
            history: Vec::new(),
            timings: BTreeMap::from([("setup_seconds".into(), 0.1)]),
            elapsed_seconds: 0.2,
            backend_metrics: BTreeMap::new(),
        }
    }

    #[test]
    fn checkpoint_round_trip_and_checksum_fail_closed() {
        let path = std::env::temp_dir().join(format!(
            "rust-pw-dft-checkpoint-{}-{}.bin",
            std::process::id(),
            TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let written_hash = write_atomic(&path, &fixture()).unwrap();
        let loaded = read(&path).unwrap();
        assert_eq!(loaded.file_sha256, written_hash);
        assert_eq!(loaded.checkpoint.iteration_completed, 3);
        assert_eq!(
            loaded.checkpoint.k_points[0].orbitals[0][0],
            Complex64::new(1.0, 0.0)
        );

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
