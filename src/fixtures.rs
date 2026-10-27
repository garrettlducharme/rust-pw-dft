use crate::provenance::sha256;
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Component, Path, PathBuf};

pub const FIXTURE_MANIFEST_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixtureManifest {
    pub schema_version: u32,
    pub fixtures: Vec<FixtureEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FixtureKind {
    Input,
    Pseudopotential,
    ReferenceResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RedistributionPolicy {
    ProjectTracked,
    NotRedistributed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixtureEntry {
    pub id: String,
    pub kind: FixtureKind,
    pub path: PathBuf,
    pub sha256: String,
    pub source: String,
    pub source_revision: String,
    pub license: String,
    pub redistribution: RedistributionPolicy,
    pub description: String,
    #[serde(default)]
    pub reference_command: Option<String>,
    #[serde(default)]
    pub reference_input_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct VerifiedFixture {
    pub id: String,
    pub kind: FixtureKind,
    pub path: PathBuf,
    pub sha256: String,
    pub source: String,
    pub source_revision: String,
    pub license: String,
    pub redistribution: RedistributionPolicy,
}

#[derive(Debug, Clone, Serialize)]
pub struct FixtureVerification {
    pub schema_version: u32,
    pub manifest_path: PathBuf,
    pub verified_count: usize,
    pub fixtures: Vec<VerifiedFixture>,
}

fn validate_relative_path(path: &Path) -> Result<()> {
    ensure!(!path.as_os_str().is_empty(), "fixture path is empty");
    ensure!(!path.is_absolute(), "fixture path must be relative");
    ensure!(
        path.components()
            .all(|component| matches!(component, Component::Normal(_))),
        "fixture path may contain only normal relative components"
    );
    Ok(())
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    ensure!(
        value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "{label} must be a 64-character hexadecimal SHA-256"
    );
    Ok(())
}

pub fn load_manifest(path: &Path) -> Result<FixtureManifest> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("failed to read fixture manifest {}", path.display()))?;
    let manifest: FixtureManifest = toml::from_str(&contents)
        .with_context(|| format!("invalid fixture manifest {}", path.display()))?;
    ensure!(
        manifest.schema_version == FIXTURE_MANIFEST_SCHEMA_VERSION,
        "unsupported fixture manifest schema_version {}; expected {}",
        manifest.schema_version,
        FIXTURE_MANIFEST_SCHEMA_VERSION
    );
    ensure!(!manifest.fixtures.is_empty(), "fixture manifest is empty");
    let mut identifiers = BTreeSet::new();
    for fixture in &manifest.fixtures {
        ensure!(!fixture.id.trim().is_empty(), "fixture id is empty");
        ensure!(
            identifiers.insert(fixture.id.as_str()),
            "duplicate fixture id {}",
            fixture.id
        );
        validate_relative_path(&fixture.path)?;
        validate_sha256(&fixture.sha256, "fixture sha256")?;
        if let Some(reference_hash) = &fixture.reference_input_sha256 {
            validate_sha256(reference_hash, "reference_input_sha256")?;
        }
        ensure!(
            !fixture.source.trim().is_empty()
                && !fixture.source_revision.trim().is_empty()
                && !fixture.license.trim().is_empty()
                && !fixture.description.trim().is_empty(),
            "fixture {} has incomplete provenance",
            fixture.id
        );
    }
    Ok(manifest)
}

pub fn verify_manifest(
    manifest_path: &Path,
    repository_root: &Path,
) -> Result<FixtureVerification> {
    let manifest = load_manifest(manifest_path)?;
    let mut verified = Vec::with_capacity(manifest.fixtures.len());
    for fixture in manifest.fixtures {
        let path = repository_root.join(&fixture.path);
        if !path.is_file() {
            bail!(
                "fixture {} is missing at {}; source: {}",
                fixture.id,
                path.display(),
                fixture.source
            );
        }
        let actual = sha256(&path)?;
        ensure!(
            actual.eq_ignore_ascii_case(&fixture.sha256),
            "fixture {} checksum mismatch: expected {}, found {}",
            fixture.id,
            fixture.sha256,
            actual
        );
        verified.push(VerifiedFixture {
            id: fixture.id,
            kind: fixture.kind,
            path: fixture.path,
            sha256: actual,
            source: fixture.source,
            source_revision: fixture.source_revision,
            license: fixture.license,
            redistribution: fixture.redistribution,
        });
    }
    Ok(FixtureVerification {
        schema_version: manifest.schema_version,
        manifest_path: manifest_path.to_path_buf(),
        verified_count: verified.len(),
        fixtures: verified,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static SEQUENCE: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn fixture_manifest_verifies_and_rejects_tampering_and_traversal() {
        let root = std::env::temp_dir().join(format!(
            "rust-pw-dft-fixtures-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(root.join("data")).unwrap();
        fs::write(root.join("data/value.txt"), b"verified fixture\n").unwrap();
        let expected = sha256(&root.join("data/value.txt")).unwrap();
        let manifest_path = root.join("manifest.toml");
        fs::write(
            &manifest_path,
            format!(
                r#"schema_version = 1
[[fixtures]]
id = "value"
kind = "input"
path = "data/value.txt"
sha256 = "{expected}"
source = "unit-test"
source_revision = "1"
license = "CC0-1.0"
redistribution = "project-tracked"
description = "test fixture"
"#
            ),
        )
        .unwrap();
        assert_eq!(
            verify_manifest(&manifest_path, &root)
                .unwrap()
                .verified_count,
            1
        );

        fs::write(root.join("data/value.txt"), b"tampered\n").unwrap();
        assert!(
            verify_manifest(&manifest_path, &root)
                .unwrap_err()
                .to_string()
                .contains("checksum mismatch")
        );

        let unsafe_manifest = fs::read_to_string(&manifest_path)
            .unwrap()
            .replace("data/value.txt", "../value.txt");
        fs::write(&manifest_path, unsafe_manifest).unwrap();
        assert!(
            load_manifest(&manifest_path)
                .unwrap_err()
                .to_string()
                .contains("normal relative components")
        );
        fs::remove_dir_all(root).unwrap();
    }
}
