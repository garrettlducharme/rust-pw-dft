use crate::kpoints::{KPointSamplingInput, OccupationScheme};
use crate::xc::XcFunctional;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

pub const INPUT_SCHEMA_VERSION: u32 = 3;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimulationInput {
    pub schema_version: u32,
    pub cell: CellInput,
    pub atoms: Vec<AtomInput>,
    pub pseudopotentials: BTreeMap<String, PathBuf>,
    pub cutoff_ry: f64,
    pub xc: XcFunctional,
    #[serde(default)]
    pub scf_options: ScfOptions,
    #[serde(default)]
    pub k_points: KPointSamplingInput,
    #[serde(default)]
    pub occupations: OccupationScheme,
    #[serde(default = "one")]
    pub spin_components: u8,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub magnetization_electrons: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CellInput {
    /// Direct-lattice vectors in bohr, stored as rows.
    pub lattice_bohr: [[f64; 3]; 3],
    #[serde(default = "fixed_true")]
    pub fixed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AtomInput {
    pub element: String,
    pub fractional: [f64; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScfOptions {
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,
    #[serde(default = "default_energy_tolerance")]
    pub energy_tolerance_ry: f64,
    #[serde(default = "default_density_tolerance")]
    pub density_tolerance: f64,
    #[serde(default = "default_residual_tolerance")]
    pub orbital_residual_tolerance: f64,
    #[serde(default = "default_mixing")]
    pub mixing_beta: f64,
    #[serde(default = "default_seed")]
    pub seed: u64,
}

impl Default for ScfOptions {
    fn default() -> Self {
        Self {
            max_iterations: default_max_iterations(),
            energy_tolerance_ry: default_energy_tolerance(),
            density_tolerance: default_density_tolerance(),
            orbital_residual_tolerance: default_residual_tolerance(),
            mixing_beta: default_mixing(),
            seed: default_seed(),
        }
    }
}

const fn one() -> u8 {
    1
}
const fn fixed_true() -> bool {
    true
}
const fn default_max_iterations() -> usize {
    80
}
const fn default_energy_tolerance() -> f64 {
    1.0e-8
}
const fn default_density_tolerance() -> f64 {
    1.0e-8
}
const fn default_residual_tolerance() -> f64 {
    1.0e-8
}
const fn default_mixing() -> f64 {
    0.3
}
const fn default_seed() -> u64 {
    0x05ee_dd17
}

impl SimulationInput {
    pub fn from_path(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("failed to read input {}", path.display()))?;
        let mut value: Self =
            toml::from_str(&text).with_context(|| format!("invalid input {}", path.display()))?;
        let base = path.parent().unwrap_or_else(|| Path::new("."));
        for pseudo in value.pseudopotentials.values_mut() {
            if pseudo.is_relative() {
                *pseudo = base.join(&*pseudo);
            }
        }
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=INPUT_SCHEMA_VERSION).contains(&self.schema_version),
            "unsupported input schema_version {}; expected 1 through {}",
            self.schema_version,
            INPUT_SCHEMA_VERSION
        );
        ensure!(self.cell.fixed, "cell relaxation is unsupported");
        self.k_points.resolve()?;
        self.occupations.validate()?;
        if self.schema_version == 1 {
            ensure!(
                self.k_points.is_legacy_gamma() && self.occupations == OccupationScheme::Fixed,
                "input schema 1 supports only legacy Gamma-point fixed occupations"
            );
        }
        ensure!(
            matches!(self.spin_components, 1 | 2),
            "spin_components must be 1 or 2"
        );
        if self.spin_components == 1 {
            ensure!(
                self.magnetization_electrons.is_none(),
                "magnetization_electrons requires spin_components = 2"
            );
        } else {
            ensure!(
                self.schema_version >= 3,
                "collinear spin requires input schema 3"
            );
            ensure!(
                self.xc == XcFunctional::LdaPz,
                "collinear spin currently supports only lda-pz"
            );
            ensure!(
                self.k_points.is_legacy_gamma() && self.occupations == OccupationScheme::Fixed,
                "collinear spin currently requires Gamma-point fixed occupations"
            );
            ensure!(
                self.magnetization_electrons
                    .is_some_and(|value| value.is_finite()),
                "collinear spin requires a finite magnetization_electrons"
            );
        }
        ensure!(
            self.cutoff_ry.is_finite() && self.cutoff_ry > 0.0,
            "cutoff_ry must be positive and finite"
        );
        ensure!(!self.atoms.is_empty(), "at least one atom is required");
        ensure!(
            self.scf_options.max_iterations > 0,
            "max_iterations must be positive"
        );
        ensure!(
            (0.0..=1.0).contains(&self.scf_options.mixing_beta)
                && self.scf_options.mixing_beta > 0.0,
            "mixing_beta must be in (0, 1]"
        );
        let mut used_elements = BTreeSet::new();
        for atom in &self.atoms {
            ensure!(
                !atom.element.trim().is_empty() && atom.element == atom.element.trim(),
                "atom element labels must be nonempty and have no surrounding whitespace"
            );
            ensure!(
                self.pseudopotentials.contains_key(&atom.element),
                "no pseudopotential was supplied for {}",
                atom.element
            );
            ensure!(
                atom.fractional.iter().all(|x| x.is_finite()),
                "atom coordinates must be finite"
            );
            used_elements.insert(atom.element.as_str());
        }
        ensure!(
            self.pseudopotentials
                .keys()
                .all(|element| used_elements.contains(element.as_str())),
            "pseudopotential map contains an element absent from the atom list"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unknown_input_keys() {
        let bad = r#"
cutoff_ry = 12.0
schema_version = 1
xc = "lda-pz"
k_points = "gamma"
spin_components = 1
forces = true
pseudopotentials = { Si = "Si.UPF" }
atoms = [{ element = "Si", fractional = [0.0, 0.0, 0.0] }]
[cell]
lattice_bohr = [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]
"#;
        assert!(toml::from_str::<SimulationInput>(bad).is_err());
    }

    #[test]
    fn requires_supported_input_schema_version() {
        let missing = r#"
cutoff_ry = 12.0
xc = "lda-pz"
pseudopotentials = { Si = "Si.UPF" }
atoms = [{ element = "Si", fractional = [0.0, 0.0, 0.0] }]
[cell]
lattice_bohr = [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]
"#;
        assert!(toml::from_str::<SimulationInput>(missing).is_err());

        let mut unsupported = missing.replacen("cutoff_ry", "schema_version = 4\ncutoff_ry", 1);
        let parsed: SimulationInput = toml::from_str(&unsupported).unwrap();
        assert!(parsed.validate().is_err());
        unsupported = unsupported.replacen("schema_version = 4", "schema_version = 3", 1);
        assert!(
            toml::from_str::<SimulationInput>(&unsupported)
                .unwrap()
                .validate()
                .is_ok()
        );
    }

    #[test]
    fn schema_three_admits_only_bounded_collinear_lda_contract() {
        let source = r#"
cutoff_ry = 20.0
schema_version = 3
xc = "lda-pz"
spin_components = 2
magnetization_electrons = 1.0
pseudopotentials = { Al = "Al.UPF" }
atoms = [{ element = "Al", fractional = [0.0, 0.0, 0.0] }]
[cell]
lattice_bohr = [[15.0,0.0,0.0],[0.0,15.0,0.0],[0.0,0.0,15.0]]
"#;
        let input: SimulationInput = toml::from_str(source).unwrap();
        input.validate().unwrap();

        let mut missing_magnetization = input.clone();
        missing_magnetization.magnetization_electrons = None;
        assert!(missing_magnetization.validate().is_err());
        let mut pbe = input.clone();
        pbe.xc = XcFunctional::Pbe;
        assert!(pbe.validate().is_err());
        let mut unpolarized = input;
        unpolarized.spin_components = 1;
        assert!(unpolarized.validate().is_err());
    }

    #[test]
    fn rejects_out_of_scope_physics() {
        let mut input: SimulationInput = toml::from_str(
            r#"
cutoff_ry = 12.0
schema_version = 1
xc = "lda-pz"
pseudopotentials = { Si = "Si.UPF" }
atoms = [{ element = "Si", fractional = [0.0, 0.0, 0.0] }]
[cell]
lattice_bohr = [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]
"#,
        )
        .unwrap();

        input.k_points = KPointSamplingInput::Legacy("2x2x2".into());
        assert!(input.validate().is_err());
        input.k_points = KPointSamplingInput::Legacy("gamma".into());
        input.spin_components = 2;
        assert!(input.validate().is_err());
        input.spin_components = 1;
        input.cell.fixed = false;
        assert!(input.validate().is_err());
        input.cell.fixed = true;
        input.atoms[0].element = "Al".into();
        assert!(input.validate().is_err());
        input.pseudopotentials.insert("Al".into(), "Al.UPF".into());
        input.pseudopotentials.remove("Si");
        assert!(input.validate().is_ok());
        input.pseudopotentials.insert("O".into(), "O.UPF".into());
        assert!(input.validate().is_err());
    }

    #[test]
    fn admits_pbe_and_rejects_unknown_xc_during_parsing() {
        let pbe = r#"
cutoff_ry = 12.0
schema_version = 1
xc = "pbe"
pseudopotentials = { Si = "Si.UPF" }
atoms = [{ element = "Si", fractional = [0.0, 0.0, 0.0] }]
[cell]
lattice_bohr = [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]
"#;
        let parsed = toml::from_str::<SimulationInput>(pbe).unwrap();
        assert_eq!(parsed.xc, XcFunctional::Pbe);
        let error = toml::from_str::<SimulationInput>(&pbe.replace("pbe", "scan")).unwrap_err();
        assert!(error.to_string().contains("unknown variant `scan`"));
    }

    #[test]
    fn admits_exact_multi_species_pseudopotential_map() {
        let input: SimulationInput = toml::from_str(
            r#"
cutoff_ry = 100.0
schema_version = 1
xc = "pbe"
pseudopotentials = { Mg = "Mg.UPF", O = "O.UPF" }
atoms = [
  { element = "Mg", fractional = [0.0, 0.0, 0.0] },
  { element = "O", fractional = [0.5, 0.5, 0.5] },
]
[cell]
lattice_bohr = [[0.0,4.0,4.0],[4.0,0.0,4.0],[4.0,4.0,0.0]]
"#,
        )
        .unwrap();
        input.validate().unwrap();
    }

    #[test]
    fn schema_two_accepts_weighted_k_points_and_smearing() {
        let input: SimulationInput = toml::from_str(
            r#"
cutoff_ry = 30.0
schema_version = 2
xc = "pbe"
pseudopotentials = { Al = "Al.UPF" }
atoms = [{ element = "Al", fractional = [0.0, 0.0, 0.0] }]

[k_points]
kind = "monkhorst-pack"
grid = [2, 2, 2]
shift = [0.5, 0.5, 0.5]
reduce_time_reversal = true

[occupations]
kind = "fermi-dirac"
smearing_ry = 0.01
extra_bands = 4

[cell]
lattice_bohr = [[7.5,0.0,0.0],[0.0,7.5,0.0],[0.0,0.0,7.5]]
"#,
        )
        .unwrap();
        input.validate().unwrap();
        assert_eq!(input.k_points.resolve().unwrap().len(), 4);
        assert_eq!(input.occupations.band_count(3.0).unwrap(), 6);
    }
}
