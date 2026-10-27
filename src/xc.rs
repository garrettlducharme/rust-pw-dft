use crate::fft::Fft3d;
use crate::lattice::Lattice;
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Exchange-correlation models admitted by the versioned input contract.
///
/// Keeping this as a closed, serializable enum makes unsupported physics fail
/// during input parsing while still providing one dispatch point for future
/// semilocal functionals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum XcFunctional {
    LdaPz,
}

impl XcFunctional {
    pub const fn identifier(self) -> &'static str {
        match self {
            Self::LdaPz => "lda-pz",
        }
    }

    pub fn matches_upf_label(self, label: &str) -> bool {
        match self {
            Self::LdaPz => label.eq_ignore_ascii_case("PZ"),
        }
    }

    pub fn evaluate_unpolarized(
        self,
        fft: &Fft3d,
        lattice: &Lattice,
        density: &[f64],
    ) -> Result<XcEvaluation> {
        match self {
            Self::LdaPz => LdaPz.evaluate_unpolarized(fft, lattice, density),
        }
    }
}

#[derive(Debug, Clone)]
pub struct XcEvaluation {
    pub potential_ry: Vec<f64>,
    pub energy_ry: f64,
    pub integral_rho_vxc_ry: f64,
}

/// Grid-level XC boundary. PBE and later semilocal models can use the FFT and
/// lattice to form gradients while the SCF and Hartree paths remain unchanged.
pub trait ExchangeCorrelation {
    fn functional(&self) -> XcFunctional;

    fn evaluate_unpolarized(
        &self,
        fft: &Fft3d,
        lattice: &Lattice,
        density: &[f64],
    ) -> Result<XcEvaluation>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LdaPz;

impl ExchangeCorrelation for LdaPz {
    fn functional(&self) -> XcFunctional {
        XcFunctional::LdaPz
    }

    fn evaluate_unpolarized(
        &self,
        fft: &Fft3d,
        lattice: &Lattice,
        density: &[f64],
    ) -> Result<XcEvaluation> {
        ensure!(
            density.len() == fft.len(),
            "XC density has the wrong grid size"
        );
        ensure!(
            density
                .iter()
                .all(|value| value.is_finite() && *value >= -1.0e-12),
            "XC density is non-finite or materially negative"
        );
        let cell_weight = lattice.volume / fft.len() as f64;
        let mut potential_ry = Vec::with_capacity(density.len());
        let mut energy_ry = 0.0;
        let mut integral_rho_vxc_ry = 0.0;
        for &rho in density {
            let nonnegative_density = rho.max(0.0);
            let (epsilon_xc_ha, v_xc_ha) = lda_pz_unpolarized(nonnegative_density);
            let epsilon_xc_ry = 2.0 * epsilon_xc_ha;
            let v_xc_ry = 2.0 * v_xc_ha;
            energy_ry += nonnegative_density * epsilon_xc_ry * cell_weight;
            integral_rho_vxc_ry += nonnegative_density * v_xc_ry * cell_weight;
            potential_ry.push(v_xc_ry);
        }
        ensure!(
            energy_ry.is_finite()
                && integral_rho_vxc_ry.is_finite()
                && potential_ry.iter().all(|value| value.is_finite()),
            "XC evaluation produced a non-finite value"
        );
        Ok(XcEvaluation {
            potential_ry,
            energy_ry,
            integral_rho_vxc_ry,
        })
    }
}

/// Perdew-Zunger (1981) parameterization of the unpolarized Ceperley-Alder LDA.
/// Returns `(epsilon_xc, v_xc)` in Hartree for a density in electrons/bohr^3.
pub fn lda_pz_unpolarized(density: f64) -> (f64, f64) {
    if density <= 0.0 || !density.is_finite() {
        return (0.0, 0.0);
    }
    let rs = (3.0 / (4.0 * PI * density)).cbrt();
    let ex = -0.458_165_293_283_142_9 / rs;
    let vx = (4.0 / 3.0) * ex;
    let (ec, dec_drs) = if rs < 1.0 {
        let (a, b, c, d) = (0.0311, -0.048, 0.0020, -0.0116);
        let ec = a * rs.ln() + b + c * rs * rs.ln() + d * rs;
        let derivative = a / rs + c * (rs.ln() + 1.0) + d;
        (ec, derivative)
    } else {
        let (gamma, beta1, beta2) = (-0.1423, 1.0529, 0.3334);
        let denominator = 1.0 + beta1 * rs.sqrt() + beta2 * rs;
        let ec = gamma / denominator;
        let derivative = -gamma * (0.5 * beta1 / rs.sqrt() + beta2) / denominator.powi(2);
        (ec, derivative)
    };
    let vc = ec - (rs / 3.0) * dec_drs;
    (ex + ec, vx + vc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::TAU;

    #[test]
    fn known_rs_two_value_is_stable() {
        let density = 3.0 / (4.0 * PI * 8.0);
        let (epsilon, potential) = lda_pz_unpolarized(density);
        assert!((epsilon - (-0.274_173_860_275_419_85)).abs() < 1.0e-12);
        assert!((potential - (-0.357_256_470_778_624_66)).abs() < 1.0e-12);
    }

    #[test]
    fn potential_is_energy_density_derivative() {
        for density in [1.0e-6, 1.0e-4, 1.0e-2, 0.1, 1.0] {
            let step = density * 1.0e-5;
            let energy_density = |rho: f64| {
                let (epsilon, _) = lda_pz_unpolarized(rho);
                rho * epsilon
            };
            let finite_difference =
                (energy_density(density + step) - energy_density(density - step)) / (2.0 * step);
            let (_, potential) = lda_pz_unpolarized(density);
            assert_relative_eq!(finite_difference, potential, epsilon = 2.0e-9);
        }
    }

    #[test]
    fn grid_evaluation_preserves_units_and_integrals() {
        let fft = Fft3d::new([3, 3, 3]).unwrap();
        let lattice = Lattice::new([[TAU, 0.0, 0.0], [0.0, TAU, 0.0], [0.0, 0.0, TAU]]).unwrap();
        let density = vec![0.02; fft.len()];
        let result = XcFunctional::LdaPz
            .evaluate_unpolarized(&fft, &lattice, &density)
            .unwrap();
        let (epsilon_ha, potential_ha) = lda_pz_unpolarized(0.02);
        assert_relative_eq!(
            result.potential_ry[0],
            2.0 * potential_ha,
            epsilon = 1.0e-14
        );
        assert_relative_eq!(
            result.energy_ry,
            lattice.volume * 0.02 * 2.0 * epsilon_ha,
            epsilon = 1.0e-13
        );
        assert_relative_eq!(
            result.integral_rho_vxc_ry,
            lattice.volume * 0.02 * 2.0 * potential_ha,
            epsilon = 1.0e-13
        );
    }

    #[test]
    fn vacuum_is_finite() {
        assert_eq!(lda_pz_unpolarized(0.0), (0.0, 0.0));
    }
}
