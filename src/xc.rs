use crate::fft::Fft3d;
use crate::lattice::Lattice;
use anyhow::{Result, ensure};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Exchange-correlation models admitted by the versioned input contract.
///
/// Keeping this as a closed, serializable enum makes unsupported physics fail
/// during input parsing while still providing one dispatch point for future
/// semilocal functionals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum XcFunctional {
    LdaPz,
    Pbe,
}

impl XcFunctional {
    pub const fn identifier(self) -> &'static str {
        match self {
            Self::LdaPz => "lda-pz",
            Self::Pbe => "pbe",
        }
    }

    pub fn matches_upf_label(self, label: &str) -> bool {
        match self {
            Self::LdaPz => label.eq_ignore_ascii_case("PZ"),
            Self::Pbe => label.eq_ignore_ascii_case("PBE"),
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
            Self::Pbe => Pbe.evaluate_unpolarized(fft, lattice, density),
        }
    }
}

#[derive(Debug, Clone)]
pub struct XcEvaluation {
    pub potential_ry: Vec<f64>,
    pub energy_ry: f64,
    pub integral_rho_vxc_ry: f64,
}

/// Pointwise unpolarized GGA result in Hartree atomic units. `sigma` denotes
/// `|grad n|^2`, so `vsigma` is the derivative of the energy density with
/// respect to that squared gradient.
#[derive(Debug, Clone, Copy)]
pub struct GgaPoint {
    pub epsilon_xc_ha: f64,
    pub vrho_ha: f64,
    pub vsigma_ha_bohr5: f64,
}

#[derive(Debug, Clone, Copy)]
struct Dual2 {
    value: f64,
    density_derivative: f64,
    sigma_derivative: f64,
}

impl Dual2 {
    const fn constant(value: f64) -> Self {
        Self {
            value,
            density_derivative: 0.0,
            sigma_derivative: 0.0,
        }
    }

    const fn density(value: f64) -> Self {
        Self {
            value,
            density_derivative: 1.0,
            sigma_derivative: 0.0,
        }
    }

    const fn sigma(value: f64) -> Self {
        Self {
            value,
            density_derivative: 0.0,
            sigma_derivative: 1.0,
        }
    }

    fn ln(self) -> Self {
        let scale = self.value.recip();
        Self {
            value: self.value.ln(),
            density_derivative: self.density_derivative * scale,
            sigma_derivative: self.sigma_derivative * scale,
        }
    }

    fn exp(self) -> Self {
        let value = self.value.exp();
        Self {
            value,
            density_derivative: self.density_derivative * value,
            sigma_derivative: self.sigma_derivative * value,
        }
    }

    fn powf(self, exponent: f64) -> Self {
        let value = self.value.powf(exponent);
        let scale = exponent * self.value.powf(exponent - 1.0);
        Self {
            value,
            density_derivative: self.density_derivative * scale,
            sigma_derivative: self.sigma_derivative * scale,
        }
    }
}

impl From<f64> for Dual2 {
    fn from(value: f64) -> Self {
        Self::constant(value)
    }
}

impl Add for Dual2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value + rhs.value,
            density_derivative: self.density_derivative + rhs.density_derivative,
            sigma_derivative: self.sigma_derivative + rhs.sigma_derivative,
        }
    }
}

impl Add<f64> for Dual2 {
    type Output = Self;

    fn add(self, rhs: f64) -> Self::Output {
        self + Self::constant(rhs)
    }
}

impl Add<Dual2> for f64 {
    type Output = Dual2;

    fn add(self, rhs: Dual2) -> Self::Output {
        Dual2::constant(self) + rhs
    }
}

impl Sub for Dual2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value - rhs.value,
            density_derivative: self.density_derivative - rhs.density_derivative,
            sigma_derivative: self.sigma_derivative - rhs.sigma_derivative,
        }
    }
}

impl Sub<f64> for Dual2 {
    type Output = Self;

    fn sub(self, rhs: f64) -> Self::Output {
        self - Self::constant(rhs)
    }
}

impl Sub<Dual2> for f64 {
    type Output = Dual2;

    fn sub(self, rhs: Dual2) -> Self::Output {
        Dual2::constant(self) - rhs
    }
}

impl Mul for Dual2 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value * rhs.value,
            density_derivative: self.density_derivative * rhs.value
                + self.value * rhs.density_derivative,
            sigma_derivative: self.sigma_derivative * rhs.value + self.value * rhs.sigma_derivative,
        }
    }
}

impl Mul<f64> for Dual2 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        self * Self::constant(rhs)
    }
}

impl Mul<Dual2> for f64 {
    type Output = Dual2;

    fn mul(self, rhs: Dual2) -> Self::Output {
        Dual2::constant(self) * rhs
    }
}

impl Div for Dual2 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.powf(-1.0)
    }
}

impl Div<f64> for Dual2 {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        self / Self::constant(rhs)
    }
}

impl Div<Dual2> for f64 {
    type Output = Dual2;

    fn div(self, rhs: Dual2) -> Self::Output {
        Dual2::constant(self) / rhs
    }
}

impl Neg for Dual2 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            value: -self.value,
            density_derivative: -self.density_derivative,
            sigma_derivative: -self.sigma_derivative,
        }
    }
}

fn pw92_unpolarized_correlation(density: Dual2) -> Dual2 {
    let rs = (3.0 / (4.0 * PI * density)).powf(1.0 / 3.0);
    let sqrt_rs = rs.powf(0.5);
    let denominator = 2.0
        * 0.031_090_7
        * (7.5957 * sqrt_rs + 3.5876 * rs + 1.6382 * rs * sqrt_rs + 0.49294 * rs * rs);
    -2.0 * 0.031_090_7 * (1.0 + 0.21370 * rs) * (1.0 + denominator.powf(-1.0)).ln()
}

/// Unpolarized PBE exchange plus correlation for one density/gradient point.
///
/// The implementation follows the original PBE form and differentiates the
/// energy density by forward-mode dual numbers. PBE is not yet an admitted SCF
/// input; this point kernel is validated first against a pinned Libxc oracle.
pub fn pbe_unpolarized_point(density: f64, sigma: f64) -> Result<GgaPoint> {
    ensure!(
        density.is_finite() && density > 0.0,
        "PBE density must be positive and finite"
    );
    ensure!(
        sigma.is_finite() && sigma >= 0.0,
        "PBE sigma must be nonnegative and finite"
    );

    let density_dual = Dual2::density(density);
    let sigma_dual = Dual2::sigma(sigma);
    let kf = (3.0 * PI * PI * density_dual).powf(1.0 / 3.0);

    const KAPPA: f64 = 0.804;
    const MU: f64 = 0.219_514_972_764_517_1;
    let exchange_lda = -3.0 * kf / (4.0 * PI);
    let reduced_exchange_gradient_squared =
        sigma_dual / (4.0 * kf * kf * density_dual * density_dual);
    let exchange_enhancement =
        1.0 + KAPPA - KAPPA / (1.0 + (MU / KAPPA) * reduced_exchange_gradient_squared);
    let exchange_energy_density = density_dual * exchange_lda * exchange_enhancement;

    const BETA: f64 = 0.066_724_550_603_149_22;
    const GAMMA: f64 = 0.031_090_690_869_654_895;
    let correlation_lda = pw92_unpolarized_correlation(density_dual);
    let ks = (4.0 * kf / PI).powf(0.5);
    let reduced_correlation_gradient_squared =
        sigma_dual / (4.0 * ks * ks * density_dual * density_dual);
    let a = (BETA / GAMMA) / ((-correlation_lda / GAMMA).exp() - 1.0);
    let at2 = a * reduced_correlation_gradient_squared;
    let correlation_gradient = GAMMA
        * (1.0
            + (BETA / GAMMA) * reduced_correlation_gradient_squared * (1.0 + at2)
                / (1.0 + at2 + at2 * at2))
            .ln();
    let correlation_energy_density = density_dual * (correlation_lda + correlation_gradient);
    let total = exchange_energy_density + correlation_energy_density;
    ensure!(
        total.value.is_finite()
            && total.density_derivative.is_finite()
            && total.sigma_derivative.is_finite(),
        "PBE evaluation produced a non-finite value"
    );
    Ok(GgaPoint {
        epsilon_xc_ha: total.value / density,
        vrho_ha: total.density_derivative,
        vsigma_ha_bohr5: total.sigma_derivative,
    })
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

#[derive(Debug, Clone, Copy, Default)]
pub struct Pbe;

fn derivative_frequency(index: usize, extent: usize) -> i32 {
    if extent.is_multiple_of(2) && index == extent / 2 {
        // The Nyquist mode has no unique sign on an even real grid. Setting it
        // to zero preserves a real, anti-Hermitian discrete derivative.
        0
    } else if index <= extent / 2 {
        index as i32
    } else {
        index as i32 - extent as i32
    }
}

fn reciprocal_at_flat_index(lattice: &Lattice, shape: [usize; 3], flat: usize) -> [f64; 3] {
    let yz = shape[1] * shape[2];
    let x = flat / yz;
    let remainder = flat % yz;
    let y = remainder / shape[2];
    let z = remainder % shape[2];
    lattice.cartesian_g([
        derivative_frequency(x, shape[0]),
        derivative_frequency(y, shape[1]),
        derivative_frequency(z, shape[2]),
    ])
}

fn spectral_gradient(fft: &Fft3d, lattice: &Lattice, values: &[f64]) -> Result<[Vec<f64>; 3]> {
    ensure!(values.len() == fft.len(), "gradient input has wrong size");
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "gradient input contains non-finite values"
    );
    let mut coefficients: Vec<_> = values
        .iter()
        .map(|&value| Complex64::new(value, 0.0))
        .collect();
    fft.forward(&mut coefficients)?;
    let shape = fft.shape();
    let mut gradient: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; fft.len()]);
    for axis in 0..3 {
        let mut derivative = coefficients.clone();
        for (flat, value) in derivative.iter_mut().enumerate() {
            let g = reciprocal_at_flat_index(lattice, shape, flat);
            *value *= Complex64::new(0.0, g[axis]);
        }
        fft.inverse(&mut derivative)?;
        let maximum_real = derivative
            .iter()
            .map(|value| value.re.abs())
            .fold(0.0_f64, f64::max);
        let maximum_imaginary = derivative
            .iter()
            .map(|value| value.im.abs())
            .fold(0.0_f64, f64::max);
        ensure!(
            maximum_imaginary <= 1.0e-11 * maximum_real.max(1.0),
            "spectral gradient has a material imaginary component"
        );
        for (output, value) in gradient[axis].iter_mut().zip(derivative) {
            *output = value.re;
        }
    }
    Ok(gradient)
}

fn spectral_divergence(fft: &Fft3d, lattice: &Lattice, vector: &[Vec<f64>; 3]) -> Result<Vec<f64>> {
    ensure!(
        vector.iter().all(|component| component.len() == fft.len()),
        "divergence input has wrong size"
    );
    let shape = fft.shape();
    let mut divergence = vec![0.0; fft.len()];
    for axis in 0..3 {
        let mut coefficients: Vec<_> = vector[axis]
            .iter()
            .map(|&value| Complex64::new(value, 0.0))
            .collect();
        fft.forward(&mut coefficients)?;
        for (flat, value) in coefficients.iter_mut().enumerate() {
            let g = reciprocal_at_flat_index(lattice, shape, flat);
            *value *= Complex64::new(0.0, g[axis]);
        }
        fft.inverse(&mut coefficients)?;
        let maximum_real = coefficients
            .iter()
            .map(|value| value.re.abs())
            .fold(0.0_f64, f64::max);
        let maximum_imaginary = coefficients
            .iter()
            .map(|value| value.im.abs())
            .fold(0.0_f64, f64::max);
        ensure!(
            maximum_imaginary <= 1.0e-11 * maximum_real.max(1.0),
            "spectral divergence has a material imaginary component"
        );
        for (output, value) in divergence.iter_mut().zip(coefficients) {
            *output += value.re;
        }
    }
    Ok(divergence)
}

impl ExchangeCorrelation for Pbe {
    fn functional(&self) -> XcFunctional {
        XcFunctional::Pbe
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
            "PBE grid density is non-finite or materially negative"
        );
        let gradient = spectral_gradient(fft, lattice, density)?;
        let cell_weight = lattice.volume / fft.len() as f64;
        let mut energy_ry = 0.0;
        let mut vrho_ha = Vec::with_capacity(fft.len());
        let mut flux: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; fft.len()]);
        for flat in 0..fft.len() {
            if density[flat] <= 1.0e-14 {
                vrho_ha.push(0.0);
                continue;
            }
            let sigma = gradient
                .iter()
                .map(|component| component[flat] * component[flat])
                .sum();
            let point = pbe_unpolarized_point(density[flat], sigma)?;
            energy_ry += 2.0 * density[flat] * point.epsilon_xc_ha * cell_weight;
            vrho_ha.push(point.vrho_ha);
            for axis in 0..3 {
                flux[axis][flat] = 2.0 * point.vsigma_ha_bohr5 * gradient[axis][flat];
            }
        }
        let divergence_ha = spectral_divergence(fft, lattice, &flux)?;
        let potential_ry: Vec<_> = vrho_ha
            .into_iter()
            .zip(divergence_ha)
            .map(|(vrho, divergence)| 2.0 * (vrho - divergence))
            .collect();
        let integral_rho_vxc_ry: f64 = density
            .iter()
            .zip(&potential_ry)
            .map(|(&rho, &potential)| rho.max(0.0) * potential * cell_weight)
            .sum();
        ensure!(
            energy_ry.is_finite()
                && integral_rho_vxc_ry.is_finite()
                && potential_ry.iter().all(|value| value.is_finite()),
            "PBE grid evaluation produced a non-finite value"
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
    use serde::Deserialize;
    use std::f64::consts::TAU;
    use std::fs;

    #[derive(Deserialize)]
    struct PbeReference {
        schema_version: u32,
        generator: String,
        exchange_functional: String,
        correlation_functional: String,
        spin: String,
        points: Vec<PbeReferencePoint>,
    }

    #[derive(Deserialize)]
    struct PbeReferencePoint {
        density_bohr3: f64,
        sigma_bohr8: f64,
        epsilon_xc_ha: f64,
        vrho_ha: f64,
        vsigma_ha_bohr5: f64,
    }

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
    fn pbe_points_match_libxc_7_1_2() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/xc/libxc-7.1.2-pbe-unpolarized.json");
        let reference: PbeReference =
            serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(reference.schema_version, 1);
        assert_eq!(reference.generator, "Libxc 7.1.2");
        assert_eq!(reference.exchange_functional, "GGA_X_PBE (101)");
        assert_eq!(reference.correlation_functional, "GGA_C_PBE (130)");
        assert_eq!(reference.spin, "unpolarized");
        assert_eq!(reference.points.len(), 7);
        for expected in reference.points {
            let actual =
                pbe_unpolarized_point(expected.density_bohr3, expected.sigma_bohr8).unwrap();
            assert_relative_eq!(
                actual.epsilon_xc_ha,
                expected.epsilon_xc_ha,
                epsilon = 2.0e-8,
                max_relative = 2.0e-8
            );
            assert_relative_eq!(
                actual.vrho_ha,
                expected.vrho_ha,
                epsilon = 3.0e-8,
                max_relative = 3.0e-8
            );
            assert_relative_eq!(
                actual.vsigma_ha_bohr5,
                expected.vsigma_ha_bohr5,
                epsilon = 2.0e-8,
                max_relative = 2.0e-8
            );
        }
    }

    #[test]
    fn pbe_derivatives_match_energy_finite_differences() {
        let density = 0.03;
        let sigma = 0.002;
        let density_step = density * 1.0e-5;
        let sigma_step = sigma * 1.0e-5;
        let energy_density = |rho: f64, gradient_squared: f64| {
            rho * pbe_unpolarized_point(rho, gradient_squared)
                .unwrap()
                .epsilon_xc_ha
        };
        let density_derivative = (energy_density(density + density_step, sigma)
            - energy_density(density - density_step, sigma))
            / (2.0 * density_step);
        let sigma_derivative = (energy_density(density, sigma + sigma_step)
            - energy_density(density, sigma - sigma_step))
            / (2.0 * sigma_step);
        let analytic = pbe_unpolarized_point(density, sigma).unwrap();
        assert_relative_eq!(
            analytic.vrho_ha,
            density_derivative,
            epsilon = 1.0e-9,
            max_relative = 1.0e-8
        );
        assert_relative_eq!(
            analytic.vsigma_ha_bohr5,
            sigma_derivative,
            epsilon = 1.0e-9,
            max_relative = 1.0e-8
        );
    }

    #[test]
    fn pbe_grid_uniform_limit_matches_point_kernel() {
        let fft = Fft3d::new([5, 5, 5]).unwrap();
        let lattice = Lattice::new([[TAU, 0.0, 0.0], [0.0, TAU, 0.0], [0.0, 0.0, TAU]]).unwrap();
        let density_value = 0.02;
        let density = vec![density_value; fft.len()];
        let grid = Pbe.evaluate_unpolarized(&fft, &lattice, &density).unwrap();
        let point = pbe_unpolarized_point(density_value, 0.0).unwrap();
        assert_relative_eq!(
            grid.energy_ry,
            2.0 * lattice.volume * density_value * point.epsilon_xc_ha,
            epsilon = 2.0e-13
        );
        for potential in grid.potential_ry {
            assert_relative_eq!(potential, 2.0 * point.vrho_ha, epsilon = 2.0e-13);
        }
        assert_relative_eq!(
            grid.integral_rho_vxc_ry,
            2.0 * lattice.volume * density_value * point.vrho_ha,
            epsilon = 2.0e-13
        );
    }

    #[test]
    fn pbe_grid_potential_is_variational_derivative() {
        let shape = [9, 7, 5];
        let fft = Fft3d::new(shape).unwrap();
        let lattice = Lattice::new([[TAU, 0.0, 0.0], [0.0, TAU, 0.0], [0.0, 0.0, TAU]]).unwrap();
        let mut density = Vec::with_capacity(fft.len());
        let mut perturbation = Vec::with_capacity(fft.len());
        for x in 0..shape[0] {
            for y in 0..shape[1] {
                for z in 0..shape[2] {
                    let rx = TAU * x as f64 / shape[0] as f64;
                    let ry = TAU * y as f64 / shape[1] as f64;
                    let rz = TAU * z as f64 / shape[2] as f64;
                    density.push(
                        0.05 + 0.010 * rx.cos() + 0.006 * (2.0 * ry).sin() + 0.004 * rz.cos(),
                    );
                    perturbation.push((2.0 * rx).cos() - 0.3 * ry.sin() + 0.2 * (2.0 * rz).cos());
                }
            }
        }
        let evaluation = Pbe.evaluate_unpolarized(&fft, &lattice, &density).unwrap();
        let step = 1.0e-6;
        let density_plus: Vec<_> = density
            .iter()
            .zip(&perturbation)
            .map(|(&rho, &direction)| rho + step * direction)
            .collect();
        let density_minus: Vec<_> = density
            .iter()
            .zip(&perturbation)
            .map(|(&rho, &direction)| rho - step * direction)
            .collect();
        let energy_plus = Pbe
            .evaluate_unpolarized(&fft, &lattice, &density_plus)
            .unwrap()
            .energy_ry;
        let energy_minus = Pbe
            .evaluate_unpolarized(&fft, &lattice, &density_minus)
            .unwrap()
            .energy_ry;
        let finite_difference = (energy_plus - energy_minus) / (2.0 * step);
        let cell_weight = lattice.volume / fft.len() as f64;
        let potential_derivative: f64 = evaluation
            .potential_ry
            .iter()
            .zip(&perturbation)
            .map(|(&potential, &direction)| potential * direction * cell_weight)
            .sum();
        assert_relative_eq!(
            potential_derivative,
            finite_difference,
            epsilon = 2.0e-8,
            max_relative = 2.0e-8
        );
    }

    #[test]
    fn pbe_grid_vacuum_is_finite() {
        let fft = Fft3d::new([5, 5, 5]).unwrap();
        let lattice = Lattice::new([[TAU, 0.0, 0.0], [0.0, TAU, 0.0], [0.0, 0.0, TAU]]).unwrap();
        let result = Pbe
            .evaluate_unpolarized(&fft, &lattice, &vec![0.0; fft.len()])
            .unwrap();
        assert_eq!(result.energy_ry, 0.0);
        assert_eq!(result.integral_rho_vxc_ry, 0.0);
        assert!(result.potential_ry.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn vacuum_is_finite() {
        assert_eq!(lda_pz_unpolarized(0.0), (0.0, 0.0));
    }
}
