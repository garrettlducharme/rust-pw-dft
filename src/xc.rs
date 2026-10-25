use std::f64::consts::PI;

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

    #[test]
    fn known_rs_two_value_is_stable() {
        let density = 3.0 / (4.0 * PI * 8.0);
        let (epsilon, potential) = lda_pz_unpolarized(density);
        assert!((epsilon - (-0.274_173_860_275_419_85)).abs() < 1.0e-12);
        assert!((potential - (-0.357_256_470_778_624_66)).abs() < 1.0e-12);
    }

    #[test]
    fn vacuum_is_finite() {
        assert_eq!(lda_pz_unpolarized(0.0), (0.0, 0.0));
    }
}
