use anyhow::{Result, ensure};
use faer::linalg::solvers::SelfAdjointEigen;
use faer::{Mat, Side};
use num_complex::Complex64;

#[derive(Debug, Clone)]
pub struct ProjectedEigenpairs {
    pub eigenvalues: Vec<f64>,
    /// Column-major eigenvectors; each inner vector is one eigenvector.
    pub eigenvectors: Vec<Vec<Complex64>>,
}

/// Solve a small CPU-FP64 Hermitian projected problem using `faer`.
///
/// This is the Rayleigh-Ritz precision boundary shared by CPU and Metal
/// backends.  `matrix` is row-major and must contain the complete matrix.
pub fn projected_hermitian_eigensolve(
    matrix: &[Complex64],
    dimension: usize,
) -> Result<ProjectedEigenpairs> {
    ensure!(dimension > 0, "projected matrix must be nonempty");
    ensure!(
        matrix.len() == dimension * dimension,
        "projected matrix has the wrong size"
    );
    let scale = matrix.iter().map(|z| z.norm()).fold(0.0_f64, f64::max);
    let tolerance = 128.0 * f64::EPSILON * scale.max(1.0);
    for i in 0..dimension {
        ensure!(
            matrix[i * dimension + i].im.abs() <= tolerance,
            "projected matrix diagonal is not real"
        );
        for j in 0..i {
            ensure!(
                (matrix[i * dimension + j] - matrix[j * dimension + i].conj()).norm() <= tolerance,
                "projected matrix is not Hermitian"
            );
        }
    }
    let faer_matrix = Mat::from_fn(dimension, dimension, |i, j| matrix[i * dimension + j]);
    let decomposition = SelfAdjointEigen::new(faer_matrix.as_ref(), Side::Lower)
        .map_err(|error| anyhow::anyhow!("projected eigensolve failed: {error:?}"))?;
    let diagonal = decomposition.S().column_vector();
    let eigenvalues = (0..dimension).map(|i| diagonal[i].re).collect();
    let vectors = decomposition.U();
    let eigenvectors = (0..dimension)
        .map(|column| (0..dimension).map(|row| vectors[(row, column)]).collect())
        .collect();
    Ok(ProjectedEigenpairs {
        eigenvalues,
        eigenvectors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_complex_hermitian_projection() {
        let matrix = [
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(2.0, 0.0),
        ];
        let result = projected_hermitian_eigensolve(&matrix, 2).unwrap();
        assert!((result.eigenvalues[0] - 1.0).abs() < 1.0e-12);
        assert!((result.eigenvalues[1] - 3.0).abs() < 1.0e-12);
        for (value, vector) in result.eigenvalues.iter().zip(&result.eigenvectors) {
            let residual = (0..2)
                .map(|row| {
                    let applied: Complex64 = (0..2)
                        .map(|column| matrix[row * 2 + column] * vector[column])
                        .sum();
                    (applied - *value * vector[row]).norm_sqr()
                })
                .sum::<f64>()
                .sqrt();
            assert!(residual < 1.0e-12);
        }
    }

    #[test]
    fn rejects_nonhermitian_projection() {
        let matrix = [
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(2.0, 0.0),
        ];
        assert!(projected_hermitian_eigensolve(&matrix, 2).is_err());
    }
}
