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

#[derive(Debug, Clone)]
pub struct DavidsonOptions {
    pub roots: usize,
    pub residual_tolerance: f64,
    pub max_iterations: usize,
    pub max_subspace: usize,
}

#[derive(Debug, Clone)]
pub struct DavidsonResult {
    pub converged: bool,
    pub iterations: usize,
    pub hamiltonian_applications: usize,
    pub eigenvalues: Vec<f64>,
    pub eigenvectors: Vec<Vec<Complex64>>,
    pub residual_norms: Vec<f64>,
}

fn inner_product(left: &[Complex64], right: &[Complex64]) -> Complex64 {
    left.iter().zip(right).map(|(a, b)| a.conj() * b).sum()
}

fn cross_gram_cpu(left: &[Vec<Complex64>], right: &[Vec<Complex64>]) -> Vec<Complex64> {
    left.iter()
        .flat_map(|left_vector| {
            right
                .iter()
                .map(|right_vector| inner_product(left_vector, right_vector))
        })
        .collect()
}

fn vector_norms_cpu(vectors: &[Vec<Complex64>]) -> Vec<f64> {
    vectors
        .iter()
        .map(|vector| inner_product(vector, vector).re.max(0.0).sqrt())
        .collect()
}

fn orthonormalize_candidate(
    mut candidate: Vec<Complex64>,
    basis: &[Vec<Complex64>],
) -> Option<Vec<Complex64>> {
    for _ in 0..2 {
        for vector in basis {
            let overlap = inner_product(vector, &candidate);
            for (value, &basis_value) in candidate.iter_mut().zip(vector) {
                *value -= overlap * basis_value;
            }
        }
    }
    let norm = inner_product(&candidate, &candidate).re.max(0.0).sqrt();
    if norm <= 1.0e-11 {
        return None;
    }
    for value in &mut candidate {
        *value /= norm;
    }
    Some(candidate)
}

fn combine_basis(basis: &[Vec<Complex64>], coefficients: &[Complex64]) -> Vec<Complex64> {
    let dimension = basis[0].len();
    let mut result = vec![Complex64::default(); dimension];
    for (vector, &coefficient) in basis.iter().zip(coefficients) {
        for (value, &basis_value) in result.iter_mut().zip(vector) {
            *value += coefficient * basis_value;
        }
    }
    result
}

/// Block Davidson eigensolver with diagonal kinetic preconditioning.
///
/// The operator remains matrix-free; only the Rayleigh-Ritz projection crosses
/// the CPU-FP64 `faer` boundary. Initial vectors may be reused from the prior
/// SCF iteration. The result reports non-convergence rather than silently
/// substituting a dense solve.
pub fn block_davidson<F>(
    kinetic_diagonal: &[f64],
    initial_vectors: Option<&[Vec<Complex64>]>,
    options: &DavidsonOptions,
    mut apply: F,
) -> Result<DavidsonResult>
where
    F: FnMut(&[Complex64]) -> Result<Vec<Complex64>>,
{
    block_davidson_batched(kinetic_diagonal, initial_vectors, options, |vectors| {
        vectors.iter().map(|vector| apply(vector)).collect()
    })
}

/// Batched block-Davidson boundary used by accelerator backends.
///
/// Every call receives the complete block of new basis vectors, allowing one
/// FFT graph execution to process all states. Returned vectors must preserve
/// input order.
pub fn block_davidson_batched<F>(
    kinetic_diagonal: &[f64],
    initial_vectors: Option<&[Vec<Complex64>]>,
    options: &DavidsonOptions,
    apply_batch: F,
) -> Result<DavidsonResult>
where
    F: FnMut(&[Vec<Complex64>]) -> Result<Vec<Vec<Complex64>>>,
{
    block_davidson_batched_with_gram(
        kinetic_diagonal,
        initial_vectors,
        options,
        apply_batch,
        |left, right| Ok(cross_gram_cpu(left, right)),
    )
}

/// Batched block Davidson with an explicit cross-Gram backend.
///
/// `cross_gram(left, right)` returns the row-major rectangular matrix
/// `<left_i|right_j>`. Accelerator backends may compute blockwise FP32
/// partials, but must finish accumulation before returning complex FP64.
pub fn block_davidson_batched_with_gram<F, G>(
    kinetic_diagonal: &[f64],
    initial_vectors: Option<&[Vec<Complex64>]>,
    options: &DavidsonOptions,
    apply_batch: F,
    cross_gram: G,
) -> Result<DavidsonResult>
where
    F: FnMut(&[Vec<Complex64>]) -> Result<Vec<Vec<Complex64>>>,
    G: FnMut(&[Vec<Complex64>], &[Vec<Complex64>]) -> Result<Vec<Complex64>>,
{
    block_davidson_batched_with_reductions(
        kinetic_diagonal,
        initial_vectors,
        options,
        apply_batch,
        cross_gram,
        |vectors| Ok(vector_norms_cpu(vectors)),
    )
}

/// Batched block Davidson with explicit cross-Gram and direct-norm backends.
///
/// The norm backend receives fully formed residual vectors and must return one
/// nonnegative FP64 norm per vector. Computing direct residual norms avoids the
/// cancellation inherent in deriving them from projected matrices.
pub fn block_davidson_batched_with_reductions<F, G, N>(
    kinetic_diagonal: &[f64],
    initial_vectors: Option<&[Vec<Complex64>]>,
    options: &DavidsonOptions,
    mut apply_batch: F,
    mut cross_gram: G,
    mut vector_norms: N,
) -> Result<DavidsonResult>
where
    F: FnMut(&[Vec<Complex64>]) -> Result<Vec<Vec<Complex64>>>,
    G: FnMut(&[Vec<Complex64>], &[Vec<Complex64>]) -> Result<Vec<Complex64>>,
    N: FnMut(&[Vec<Complex64>]) -> Result<Vec<f64>>,
{
    let dimension = kinetic_diagonal.len();
    ensure!(dimension > 0, "Davidson dimension must be nonzero");
    ensure!(
        options.roots > 0 && options.roots <= dimension,
        "invalid Davidson root count"
    );
    ensure!(
        options.residual_tolerance.is_finite() && options.residual_tolerance > 0.0,
        "invalid Davidson residual tolerance"
    );
    ensure!(
        options.max_iterations > 0,
        "invalid Davidson iteration limit"
    );
    ensure!(
        options.max_subspace > options.roots && options.max_subspace <= dimension,
        "Davidson max_subspace must be in roots+1..=dimension"
    );
    ensure!(
        kinetic_diagonal.iter().all(|value| value.is_finite()),
        "kinetic diagonal contains non-finite values"
    );

    let mut basis = Vec::new();
    if let Some(initial) = initial_vectors {
        for vector in initial {
            ensure!(
                vector.len() == dimension,
                "initial Davidson vector has wrong size"
            );
            if let Some(vector) = orthonormalize_candidate(vector.clone(), &basis) {
                basis.push(vector);
            }
            if basis.len() == options.roots {
                break;
            }
        }
    }
    let mut kinetic_order: Vec<_> = (0..dimension).collect();
    kinetic_order.sort_by(|&a, &b| kinetic_diagonal[a].total_cmp(&kinetic_diagonal[b]));
    for index in kinetic_order {
        if basis.len() == options.roots {
            break;
        }
        let mut vector = vec![Complex64::default(); dimension];
        vector[index] = Complex64::new(1.0, 0.0);
        if let Some(vector) = orthonormalize_candidate(vector, &basis) {
            basis.push(vector);
        }
    }
    ensure!(
        basis.len() == options.roots,
        "failed to initialize Davidson basis"
    );

    let mut applied_basis = apply_batch(&basis)?;
    ensure!(
        applied_basis.len() == basis.len(),
        "Hamiltonian batch result has wrong state count"
    );
    for applied in &applied_basis {
        ensure!(
            applied.len() == dimension,
            "Hamiltonian result has wrong size"
        );
    }
    applied_basis.reserve(options.max_subspace - applied_basis.len());
    let mut applications = basis.len();
    let mut last_result = DavidsonResult {
        converged: false,
        iterations: 0,
        hamiltonian_applications: applications,
        eigenvalues: Vec::new(),
        eigenvectors: Vec::new(),
        residual_norms: Vec::new(),
    };

    for iteration in 1..=options.max_iterations {
        let subspace = basis.len();
        let raw_projection = cross_gram(&basis, &applied_basis)?;
        ensure!(
            raw_projection.len() == subspace * subspace,
            "cross-Gram backend returned the wrong matrix size"
        );
        let mut projected = vec![Complex64::default(); subspace * subspace];
        for i in 0..subspace {
            for j in 0..=i {
                let lower = raw_projection[i * subspace + j];
                let upper = raw_projection[j * subspace + i].conj();
                let value = 0.5 * (lower + upper);
                projected[i * subspace + j] = if i == j {
                    Complex64::new(value.re, 0.0)
                } else {
                    value
                };
                projected[j * subspace + i] = projected[i * subspace + j].conj();
            }
        }
        let projected_pairs = projected_hermitian_eigensolve(&projected, subspace)?;
        let eigenvalues = projected_pairs.eigenvalues[..options.roots].to_vec();
        let mut ritz_vectors = Vec::with_capacity(options.roots);
        let mut applied_ritz = Vec::with_capacity(options.roots);
        let mut residual_vectors = Vec::with_capacity(options.roots);
        for (root, &eigenvalue) in eigenvalues.iter().enumerate() {
            let vector = combine_basis(&basis, &projected_pairs.eigenvectors[root]);
            let applied = combine_basis(&applied_basis, &projected_pairs.eigenvectors[root]);
            let residual: Vec<_> = applied
                .iter()
                .zip(&vector)
                .map(|(hpsi, psi)| *hpsi - eigenvalue * psi)
                .collect();
            ritz_vectors.push(vector);
            applied_ritz.push(applied);
            residual_vectors.push(residual);
        }
        let residual_norms = vector_norms(&residual_vectors)?;
        ensure!(
            residual_norms.len() == options.roots,
            "norm backend returned the wrong vector count"
        );
        ensure!(
            residual_norms
                .iter()
                .all(|norm| norm.is_finite() && *norm >= 0.0),
            "norm backend returned an invalid residual norm"
        );
        let converged = residual_norms
            .iter()
            .all(|&norm| norm <= options.residual_tolerance);
        last_result = DavidsonResult {
            converged,
            iterations: iteration,
            hamiltonian_applications: applications,
            eigenvalues,
            eigenvectors: ritz_vectors.clone(),
            residual_norms: residual_norms.clone(),
        };
        if converged {
            return Ok(last_result);
        }

        let mut corrections = Vec::new();
        for (root, &residual_norm) in residual_norms.iter().enumerate() {
            if residual_norm <= options.residual_tolerance {
                continue;
            }
            let mut correction: Vec<_> = residual_vectors[root]
                .iter()
                .zip(kinetic_diagonal)
                .map(|(&residual, &kinetic)| {
                    let denominator = kinetic - last_result.eigenvalues[root];
                    let safe_denominator = if denominator.abs() < 0.25 {
                        0.25_f64.copysign(denominator)
                    } else {
                        denominator
                    };
                    -residual / safe_denominator
                })
                .collect();
            let mut exclusion = basis.clone();
            exclusion.extend(corrections.iter().cloned());
            if let Some(orthogonal) =
                orthonormalize_candidate(std::mem::take(&mut correction), &exclusion)
            {
                corrections.push(orthogonal);
            }
        }
        if corrections.is_empty() {
            return Ok(last_result);
        }

        if basis.len() + corrections.len() > options.max_subspace {
            basis = ritz_vectors;
            applied_basis = applied_ritz;
            let mut restarted_corrections = Vec::new();
            for correction in corrections {
                let mut exclusion = basis.clone();
                exclusion.extend(restarted_corrections.iter().cloned());
                if let Some(orthogonal) = orthonormalize_candidate(correction, &exclusion) {
                    restarted_corrections.push(orthogonal);
                }
            }
            corrections = restarted_corrections;
        }
        let applied_corrections = apply_batch(&corrections)?;
        ensure!(
            applied_corrections.len() == corrections.len(),
            "Hamiltonian batch result has wrong state count"
        );
        for (correction, applied) in corrections.into_iter().zip(applied_corrections) {
            ensure!(
                applied.len() == dimension,
                "Hamiltonian result has wrong size"
            );
            basis.push(correction);
            applied_basis.push(applied);
            applications += 1;
        }
    }
    last_result.hamiltonian_applications = applications;
    Ok(last_result)
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

    #[test]
    fn block_davidson_matches_dense_complex_hermitian_solution() {
        let dimension = 6;
        let mut matrix = vec![Complex64::default(); dimension * dimension];
        for i in 0..dimension {
            matrix[i * dimension + i] = Complex64::new((i + 1) as f64, 0.0);
        }
        for i in 0..dimension - 1 {
            let coupling = Complex64::new(0.11 * (i + 1) as f64, -0.03);
            matrix[i * dimension + i + 1] = coupling;
            matrix[(i + 1) * dimension + i] = coupling.conj();
        }
        let dense = projected_hermitian_eigensolve(&matrix, dimension).unwrap();
        let kinetic: Vec<_> = (0..dimension).map(|i| (i + 1) as f64).collect();
        let options = DavidsonOptions {
            roots: 2,
            residual_tolerance: 1.0e-11,
            max_iterations: 20,
            max_subspace: dimension,
        };
        let davidson = block_davidson(&kinetic, None, &options, |vector| {
            Ok((0..dimension)
                .map(|row| {
                    (0..dimension)
                        .map(|column| matrix[row * dimension + column] * vector[column])
                        .sum()
                })
                .collect())
        })
        .unwrap();
        assert!(davidson.converged);
        for root in 0..2 {
            assert!((davidson.eigenvalues[root] - dense.eigenvalues[root]).abs() < 1.0e-11);
            assert!(davidson.residual_norms[root] < 1.0e-11);
        }
        assert!(davidson.hamiltonian_applications <= dimension);

        let mut gram_calls = 0;
        let accelerated_reduction = block_davidson_batched_with_gram(
            &kinetic,
            None,
            &options,
            |vectors| {
                Ok(vectors
                    .iter()
                    .map(|vector| {
                        (0..dimension)
                            .map(|row| {
                                (0..dimension)
                                    .map(|column| matrix[row * dimension + column] * vector[column])
                                    .sum()
                            })
                            .collect()
                    })
                    .collect())
            },
            |left, right| {
                gram_calls += 1;
                Ok(cross_gram_cpu(left, right))
            },
        )
        .unwrap();
        assert!(accelerated_reduction.converged);
        assert!(gram_calls > 0);
        for root in 0..2 {
            assert!(
                (accelerated_reduction.eigenvalues[root] - dense.eigenvalues[root]).abs() < 1.0e-11
            );
        }
    }

    #[test]
    fn rejects_invalid_accelerator_norm_results() {
        let kinetic = [1.0, 2.0];
        let options = DavidsonOptions {
            roots: 1,
            residual_tolerance: 1.0e-10,
            max_iterations: 2,
            max_subspace: 2,
        };
        let apply = |vectors: &[Vec<Complex64>]| {
            Ok(vectors
                .iter()
                .map(|vector| vec![vector[0], 2.0 * vector[1]])
                .collect())
        };

        let wrong_count = block_davidson_batched_with_reductions(
            &kinetic,
            None,
            &options,
            apply,
            |left, right| Ok(cross_gram_cpu(left, right)),
            |_| Ok(Vec::new()),
        )
        .unwrap_err();
        assert!(wrong_count.to_string().contains("wrong vector count"));

        let invalid_value = block_davidson_batched_with_reductions(
            &kinetic,
            None,
            &options,
            apply,
            |left, right| Ok(cross_gram_cpu(left, right)),
            |_| Ok(vec![f64::NAN]),
        )
        .unwrap_err();
        assert!(invalid_value.to_string().contains("invalid residual norm"));
    }
}
