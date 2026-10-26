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

/// Stateful boundary for an eigensolver whose full-vector state remains on an
/// accelerator. Implementations may exchange compact projected matrices and
/// convergence scalars, but normal iterations must not upload or download a
/// complete wavefunction.
pub trait ResidentDavidsonBackend {
    fn initialize(&mut self, basis: &[Vec<Complex64>]) -> Result<()>;
    fn resident_count(&self) -> usize;
    fn projected_hamiltonian(&mut self, subspace: usize) -> Result<Vec<Complex64>>;
    fn form_ritz_residuals(
        &mut self,
        coefficients: &[Vec<Complex64>],
        eigenvalues: &[f64],
    ) -> Result<()>;
    fn residual_norms(&mut self, roots: usize) -> Result<Vec<f64>>;
    fn precondition(&mut self, active_roots: &[bool]) -> Result<()>;
    fn restart_from_ritz(&mut self, roots: usize) -> Result<()>;
    fn orthonormalize_corrections(&mut self, active_roots: &[bool]) -> Result<usize>;
    fn apply_accepted_corrections(&mut self, accepted_count: usize) -> Result<()>;
    fn download_ritz(&mut self, roots: usize) -> Result<Vec<Vec<Complex64>>>;
    fn full_vector_transfer_bytes(&self) -> (usize, usize);
    fn verify_iteration_residency(
        &mut self,
        expected_upload_bytes: usize,
        expected_download_bytes: usize,
    ) -> Result<()>;
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

fn validate_davidson_inputs(kinetic_diagonal: &[f64], options: &DavidsonOptions) -> Result<()> {
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
    Ok(())
}

pub(crate) fn initialize_davidson_basis(
    kinetic_diagonal: &[f64],
    initial_vectors: Option<&[Vec<Complex64>]>,
    options: &DavidsonOptions,
) -> Result<Vec<Vec<Complex64>>> {
    validate_davidson_inputs(kinetic_diagonal, options)?;
    let dimension = kinetic_diagonal.len();
    let mut basis = Vec::with_capacity(options.max_subspace);
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
    Ok(basis)
}

pub(crate) fn solve_raw_projected_hamiltonian(
    raw_projection: &[Complex64],
    subspace: usize,
) -> Result<ProjectedEigenpairs> {
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
    projected_hermitian_eigensolve(&projected, subspace)
}

fn download_orthonormalized_ritz<B: ResidentDavidsonBackend>(
    backend: &mut B,
    roots: usize,
    dimension: usize,
) -> Result<Vec<Vec<Complex64>>> {
    let downloaded = backend.download_ritz(roots)?;
    ensure!(
        downloaded.len() == roots && downloaded.iter().all(|vector| vector.len() == dimension),
        "resident Ritz download returned the wrong shape"
    );
    let mut orthonormal = Vec::with_capacity(roots);
    for vector in downloaded {
        let vector = orthonormalize_candidate(vector, &orthonormal)
            .ok_or_else(|| anyhow::anyhow!("resident Ritz download is rank deficient"))?;
        orthonormal.push(vector);
    }
    Ok(orthonormal)
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
    let mut basis = initialize_davidson_basis(kinetic_diagonal, initial_vectors, options)?;

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
        let projected_pairs = solve_raw_projected_hamiltonian(&raw_projection, subspace)?;
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

/// Block Davidson driver for an accelerator-resident basis and Hamiltonian
/// journal.
///
/// The initial basis is uploaded once. Each normal iteration exchanges only a
/// compact projected matrix, Ritz coefficients, eigenvalues, reduction
/// partials, and convergence decisions. A complete vector is downloaded only
/// when the solve terminates, and any other full-vector transfer fails closed.
pub fn block_davidson_resident<B>(
    kinetic_diagonal: &[f64],
    initial_vectors: Option<&[Vec<Complex64>]>,
    options: &DavidsonOptions,
    backend: &mut B,
) -> Result<DavidsonResult>
where
    B: ResidentDavidsonBackend,
{
    let initial_basis = initialize_davidson_basis(kinetic_diagonal, initial_vectors, options)?;
    backend.initialize(&initial_basis)?;
    ensure!(
        backend.resident_count() == initial_basis.len(),
        "resident Davidson initialization produced the wrong basis size"
    );
    let (expected_upload_bytes, expected_download_bytes) = backend.full_vector_transfer_bytes();
    let mut applications = initial_basis.len();
    let mut last_result: DavidsonResult;

    for iteration in 1..=options.max_iterations {
        let subspace = backend.resident_count();
        ensure!(
            subspace >= options.roots && subspace <= options.max_subspace,
            "resident Davidson subspace is out of range"
        );
        let raw_projection = backend.projected_hamiltonian(subspace)?;
        let projected_pairs = solve_raw_projected_hamiltonian(&raw_projection, subspace)?;
        ensure!(
            projected_pairs.eigenvalues.len() >= options.roots
                && projected_pairs.eigenvectors.len() >= options.roots,
            "projected eigensolve returned too few roots"
        );
        let eigenvalues = projected_pairs.eigenvalues[..options.roots].to_vec();
        let coefficients = projected_pairs.eigenvectors[..options.roots].to_vec();
        backend.form_ritz_residuals(&coefficients, &eigenvalues)?;
        let residual_norms = backend.residual_norms(options.roots)?;
        ensure!(
            residual_norms.len() == options.roots,
            "resident norm backend returned the wrong vector count"
        );
        ensure!(
            residual_norms
                .iter()
                .all(|norm| norm.is_finite() && *norm >= 0.0),
            "resident norm backend returned an invalid residual norm"
        );
        let converged = residual_norms
            .iter()
            .all(|&norm| norm <= options.residual_tolerance);
        last_result = DavidsonResult {
            converged,
            iterations: iteration,
            hamiltonian_applications: applications,
            eigenvalues,
            eigenvectors: Vec::new(),
            residual_norms,
        };

        backend.verify_iteration_residency(expected_upload_bytes, expected_download_bytes)?;
        if converged || iteration == options.max_iterations {
            last_result.eigenvectors =
                download_orthonormalized_ritz(backend, options.roots, kinetic_diagonal.len())?;
            return Ok(last_result);
        }

        let active_roots: Vec<_> = last_result
            .residual_norms
            .iter()
            .map(|&norm| norm > options.residual_tolerance)
            .collect();
        let active_count = active_roots.iter().filter(|&&active| active).count();
        ensure!(
            active_count > 0,
            "unconverged resident solve has no active roots"
        );
        backend.precondition(&active_roots)?;

        if subspace + active_count > options.max_subspace {
            backend.restart_from_ritz(options.roots)?;
            ensure!(
                backend.resident_count() == options.roots,
                "resident Davidson restart produced the wrong basis size"
            );
        }
        let basis_before_append = backend.resident_count();
        let accepted = backend.orthonormalize_corrections(&active_roots)?;
        if accepted == 0 {
            last_result.eigenvectors =
                download_orthonormalized_ritz(backend, options.roots, kinetic_diagonal.len())?;
            return Ok(last_result);
        }
        backend.apply_accepted_corrections(accepted)?;
        applications += accepted;
        ensure!(
            backend.resident_count() == basis_before_append + accepted,
            "resident Davidson append produced the wrong basis size"
        );
        ensure!(
            backend.resident_count() <= options.max_subspace,
            "resident Davidson append exceeded its subspace"
        );
    }
    unreachable!("positive Davidson iteration limit must terminate in the loop")
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

    struct MinimalResidentBackend {
        basis: Vec<Vec<Complex64>>,
        upload_bytes: usize,
        download_bytes: usize,
        leak_during_iteration: bool,
        invariant_checks: usize,
    }

    impl MinimalResidentBackend {
        fn new(leak_during_iteration: bool) -> Self {
            Self {
                basis: Vec::new(),
                upload_bytes: 0,
                download_bytes: 0,
                leak_during_iteration,
                invariant_checks: 0,
            }
        }
    }

    impl ResidentDavidsonBackend for MinimalResidentBackend {
        fn initialize(&mut self, basis: &[Vec<Complex64>]) -> Result<()> {
            self.basis = basis.to_vec();
            self.upload_bytes = basis.iter().map(|vector| vector.len() * 16).sum();
            Ok(())
        }

        fn resident_count(&self) -> usize {
            self.basis.len()
        }

        fn projected_hamiltonian(&mut self, subspace: usize) -> Result<Vec<Complex64>> {
            ensure!(subspace == 1, "test backend only supports one state");
            Ok(vec![Complex64::new(1.0, 0.0)])
        }

        fn form_ritz_residuals(
            &mut self,
            _coefficients: &[Vec<Complex64>],
            _eigenvalues: &[f64],
        ) -> Result<()> {
            Ok(())
        }

        fn residual_norms(&mut self, roots: usize) -> Result<Vec<f64>> {
            ensure!(roots == 1, "test backend only supports one root");
            Ok(vec![0.0])
        }

        fn precondition(&mut self, _active_roots: &[bool]) -> Result<()> {
            anyhow::bail!("a converged test solve must not precondition")
        }

        fn restart_from_ritz(&mut self, _roots: usize) -> Result<()> {
            anyhow::bail!("a converged test solve must not restart")
        }

        fn orthonormalize_corrections(&mut self, _active_roots: &[bool]) -> Result<usize> {
            anyhow::bail!("a converged test solve must not form corrections")
        }

        fn apply_accepted_corrections(&mut self, _accepted_count: usize) -> Result<()> {
            anyhow::bail!("a converged test solve must not append corrections")
        }

        fn download_ritz(&mut self, roots: usize) -> Result<Vec<Vec<Complex64>>> {
            ensure!(roots == 1, "test backend only supports one root");
            self.download_bytes += self.basis[0].len() * 16;
            Ok(self.basis.clone())
        }

        fn full_vector_transfer_bytes(&self) -> (usize, usize) {
            (self.upload_bytes, self.download_bytes)
        }

        fn verify_iteration_residency(
            &mut self,
            expected_upload_bytes: usize,
            expected_download_bytes: usize,
        ) -> Result<()> {
            if self.leak_during_iteration {
                self.download_bytes += self.basis[0].len() * 16;
            }
            ensure!(
                self.upload_bytes == expected_upload_bytes
                    && self.download_bytes == expected_download_bytes,
                "resident Davidson iteration crossed a full-vector boundary"
            );
            self.invariant_checks += 1;
            Ok(())
        }
    }

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

    #[test]
    fn resident_driver_allows_only_initial_upload_and_final_download() {
        let kinetic = [1.0, 2.0];
        let options = DavidsonOptions {
            roots: 1,
            residual_tolerance: 1.0e-12,
            max_iterations: 2,
            max_subspace: 2,
        };
        let mut backend = MinimalResidentBackend::new(false);
        let result = block_davidson_resident(&kinetic, None, &options, &mut backend).unwrap();
        assert!(result.converged);
        assert_eq!(result.iterations, 1);
        assert_eq!(result.eigenvalues, vec![1.0]);
        assert_eq!(backend.invariant_checks, 1);
        assert_eq!(backend.full_vector_transfer_bytes(), (32, 32));
    }

    #[test]
    fn resident_driver_fails_closed_on_iteration_vector_transfer() {
        let kinetic = [1.0, 2.0];
        let options = DavidsonOptions {
            roots: 1,
            residual_tolerance: 1.0e-12,
            max_iterations: 2,
            max_subspace: 2,
        };
        let mut backend = MinimalResidentBackend::new(true);
        let error = block_davidson_resident(&kinetic, None, &options, &mut backend).unwrap_err();
        assert!(error.to_string().contains("crossed a full-vector boundary"));
    }
}
