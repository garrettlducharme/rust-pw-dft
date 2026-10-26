use anyhow::{Result, bail};
use num_complex::Complex32;

#[derive(Debug)]
pub struct ResidentRitzVectors {
    pub ritz: Vec<Vec<num_complex::Complex64>>,
    pub applied_ritz: Vec<Vec<num_complex::Complex64>>,
    pub residuals: Vec<Vec<num_complex::Complex64>>,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod imp {
    use super::*;
    use std::ffi::CStr;
    use std::marker::PhantomData;
    use std::os::raw::{c_char, c_int, c_void};
    use std::ptr::NonNull;
    use std::rc::Rc;

    unsafe extern "C" {
        fn rpd_metal_device_name(buffer: *mut c_char, capacity: usize) -> c_int;
        fn rpd_metal_fft_roundtrip(
            input_interleaved: *const f32,
            output_interleaved: *mut f32,
            nx: usize,
            ny: usize,
            nz: usize,
            batch: usize,
            error_buffer: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn rpd_metal_apply_local(
            input_interleaved: *const f32,
            potential_interleaved: *const f32,
            kinetic_interleaved: *const f32,
            output_interleaved: *mut f32,
            nx: usize,
            ny: usize,
            nz: usize,
            batch: usize,
            error_buffer: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn rpd_metal_apply_dense(
            input_interleaved: *const f32,
            matrix_interleaved: *const f32,
            output_interleaved: *mut f32,
            dimension: usize,
            batch: usize,
            error_buffer: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn rpd_metal_hamiltonian_create(
            kinetic_interleaved: *const f32,
            basis_indices: *const i32,
            projector_conjugate_interleaved: *const f32,
            d_transpose_interleaved: *const f32,
            projector_transpose_interleaved: *const f32,
            nx: usize,
            ny: usize,
            nz: usize,
            dimension: usize,
            projector_count: usize,
            max_batch: usize,
            error_buffer: *mut c_char,
            error_capacity: usize,
        ) -> *mut c_void;
        fn rpd_metal_hamiltonian_apply(
            context: *mut c_void,
            grid_input_interleaved: *const f32,
            compact_input_interleaved: *const f32,
            potential_interleaved: *const f32,
            compact_output_interleaved: *mut f32,
            batch: usize,
            error_buffer: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn rpd_metal_hamiltonian_destroy(context: *mut c_void);
        fn rpd_metal_hamiltonian_peak_allocated_bytes(context: *mut c_void) -> u64;
        fn rpd_metal_hamiltonian_partial_gram(
            context: *mut c_void,
            left_input_interleaved: *const f32,
            right_input_interleaved: *const f32,
            partial_output_interleaved: *mut f32,
            left_batch: usize,
            right_batch: usize,
            error_buffer: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn rpd_metal_hamiltonian_resident_begin(
            context: *mut c_void,
            error_buffer: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn rpd_metal_hamiltonian_resident_count(context: *mut c_void) -> usize;
        fn rpd_metal_hamiltonian_resident_sync(
            context: *mut c_void,
            basis_interleaved: *const f32,
            applied_interleaved: *const f32,
            batch: usize,
            error_buffer: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn rpd_metal_hamiltonian_resident_partial_gram(
            context: *mut c_void,
            partial_output_interleaved: *mut f32,
            subspace: usize,
            error_buffer: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn rpd_metal_hamiltonian_resident_form_ritz_residuals(
            context: *mut c_void,
            coefficients_interleaved: *const f32,
            eigenvalues_interleaved: *const f32,
            subspace: usize,
            roots: usize,
            error_buffer: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn rpd_metal_hamiltonian_resident_read_ritz_residuals(
            context: *mut c_void,
            ritz_output_interleaved: *mut f32,
            applied_ritz_output_interleaved: *mut f32,
            residual_output_interleaved: *mut f32,
            roots: usize,
            error_buffer: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn rpd_metal_hamiltonian_resident_partial_residual_norms(
            context: *mut c_void,
            partial_output_interleaved: *mut f32,
            roots: usize,
            error_buffer: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn rpd_metal_hamiltonian_partial_norms(
            context: *mut c_void,
            input_interleaved: *const f32,
            partial_output_interleaved: *mut f32,
            batch: usize,
            error_buffer: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn rpd_metal_hamiltonian_reduction_chunks(context: *mut c_void) -> usize;
    }

    pub struct HamiltonianContext {
        raw: NonNull<c_void>,
        shape: [usize; 3],
        dimension: usize,
        projector_count: usize,
        max_batch: usize,
        reduction_chunks: usize,
        persistent_buffer_bytes: usize,
        gram_reduction_calls: usize,
        gram_reduction_pairs: usize,
        resident_projection_calls: usize,
        resident_projection_resyncs: usize,
        norm_reduction_calls: usize,
        norm_reduction_vectors: usize,
        resident_ritz_updates: usize,
        resident_coefficient_upload_bytes: usize,
        resident_vector_download_bytes: usize,
        _not_send_or_sync: PhantomData<Rc<()>>,
    }

    impl HamiltonianContext {
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            shape: [usize; 3],
            dimension: usize,
            projector_count: usize,
            max_batch: usize,
            kinetic: &[Complex32],
            basis_indices: &[i32],
            projector_conjugate: &[Complex32],
            d_transpose: &[Complex32],
            projector_transpose: &[Complex32],
        ) -> Result<Self> {
            let grid_points = shape.iter().product::<usize>();
            if shape.contains(&0) || dimension == 0 || projector_count == 0 || max_batch == 0 {
                bail!("Metal Hamiltonian context dimensions must be nonzero");
            }
            if kinetic.len() != grid_points {
                bail!("Metal kinetic grid has the wrong size");
            }
            if basis_indices.len() != dimension * max_batch {
                bail!("Metal basis-index matrix has the wrong size");
            }
            if projector_conjugate.len() != dimension * projector_count
                || projector_transpose.len() != projector_count * dimension
            {
                bail!("Metal projector matrix has the wrong size");
            }
            if d_transpose.len() != projector_count * projector_count {
                bail!("Metal projector coupling matrix has the wrong size");
            }
            let mut error = vec![0_i8; 2048];
            let raw = unsafe {
                rpd_metal_hamiltonian_create(
                    kinetic.as_ptr().cast::<f32>(),
                    basis_indices.as_ptr(),
                    projector_conjugate.as_ptr().cast::<f32>(),
                    d_transpose.as_ptr().cast::<f32>(),
                    projector_transpose.as_ptr().cast::<f32>(),
                    shape[0],
                    shape[1],
                    shape[2],
                    dimension,
                    projector_count,
                    max_batch,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            let Some(raw) = NonNull::new(raw) else {
                let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
                bail!("Metal Hamiltonian context creation failed: {message}");
            };
            let reduction_chunks = unsafe { rpd_metal_hamiltonian_reduction_chunks(raw.as_ptr()) };
            if reduction_chunks == 0 {
                unsafe { rpd_metal_hamiltonian_destroy(raw.as_ptr()) };
                bail!("Metal Hamiltonian reported zero reduction chunks");
            }
            let reduction_dimension = reduction_chunks * 256;
            let complex_bytes = std::mem::size_of::<Complex32>();
            let persistent_buffer_bytes = complex_bytes
                * (grid_points * max_batch
                    + 2 * dimension * max_batch
                    + 2 * grid_points
                    + 2 * dimension * projector_count
                    + projector_count * projector_count
                    + 6 * reduction_dimension * max_batch
                    + max_batch * max_batch
                    + max_batch)
                + std::mem::size_of_val(basis_indices);
            Ok(Self {
                raw,
                shape,
                dimension,
                projector_count,
                max_batch,
                reduction_chunks,
                persistent_buffer_bytes,
                gram_reduction_calls: 0,
                gram_reduction_pairs: 0,
                resident_projection_calls: 0,
                resident_projection_resyncs: 0,
                norm_reduction_calls: 0,
                norm_reduction_vectors: 0,
                resident_ritz_updates: 0,
                resident_coefficient_upload_bytes: 0,
                resident_vector_download_bytes: 0,
                _not_send_or_sync: PhantomData,
            })
        }

        pub fn apply(
            &mut self,
            grid_input: &[Complex32],
            compact_input: &[Complex32],
            potential: &[Complex32],
            batch: usize,
        ) -> Result<Vec<Complex32>> {
            let grid_points = self.shape.iter().product::<usize>();
            if batch == 0 || batch > self.max_batch {
                bail!(
                    "Metal Hamiltonian batch {batch} exceeds configured maximum {}",
                    self.max_batch
                );
            }
            if grid_input.len() != grid_points * batch {
                bail!("Metal grid input has the wrong size");
            }
            if compact_input.len() != self.dimension * batch {
                bail!("Metal compact input has the wrong size");
            }
            if potential.len() != grid_points {
                bail!("Metal local potential has the wrong size");
            }
            let mut compact_output = vec![Complex32::default(); compact_input.len()];
            let mut error = vec![0_i8; 2048];
            let status = unsafe {
                rpd_metal_hamiltonian_apply(
                    self.raw.as_ptr(),
                    grid_input.as_ptr().cast::<f32>(),
                    compact_input.as_ptr().cast::<f32>(),
                    potential.as_ptr().cast::<f32>(),
                    compact_output.as_mut_ptr().cast::<f32>(),
                    batch,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
                bail!("resident Metal Hamiltonian failed ({status}): {message}");
            }
            Ok(compact_output)
        }

        /// Start a Davidson solve whose successive Hamiltonian batches will be
        /// journaled in persistent compact-FP32 Metal buffers.
        pub fn begin_resident_davidson(&mut self) -> Result<()> {
            let mut error = vec![0_i8; 2048];
            let status = unsafe {
                rpd_metal_hamiltonian_resident_begin(
                    self.raw.as_ptr(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
                bail!("failed to begin resident Metal Davidson solve ({status}): {message}");
            }
            Ok(())
        }

        /// Project a resident Davidson basis against its resident Hamiltonian
        /// images. Normal append iterations require no host-vector upload;
        /// subspace restarts are detected by a row-count mismatch and resynced.
        pub fn resident_cross_gram(
            &mut self,
            basis: &[Vec<num_complex::Complex64>],
            applied_basis: &[Vec<num_complex::Complex64>],
        ) -> Result<Vec<num_complex::Complex64>> {
            if basis.is_empty() || basis.len() != applied_basis.len() {
                bail!("resident Metal projection requires equal nonempty batches");
            }
            if basis.len() > self.max_batch {
                bail!("resident Metal projection exceeds the configured subspace");
            }
            if basis
                .iter()
                .chain(applied_basis)
                .any(|vector| vector.len() != self.dimension)
            {
                bail!("resident Metal projection vector has the wrong dimension");
            }
            let resident_count = unsafe { rpd_metal_hamiltonian_resident_count(self.raw.as_ptr()) };
            if resident_count != basis.len() {
                let basis_values: Vec<_> = basis
                    .iter()
                    .flatten()
                    .map(|value| Complex32::new(value.re as f32, value.im as f32))
                    .collect();
                let applied_values: Vec<_> = applied_basis
                    .iter()
                    .flatten()
                    .map(|value| Complex32::new(value.re as f32, value.im as f32))
                    .collect();
                let mut error = vec![0_i8; 2048];
                let status = unsafe {
                    rpd_metal_hamiltonian_resident_sync(
                        self.raw.as_ptr(),
                        basis_values.as_ptr().cast::<f32>(),
                        applied_values.as_ptr().cast::<f32>(),
                        basis.len(),
                        error.as_mut_ptr(),
                        error.len(),
                    )
                };
                if status != 0 {
                    let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
                    bail!("resident Metal projection resync failed ({status}): {message}");
                }
                self.resident_projection_resyncs += 1;
            }

            let pair_count = basis.len() * basis.len();
            let mut partials = vec![Complex32::default(); pair_count * self.reduction_chunks];
            let mut error = vec![0_i8; 2048];
            let status = unsafe {
                rpd_metal_hamiltonian_resident_partial_gram(
                    self.raw.as_ptr(),
                    partials.as_mut_ptr().cast::<f32>(),
                    basis.len(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
                bail!("resident Metal projection failed ({status}): {message}");
            }
            self.gram_reduction_calls += 1;
            self.gram_reduction_pairs += pair_count;
            self.resident_projection_calls += 1;
            Ok((0..pair_count)
                .map(|pair| {
                    partials[pair * self.reduction_chunks..(pair + 1) * self.reduction_chunks]
                        .iter()
                        .fold(num_complex::Complex64::default(), |sum, value| {
                            sum + num_complex::Complex64::new(value.re as f64, value.im as f64)
                        })
                })
                .collect())
        }

        /// Form Ritz vectors, applied Ritz vectors, and residuals from the
        /// resident Davidson journal. Only the compact coefficient matrix and
        /// eigenvalues cross from the CPU; all full vectors remain in Metal
        /// buffers until explicitly downloaded for validation or final output.
        pub fn form_resident_ritz_residuals(
            &mut self,
            coefficients: &[Vec<num_complex::Complex64>],
            eigenvalues: &[f64],
        ) -> Result<()> {
            if coefficients.is_empty() || coefficients.len() != eigenvalues.len() {
                bail!("resident Ritz update requires one coefficient row per eigenvalue");
            }
            if coefficients.len() > self.max_batch {
                bail!("resident Ritz root count exceeds capacity");
            }
            let subspace = coefficients[0].len();
            if subspace == 0 || subspace > self.max_batch {
                bail!("resident Ritz subspace is empty or exceeds capacity");
            }
            if coefficients.iter().any(|row| {
                row.len() != subspace
                    || row
                        .iter()
                        .any(|value| !value.re.is_finite() || !value.im.is_finite())
            }) || eigenvalues.iter().any(|value| !value.is_finite())
            {
                bail!("resident Ritz coefficients or eigenvalues are invalid");
            }
            let coefficient_values: Vec<_> = coefficients
                .iter()
                .flatten()
                .map(|value| Complex32::new(value.re as f32, value.im as f32))
                .collect();
            let eigenvalue_values: Vec<_> = eigenvalues
                .iter()
                .map(|&value| Complex32::new(value as f32, 0.0))
                .collect();
            let mut error = vec![0_i8; 2048];
            let status = unsafe {
                rpd_metal_hamiltonian_resident_form_ritz_residuals(
                    self.raw.as_ptr(),
                    coefficient_values.as_ptr().cast::<f32>(),
                    eigenvalue_values.as_ptr().cast::<f32>(),
                    subspace,
                    coefficients.len(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
                bail!("resident Metal Ritz update failed ({status}): {message}");
            }
            self.resident_ritz_updates += 1;
            self.resident_coefficient_upload_bytes += (coefficient_values.len()
                + eigenvalue_values.len())
                * std::mem::size_of::<Complex32>();
            Ok(())
        }

        /// Download resident Ritz, applied-Ritz, and residual vectors. This is
        /// an explicit validation/final-output boundary and is byte-counted.
        pub fn read_resident_ritz_residuals(
            &mut self,
            roots: usize,
        ) -> Result<ResidentRitzVectors> {
            if roots == 0 || roots > self.max_batch {
                bail!("resident Ritz download root count is invalid");
            }
            let value_count = roots * self.dimension;
            let mut ritz = vec![Complex32::default(); value_count];
            let mut applied_ritz = vec![Complex32::default(); value_count];
            let mut residuals = vec![Complex32::default(); value_count];
            let mut error = vec![0_i8; 2048];
            let status = unsafe {
                rpd_metal_hamiltonian_resident_read_ritz_residuals(
                    self.raw.as_ptr(),
                    ritz.as_mut_ptr().cast::<f32>(),
                    applied_ritz.as_mut_ptr().cast::<f32>(),
                    residuals.as_mut_ptr().cast::<f32>(),
                    roots,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
                bail!("resident Metal Ritz download failed ({status}): {message}");
            }
            self.resident_vector_download_bytes +=
                3 * value_count * std::mem::size_of::<Complex32>();
            let unpack = |values: Vec<Complex32>| {
                values
                    .chunks_exact(self.dimension)
                    .map(|row| {
                        row.iter()
                            .map(|value| {
                                num_complex::Complex64::new(value.re as f64, value.im as f64)
                            })
                            .collect()
                    })
                    .collect()
            };
            Ok(ResidentRitzVectors {
                ritz: unpack(ritz),
                applied_ritz: unpack(applied_ritz),
                residuals: unpack(residuals),
            })
        }

        /// Reduce norms directly from resident residual buffers without a
        /// full-vector upload or download.
        pub fn resident_residual_norms(&mut self, roots: usize) -> Result<Vec<f64>> {
            if roots == 0 || roots > self.max_batch {
                bail!("resident residual-norm root count is invalid");
            }
            let mut partials = vec![Complex32::default(); roots * self.reduction_chunks];
            let mut error = vec![0_i8; 2048];
            let status = unsafe {
                rpd_metal_hamiltonian_resident_partial_residual_norms(
                    self.raw.as_ptr(),
                    partials.as_mut_ptr().cast::<f32>(),
                    roots,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
                bail!("resident Metal residual norms failed ({status}): {message}");
            }
            let mut norms = Vec::with_capacity(roots);
            for root in 0..roots {
                let sum = partials
                    [root * self.reduction_chunks..(root + 1) * self.reduction_chunks]
                    .iter()
                    .fold(num_complex::Complex64::default(), |sum, value| {
                        sum + num_complex::Complex64::new(value.re as f64, value.im as f64)
                    });
                if !sum.re.is_finite()
                    || !sum.im.is_finite()
                    || sum.im.abs() > 1.0e-6 * sum.re.abs().max(1.0e-20)
                {
                    bail!("resident Metal norm reduction returned an invalid complex sum");
                }
                norms.push(sum.re.max(0.0).sqrt());
            }
            self.norm_reduction_calls += 1;
            self.norm_reduction_vectors += roots;
            Ok(norms)
        }

        /// Return direct residual norms from chunked complex-FP32 Metal
        /// partials with final complex-FP64 accumulation on the CPU.
        pub fn residual_norms(
            &mut self,
            vectors: &[Vec<num_complex::Complex64>],
        ) -> Result<Vec<f64>> {
            if vectors.is_empty() || vectors.len() > self.max_batch {
                bail!("Metal residual-norm batch is empty or exceeds capacity");
            }
            if vectors.iter().any(|vector| vector.len() != self.dimension) {
                bail!("Metal residual-norm vector has the wrong dimension");
            }
            let values: Vec<_> = vectors
                .iter()
                .flatten()
                .map(|value| Complex32::new(value.re as f32, value.im as f32))
                .collect();
            let mut partials = vec![Complex32::default(); vectors.len() * self.reduction_chunks];
            let mut error = vec![0_i8; 2048];
            let status = unsafe {
                rpd_metal_hamiltonian_partial_norms(
                    self.raw.as_ptr(),
                    values.as_ptr().cast::<f32>(),
                    partials.as_mut_ptr().cast::<f32>(),
                    vectors.len(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
                bail!("Metal partial norm reduction failed ({status}): {message}");
            }
            let mut norms = Vec::with_capacity(vectors.len());
            for vector in 0..vectors.len() {
                let sum = partials
                    [vector * self.reduction_chunks..(vector + 1) * self.reduction_chunks]
                    .iter()
                    .fold(num_complex::Complex64::default(), |sum, value| {
                        sum + num_complex::Complex64::new(value.re as f64, value.im as f64)
                    });
                if !sum.re.is_finite()
                    || !sum.im.is_finite()
                    || sum.im.abs() > 1.0e-6 * sum.re.abs().max(1.0e-20)
                {
                    bail!("Metal residual-norm reduction returned an invalid complex sum");
                }
                norms.push(sum.re.max(0.0).sqrt());
            }
            self.norm_reduction_calls += 1;
            self.norm_reduction_vectors += vectors.len();
            Ok(norms)
        }

        /// Return the rectangular cross-Gram matrix `<left_i|right_j>`.
        /// Metal produces one complex-FP32 partial per 256 coefficients; Rust
        /// performs the final accumulation in complex FP64.
        pub fn cross_gram(
            &mut self,
            left: &[Vec<num_complex::Complex64>],
            right: &[Vec<num_complex::Complex64>],
        ) -> Result<Vec<num_complex::Complex64>> {
            if left.is_empty() || right.is_empty() {
                bail!("Metal cross-Gram batches must be nonempty");
            }
            if left.len() > self.max_batch || right.len() > self.max_batch {
                bail!("Metal cross-Gram batch exceeds the configured maximum");
            }
            if left
                .iter()
                .chain(right)
                .any(|vector| vector.len() != self.dimension)
            {
                bail!("Metal cross-Gram vector has the wrong dimension");
            }
            let left_values: Vec<_> = left
                .iter()
                .flatten()
                .map(|value| Complex32::new(value.re as f32, value.im as f32))
                .collect();
            let right_values: Vec<_> = right
                .iter()
                .flatten()
                .map(|value| Complex32::new(value.re as f32, value.im as f32))
                .collect();
            let pair_count = left.len() * right.len();
            let mut partials = vec![Complex32::default(); pair_count * self.reduction_chunks];
            let mut error = vec![0_i8; 2048];
            let status = unsafe {
                rpd_metal_hamiltonian_partial_gram(
                    self.raw.as_ptr(),
                    left_values.as_ptr().cast::<f32>(),
                    right_values.as_ptr().cast::<f32>(),
                    partials.as_mut_ptr().cast::<f32>(),
                    left.len(),
                    right.len(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
                bail!("Metal partial Gram reduction failed ({status}): {message}");
            }
            self.gram_reduction_calls += 1;
            self.gram_reduction_pairs += pair_count;
            Ok((0..pair_count)
                .map(|pair| {
                    partials[pair * self.reduction_chunks..(pair + 1) * self.reduction_chunks]
                        .iter()
                        .fold(num_complex::Complex64::default(), |sum, value| {
                            sum + num_complex::Complex64::new(value.re as f64, value.im as f64)
                        })
                })
                .collect())
        }

        pub fn gram_reduction_calls(&self) -> usize {
            self.gram_reduction_calls
        }

        pub fn gram_reduction_pairs(&self) -> usize {
            self.gram_reduction_pairs
        }

        pub fn resident_projection_calls(&self) -> usize {
            self.resident_projection_calls
        }

        pub fn resident_projection_resyncs(&self) -> usize {
            self.resident_projection_resyncs
        }

        pub fn norm_reduction_calls(&self) -> usize {
            self.norm_reduction_calls
        }

        pub fn norm_reduction_vectors(&self) -> usize {
            self.norm_reduction_vectors
        }

        pub fn resident_ritz_updates(&self) -> usize {
            self.resident_ritz_updates
        }

        pub fn resident_coefficient_upload_bytes(&self) -> usize {
            self.resident_coefficient_upload_bytes
        }

        pub fn resident_vector_download_bytes(&self) -> usize {
            self.resident_vector_download_bytes
        }

        pub fn projector_count(&self) -> usize {
            self.projector_count
        }

        pub fn persistent_buffer_bytes(&self) -> usize {
            self.persistent_buffer_bytes
        }

        pub fn peak_device_allocated_bytes(&self) -> u64 {
            unsafe { rpd_metal_hamiltonian_peak_allocated_bytes(self.raw.as_ptr()) }
        }
    }

    impl Drop for HamiltonianContext {
        fn drop(&mut self) {
            unsafe { rpd_metal_hamiltonian_destroy(self.raw.as_ptr()) };
        }
    }

    pub fn device_name() -> Result<String> {
        let mut buffer = vec![0_i8; 512];
        let status = unsafe { rpd_metal_device_name(buffer.as_mut_ptr(), buffer.len()) };
        if status != 0 {
            bail!("no usable Metal device (status {status})");
        }
        Ok(unsafe { CStr::from_ptr(buffer.as_ptr()) }
            .to_string_lossy()
            .into_owned())
    }

    pub fn fft_roundtrip(
        values: &[Complex32],
        shape: [usize; 3],
        batch: usize,
    ) -> Result<Vec<Complex32>> {
        let expected = shape.iter().product::<usize>() * batch;
        if values.len() != expected {
            bail!(
                "Metal FFT input has {} values; expected {expected}",
                values.len()
            );
        }
        let mut output = vec![Complex32::default(); expected];
        let mut error = vec![0_i8; 2048];
        let status = unsafe {
            rpd_metal_fft_roundtrip(
                values.as_ptr().cast::<f32>(),
                output.as_mut_ptr().cast::<f32>(),
                shape[0],
                shape[1],
                shape[2],
                batch,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
            bail!("Metal FFT failed ({status}): {message}");
        }
        Ok(output)
    }

    pub fn apply_local_fft(
        values: &[Complex32],
        potential: &[f32],
        shape: [usize; 3],
        batch: usize,
    ) -> Result<Vec<Complex32>> {
        apply_local_hamiltonian(
            values,
            potential,
            &vec![0.0; shape.iter().product()],
            shape,
            batch,
        )
    }

    pub fn apply_local_hamiltonian(
        values: &[Complex32],
        potential: &[f32],
        kinetic: &[f32],
        shape: [usize; 3],
        batch: usize,
    ) -> Result<Vec<Complex32>> {
        let grid_points = shape.iter().product::<usize>();
        let expected = grid_points * batch;
        if values.len() != expected {
            bail!(
                "Metal local-operator input has {} values; expected {expected}",
                values.len()
            );
        }
        if potential.len() != grid_points {
            bail!(
                "Metal local potential has {} values; expected {grid_points}",
                potential.len()
            );
        }
        if kinetic.len() != grid_points {
            bail!(
                "Metal kinetic diagonal has {} values; expected {grid_points}",
                kinetic.len()
            );
        }
        let complex_potential: Vec<_> = potential
            .iter()
            .map(|&value| Complex32::new(value, 0.0))
            .collect();
        let complex_kinetic: Vec<_> = kinetic
            .iter()
            .map(|&value| Complex32::new(value, 0.0))
            .collect();
        let mut output = vec![Complex32::default(); expected];
        let mut error = vec![0_i8; 2048];
        let status = unsafe {
            rpd_metal_apply_local(
                values.as_ptr().cast::<f32>(),
                complex_potential.as_ptr().cast::<f32>(),
                complex_kinetic.as_ptr().cast::<f32>(),
                output.as_mut_ptr().cast::<f32>(),
                shape[0],
                shape[1],
                shape[2],
                batch,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
            bail!("Metal local operator failed ({status}): {message}");
        }
        Ok(output)
    }

    pub fn apply_dense_operator(
        values: &[Complex32],
        matrix: &[Complex32],
        dimension: usize,
        batch: usize,
    ) -> Result<Vec<Complex32>> {
        if dimension == 0 || batch == 0 {
            bail!("Metal dense operator requires nonzero dimensions");
        }
        if values.len() != dimension * batch {
            bail!(
                "Metal dense-operator input has {} values; expected {}",
                values.len(),
                dimension * batch
            );
        }
        if matrix.len() != dimension * dimension {
            bail!(
                "Metal dense matrix has {} values; expected {}",
                matrix.len(),
                dimension * dimension
            );
        }
        // MPSGraph consumes row batches on the left. Transpose the conventional
        // column-action matrix so row-vector multiplication produces H|psi>.
        let mut transpose = vec![Complex32::default(); matrix.len()];
        for row in 0..dimension {
            for column in 0..dimension {
                transpose[row * dimension + column] = matrix[column * dimension + row];
            }
        }
        let mut output = vec![Complex32::default(); values.len()];
        let mut error = vec![0_i8; 2048];
        let status = unsafe {
            rpd_metal_apply_dense(
                values.as_ptr().cast::<f32>(),
                transpose.as_ptr().cast::<f32>(),
                output.as_mut_ptr().cast::<f32>(),
                dimension,
                batch,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
            bail!("Metal dense operator failed ({status}): {message}");
        }
        Ok(output)
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
mod imp {
    use super::*;
    pub struct HamiltonianContext;
    impl HamiltonianContext {
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            _shape: [usize; 3],
            _dimension: usize,
            _projector_count: usize,
            _max_batch: usize,
            _kinetic: &[Complex32],
            _basis_indices: &[i32],
            _projector_conjugate: &[Complex32],
            _d_transpose: &[Complex32],
            _projector_transpose: &[Complex32],
        ) -> Result<Self> {
            bail!("binary was built without the macOS-only `metal` feature")
        }
        pub fn apply(
            &mut self,
            _grid_input: &[Complex32],
            _compact_input: &[Complex32],
            _potential: &[Complex32],
            _batch: usize,
        ) -> Result<Vec<Complex32>> {
            bail!("binary was built without the macOS-only `metal` feature")
        }
        pub fn cross_gram(
            &mut self,
            _left: &[Vec<num_complex::Complex64>],
            _right: &[Vec<num_complex::Complex64>],
        ) -> Result<Vec<num_complex::Complex64>> {
            bail!("binary was built without the macOS-only `metal` feature")
        }
        pub fn form_resident_ritz_residuals(
            &mut self,
            _coefficients: &[Vec<num_complex::Complex64>],
            _eigenvalues: &[f64],
        ) -> Result<()> {
            bail!("binary was built without the macOS-only `metal` feature")
        }
        pub fn read_resident_ritz_residuals(
            &mut self,
            _roots: usize,
        ) -> Result<ResidentRitzVectors> {
            bail!("binary was built without the macOS-only `metal` feature")
        }
        pub fn resident_residual_norms(&mut self, _roots: usize) -> Result<Vec<f64>> {
            bail!("binary was built without the macOS-only `metal` feature")
        }
        pub fn begin_resident_davidson(&mut self) -> Result<()> {
            bail!("binary was built without the macOS-only `metal` feature")
        }
        pub fn resident_cross_gram(
            &mut self,
            _basis: &[Vec<num_complex::Complex64>],
            _applied_basis: &[Vec<num_complex::Complex64>],
        ) -> Result<Vec<num_complex::Complex64>> {
            bail!("binary was built without the macOS-only `metal` feature")
        }
        pub fn residual_norms(
            &mut self,
            _vectors: &[Vec<num_complex::Complex64>],
        ) -> Result<Vec<f64>> {
            bail!("binary was built without the macOS-only `metal` feature")
        }
        pub fn gram_reduction_calls(&self) -> usize {
            0
        }
        pub fn gram_reduction_pairs(&self) -> usize {
            0
        }
        pub fn resident_projection_calls(&self) -> usize {
            0
        }
        pub fn resident_projection_resyncs(&self) -> usize {
            0
        }
        pub fn norm_reduction_calls(&self) -> usize {
            0
        }
        pub fn norm_reduction_vectors(&self) -> usize {
            0
        }
        pub fn resident_ritz_updates(&self) -> usize {
            0
        }
        pub fn resident_coefficient_upload_bytes(&self) -> usize {
            0
        }
        pub fn resident_vector_download_bytes(&self) -> usize {
            0
        }
        pub fn projector_count(&self) -> usize {
            0
        }
        pub fn persistent_buffer_bytes(&self) -> usize {
            0
        }
        pub fn peak_device_allocated_bytes(&self) -> u64 {
            0
        }
    }
    pub fn device_name() -> Result<String> {
        bail!("binary was built without the macOS-only `metal` feature")
    }
    pub fn fft_roundtrip(
        _values: &[Complex32],
        _shape: [usize; 3],
        _batch: usize,
    ) -> Result<Vec<Complex32>> {
        bail!("binary was built without the macOS-only `metal` feature")
    }
    pub fn apply_local_fft(
        _values: &[Complex32],
        _potential: &[f32],
        _shape: [usize; 3],
        _batch: usize,
    ) -> Result<Vec<Complex32>> {
        bail!("binary was built without the macOS-only `metal` feature")
    }
    pub fn apply_local_hamiltonian(
        _values: &[Complex32],
        _potential: &[f32],
        _kinetic: &[f32],
        _shape: [usize; 3],
        _batch: usize,
    ) -> Result<Vec<Complex32>> {
        bail!("binary was built without the macOS-only `metal` feature")
    }
    pub fn apply_dense_operator(
        _values: &[Complex32],
        _matrix: &[Complex32],
        _dimension: usize,
        _batch: usize,
    ) -> Result<Vec<Complex32>> {
        bail!("binary was built without the macOS-only `metal` feature")
    }
}

pub use imp::{
    HamiltonianContext, apply_dense_operator, apply_local_fft, apply_local_hamiltonian,
    device_name, fft_roundtrip,
};

#[cfg(all(test, feature = "metal", target_os = "macos"))]
mod tests {
    use super::*;
    use crate::fft::Fft3d;
    use num_complex::Complex64;

    #[test]
    fn metal_fft_roundtrip_reports_error() {
        let shape = [11, 9, 7];
        let input: Vec<_> = (0..shape.iter().product())
            .map(|i| Complex32::new((i as f32 * 0.17).sin(), (i as f32 * 0.23).cos()))
            .collect();
        let output = fft_roundtrip(&input, shape, 1).unwrap();
        let reference_norm: f64 = input.iter().map(|z| z.norm_sqr() as f64).sum();
        let error: f64 = input
            .iter()
            .zip(&output)
            .map(|(a, b)| (*a - *b).norm_sqr() as f64)
            .sum();
        let relative_l2 = (error / reference_norm).sqrt();
        eprintln!("Metal FFT measured relative L2 error: {relative_l2:e}");
        assert!(relative_l2 < 1.0e-5);
    }

    #[test]
    fn metal_local_fft_chain_matches_cpu_reference() {
        let shape = [7, 6, 5];
        let batch = 3;
        let grid_points = shape.iter().product::<usize>();
        let input: Vec<_> = (0..grid_points * batch)
            .map(|i| Complex32::new((i as f32 * 0.13).sin(), (i as f32 * 0.07).cos()))
            .collect();
        let potential: Vec<_> = (0..grid_points)
            .map(|i| 0.7 + 0.2 * (i as f32 * 0.11).cos())
            .collect();
        let kinetic: Vec<_> = (0..grid_points).map(|i| 0.03 * (i % 17) as f32).collect();
        let metal = apply_local_hamiltonian(&input, &potential, &kinetic, shape, batch).unwrap();

        let fft = Fft3d::new(shape).unwrap();
        let mut cpu = Vec::with_capacity(input.len());
        for state in 0..batch {
            let start = state * grid_points;
            let mut values: Vec<_> = input[start..start + grid_points]
                .iter()
                .map(|value| Complex64::new(value.re as f64, value.im as f64))
                .collect();
            fft.inverse(&mut values).unwrap();
            for (value, &local) in values.iter_mut().zip(&potential) {
                *value *= local as f64;
            }
            fft.forward(&mut values).unwrap();
            for ((value, original), &diagonal) in values
                .iter_mut()
                .zip(&input[start..start + grid_points])
                .zip(&kinetic)
            {
                *value += diagonal as f64 * Complex64::new(original.re as f64, original.im as f64);
            }
            cpu.extend(
                values
                    .iter()
                    .map(|value| Complex32::new(value.re as f32, value.im as f32)),
            );
        }
        let reference_norm: f64 = cpu.iter().map(|value| value.norm_sqr() as f64).sum();
        let error: f64 = cpu
            .iter()
            .zip(&metal)
            .map(|(a, b)| (*a - *b).norm_sqr() as f64)
            .sum();
        let relative_l2 = (error / reference_norm).sqrt();
        eprintln!("Metal local FFT chain relative L2 error: {relative_l2:e}");
        assert!(relative_l2 < 1.0e-5);
    }

    #[test]
    fn metal_dense_complex_operator_matches_cpu_reference() {
        let dimension = 5;
        let batch = 3;
        let values: Vec<_> = (0..dimension * batch)
            .map(|i| Complex32::new((i as f32 * 0.31).sin(), (i as f32 * 0.23).cos()))
            .collect();
        let mut matrix = vec![Complex32::default(); dimension * dimension];
        for row in 0..dimension {
            for column in 0..dimension {
                matrix[row * dimension + column] = Complex32::new(
                    0.1 * (row + column + 1) as f32,
                    0.03 * (row as f32 - column as f32),
                );
            }
        }
        let metal = apply_dense_operator(&values, &matrix, dimension, batch).unwrap();
        let mut cpu = vec![Complex32::default(); values.len()];
        for state in 0..batch {
            for row in 0..dimension {
                cpu[state * dimension + row] = (0..dimension)
                    .map(|column| {
                        matrix[row * dimension + column] * values[state * dimension + column]
                    })
                    .sum();
            }
        }
        let reference_norm: f64 = cpu.iter().map(|value| value.norm_sqr() as f64).sum();
        let error: f64 = cpu
            .iter()
            .zip(&metal)
            .map(|(a, b)| (*a - *b).norm_sqr() as f64)
            .sum();
        let relative_l2 = (error / reference_norm).sqrt();
        eprintln!("Metal dense complex operator relative L2 error: {relative_l2:e}");
        assert!(relative_l2 < 1.0e-5);
    }
}
