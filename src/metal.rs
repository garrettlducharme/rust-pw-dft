use anyhow::{Result, bail};
use num_complex::Complex32;

#[cfg(all(feature = "metal", target_os = "macos"))]
mod imp {
    use super::*;
    use std::ffi::CStr;
    use std::os::raw::{c_char, c_int};

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
    apply_dense_operator, apply_local_fft, apply_local_hamiltonian, device_name, fft_roundtrip,
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
