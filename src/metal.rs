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
}

pub use imp::{device_name, fft_roundtrip};

#[cfg(all(test, feature = "metal", target_os = "macos"))]
mod tests {
    use super::*;

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
}
