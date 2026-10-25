use anyhow::{Result, ensure};
use num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

#[derive(Clone)]
pub struct Fft3d {
    shape: [usize; 3],
    forward: [Arc<dyn Fft<f64>>; 3],
    inverse: [Arc<dyn Fft<f64>>; 3],
}

impl Fft3d {
    pub fn new(shape: [usize; 3]) -> Result<Self> {
        ensure!(
            shape.iter().all(|&n| n > 0),
            "FFT dimensions must be nonzero"
        );
        let mut planner = FftPlanner::new();
        let forward = shape.map(|n| planner.plan_fft_forward(n));
        let inverse = shape.map(|n| planner.plan_fft_inverse(n));
        Ok(Self {
            shape,
            forward,
            inverse,
        })
    }

    pub fn shape(&self) -> [usize; 3] {
        self.shape
    }
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn is_empty(&self) -> bool {
        false
    }

    pub fn forward(&self, values: &mut [Complex64]) -> Result<()> {
        self.transform(values, false)
    }

    pub fn inverse(&self, values: &mut [Complex64]) -> Result<()> {
        self.transform(values, true)?;
        let scale = 1.0 / self.len() as f64;
        for value in values {
            *value *= scale;
        }
        Ok(())
    }

    fn transform(&self, values: &mut [Complex64], inverse: bool) -> Result<()> {
        ensure!(values.len() == self.len(), "FFT buffer has wrong length");
        let [nx, ny, nz] = self.shape;
        let plans = if inverse {
            &self.inverse
        } else {
            &self.forward
        };
        let mut line = vec![Complex64::default(); nx.max(ny).max(nz)];

        for x in 0..nx {
            for y in 0..ny {
                for z in 0..nz {
                    line[z] = values[(x * ny + y) * nz + z];
                }
                plans[2].process(&mut line[..nz]);
                for z in 0..nz {
                    values[(x * ny + y) * nz + z] = line[z];
                }
            }
        }
        for x in 0..nx {
            for z in 0..nz {
                for y in 0..ny {
                    line[y] = values[(x * ny + y) * nz + z];
                }
                plans[1].process(&mut line[..ny]);
                for y in 0..ny {
                    values[(x * ny + y) * nz + z] = line[y];
                }
            }
        }
        for y in 0..ny {
            for z in 0..nz {
                for x in 0..nx {
                    line[x] = values[(x * ny + y) * nz + z];
                }
                plans[0].process(&mut line[..nx]);
                for x in 0..nx {
                    values[(x * ny + y) * nz + z] = line[x];
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_parseval_are_fp64_accurate() {
        let fft = Fft3d::new([7, 6, 5]).unwrap();
        let original: Vec<_> = (0..fft.len())
            .map(|i| Complex64::new((i as f64 * 0.31).sin(), (i as f64 * 0.17).cos()))
            .collect();
        let norm_real: f64 = original.iter().map(Complex64::norm_sqr).sum();
        let mut frequency = original.clone();
        fft.forward(&mut frequency).unwrap();
        let norm_frequency: f64 =
            frequency.iter().map(Complex64::norm_sqr).sum::<f64>() / fft.len() as f64;
        assert!((norm_real - norm_frequency).abs() / norm_real < 1.0e-12);
        fft.inverse(&mut frequency).unwrap();
        let error = frequency
            .iter()
            .zip(&original)
            .map(|(a, b)| (*a - *b).norm_sqr())
            .sum::<f64>()
            .sqrt()
            / norm_real.sqrt();
        assert!(error < 1.0e-12, "relative L2 error was {error:e}");
    }
}
