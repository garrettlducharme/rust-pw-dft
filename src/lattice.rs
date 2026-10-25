use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::f64::consts::TAU;

pub type Vec3 = [f64; 3];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lattice {
    pub direct: [Vec3; 3],
    pub reciprocal: [Vec3; 3],
    pub volume: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GVector {
    pub index: [i32; 3],
}

fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn scale(v: Vec3, s: f64) -> Vec3 {
    [v[0] * s, v[1] * s, v[2] * s]
}

impl Lattice {
    pub fn new(direct: [Vec3; 3]) -> Result<Self> {
        let volume = dot(direct[0], cross(direct[1], direct[2]));
        ensure!(
            volume.is_finite() && volume.abs() > 1.0e-12,
            "cell is singular"
        );
        ensure!(volume > 0.0, "cell vectors must have positive orientation");
        let reciprocal = [
            scale(cross(direct[1], direct[2]), TAU / volume),
            scale(cross(direct[2], direct[0]), TAU / volume),
            scale(cross(direct[0], direct[1]), TAU / volume),
        ];
        Ok(Self {
            direct,
            reciprocal,
            volume,
        })
    }

    pub fn cartesian_g(&self, index: [i32; 3]) -> Vec3 {
        let mut g = [0.0; 3];
        for (i, &n) in index.iter().enumerate() {
            for (j, value) in g.iter_mut().enumerate() {
                *value += n as f64 * self.reciprocal[i][j];
            }
        }
        g
    }

    /// Enumerate Gamma-point plane waves using kinetic energy in Rydberg.
    /// In atomic units, `E_kin(Ry) = |G|^2`.
    pub fn enumerate_g_vectors(&self, cutoff_ry: f64) -> Vec<GVector> {
        let reciprocal_norms = self.reciprocal.map(|v| dot(v, v).sqrt());
        let bounds = reciprocal_norms.map(|norm| (cutoff_ry.sqrt() / norm).ceil() as i32 + 1);
        let mut result = Vec::new();
        for i in -bounds[0]..=bounds[0] {
            for j in -bounds[1]..=bounds[1] {
                for k in -bounds[2]..=bounds[2] {
                    let g = self.cartesian_g([i, j, k]);
                    if dot(g, g) <= cutoff_ry * (1.0 + 16.0 * f64::EPSILON) {
                        result.push(GVector { index: [i, j, k] });
                    }
                }
            }
        }
        result.sort_by_key(|g| (g.index[0], g.index[1], g.index[2]));
        result
    }

    pub fn density_fft_grid(g_vectors: &[GVector]) -> [usize; 3] {
        let mut max_index = [0_i32; 3];
        for g in g_vectors {
            for (axis, value) in max_index.iter_mut().enumerate() {
                *value = (*value).max(g.index[axis].abs());
            }
        }
        max_index.map(|n| next_smooth((4 * n + 1).max(1) as usize))
    }
}

fn next_smooth(mut n: usize) -> usize {
    loop {
        let mut remainder = n;
        for factor in [2, 3, 5] {
            while remainder.is_multiple_of(factor) {
                remainder /= factor;
            }
        }
        if remainder == 1 {
            return n;
        }
        n += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn direct_and_reciprocal_are_dual() {
        let lattice = Lattice::new([[5.0, 0.2, 0.1], [0.0, 4.0, 0.3], [0.0, 0.0, 3.0]]).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(
                    dot(lattice.direct[i], lattice.reciprocal[j]),
                    if i == j { TAU } else { 0.0 },
                    epsilon = 1.0e-12
                );
            }
        }
    }

    #[test]
    fn cutoff_is_inclusive_and_grid_avoids_aliasing() {
        let lattice = Lattice::new([[TAU, 0.0, 0.0], [0.0, TAU, 0.0], [0.0, 0.0, TAU]]).unwrap();
        let gs = lattice.enumerate_g_vectors(1.0);
        assert_eq!(gs.len(), 7);
        assert_eq!(Lattice::density_fft_grid(&gs), [5, 5, 5]);
    }
}
