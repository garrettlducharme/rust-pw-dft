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

    pub fn cartesian_k(&self, reduced: Vec3) -> Vec3 {
        let mut k = [0.0; 3];
        for (i, &coefficient) in reduced.iter().enumerate() {
            for (j, value) in k.iter_mut().enumerate() {
                *value += coefficient * self.reciprocal[i][j];
            }
        }
        k
    }

    /// Enumerate Gamma-point plane waves using kinetic energy in Rydberg.
    /// In atomic units, `E_kin(Ry) = |G|^2`.
    pub fn enumerate_g_vectors(&self, cutoff_ry: f64) -> Vec<GVector> {
        self.enumerate_g_vectors_at_k(cutoff_ry, [0.0; 3])
    }

    /// Enumerate the k-dependent plane-wave sphere using
    /// `E_kin(Ry) = |G + k|^2`.
    ///
    /// The integer bounds follow from
    /// `(G + k) · a_i = 2π(n_i + k_i)`, so they remain conservative for
    /// skewed cells where bounds based only on individual reciprocal-vector
    /// norms can miss cancellation directions.
    pub fn enumerate_g_vectors_at_k(&self, cutoff_ry: f64, k_reduced: Vec3) -> Vec<GVector> {
        let radius = cutoff_ry.sqrt();
        let bounds: [i32; 3] = std::array::from_fn(|axis| {
            let direct_norm = dot(self.direct[axis], self.direct[axis]).sqrt();
            (radius * direct_norm / TAU + k_reduced[axis].abs()).ceil() as i32 + 1
        });
        let k_cartesian = self.cartesian_k(k_reduced);
        let mut result = Vec::new();
        for i in -bounds[0]..=bounds[0] {
            for j in -bounds[1]..=bounds[1] {
                for k in -bounds[2]..=bounds[2] {
                    let mut g_plus_k = self.cartesian_g([i, j, k]);
                    for axis in 0..3 {
                        g_plus_k[axis] += k_cartesian[axis];
                    }
                    if dot(g_plus_k, g_plus_k) <= cutoff_ry * (1.0 + 16.0 * f64::EPSILON) {
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

    #[test]
    fn shifted_plane_wave_sphere_uses_g_plus_k() {
        let lattice = Lattice::new([[TAU, 0.0, 0.0], [0.0, TAU, 0.0], [0.0, 0.0, TAU]]).unwrap();
        let gs = lattice.enumerate_g_vectors_at_k(1.0, [0.5, 0.0, 0.0]);
        assert_eq!(gs.len(), 2);
        assert!(gs.iter().any(|g| g.index == [-1, 0, 0]));
        assert!(gs.iter().any(|g| g.index == [0, 0, 0]));
    }

    #[test]
    fn gamma_enumeration_delegates_exactly_to_shifted_path() {
        let lattice = Lattice::new([[5.0, 0.2, 0.1], [0.0, 4.0, 0.3], [0.0, 0.0, 3.0]]).unwrap();
        assert_eq!(
            lattice.enumerate_g_vectors(8.0),
            lattice.enumerate_g_vectors_at_k(8.0, [0.0; 3])
        );
    }
}
