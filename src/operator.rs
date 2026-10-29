//! CPU-FP64 Kohn-Sham Hamiltonian building blocks.
//!
//! The radial transforms use the conventional plane-wave normalization
//! `exp(i G.r) / sqrt(volume)`. Energies and pseudopotentials are in Rydberg.

use crate::fft::Fft3d;
use crate::input::AtomInput;
use crate::lattice::{GVector, Lattice, Vec3};
use crate::pseudo::{NormConservingPseudo, Projector};
use crate::xc::XcFunctional;
use anyhow::{Result, ensure};
#[cfg(all(feature = "metal", target_os = "macos"))]
use num_complex::Complex32;
use num_complex::Complex64;
use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::{PI, TAU};
use std::sync::OnceLock;

const FOUR_PI: f64 = 4.0 * PI;
const ELECTRON_CHARGE_SQUARED_RY: f64 = 2.0;
const RADIAL_NOISE_CUTOFF_BOHR: f64 = 10.0;

fn norm_squared(v: Vec3) -> f64 {
    v.iter().map(|x| x * x).sum()
}

fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn subtract(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(v: Vec3, factor: f64) -> Vec3 {
    [v[0] * factor, v[1] * factor, v[2] * factor]
}

fn fractional_to_cartesian(lattice: &Lattice, fractional: [f64; 3]) -> Vec3 {
    let mut result = [0.0; 3];
    for (axis, coefficient) in fractional.into_iter().enumerate() {
        result = add(result, scale(lattice.direct[axis], coefficient));
    }
    result
}

fn flat_index_for_shape(index: [i32; 3], shape: [usize; 3]) -> usize {
    let x = index[0].rem_euclid(shape[0] as i32) as usize;
    let y = index[1].rem_euclid(shape[1] as i32) as usize;
    let z = index[2].rem_euclid(shape[2] as i32) as usize;
    (x * shape[1] + y) * shape[2] + z
}

fn signed_frequency(index: usize, extent: usize) -> i32 {
    if index <= extent / 2 {
        index as i32
    } else {
        index as i32 - extent as i32
    }
}

/// Project numerically constructed reciprocal coefficients onto the exact
/// Hermitian subspace required by a real-space field. The pre-projection
/// mismatch remains fail-closed so this cannot hide a broken structure factor.
fn enforce_hermitian_grid(values: &mut [Complex64], shape: [usize; 3]) -> Result<()> {
    ensure!(
        values.len() == shape.iter().product::<usize>(),
        "Hermitian grid has the wrong size"
    );
    let mut maximum_mismatch = 0.0_f64;
    let mut maximum_scale = 0.0_f64;
    for x in 0..shape[0] {
        for y in 0..shape[1] {
            for z in 0..shape[2] {
                let flat = (x * shape[1] + y) * shape[2] + z;
                let partner_x = (shape[0] - x) % shape[0];
                let partner_y = (shape[1] - y) % shape[1];
                let partner_z = (shape[2] - z) % shape[2];
                let partner = (partner_x * shape[1] + partner_y) * shape[2] + partner_z;
                maximum_mismatch =
                    maximum_mismatch.max((values[flat] - values[partner].conj()).norm());
                maximum_scale = maximum_scale.max(values[flat].norm());
            }
        }
    }
    let relative_mismatch = if maximum_scale == 0.0 {
        0.0
    } else {
        maximum_mismatch / maximum_scale
    };
    ensure!(
        relative_mismatch < 1.0e-10,
        "reciprocal field is not Hermitian (relative mismatch {relative_mismatch:e})"
    );

    for x in 0..shape[0] {
        for y in 0..shape[1] {
            for z in 0..shape[2] {
                let flat = (x * shape[1] + y) * shape[2] + z;
                let partner_x = (shape[0] - x) % shape[0];
                let partner_y = (shape[1] - y) % shape[1];
                let partner_z = (shape[2] - z) % shape[2];
                let partner = (partner_x * shape[1] + partner_y) * shape[2] + partner_z;
                if flat <= partner {
                    let average = 0.5 * (values[flat] + values[partner].conj());
                    values[flat] = average;
                    values[partner] = average.conj();
                }
            }
        }
    }
    Ok(())
}

fn weighted_simpson(values: &[f64], radial_weights: &[f64]) -> Result<f64> {
    ensure!(
        values.len() == radial_weights.len() && values.len() >= 3,
        "radial integral arrays have incompatible lengths"
    );
    let n = values.len();
    let mut sum = 0.0;
    for i in 1..n - 1 {
        let coefficient = if i % 2 == 1 { 4.0 } else { 2.0 };
        sum += coefficient * values[i] * radial_weights[i];
    }
    if n % 2 == 1 {
        sum += values[0] * radial_weights[0] + values[n - 1] * radial_weights[n - 1];
    } else {
        sum += values[0] * radial_weights[0] - 0.25 * values[n - 3] * radial_weights[n - 3]
            + values[n - 2] * radial_weights[n - 2]
            + 1.25 * values[n - 1] * radial_weights[n - 1];
    }
    Ok(sum / 3.0)
}

/// Numerical UPF tails beyond 10 bohr can contain amplified cancellation
/// noise in `r V_loc + Z e²`. The standard QE transform truncates there and
/// forces an odd Simpson mesh; use the same well-defined convention for every
/// radial field whose physical tail has already vanished.
fn radial_integration_points(pseudo: &NormConservingPseudo) -> Result<usize> {
    let mut points = pseudo
        .radial_grid
        .iter()
        .position(|radius| *radius > RADIAL_NOISE_CUTOFF_BOHR)
        .map_or(pseudo.radial_grid.len(), |index| index + 1);
    if points.is_multiple_of(2) {
        points -= 1;
    }
    ensure!(points >= 3, "UPF radial integration mesh is too short");
    Ok(points)
}

fn spherical_bessel(angular_momentum: usize, x: f64) -> Result<f64> {
    let ax = x.abs();
    match angular_momentum {
        0 if ax < 1.0e-5 => {
            let x2 = x * x;
            Ok(1.0 - x2 / 6.0 + x2 * x2 / 120.0)
        }
        0 => Ok(x.sin() / x),
        1 if ax < 1.0e-4 => {
            let x2 = x * x;
            Ok(x * (1.0 / 3.0 - x2 / 30.0 + x2 * x2 / 840.0))
        }
        1 => Ok(x.sin() / (x * x) - x.cos() / x),
        _ => anyhow::bail!("only l=0 and l=1 projector channels are supported"),
    }
}

/// Fourier coefficient of one atom's local potential, including `1 / volume`.
pub fn local_form_factor_ry(pseudo: &NormConservingPseudo, q: f64, volume: f64) -> Result<f64> {
    ensure!(q.is_finite() && q >= 0.0, "invalid reciprocal magnitude");
    ensure!(volume.is_finite() && volume > 0.0, "invalid cell volume");
    let z_e2 = pseudo.valence_charge * ELECTRON_CHARGE_SQUARED_RY;
    let points = radial_integration_points(pseudo)?;
    let values: Vec<f64> = if q < 1.0e-12 {
        pseudo
            .radial_grid
            .iter()
            .zip(&pseudo.local_potential_ry)
            .take(points)
            .map(|(&r, &v)| r * (r * v + z_e2))
            .collect()
    } else {
        pseudo
            .radial_grid
            .iter()
            .zip(&pseudo.local_potential_ry)
            .take(points)
            .map(|(&r, &v)| (r * v + z_e2 * libm::erf(r)) * (q * r).sin() / q)
            .collect()
    };
    let short_range =
        FOUR_PI * weighted_simpson(&values, &pseudo.radial_weights[..points])? / volume;
    if q < 1.0e-12 {
        Ok(short_range)
    } else {
        Ok(short_range - FOUR_PI * z_e2 * (-0.25 * q * q).exp() / (volume * q * q))
    }
}

/// Radial part of a separable projector, including `4 pi / sqrt(volume)`.
pub fn projector_radial_form_factor(
    pseudo: &NormConservingPseudo,
    projector: &Projector,
    q: f64,
    volume: f64,
) -> Result<f64> {
    ensure!(q.is_finite() && q >= 0.0, "invalid reciprocal magnitude");
    let n = projector.cutoff_points;
    let mut integrand = Vec::with_capacity(n);
    for i in 0..n {
        let r = pseudo.radial_grid[i];
        integrand
            .push(projector.beta[i] * spherical_bessel(projector.angular_momentum, q * r)? * r);
    }
    Ok(FOUR_PI * weighted_simpson(&integrand, &pseudo.radial_weights[..n])? / volume.sqrt())
}

pub fn atomic_density_form_factor(
    pseudo: &NormConservingPseudo,
    q: f64,
    volume: f64,
) -> Result<f64> {
    ensure!(q.is_finite() && q >= 0.0, "invalid reciprocal magnitude");
    let points = radial_integration_points(pseudo)?;
    let values: Vec<_> = pseudo
        .atomic_density
        .iter()
        .zip(&pseudo.radial_grid)
        .take(points)
        .map(|(&density, &r)| density * spherical_bessel(0, q * r).unwrap_or_default())
        .collect();
    Ok(weighted_simpson(&values, &pseudo.radial_weights[..points])? / volume)
}

fn real_spherical_harmonic(l: usize, component: usize, g: Vec3) -> Result<f64> {
    let length = norm_squared(g).sqrt();
    match (l, component) {
        (0, 0) => Ok(1.0 / FOUR_PI.sqrt()),
        (1, _) if length < 1.0e-14 => Ok(0.0),
        (1, 0) => Ok((3.0 / FOUR_PI).sqrt() * g[2] / length),
        (1, 1) => Ok(-(3.0 / FOUR_PI).sqrt() * g[0] / length),
        (1, 2) => Ok(-(3.0 / FOUR_PI).sqrt() * g[1] / length),
        _ => anyhow::bail!("unsupported real spherical-harmonic channel l={l}, m={component}"),
    }
}

#[derive(Clone)]
struct AtomProjectors {
    /// One coefficient vector for each `(radial projector, real m)` channel.
    columns: Vec<Vec<Complex64>>,
    radial_projector: Vec<usize>,
    component: Vec<usize>,
    dij_ry: Vec<Vec<f64>>,
}

#[derive(Clone)]
struct DenseReferenceData {
    lattice: Lattice,
    g_vectors: Vec<GVector>,
    atoms: Vec<AtomInput>,
    pseudos: BTreeMap<String, NormConservingPseudo>,
}

/// Dense-reference CPU Hamiltonian for operator validation and small cells.
///
/// The local operator is deliberately formed directly in reciprocal space. This
/// is not the production FFT path; it provides an independent, transparent
/// reference against which the later FFT-resident CPU and Metal paths can be tested.
#[derive(Clone)]
pub struct CpuHamiltonian {
    k_reduced: [f64; 3],
    wavefunction_cutoff_ry: f64,
    kinetic_ry: Vec<f64>,
    local_ry: OnceLock<Vec<Complex64>>,
    dense_reference: DenseReferenceData,
    atom_projectors: Vec<AtomProjectors>,
    fft: Fft3d,
    basis_grid_indices: Vec<usize>,
    ionic_local_real_ry: Vec<f64>,
    nonlocal_dense_ry: OnceLock<Vec<Complex64>>,
}

impl CpuHamiltonian {
    pub fn new(
        lattice: &Lattice,
        g_vectors: &[GVector],
        atoms: &[AtomInput],
        pseudos: &BTreeMap<String, NormConservingPseudo>,
        wavefunction_cutoff_ry: f64,
    ) -> Result<Self> {
        Self::new_at_k(
            lattice,
            g_vectors,
            atoms,
            pseudos,
            wavefunction_cutoff_ry,
            [0.0; 3],
            Lattice::density_fft_grid(g_vectors),
        )
    }

    pub fn new_at_k(
        lattice: &Lattice,
        g_vectors: &[GVector],
        atoms: &[AtomInput],
        pseudos: &BTreeMap<String, NormConservingPseudo>,
        wavefunction_cutoff_ry: f64,
        k_reduced: [f64; 3],
        grid_shape: [usize; 3],
    ) -> Result<Self> {
        ensure!(!g_vectors.is_empty(), "Hamiltonian basis is empty");
        ensure!(
            wavefunction_cutoff_ry.is_finite() && wavefunction_cutoff_ry > 0.0,
            "wavefunction cutoff must be positive"
        );
        ensure!(
            atoms.iter().all(|atom| pseudos.contains_key(&atom.element)),
            "an atom has no loaded pseudopotential"
        );
        ensure!(
            k_reduced.iter().all(|value| value.is_finite()),
            "k-point coordinates must be finite"
        );
        ensure!(
            grid_shape.iter().all(|&size| size > 0),
            "FFT grid dimensions must be positive"
        );
        let k_cartesian = lattice.cartesian_k(k_reduced);
        let cartesian: Vec<_> = g_vectors
            .iter()
            .map(|g| {
                let mut value = lattice.cartesian_g(g.index);
                for axis in 0..3 {
                    value[axis] += k_cartesian[axis];
                }
                value
            })
            .collect();
        let kinetic_ry = cartesian.iter().map(|&g| norm_squared(g)).collect();
        let dimension = g_vectors.len();
        let fft = Fft3d::new(grid_shape)?;
        let basis_grid_indices = g_vectors
            .iter()
            .map(|g| flat_index_for_shape(g.index, grid_shape))
            .collect::<Vec<_>>();
        let unique_grid_indices = basis_grid_indices.iter().copied().collect::<BTreeSet<_>>();
        ensure!(
            unique_grid_indices.len() == g_vectors.len(),
            "FFT grid aliases distinct plane waves"
        );

        let mut atom_projectors = Vec::with_capacity(atoms.len());
        for atom in atoms {
            let pseudo = &pseudos[&atom.element];
            let mut columns = Vec::new();
            let mut radial_projector = Vec::new();
            let mut component = Vec::new();
            for (projector_index, projector) in pseudo.projectors.iter().enumerate() {
                for m in 0..(2 * projector.angular_momentum + 1) {
                    let mut coefficients = Vec::with_capacity(dimension);
                    let minus_i_to_l =
                        Complex64::new(0.0, -1.0).powu(projector.angular_momentum as u32);
                    for (g_vector, &g) in g_vectors.iter().zip(&cartesian) {
                        let q = norm_squared(g).sqrt();
                        let radial =
                            projector_radial_form_factor(pseudo, projector, q, lattice.volume)?;
                        let angular = real_spherical_harmonic(projector.angular_momentum, m, g)?;
                        let phase = -TAU
                            * g_vector
                                .index
                                .iter()
                                .zip(k_reduced)
                                .zip(atom.fractional)
                                .map(|((&n, k), f)| (n as f64 + k) * f)
                                .sum::<f64>();
                        coefficients.push(
                            radial * angular * minus_i_to_l * Complex64::from_polar(1.0, phase),
                        );
                    }
                    columns.push(coefficients);
                    radial_projector.push(projector_index);
                    component.push(m);
                }
            }
            atom_projectors.push(AtomProjectors {
                columns,
                radial_projector,
                component,
                dij_ry: pseudo.dij_ry.clone(),
            });
        }
        let mut ionic_coefficients = vec![Complex64::default(); fft.len()];
        for x in 0..grid_shape[0] {
            for y in 0..grid_shape[1] {
                for z in 0..grid_shape[2] {
                    let index = [
                        signed_frequency(x, grid_shape[0]),
                        signed_frequency(y, grid_shape[1]),
                        signed_frequency(z, grid_shape[2]),
                    ];
                    let q2 = norm_squared(lattice.cartesian_g(index));
                    if q2 > 4.0 * wavefunction_cutoff_ry * (1.0 + 32.0 * f64::EPSILON) {
                        continue;
                    }
                    let q = q2.sqrt();
                    let coefficient = atoms.iter().try_fold(
                        Complex64::default(),
                        |sum, atom| -> Result<Complex64> {
                            let pseudo = &pseudos[&atom.element];
                            let radial = local_form_factor_ry(pseudo, q, lattice.volume)?;
                            let phase = -TAU
                                * index
                                    .iter()
                                    .zip(atom.fractional)
                                    .map(|(&n, f)| n as f64 * f)
                                    .sum::<f64>();
                            Ok(sum + radial * Complex64::from_polar(1.0, phase))
                        },
                    )?;
                    ionic_coefficients[(x * grid_shape[1] + y) * grid_shape[2] + z] = coefficient;
                }
            }
        }
        enforce_hermitian_grid(&mut ionic_coefficients, grid_shape)?;
        fft.inverse(&mut ionic_coefficients)?;
        let grid_points = fft.len() as f64;
        let maximum_imaginary = ionic_coefficients
            .iter()
            .map(|value| value.im.abs())
            .fold(0.0_f64, f64::max);
        let maximum_real = ionic_coefficients
            .iter()
            .map(|value| value.re.abs())
            .fold(0.0_f64, f64::max);
        let relative_imaginary = maximum_imaginary / maximum_real.max(1.0);
        ensure!(
            relative_imaginary < 1.0e-12,
            "ionic local potential is not real (absolute {maximum_imaginary:e}, relative {relative_imaginary:e})"
        );
        let ionic_local_real_ry = ionic_coefficients
            .iter()
            .map(|value| value.re * grid_points)
            .collect();
        Ok(Self {
            k_reduced,
            wavefunction_cutoff_ry,
            kinetic_ry,
            local_ry: OnceLock::new(),
            dense_reference: DenseReferenceData {
                lattice: lattice.clone(),
                g_vectors: g_vectors.to_vec(),
                atoms: atoms.to_vec(),
                pseudos: pseudos.clone(),
            },
            atom_projectors,
            fft,
            basis_grid_indices,
            ionic_local_real_ry,
            nonlocal_dense_ry: OnceLock::new(),
        })
    }

    pub fn dimension(&self) -> usize {
        self.kinetic_ry.len()
    }

    pub fn k_reduced(&self) -> [f64; 3] {
        self.k_reduced
    }

    pub fn kinetic_ry(&self) -> &[f64] {
        &self.kinetic_ry
    }

    pub fn grid_shape(&self) -> [usize; 3] {
        self.fft.shape()
    }

    pub fn fft(&self) -> &Fft3d {
        &self.fft
    }

    pub fn ionic_local_real_ry(&self) -> &[f64] {
        &self.ionic_local_real_ry
    }

    /// Hellmann-Feynman force from the local ionic pseudopotential at fixed
    /// density, in Rydberg per bohr.
    pub fn local_ionic_forces_ry_bohr(&self, density: &[f64]) -> Result<Vec<Vec3>> {
        ensure!(
            density.len() == self.fft.len() && density.iter().all(|value| value.is_finite()),
            "force density is invalid"
        );
        let grid_points = self.fft.len() as f64;
        let mut density_g: Vec<_> = density
            .iter()
            .map(|&value| Complex64::new(value, 0.0))
            .collect();
        self.fft.forward(&mut density_g)?;
        for coefficient in &mut density_g {
            *coefficient /= grid_points;
        }

        let reference = &self.dense_reference;
        let shape = self.fft.shape();
        let mut forces = vec![[0.0; 3]; reference.atoms.len()];
        for x in 0..shape[0] {
            for y in 0..shape[1] {
                for z in 0..shape[2] {
                    let flat = (x * shape[1] + y) * shape[2] + z;
                    let index = [
                        signed_frequency(x, shape[0]),
                        signed_frequency(y, shape[1]),
                        signed_frequency(z, shape[2]),
                    ];
                    let g = reference.lattice.cartesian_g(index);
                    let q2 = norm_squared(g);
                    if q2 <= 1.0e-20
                        || q2 > 4.0 * self.wavefunction_cutoff_ry * (1.0 + 32.0 * f64::EPSILON)
                    {
                        continue;
                    }
                    let q = q2.sqrt();
                    for (atom_index, atom) in reference.atoms.iter().enumerate() {
                        let radial = local_form_factor_ry(
                            &reference.pseudos[&atom.element],
                            q,
                            reference.lattice.volume,
                        )?;
                        let phase = -TAU
                            * index
                                .iter()
                                .zip(atom.fractional)
                                .map(|(&n, f)| n as f64 * f)
                                .sum::<f64>();
                        let potential_g = radial * Complex64::from_polar(1.0, phase);
                        let product = density_g[flat] * potential_g.conj();
                        for axis in 0..3 {
                            forces[atom_index][axis] +=
                                reference.lattice.volume * g[axis] * product.im;
                        }
                    }
                }
            }
        }
        ensure!(
            forces
                .iter()
                .flatten()
                .all(|component| component.is_finite()),
            "local ionic force contains a non-finite value"
        );
        Ok(forces)
    }

    /// Hellmann-Feynman force from separable nonlocal projectors for one
    /// k-point, before applying its Brillouin-zone weight.
    pub fn nonlocal_ionic_forces_ry_bohr(
        &self,
        orbitals: &[Vec<Complex64>],
        occupations: &[f64],
    ) -> Result<Vec<Vec3>> {
        ensure!(
            orbitals.len() == occupations.len() && !orbitals.is_empty(),
            "nonlocal force orbital and occupation counts differ"
        );
        ensure!(
            orbitals.iter().all(|orbital| {
                orbital.len() == self.dimension()
                    && orbital
                        .iter()
                        .all(|value| value.re.is_finite() && value.im.is_finite())
            }) && occupations
                .iter()
                .all(|value| value.is_finite() && (0.0..=2.0).contains(value)),
            "nonlocal force inputs are invalid"
        );
        let reference = &self.dense_reference;
        let k_cartesian = reference.lattice.cartesian_k(self.k_reduced);
        let momenta: Vec<_> = reference
            .g_vectors
            .iter()
            .map(|g_vector| {
                let mut value = reference.lattice.cartesian_g(g_vector.index);
                for axis in 0..3 {
                    value[axis] += k_cartesian[axis];
                }
                value
            })
            .collect();
        let mut forces = vec![[0.0; 3]; reference.atoms.len()];
        for (orbital, &occupation) in orbitals.iter().zip(occupations) {
            if occupation == 0.0 {
                continue;
            }
            for (atom_index, atom) in self.atom_projectors.iter().enumerate() {
                let overlaps: Vec<Complex64> = atom
                    .columns
                    .iter()
                    .map(|beta| {
                        beta.iter()
                            .zip(orbital)
                            .map(|(coefficient, psi)| coefficient.conj() * psi)
                            .sum()
                    })
                    .collect();
                let weighted: Vec<Complex64> = (0..atom.columns.len())
                    .map(|left| {
                        overlaps
                            .iter()
                            .enumerate()
                            .filter(|(right, _)| atom.component[left] == atom.component[*right])
                            .map(|(right, &overlap)| {
                                atom.dij_ry[atom.radial_projector[left]]
                                    [atom.radial_projector[right]]
                                    * overlap
                            })
                            .sum()
                    })
                    .collect();
                for axis in 0..3 {
                    let derivative_overlaps: Vec<Complex64> = atom
                        .columns
                        .iter()
                        .map(|beta| {
                            beta.iter()
                                .zip(orbital)
                                .zip(&momenta)
                                .map(|((coefficient, psi), momentum)| {
                                    Complex64::new(0.0, momentum[axis]) * coefficient.conj() * psi
                                })
                                .sum()
                        })
                        .collect();
                    let derivative_energy = 2.0
                        * derivative_overlaps
                            .iter()
                            .zip(&weighted)
                            .map(|(derivative, value)| (derivative.conj() * value).re)
                            .sum::<f64>();
                    forces[atom_index][axis] -= occupation * derivative_energy;
                }
            }
        }
        ensure!(
            forces
                .iter()
                .flatten()
                .all(|component| component.is_finite()),
            "nonlocal ionic force contains a non-finite value"
        );
        Ok(forces)
    }

    fn add_nonlocal(&self, wavefunction: &[Complex64], result: &mut [Complex64]) {
        let n = self.dimension();
        for atom in &self.atom_projectors {
            let overlaps: Vec<Complex64> = atom
                .columns
                .iter()
                .map(|beta| {
                    beta.iter()
                        .zip(wavefunction)
                        .map(|(b, psi)| b.conj() * psi)
                        .sum()
                })
                .collect();
            for (left, beta) in atom.columns.iter().enumerate() {
                let mut weighted = Complex64::default();
                for (right, &overlap) in overlaps.iter().enumerate() {
                    if atom.component[left] == atom.component[right] {
                        weighted += atom.dij_ry[atom.radial_projector[left]]
                            [atom.radial_projector[right]]
                            * overlap;
                    }
                }
                for row in 0..n {
                    result[row] += beta[row] * weighted;
                }
            }
        }
    }

    fn build_local_dense_matrix(&self) -> Result<Vec<Complex64>> {
        let n = self.dimension();
        let reference = &self.dense_reference;
        let mut matrix = vec![Complex64::default(); n * n];
        for row in 0..n {
            for column in 0..n {
                let delta_index = [
                    reference.g_vectors[row].index[0] - reference.g_vectors[column].index[0],
                    reference.g_vectors[row].index[1] - reference.g_vectors[column].index[1],
                    reference.g_vectors[row].index[2] - reference.g_vectors[column].index[2],
                ];
                let q = norm_squared(reference.lattice.cartesian_g(delta_index)).sqrt();
                let coefficient = reference.atoms.iter().try_fold(
                    Complex64::default(),
                    |sum, atom| -> Result<Complex64> {
                        let pseudo = &reference.pseudos[&atom.element];
                        let radial = local_form_factor_ry(pseudo, q, reference.lattice.volume)?;
                        let phase = -TAU
                            * delta_index
                                .iter()
                                .zip(atom.fractional)
                                .map(|(&n, f)| n as f64 * f)
                                .sum::<f64>();
                        Ok(sum + radial * Complex64::from_polar(1.0, phase))
                    },
                )?;
                matrix[row * n + column] = coefficient;
            }
        }
        Ok(matrix)
    }

    fn local_dense_matrix(&self) -> Result<&[Complex64]> {
        if self.local_ry.get().is_none() {
            let matrix = self.build_local_dense_matrix()?;
            let _ = self.local_ry.set(matrix);
        }
        Ok(self
            .local_ry
            .get()
            .expect("local dense matrix was initialized"))
    }

    fn build_nonlocal_dense_matrix(&self) -> Vec<Complex64> {
        let n = self.dimension();
        let mut matrix = vec![Complex64::default(); n * n];
        for column in 0..n {
            let mut basis = vec![Complex64::default(); n];
            basis[column] = Complex64::new(1.0, 0.0);
            let mut applied = vec![Complex64::default(); n];
            self.add_nonlocal(&basis, &mut applied);
            for row in 0..n {
                matrix[row * n + column] = applied[row];
            }
        }
        matrix
    }

    pub fn nonlocal_dense_matrix(&self) -> &[Complex64] {
        self.nonlocal_dense_ry
            .get_or_init(|| self.build_nonlocal_dense_matrix())
    }

    pub fn apply(&self, wavefunction: &[Complex64]) -> Result<Vec<Complex64>> {
        let n = self.dimension();
        ensure!(
            wavefunction.len() == n,
            "wavefunction has the wrong dimension"
        );
        let local_ry = self.local_dense_matrix()?;
        let mut result = vec![Complex64::default(); n];
        for row in 0..n {
            result[row] = self.kinetic_ry[row] * wavefunction[row];
            for (column, &coefficient) in wavefunction.iter().enumerate() {
                result[row] += local_ry[row * n + column] * coefficient;
            }
        }
        self.add_nonlocal(wavefunction, &mut result);
        Ok(result)
    }

    /// Apply only the kinetic and FFT-local terms. This is the exact boundary
    /// implemented by the first Metal Hamiltonian graph.
    pub fn apply_local_kinetic_fft(
        &self,
        wavefunction: &[Complex64],
        hartree_xc_real_ry: &[f64],
    ) -> Result<Vec<Complex64>> {
        let n = self.dimension();
        ensure!(
            wavefunction.len() == n,
            "wavefunction has the wrong dimension"
        );
        ensure!(
            hartree_xc_real_ry.len() == self.fft.len(),
            "real-space potential has the wrong grid size"
        );
        let mut grid = vec![Complex64::default(); self.fft.len()];
        for (basis, &grid_index) in self.basis_grid_indices.iter().enumerate() {
            grid[grid_index] = wavefunction[basis];
        }
        self.fft.inverse(&mut grid)?;
        for ((value, &ionic), &electronic) in grid
            .iter_mut()
            .zip(&self.ionic_local_real_ry)
            .zip(hartree_xc_real_ry)
        {
            *value *= ionic + electronic;
        }
        self.fft.forward(&mut grid)?;
        Ok(self
            .basis_grid_indices
            .iter()
            .enumerate()
            .map(|(basis, &grid_index)| {
                grid[grid_index] + self.kinetic_ry[basis] * wavefunction[basis]
            })
            .collect())
    }

    /// Apply the FFT local path plus kinetic and nonlocal terms. The supplied
    /// potential contains Hartree and XC only; the ionic local potential is
    /// cached internally.
    pub fn apply_fft(
        &self,
        wavefunction: &[Complex64],
        hartree_xc_real_ry: &[f64],
    ) -> Result<Vec<Complex64>> {
        let mut result = self.apply_local_kinetic_fft(wavefunction, hartree_xc_real_ry)?;
        self.add_nonlocal(wavefunction, &mut result);
        Ok(result)
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn apply_local_kinetic_metal(
        &self,
        wavefunctions: &[Vec<Complex64>],
        hartree_xc_real_ry: &[f64],
    ) -> Result<Vec<Vec<Complex64>>> {
        ensure!(
            !wavefunctions.is_empty(),
            "Metal Hamiltonian batch is empty"
        );
        ensure!(
            hartree_xc_real_ry.len() == self.fft.len(),
            "real-space potential has the wrong grid size"
        );
        let mut input = vec![Complex32::default(); self.fft.len() * wavefunctions.len()];
        for (state, wavefunction) in wavefunctions.iter().enumerate() {
            ensure!(
                wavefunction.len() == self.dimension(),
                "wavefunction has the wrong dimension"
            );
            let offset = state * self.fft.len();
            for (basis, &grid_index) in self.basis_grid_indices.iter().enumerate() {
                input[offset + grid_index] =
                    Complex32::new(wavefunction[basis].re as f32, wavefunction[basis].im as f32);
            }
        }
        let potential: Vec<_> = self
            .ionic_local_real_ry
            .iter()
            .zip(hartree_xc_real_ry)
            .map(|(&ionic, &electronic)| (ionic + electronic) as f32)
            .collect();
        let mut kinetic = vec![0.0_f32; self.fft.len()];
        for (basis, &grid_index) in self.basis_grid_indices.iter().enumerate() {
            kinetic[grid_index] = self.kinetic_ry[basis] as f32;
        }
        let output = crate::metal::apply_local_hamiltonian(
            &input,
            &potential,
            &kinetic,
            self.fft.shape(),
            wavefunctions.len(),
        )?;
        Ok((0..wavefunctions.len())
            .map(|state| {
                let offset = state * self.fft.len();
                self.basis_grid_indices
                    .iter()
                    .map(|&grid_index| {
                        let value = output[offset + grid_index];
                        Complex64::new(value.re as f64, value.im as f64)
                    })
                    .collect()
            })
            .collect())
    }

    /// Complete FP32 Metal Hamiltonian validation path. This deliberately
    /// transfers packed vectors between the local FFT graph and the nonlocal
    /// matrix graph, so it is not admitted as the production Metal backend.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn apply_hamiltonian_metal_validation(
        &self,
        wavefunctions: &[Vec<Complex64>],
        hartree_xc_real_ry: &[f64],
    ) -> Result<Vec<Vec<Complex64>>> {
        let mut output = self.apply_local_kinetic_metal(wavefunctions, hartree_xc_real_ry)?;
        let packed: Vec<_> = wavefunctions
            .iter()
            .flatten()
            .map(|value| Complex32::new(value.re as f32, value.im as f32))
            .collect();
        let nonlocal_matrix: Vec<_> = self
            .nonlocal_dense_matrix()
            .iter()
            .map(|value| Complex32::new(value.re as f32, value.im as f32))
            .collect();
        let nonlocal = crate::metal::apply_dense_operator(
            &packed,
            &nonlocal_matrix,
            self.dimension(),
            wavefunctions.len(),
        )?;
        for (state, state_output) in output.iter_mut().enumerate() {
            for (coefficient, nonlocal_value) in state_output
                .iter_mut()
                .zip(&nonlocal[state * self.dimension()..(state + 1) * self.dimension()])
            {
                *coefficient += Complex64::new(nonlocal_value.re as f64, nonlocal_value.im as f64);
            }
        }
        Ok(output)
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn create_metal_hamiltonian_context(
        &self,
        max_batch: usize,
    ) -> Result<crate::metal::HamiltonianContext> {
        let projector_count: usize = self
            .atom_projectors
            .iter()
            .map(|atom| atom.columns.len())
            .sum();
        ensure!(projector_count > 0, "Metal Hamiltonian requires projectors");
        let dimension = self.dimension();
        let mut projector_conjugate = vec![Complex32::default(); dimension * projector_count];
        let mut projector_transpose = vec![Complex32::default(); projector_count * dimension];
        let mut d_transpose = vec![Complex32::default(); projector_count * projector_count];
        let mut offset = 0;
        for atom in &self.atom_projectors {
            for (channel, beta) in atom.columns.iter().enumerate() {
                let global = offset + channel;
                for (basis, value) in beta.iter().enumerate() {
                    projector_conjugate[basis * projector_count + global] =
                        Complex32::new(value.re as f32, -value.im as f32);
                    projector_transpose[global * dimension + basis] =
                        Complex32::new(value.re as f32, value.im as f32);
                }
            }
            for left in 0..atom.columns.len() {
                for right in 0..atom.columns.len() {
                    if atom.component[left] == atom.component[right] {
                        let coupling = atom.dij_ry[atom.radial_projector[left]]
                            [atom.radial_projector[right]]
                            as f32;
                        d_transpose[(offset + right) * projector_count + offset + left] =
                            Complex32::new(coupling, 0.0);
                    }
                }
            }
            offset += atom.columns.len();
        }
        let mut kinetic = vec![Complex32::default(); self.fft.len()];
        for (basis, &grid_index) in self.basis_grid_indices.iter().enumerate() {
            kinetic[grid_index] = Complex32::new(self.kinetic_ry[basis] as f32, 0.0);
        }
        ensure!(
            self.fft.len() <= i32::MAX as usize,
            "Metal FFT grid is too large for MPSGraph gather indices"
        );
        let mut basis_indices = Vec::with_capacity(dimension * max_batch);
        for _ in 0..max_batch {
            basis_indices.extend(self.basis_grid_indices.iter().map(|&index| index as i32));
        }
        crate::metal::HamiltonianContext::new(
            self.fft.shape(),
            dimension,
            projector_count,
            max_batch,
            &kinetic,
            &basis_indices,
            &projector_conjugate,
            &d_transpose,
            &projector_transpose,
        )
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn initialize_metal_resident_davidson(
        &self,
        context: &mut crate::metal::HamiltonianContext,
        wavefunctions: &[Vec<Complex64>],
        hartree_xc_real_ry: &[f64],
    ) -> Result<()> {
        ensure!(
            !wavefunctions.is_empty(),
            "resident Metal Davidson initial basis is empty"
        );
        ensure!(
            hartree_xc_real_ry.len() == self.fft.len(),
            "real-space potential has the wrong grid size"
        );
        let mut grid_input = vec![Complex32::default(); self.fft.len() * wavefunctions.len()];
        let mut compact_input = Vec::with_capacity(self.dimension() * wavefunctions.len());
        for (state, wavefunction) in wavefunctions.iter().enumerate() {
            ensure!(
                wavefunction.len() == self.dimension(),
                "wavefunction has the wrong dimension"
            );
            let grid_offset = state * self.fft.len();
            for (basis, &grid_index) in self.basis_grid_indices.iter().enumerate() {
                let value = wavefunction[basis];
                let value = Complex32::new(value.re as f32, value.im as f32);
                grid_input[grid_offset + grid_index] = value;
                compact_input.push(value);
            }
        }
        let potential: Vec<_> = self
            .ionic_local_real_ry
            .iter()
            .zip(hartree_xc_real_ry)
            .map(|(&ionic, &electronic)| Complex32::new((ionic + electronic) as f32, 0.0))
            .collect();
        context.initialize_resident_davidson(
            &grid_input,
            &compact_input,
            &potential,
            wavefunctions.len(),
        )
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn apply_hamiltonian_metal_persistent(
        &self,
        context: &mut crate::metal::HamiltonianContext,
        wavefunctions: &[Vec<Complex64>],
        hartree_xc_real_ry: &[f64],
    ) -> Result<Vec<Vec<Complex64>>> {
        ensure!(
            !wavefunctions.is_empty(),
            "Metal Hamiltonian batch is empty"
        );
        ensure!(
            hartree_xc_real_ry.len() == self.fft.len(),
            "real-space potential has the wrong grid size"
        );
        let mut grid_input = vec![Complex32::default(); self.fft.len() * wavefunctions.len()];
        let mut compact_input = Vec::with_capacity(self.dimension() * wavefunctions.len());
        for (state, wavefunction) in wavefunctions.iter().enumerate() {
            ensure!(
                wavefunction.len() == self.dimension(),
                "wavefunction has the wrong dimension"
            );
            let grid_offset = state * self.fft.len();
            for (basis, &grid_index) in self.basis_grid_indices.iter().enumerate() {
                let value = wavefunction[basis];
                let value = Complex32::new(value.re as f32, value.im as f32);
                grid_input[grid_offset + grid_index] = value;
                compact_input.push(value);
            }
        }
        let potential: Vec<_> = self
            .ionic_local_real_ry
            .iter()
            .zip(hartree_xc_real_ry)
            .map(|(&ionic, &electronic)| Complex32::new((ionic + electronic) as f32, 0.0))
            .collect();
        let compact_output =
            context.apply(&grid_input, &compact_input, &potential, wavefunctions.len())?;
        Ok((0..wavefunctions.len())
            .map(|state| {
                let compact_offset = state * self.dimension();
                (0..self.dimension())
                    .map(|basis| {
                        let value = compact_output[compact_offset + basis];
                        Complex64::new(value.re as f64, value.im as f64)
                    })
                    .collect()
            })
            .collect())
    }

    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    pub fn apply_hamiltonian_metal_validation(
        &self,
        _wavefunctions: &[Vec<Complex64>],
        _hartree_xc_real_ry: &[f64],
    ) -> Result<Vec<Vec<Complex64>>> {
        anyhow::bail!("Metal Hamiltonian requires the macOS-only `metal` feature")
    }

    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    pub fn create_metal_hamiltonian_context(
        &self,
        _max_batch: usize,
    ) -> Result<crate::metal::HamiltonianContext> {
        anyhow::bail!("Metal Hamiltonian requires the macOS-only `metal` feature")
    }

    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    pub fn initialize_metal_resident_davidson(
        &self,
        _context: &mut crate::metal::HamiltonianContext,
        _wavefunctions: &[Vec<Complex64>],
        _hartree_xc_real_ry: &[f64],
    ) -> Result<()> {
        anyhow::bail!("Metal Hamiltonian requires the macOS-only `metal` feature")
    }

    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    pub fn apply_hamiltonian_metal_persistent(
        &self,
        _context: &mut crate::metal::HamiltonianContext,
        _wavefunctions: &[Vec<Complex64>],
        _hartree_xc_real_ry: &[f64],
    ) -> Result<Vec<Vec<Complex64>>> {
        anyhow::bail!("Metal Hamiltonian requires the macOS-only `metal` feature")
    }

    pub fn dense_matrix_fft(&self, hartree_xc_real_ry: &[f64]) -> Result<Vec<Complex64>> {
        let n = self.dimension();
        let mut matrix = vec![Complex64::default(); n * n];
        for column in 0..n {
            let mut basis = vec![Complex64::default(); n];
            basis[column] = Complex64::new(1.0, 0.0);
            let applied = self.apply_fft(&basis, hartree_xc_real_ry)?;
            for row in 0..n {
                matrix[row * n + column] = applied[row];
            }
        }
        Ok(matrix)
    }

    pub fn dense_matrix(&self) -> Result<Vec<Complex64>> {
        let n = self.dimension();
        let mut matrix = vec![Complex64::default(); n * n];
        for column in 0..n {
            let mut basis = vec![Complex64::default(); n];
            basis[column] = Complex64::new(1.0, 0.0);
            let applied = self.apply(&basis)?;
            for row in 0..n {
                matrix[row * n + column] = applied[row];
            }
        }
        Ok(matrix)
    }
}

#[derive(Debug, Clone)]
pub struct ElectronicPotential {
    pub real_ry: Vec<f64>,
    pub hartree_energy_ry: f64,
    pub xc_energy_ry: f64,
    pub integral_rho_vxc_ry: f64,
    pub electron_count: f64,
}

pub fn density_from_orbitals(
    fft: &Fft3d,
    g_vectors: &[GVector],
    orbitals: &[Vec<Complex64>],
    occupations: &[f64],
    volume: f64,
) -> Result<Vec<f64>> {
    let mut density = vec![0.0; fft.len()];
    accumulate_density_from_orbitals(
        fft,
        g_vectors,
        orbitals,
        occupations,
        volume,
        1.0,
        &mut density,
    )?;
    Ok(density)
}

pub fn accumulate_density_from_orbitals(
    fft: &Fft3d,
    g_vectors: &[GVector],
    orbitals: &[Vec<Complex64>],
    occupations: &[f64],
    volume: f64,
    k_point_weight: f64,
    density: &mut [f64],
) -> Result<()> {
    ensure!(
        orbitals.len() == occupations.len(),
        "occupation count mismatch"
    );
    ensure!(
        density.len() == fft.len(),
        "density accumulation grid mismatch"
    );
    ensure!(
        k_point_weight.is_finite() && k_point_weight > 0.0,
        "k-point density weight must be positive and finite"
    );
    let indices: Vec<_> = g_vectors
        .iter()
        .map(|g| flat_index_for_shape(g.index, fft.shape()))
        .collect();
    let normalization = (fft.len() as f64).powi(2) / volume;
    for (orbital, &occupation) in orbitals.iter().zip(occupations) {
        ensure!(
            occupation.is_finite() && (0.0..=2.0).contains(&occupation),
            "orbital occupation must be finite and in [0, 2]"
        );
        ensure!(
            orbital.len() == g_vectors.len(),
            "orbital dimension mismatch"
        );
        let mut grid = vec![Complex64::default(); fft.len()];
        for (coefficient, &grid_index) in orbital.iter().zip(&indices) {
            grid[grid_index] = *coefficient;
        }
        fft.inverse(&mut grid)?;
        for (rho, value) in density.iter_mut().zip(grid) {
            *rho += k_point_weight * occupation * normalization * value.norm_sqr();
        }
    }
    Ok(())
}

pub fn atomic_starting_density(
    fft: &Fft3d,
    lattice: &Lattice,
    atoms: &[AtomInput],
    pseudos: &BTreeMap<String, NormConservingPseudo>,
    wavefunction_cutoff_ry: f64,
    electron_count: f64,
) -> Result<Vec<f64>> {
    ensure!(
        atoms.iter().all(|atom| pseudos.contains_key(&atom.element)),
        "an atom has no pseudopotential for starting-density construction"
    );
    let shape = fft.shape();
    let mut coefficients = vec![Complex64::default(); fft.len()];
    for x in 0..shape[0] {
        for y in 0..shape[1] {
            for z in 0..shape[2] {
                let index = [
                    signed_frequency(x, shape[0]),
                    signed_frequency(y, shape[1]),
                    signed_frequency(z, shape[2]),
                ];
                let q2 = norm_squared(lattice.cartesian_g(index));
                if q2 > 4.0 * wavefunction_cutoff_ry * (1.0 + 32.0 * f64::EPSILON) {
                    continue;
                }
                let q = q2.sqrt();
                let coefficient = atoms.iter().try_fold(
                    Complex64::default(),
                    |sum, atom| -> Result<Complex64> {
                        let pseudo = &pseudos[&atom.element];
                        let radial = atomic_density_form_factor(pseudo, q, lattice.volume)?;
                        let phase = -TAU
                            * index
                                .iter()
                                .zip(atom.fractional)
                                .map(|(&n, f)| n as f64 * f)
                                .sum::<f64>();
                        Ok(sum + radial * Complex64::from_polar(1.0, phase))
                    },
                )?;
                coefficients[(x * shape[1] + y) * shape[2] + z] = coefficient;
            }
        }
    }
    enforce_hermitian_grid(&mut coefficients, shape)?;
    fft.inverse(&mut coefficients)?;
    let grid_points = fft.len() as f64;
    let mut density: Vec<f64> = coefficients
        .iter()
        .map(|value| value.re * grid_points)
        .collect();
    let integrated = density.iter().sum::<f64>() * lattice.volume / grid_points;
    ensure!(
        integrated > 0.0,
        "atomic starting density has nonpositive charge"
    );
    let renormalization = electron_count / integrated;
    for value in &mut density {
        *value = (*value * renormalization).max(0.0);
    }
    let clipped_integrated = density.iter().sum::<f64>() * lattice.volume / grid_points;
    for value in &mut density {
        *value *= electron_count / clipped_integrated;
    }
    Ok(density)
}

/// Periodic point-ion Ewald energy in Rydberg. `ionic_charges` is aligned with
/// `atoms`; `eta` is the Gaussian splitting parameter in inverse bohr.
pub fn ewald_energy_ry(
    lattice: &Lattice,
    atoms: &[AtomInput],
    ionic_charges: &[f64],
    eta: f64,
) -> Result<f64> {
    ensure!(eta.is_finite() && eta > 0.0, "Ewald eta must be positive");
    ensure!(!atoms.is_empty(), "Ewald sum requires atoms");
    ensure!(
        ionic_charges.len() == atoms.len()
            && ionic_charges
                .iter()
                .all(|charge| charge.is_finite() && *charge > 0.0),
        "Ewald ionic charges are invalid"
    );
    let positions: Vec<_> = atoms
        .iter()
        .map(|atom| fractional_to_cartesian(lattice, atom.fractional))
        .collect();
    let shortest_direct = lattice
        .direct
        .iter()
        .map(|&vector| norm_squared(vector).sqrt())
        .fold(f64::INFINITY, f64::min);
    let real_cutoff = 8.0 / eta;
    let direct_bound = (real_cutoff / shortest_direct).ceil() as i32 + 2;
    let mut real_sum = 0.0;
    for (i, (&ri, &charge_i)) in positions.iter().zip(ionic_charges).enumerate() {
        for (j, (&rj, &charge_j)) in positions.iter().zip(ionic_charges).enumerate() {
            for a in -direct_bound..=direct_bound {
                for b in -direct_bound..=direct_bound {
                    for c in -direct_bound..=direct_bound {
                        if i == j && a == 0 && b == 0 && c == 0 {
                            continue;
                        }
                        let translation = add(
                            add(
                                scale(lattice.direct[0], a as f64),
                                scale(lattice.direct[1], b as f64),
                            ),
                            scale(lattice.direct[2], c as f64),
                        );
                        let distance = norm_squared(add(subtract(ri, rj), translation)).sqrt();
                        if distance <= real_cutoff {
                            real_sum += charge_i * charge_j * libm::erfc(eta * distance) / distance;
                        }
                    }
                }
            }
        }
    }
    real_sum *= 0.5;

    let reciprocal_cutoff = 2.0 * eta * (-1.0e-14_f64.ln()).sqrt();
    let shortest_reciprocal = lattice
        .reciprocal
        .iter()
        .map(|&vector| norm_squared(vector).sqrt())
        .fold(f64::INFINITY, f64::min);
    let reciprocal_bound = (reciprocal_cutoff / shortest_reciprocal).ceil() as i32 + 2;
    let mut reciprocal_sum = 0.0;
    for a in -reciprocal_bound..=reciprocal_bound {
        for b in -reciprocal_bound..=reciprocal_bound {
            for c in -reciprocal_bound..=reciprocal_bound {
                if a == 0 && b == 0 && c == 0 {
                    continue;
                }
                let g = lattice.cartesian_g([a, b, c]);
                let g2 = norm_squared(g);
                if g2.sqrt() > reciprocal_cutoff {
                    continue;
                }
                let structure: Complex64 = positions
                    .iter()
                    .zip(ionic_charges)
                    .map(|(&position, &charge)| {
                        let phase = -(g[0] * position[0] + g[1] * position[1] + g[2] * position[2]);
                        charge * Complex64::from_polar(1.0, phase)
                    })
                    .sum();
                reciprocal_sum += (-g2 / (4.0 * eta * eta)).exp() * structure.norm_sqr() / g2;
            }
        }
    }
    reciprocal_sum *= 2.0 * PI / lattice.volume;
    let total_charge = ionic_charges.iter().sum::<f64>();
    let self_energy = eta / PI.sqrt()
        * ionic_charges
            .iter()
            .map(|charge| charge * charge)
            .sum::<f64>();
    let background = PI * total_charge * total_charge / (2.0 * eta * eta * lattice.volume);
    Ok(2.0 * (real_sum + reciprocal_sum - self_energy - background))
}

/// Analytic ion-ion Ewald forces in Rydberg per bohr.
///
/// The returned Cartesian vectors are the negative derivatives of
/// [`ewald_energy_ry`] with respect to each ionic position. The same real and
/// reciprocal cutoffs are used so finite-difference checks compare an exactly
/// matched summation convention.
pub fn ewald_forces_ry_bohr(
    lattice: &Lattice,
    atoms: &[AtomInput],
    ionic_charges: &[f64],
    eta: f64,
) -> Result<Vec<Vec3>> {
    ensure!(eta.is_finite() && eta > 0.0, "Ewald eta must be positive");
    ensure!(!atoms.is_empty(), "Ewald sum requires atoms");
    ensure!(
        ionic_charges.len() == atoms.len()
            && ionic_charges
                .iter()
                .all(|charge| charge.is_finite() && *charge > 0.0),
        "Ewald ionic charges are invalid"
    );
    let positions: Vec<_> = atoms
        .iter()
        .map(|atom| fractional_to_cartesian(lattice, atom.fractional))
        .collect();
    let mut forces = vec![[0.0; 3]; atoms.len()];

    let shortest_direct = lattice
        .direct
        .iter()
        .map(|&vector| norm_squared(vector).sqrt())
        .fold(f64::INFINITY, f64::min);
    let real_cutoff = 8.0 / eta;
    let direct_bound = (real_cutoff / shortest_direct).ceil() as i32 + 2;
    for (i, (&ri, &charge_i)) in positions.iter().zip(ionic_charges).enumerate() {
        for (j, (&rj, &charge_j)) in positions.iter().zip(ionic_charges).enumerate() {
            for a in -direct_bound..=direct_bound {
                for b in -direct_bound..=direct_bound {
                    for c in -direct_bound..=direct_bound {
                        if i == j && a == 0 && b == 0 && c == 0 {
                            continue;
                        }
                        let translation = add(
                            add(
                                scale(lattice.direct[0], a as f64),
                                scale(lattice.direct[1], b as f64),
                            ),
                            scale(lattice.direct[2], c as f64),
                        );
                        let displacement = add(subtract(ri, rj), translation);
                        let distance_squared = norm_squared(displacement);
                        let distance = distance_squared.sqrt();
                        if distance <= real_cutoff {
                            let coefficient = 2.0
                                * charge_i
                                * charge_j
                                * (libm::erfc(eta * distance) / (distance_squared * distance)
                                    + 2.0 * eta / PI.sqrt()
                                        * (-eta * eta * distance_squared).exp()
                                        / distance_squared);
                            for axis in 0..3 {
                                forces[i][axis] += coefficient * displacement[axis];
                            }
                        }
                    }
                }
            }
        }
    }

    let reciprocal_cutoff = 2.0 * eta * (-1.0e-14_f64.ln()).sqrt();
    let shortest_reciprocal = lattice
        .reciprocal
        .iter()
        .map(|&vector| norm_squared(vector).sqrt())
        .fold(f64::INFINITY, f64::min);
    let reciprocal_bound = (reciprocal_cutoff / shortest_reciprocal).ceil() as i32 + 2;
    for a in -reciprocal_bound..=reciprocal_bound {
        for b in -reciprocal_bound..=reciprocal_bound {
            for c in -reciprocal_bound..=reciprocal_bound {
                if a == 0 && b == 0 && c == 0 {
                    continue;
                }
                let g = lattice.cartesian_g([a, b, c]);
                let g2 = norm_squared(g);
                if g2.sqrt() > reciprocal_cutoff {
                    continue;
                }
                let structure: Complex64 = positions
                    .iter()
                    .zip(ionic_charges)
                    .map(|(&position, &charge)| {
                        let phase = -(g[0] * position[0] + g[1] * position[1] + g[2] * position[2]);
                        charge * Complex64::from_polar(1.0, phase)
                    })
                    .sum();
                let reciprocal_prefactor =
                    -8.0 * PI / lattice.volume * (-g2 / (4.0 * eta * eta)).exp() / g2;
                for (i, (&position, &charge)) in positions.iter().zip(ionic_charges).enumerate() {
                    let phase = -(g[0] * position[0] + g[1] * position[1] + g[2] * position[2]);
                    let phase_factor = Complex64::from_polar(1.0, phase);
                    let coefficient =
                        reciprocal_prefactor * charge * (structure.conj() * phase_factor).im;
                    for axis in 0..3 {
                        forces[i][axis] += coefficient * g[axis];
                    }
                }
            }
        }
    }
    ensure!(
        forces
            .iter()
            .flatten()
            .all(|component| component.is_finite()),
        "Ewald force contains a non-finite value"
    );
    Ok(forces)
}

pub fn hartree_xc_potential(
    fft: &Fft3d,
    lattice: &Lattice,
    density: &[f64],
    functional: XcFunctional,
) -> Result<ElectronicPotential> {
    ensure!(
        density.len() == fft.len(),
        "density has the wrong grid size"
    );
    ensure!(
        density
            .iter()
            .all(|value| value.is_finite() && *value >= -1.0e-12),
        "density is non-finite or materially negative"
    );
    let grid_points = fft.len() as f64;
    let cell_weight = lattice.volume / grid_points;
    let electron_count = density.iter().sum::<f64>() * cell_weight;
    let mut rho_g: Vec<_> = density
        .iter()
        .map(|&value| Complex64::new(value, 0.0))
        .collect();
    fft.forward(&mut rho_g)?;
    for value in &mut rho_g {
        *value /= grid_points;
    }
    let shape = fft.shape();
    let mut vh_g = vec![Complex64::default(); fft.len()];
    let mut hartree_energy_ry = 0.0;
    for x in 0..shape[0] {
        for y in 0..shape[1] {
            for z in 0..shape[2] {
                let flat = (x * shape[1] + y) * shape[2] + z;
                let index = [
                    signed_frequency(x, shape[0]),
                    signed_frequency(y, shape[1]),
                    signed_frequency(z, shape[2]),
                ];
                let g2 = norm_squared(lattice.cartesian_g(index));
                if g2 > 1.0e-14 {
                    vh_g[flat] = (8.0 * PI / g2) * rho_g[flat];
                    hartree_energy_ry +=
                        0.5 * lattice.volume * (rho_g[flat].conj() * vh_g[flat]).re;
                }
            }
        }
    }
    fft.inverse(&mut vh_g)?;
    let xc = functional.evaluate_unpolarized(fft, lattice, density)?;
    let mut real_ry = Vec::with_capacity(fft.len());
    for (vh, v_xc_ry) in vh_g.into_iter().zip(&xc.potential_ry) {
        real_ry.push(vh.re * grid_points + v_xc_ry);
    }
    Ok(ElectronicPotential {
        real_ry,
        hartree_energy_ry,
        xc_energy_ry: xc.energy_ry,
        integral_rho_vxc_ry: xc.integral_rho_vxc_ry,
        electron_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(all(feature = "metal", target_os = "macos"))]
    use crate::eigensolver::block_davidson_batched_with_reductions;
    use crate::eigensolver::{DavidsonOptions, block_davidson, projected_hermitian_eigensolve};
    use crate::input::SimulationInput;
    use std::path::Path;

    #[test]
    fn special_functions_have_correct_limits() {
        assert_eq!(libm::erf(0.0), 0.0);
        assert!((libm::erf(1.0) - 0.842_700_792_949_714_9).abs() < 1.0e-15);
        assert!((spherical_bessel(0, 0.0).unwrap() - 1.0).abs() < 1.0e-15);
        assert!(spherical_bessel(1, 0.0).unwrap().abs() < 1.0e-15);
    }

    #[test]
    fn weighted_simpson_integrates_quadratic() {
        let values: Vec<_> = (0..=10).map(|i| (i as f64 / 10.0).powi(2)).collect();
        let weights = vec![0.1; 11];
        assert!((weighted_simpson(&values, &weights).unwrap() - 1.0 / 3.0).abs() < 1.0e-14);
    }

    #[test]
    fn silicon_reference_hamiltonian_is_hermitian_when_available() {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let _device_guard = crate::metal::test_device_lock();
        let input_path = Path::new("examples/si2.toml");
        if !input_path.exists() {
            return;
        }
        let input = SimulationInput::from_path(input_path).unwrap();
        if !input.pseudopotentials["Si"].exists() {
            return;
        }
        let pseudo = NormConservingPseudo::from_upf(&input.pseudopotentials["Si"]).unwrap();
        let valence_charge = pseudo.valence_charge;
        let pseudos = BTreeMap::from([("Si".to_owned(), pseudo)]);
        let lattice = Lattice::new(input.cell.lattice_bohr).unwrap();
        let g_vectors = lattice.enumerate_g_vectors(input.cutoff_ry);
        let hamiltonian = CpuHamiltonian::new(
            &lattice,
            &g_vectors,
            &input.atoms,
            &pseudos,
            input.cutoff_ry,
        )
        .unwrap();
        let matrix = hamiltonian.dense_matrix().unwrap();
        let scale = matrix.iter().map(|z| z.norm()).fold(0.0_f64, f64::max);
        let mut maximum_error = 0.0_f64;
        for i in 0..hamiltonian.dimension() {
            for j in 0..hamiltonian.dimension() {
                maximum_error = maximum_error.max(
                    (matrix[i * hamiltonian.dimension() + j]
                        - matrix[j * hamiltonian.dimension() + i].conj())
                    .norm(),
                );
            }
        }
        assert!(
            maximum_error / scale < 1.0e-12,
            "relative Hermiticity error {maximum_error:e}"
        );

        let wavefunction: Vec<_> = (0..hamiltonian.dimension())
            .map(|i| Complex64::new((i as f64 * 0.37).sin(), (i as f64 * 0.19).cos()))
            .collect();
        let direct = hamiltonian.apply(&wavefunction).unwrap();
        let fft = hamiltonian
            .apply_fft(&wavefunction, &vec![0.0; hamiltonian.fft.len()])
            .unwrap();
        let reference_norm = direct.iter().map(Complex64::norm_sqr).sum::<f64>();
        let difference = direct
            .iter()
            .zip(&fft)
            .map(|(a, b)| (*a - *b).norm_sqr())
            .sum::<f64>();
        assert!(
            (difference / reference_norm).sqrt() < 1.0e-12,
            "direct and FFT local operators disagree"
        );

        let starting_density = atomic_starting_density(
            hamiltonian.fft(),
            &lattice,
            &input.atoms,
            &pseudos,
            input.cutoff_ry,
            8.0,
        )
        .unwrap();
        let integrated =
            starting_density.iter().sum::<f64>() * lattice.volume / starting_density.len() as f64;
        assert!((integrated - 8.0).abs() < 1.0e-12);

        let electronic = hartree_xc_potential(
            hamiltonian.fft(),
            &lattice,
            &starting_density,
            XcFunctional::LdaPz,
        )
        .unwrap();
        let dense_matrix = hamiltonian.dense_matrix_fft(&electronic.real_ry).unwrap();
        let dense_eigenpairs =
            projected_hermitian_eigensolve(&dense_matrix, hamiltonian.dimension()).unwrap();
        let davidson = block_davidson(
            hamiltonian.kinetic_ry(),
            None,
            &DavidsonOptions {
                roots: 4,
                residual_tolerance: 1.0e-10,
                max_iterations: 80,
                max_subspace: 32,
            },
            |orbital| hamiltonian.apply_fft(orbital, &electronic.real_ry),
        )
        .unwrap();
        assert!(davidson.converged);
        for (davidson_value, dense_value) in davidson
            .eigenvalues
            .iter()
            .zip(&dense_eigenpairs.eigenvalues)
        {
            assert!((davidson_value - dense_value).abs() < 1.0e-10);
        }

        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            let mut wavefunctions: Vec<Vec<Complex64>> = Vec::new();
            for state in 0..3 {
                let mut values: Vec<_> = (0..hamiltonian.dimension())
                    .map(|i| {
                        Complex64::new(
                            ((i + 3 * state) as f64 * 0.17).sin(),
                            ((2 * i + state) as f64 * 0.09).cos(),
                        )
                    })
                    .collect();
                for _ in 0..2 {
                    for basis in &wavefunctions {
                        let overlap: Complex64 = basis
                            .iter()
                            .zip(&values)
                            .map(|(left, right)| left.conj() * right)
                            .sum();
                        for (value, &basis_value) in values.iter_mut().zip(basis) {
                            *value -= overlap * basis_value;
                        }
                    }
                }
                let norm = values.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
                for value in &mut values {
                    *value /= norm;
                }
                wavefunctions.push(values);
            }
            let cpu: Vec<_> = wavefunctions
                .iter()
                .map(|orbital| {
                    hamiltonian
                        .apply_local_kinetic_fft(orbital, &electronic.real_ry)
                        .unwrap()
                })
                .collect();
            let metal = hamiltonian
                .apply_local_kinetic_metal(&wavefunctions, &electronic.real_ry)
                .unwrap();
            let reference_norm = cpu.iter().flatten().map(Complex64::norm_sqr).sum::<f64>();
            let error = cpu
                .iter()
                .flatten()
                .zip(metal.iter().flatten())
                .map(|(a, b)| (*a - *b).norm_sqr())
                .sum::<f64>();
            let relative_l2 = (error / reference_norm).sqrt();
            eprintln!("silicon Metal local+kinetic fixed-density error: {relative_l2:e}");
            assert!(relative_l2 < 1.0e-5);

            let cpu_full: Vec<_> = wavefunctions
                .iter()
                .map(|orbital| hamiltonian.apply_fft(orbital, &electronic.real_ry).unwrap())
                .collect();
            let metal_full = hamiltonian
                .apply_hamiltonian_metal_validation(&wavefunctions, &electronic.real_ry)
                .unwrap();
            let full_reference_norm = cpu_full
                .iter()
                .flatten()
                .map(Complex64::norm_sqr)
                .sum::<f64>();
            let full_error = cpu_full
                .iter()
                .flatten()
                .zip(metal_full.iter().flatten())
                .map(|(a, b)| (*a - *b).norm_sqr())
                .sum::<f64>();
            let full_relative_l2 = (full_error / full_reference_norm).sqrt();
            eprintln!("silicon complete Metal fixed-density error: {full_relative_l2:e}");
            assert!(full_relative_l2 < 1.0e-5);

            let mut persistent_context = hamiltonian
                .create_metal_hamiltonian_context(32.min(hamiltonian.dimension()))
                .unwrap();
            assert!(persistent_context.projector_count() > 0);
            persistent_context.begin_resident_davidson().unwrap();
            let persistent_full = hamiltonian
                .apply_hamiltonian_metal_persistent(
                    &mut persistent_context,
                    &wavefunctions,
                    &electronic.real_ry,
                )
                .unwrap();
            let persistent_error = cpu_full
                .iter()
                .flatten()
                .zip(persistent_full.iter().flatten())
                .map(|(a, b)| (*a - *b).norm_sqr())
                .sum::<f64>();
            let persistent_relative_l2 = (persistent_error / full_reference_norm).sqrt();
            eprintln!(
                "silicon persistent compact-projector Metal error: {persistent_relative_l2:e}"
            );
            assert!(persistent_relative_l2 < 1.0e-5);

            let resident_projection = persistent_context
                .resident_cross_gram(&wavefunctions, &persistent_full)
                .unwrap();
            let mut maximum_resident_projection_error = 0.0_f64;
            for left in 0..wavefunctions.len() {
                for right in 0..wavefunctions.len() {
                    let cpu_projection: Complex64 = wavefunctions[left]
                        .iter()
                        .zip(&persistent_full[right])
                        .map(|(a, b)| a.conj() * b)
                        .sum();
                    maximum_resident_projection_error = maximum_resident_projection_error.max(
                        (resident_projection[left * wavefunctions.len() + right] - cpu_projection)
                            .norm(),
                    );
                }
            }
            eprintln!(
                "silicon resident Metal projection maximum absolute error: {maximum_resident_projection_error:e}"
            );
            assert!(maximum_resident_projection_error < 1.0e-5);
            assert_eq!(persistent_context.resident_projection_calls(), 1);
            assert_eq!(persistent_context.resident_projection_resyncs(), 0);

            let ritz_coefficients: Vec<Vec<Complex64>> = (0..2)
                .map(|root| {
                    (0..wavefunctions.len())
                        .map(|state| {
                            Complex64::new(
                                0.17 * (root + state + 1) as f64,
                                0.03 * (root as f64 - state as f64),
                            )
                        })
                        .collect()
                })
                .collect();
            // Root zero intentionally exercises the signed 0.25 Ry
            // denominator floor at the G=0 kinetic entry.
            let ritz_eigenvalues = [0.10, 0.42];
            persistent_context
                .form_resident_ritz_residuals(&ritz_coefficients, &ritz_eigenvalues)
                .unwrap();
            let resident_norms = persistent_context.resident_residual_norms(2).unwrap();
            assert_eq!(
                persistent_context.resident_coefficient_upload_bytes(),
                (2 * wavefunctions.len() + 2) * std::mem::size_of::<Complex32>()
            );
            assert_eq!(persistent_context.resident_vector_download_bytes(), 0);
            let resident_vectors = persistent_context.read_resident_ritz_residuals(2).unwrap();
            let combine = |source: &[Vec<Complex64>], coefficients: &[Complex64]| {
                let mut output = vec![Complex64::default(); hamiltonian.dimension()];
                for (vector, coefficient) in source.iter().zip(coefficients) {
                    for (value, basis_value) in output.iter_mut().zip(vector) {
                        *value += *coefficient * *basis_value;
                    }
                }
                output
            };
            let relative_l2 = |actual: &[Complex64], expected: &[Complex64]| {
                let error = actual
                    .iter()
                    .zip(expected)
                    .map(|(a, b)| (*a - *b).norm_sqr())
                    .sum::<f64>();
                let reference = expected.iter().map(Complex64::norm_sqr).sum::<f64>();
                (error / reference.max(1.0e-30)).sqrt()
            };
            let mut maximum_ritz_error = 0.0_f64;
            let mut maximum_applied_ritz_error = 0.0_f64;
            let mut maximum_residual_error = 0.0_f64;
            let mut maximum_resident_norm_error = 0.0_f64;
            for root in 0..2 {
                let expected_ritz = combine(&wavefunctions, &ritz_coefficients[root]);
                let expected_applied = combine(&persistent_full, &ritz_coefficients[root]);
                let expected_residual: Vec<_> = expected_applied
                    .iter()
                    .zip(&expected_ritz)
                    .map(|(applied, vector)| *applied - ritz_eigenvalues[root] * *vector)
                    .collect();
                maximum_ritz_error = maximum_ritz_error
                    .max(relative_l2(&resident_vectors.ritz[root], &expected_ritz));
                maximum_applied_ritz_error = maximum_applied_ritz_error.max(relative_l2(
                    &resident_vectors.applied_ritz[root],
                    &expected_applied,
                ));
                maximum_residual_error = maximum_residual_error.max(relative_l2(
                    &resident_vectors.residuals[root],
                    &expected_residual,
                ));
                let expected_norm = expected_residual
                    .iter()
                    .map(Complex64::norm_sqr)
                    .sum::<f64>()
                    .sqrt();
                maximum_resident_norm_error = maximum_resident_norm_error
                    .max((resident_norms[root] - expected_norm).abs() / expected_norm.max(1.0e-30));
            }
            eprintln!(
                "silicon resident Ritz/applied/residual/norm relative errors: {maximum_ritz_error:e} / {maximum_applied_ritz_error:e} / {maximum_residual_error:e} / {maximum_resident_norm_error:e}"
            );
            assert!(maximum_ritz_error < 2.0e-5);
            assert!(maximum_applied_ritz_error < 2.0e-5);
            assert!(maximum_residual_error < 2.0e-5);
            assert!(maximum_resident_norm_error < 2.0e-5);
            assert_eq!(persistent_context.resident_ritz_updates(), 1);

            persistent_context
                .precondition_resident_residuals(&[true, false])
                .unwrap();
            let resident_corrections = persistent_context.read_resident_corrections(2).unwrap();
            let expected_preconditioned: Vec<Vec<Complex64>> = (0..2)
                .map(|root| {
                    resident_vectors.residuals[root]
                        .iter()
                        .zip(hamiltonian.kinetic_ry())
                        .map(|(&residual, &kinetic)| {
                            let denominator = kinetic - ritz_eigenvalues[root];
                            let safe_denominator = if denominator.abs() < 0.25 {
                                0.25_f64.copysign(denominator)
                            } else {
                                denominator
                            };
                            -residual / safe_denominator
                        })
                        .collect()
                })
                .collect();
            let correction_error =
                relative_l2(&resident_corrections[0], &expected_preconditioned[0]);
            let masked_correction_norm = resident_corrections[1]
                .iter()
                .map(Complex64::norm_sqr)
                .sum::<f64>()
                .sqrt();
            eprintln!(
                "silicon resident preconditioner relative error / masked-root norm: {correction_error:e} / {masked_correction_norm:e}"
            );
            assert!(correction_error < 2.0e-5);
            assert_eq!(masked_correction_norm, 0.0);
            assert_eq!(persistent_context.resident_preconditioner_updates(), 1);
            assert!(
                persistent_context
                    .orthonormalize_resident_corrections(&[false, false])
                    .is_err()
            );
            assert!(
                persistent_context
                    .orthonormalize_resident_corrections(&[true])
                    .is_err()
            );
            assert_eq!(
                persistent_context.resident_coefficient_upload_bytes(),
                (2 * wavefunctions.len() + 2) * std::mem::size_of::<Complex32>()
                    + 2 * std::mem::size_of::<f32>()
            );
            assert_eq!(
                persistent_context.resident_vector_download_bytes(),
                4 * 2 * hamiltonian.dimension() * std::mem::size_of::<Complex32>()
            );

            let vector_downloads_before_orthogonalization =
                persistent_context.resident_vector_download_bytes();
            let full_uploads_before_orthogonalization =
                persistent_context.full_vector_upload_bytes();
            let full_downloads_before_orthogonalization =
                persistent_context.full_vector_download_bytes();
            persistent_context
                .precondition_resident_residuals(&[true, true])
                .unwrap();
            let accepted_count = persistent_context
                .orthonormalize_resident_corrections(&[true, true])
                .unwrap();
            assert_eq!(accepted_count, 2);
            assert_eq!(
                persistent_context.resident_vector_download_bytes(),
                vector_downloads_before_orthogonalization
            );
            assert_eq!(
                persistent_context.full_vector_upload_bytes(),
                full_uploads_before_orthogonalization
            );
            assert_eq!(
                persistent_context.full_vector_download_bytes(),
                full_downloads_before_orthogonalization
            );
            let accepted_corrections = persistent_context
                .read_resident_accepted_corrections(accepted_count)
                .unwrap();
            let mut cpu_accepted: Vec<Vec<Complex64>> = Vec::new();
            for raw in &expected_preconditioned {
                let mut candidate = raw.clone();
                for _ in 0..2 {
                    for vector in wavefunctions.iter().chain(&cpu_accepted) {
                        let overlap: Complex64 = vector
                            .iter()
                            .zip(&candidate)
                            .map(|(left, right)| left.conj() * right)
                            .sum();
                        for (value, &basis_value) in candidate.iter_mut().zip(vector) {
                            *value -= overlap * basis_value;
                        }
                    }
                }
                let norm = candidate
                    .iter()
                    .map(Complex64::norm_sqr)
                    .sum::<f64>()
                    .sqrt();
                assert!(norm > 1.0e-11);
                for value in &mut candidate {
                    *value /= norm;
                }
                cpu_accepted.push(candidate);
            }
            let maximum_correction_error = accepted_corrections
                .iter()
                .zip(&cpu_accepted)
                .map(|(actual, expected)| relative_l2(actual, expected))
                .fold(0.0_f64, f64::max);
            let combined_basis: Vec<&Vec<Complex64>> =
                wavefunctions.iter().chain(&accepted_corrections).collect();
            let mut maximum_orthonormality_error = 0.0_f64;
            for left in 0..combined_basis.len() {
                for right in 0..combined_basis.len() {
                    let overlap: Complex64 = combined_basis[left]
                        .iter()
                        .zip(combined_basis[right])
                        .map(|(a, b)| a.conj() * b)
                        .sum();
                    let expected = if left == right {
                        Complex64::new(1.0, 0.0)
                    } else {
                        Complex64::default()
                    };
                    maximum_orthonormality_error =
                        maximum_orthonormality_error.max((overlap - expected).norm());
                }
            }
            eprintln!(
                "silicon resident correction/vector and orthonormality errors: {maximum_correction_error:e} / {maximum_orthonormality_error:e}"
            );
            assert!(maximum_correction_error < 2.0e-5);
            assert!(maximum_orthonormality_error < 2.0e-5);
            assert_eq!(persistent_context.resident_orthogonalization_calls(), 1);
            assert_eq!(persistent_context.resident_orthogonalization_passes(), 4);
            assert_eq!(persistent_context.resident_corrections_accepted(), 2);
            assert_eq!(persistent_context.resident_corrections_rejected(), 0);
            assert!(persistent_context.resident_reduction_download_bytes() > 0);

            let dependent_coefficients = vec![ritz_coefficients[0].clone(); 2];
            let dependent_eigenvalues = [ritz_eigenvalues[0]; 2];
            persistent_context
                .form_resident_ritz_residuals(&dependent_coefficients, &dependent_eigenvalues)
                .unwrap();
            persistent_context
                .precondition_resident_residuals(&[true, true])
                .unwrap();
            let dependent_accepted = persistent_context
                .orthonormalize_resident_corrections(&[true, true])
                .unwrap();
            assert_eq!(dependent_accepted, 1);
            assert_eq!(persistent_context.resident_corrections_rejected(), 1);

            persistent_context
                .form_resident_ritz_residuals(&ritz_coefficients, &ritz_eigenvalues)
                .unwrap();
            persistent_context
                .precondition_resident_residuals(&[true, true])
                .unwrap();
            let append_count = persistent_context
                .orthonormalize_resident_corrections(&[true, true])
                .unwrap();
            assert_eq!(append_count, 2);
            let cpu_applied_corrections: Vec<_> = accepted_corrections
                .iter()
                .map(|correction| {
                    hamiltonian
                        .apply_fft(correction, &electronic.real_ry)
                        .unwrap()
                })
                .collect();
            let uploads_before_append = persistent_context.full_vector_upload_bytes();
            let downloads_before_append = persistent_context.full_vector_download_bytes();
            persistent_context
                .apply_resident_accepted_corrections(append_count)
                .unwrap();
            assert_eq!(
                persistent_context.full_vector_upload_bytes(),
                uploads_before_append
            );
            assert_eq!(
                persistent_context.full_vector_download_bytes(),
                downloads_before_append
            );
            assert_eq!(
                persistent_context.resident_correction_hamiltonian_applications(),
                1
            );
            assert_eq!(persistent_context.resident_corrections_appended(), 2);
            assert!(
                persistent_context
                    .apply_resident_accepted_corrections(append_count)
                    .is_err()
            );
            let journal_tail = persistent_context
                .read_resident_journal_tail(append_count)
                .unwrap();
            let maximum_appended_basis_error = journal_tail
                .basis
                .iter()
                .zip(&accepted_corrections)
                .map(|(actual, expected)| relative_l2(actual, expected))
                .fold(0.0_f64, f64::max);
            let maximum_appended_hamiltonian_error = journal_tail
                .applied_basis
                .iter()
                .zip(&cpu_applied_corrections)
                .map(|(actual, expected)| relative_l2(actual, expected))
                .fold(0.0_f64, f64::max);
            eprintln!(
                "silicon resident correction append basis/H errors: {maximum_appended_basis_error:e} / {maximum_appended_hamiltonian_error:e}"
            );
            assert!(maximum_appended_basis_error < 2.0e-5);
            assert!(maximum_appended_hamiltonian_error < 2.0e-5);

            let resynced_projection = persistent_context
                .resident_cross_gram(&wavefunctions[..2], &persistent_full[..2])
                .unwrap();
            for left in 0..2 {
                for right in 0..2 {
                    let cpu_projection: Complex64 = wavefunctions[left]
                        .iter()
                        .zip(&persistent_full[right])
                        .map(|(a, b)| a.conj() * b)
                        .sum();
                    assert!(
                        (resynced_projection[left * 2 + right] - cpu_projection).norm() < 1.0e-5
                    );
                }
            }
            assert_eq!(persistent_context.resident_projection_calls(), 2);
            assert_eq!(persistent_context.resident_projection_resyncs(), 1);
            assert!(persistent_context.resident_residual_norms(2).is_err());
            assert!(
                persistent_context
                    .precondition_resident_residuals(&[true, false])
                    .is_err()
            );
            assert!(
                persistent_context
                    .orthonormalize_resident_corrections(&[true, false])
                    .is_err()
            );

            let gram_calls_before = persistent_context.gram_reduction_calls();
            let gram_pairs_before = persistent_context.gram_reduction_pairs();
            let metal_gram = persistent_context
                .cross_gram(&wavefunctions, &wavefunctions)
                .unwrap();
            let mut maximum_gram_error = 0.0_f64;
            for left in 0..wavefunctions.len() {
                for right in 0..wavefunctions.len() {
                    let cpu_overlap: Complex64 = wavefunctions[left]
                        .iter()
                        .zip(&wavefunctions[right])
                        .map(|(a, b)| a.conj() * b)
                        .sum();
                    maximum_gram_error = maximum_gram_error
                        .max((metal_gram[left * wavefunctions.len() + right] - cpu_overlap).norm());
                }
            }
            eprintln!("silicon Metal partial-Gram maximum absolute error: {maximum_gram_error:e}");
            assert!(maximum_gram_error < 1.0e-5);
            let rectangular_gram = persistent_context
                .cross_gram(&wavefunctions[..2], &wavefunctions)
                .unwrap();
            for left in 0..2 {
                for right in 0..wavefunctions.len() {
                    let cpu_overlap: Complex64 = wavefunctions[left]
                        .iter()
                        .zip(&wavefunctions[right])
                        .map(|(a, b)| a.conj() * b)
                        .sum();
                    assert!(
                        (rectangular_gram[left * wavefunctions.len() + right] - cpu_overlap).norm()
                            < 1.0e-5
                    );
                }
            }
            assert_eq!(
                persistent_context.gram_reduction_calls() - gram_calls_before,
                2
            );
            assert_eq!(
                persistent_context.gram_reduction_pairs() - gram_pairs_before,
                wavefunctions.len() * wavefunctions.len() + 2 * wavefunctions.len()
            );
            let norm_calls_before = persistent_context.norm_reduction_calls();
            let norm_vectors_before = persistent_context.norm_reduction_vectors();
            let metal_norms = persistent_context.residual_norms(&wavefunctions).unwrap();
            let mut maximum_norm_relative_error = 0.0_f64;
            for (vector, &metal_norm) in wavefunctions.iter().zip(&metal_norms) {
                let cpu_norm = vector.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
                maximum_norm_relative_error = maximum_norm_relative_error
                    .max((metal_norm - cpu_norm).abs() / cpu_norm.max(1.0e-30));
            }
            eprintln!(
                "silicon Metal direct-norm maximum relative error: {maximum_norm_relative_error:e}"
            );
            assert!(maximum_norm_relative_error < 1.0e-5);
            assert_eq!(
                persistent_context.norm_reduction_calls() - norm_calls_before,
                1
            );
            assert_eq!(
                persistent_context.norm_reduction_vectors() - norm_vectors_before,
                wavefunctions.len()
            );

            persistent_context.begin_resident_davidson().unwrap();
            assert!(persistent_context.restart_resident_from_ritz(2).is_err());
            hamiltonian
                .apply_hamiltonian_metal_persistent(
                    &mut persistent_context,
                    &wavefunctions,
                    &electronic.real_ry,
                )
                .unwrap();
            persistent_context
                .form_resident_ritz_residuals(&ritz_coefficients, &ritz_eigenvalues)
                .unwrap();
            persistent_context
                .precondition_resident_residuals(&[true, true])
                .unwrap();
            let uploads_before_restart = persistent_context.full_vector_upload_bytes();
            let downloads_before_restart = persistent_context.full_vector_download_bytes();
            persistent_context.restart_resident_from_ritz(2).unwrap();
            assert_eq!(
                persistent_context.full_vector_upload_bytes(),
                uploads_before_restart
            );
            assert_eq!(
                persistent_context.full_vector_download_bytes(),
                downloads_before_restart
            );
            assert_eq!(persistent_context.resident_restarts(), 1);
            let restarted_journal = persistent_context.read_resident_journal_tail(2).unwrap();
            let maximum_restart_basis_error = restarted_journal
                .basis
                .iter()
                .zip(&resident_vectors.ritz)
                .map(|(actual, expected)| relative_l2(actual, expected))
                .fold(0.0_f64, f64::max);
            let maximum_restart_applied_error = restarted_journal
                .applied_basis
                .iter()
                .zip(&resident_vectors.applied_ritz)
                .map(|(actual, expected)| relative_l2(actual, expected))
                .fold(0.0_f64, f64::max);
            eprintln!(
                "silicon device-local Ritz restart basis/H errors: {maximum_restart_basis_error:e} / {maximum_restart_applied_error:e}"
            );
            assert!(maximum_restart_basis_error < 2.0e-5);
            assert!(maximum_restart_applied_error < 2.0e-5);
            assert!(
                persistent_context
                    .orthonormalize_resident_corrections(&[true, true])
                    .unwrap()
                    > 0
            );

            persistent_context.begin_resident_davidson().unwrap();
            let context = std::cell::RefCell::new(&mut persistent_context);
            let metal_davidson = block_davidson_batched_with_reductions(
                hamiltonian.kinetic_ry(),
                None,
                &DavidsonOptions {
                    roots: 4,
                    residual_tolerance: 1.0e-5,
                    max_iterations: 80,
                    max_subspace: 32,
                },
                |orbitals| {
                    let mut context = context.borrow_mut();
                    hamiltonian.apply_hamiltonian_metal_persistent(
                        &mut context,
                        orbitals,
                        &electronic.real_ry,
                    )
                },
                |left, right| {
                    let mut context = context.borrow_mut();
                    context.resident_cross_gram(left, right)
                },
                |vectors| {
                    let mut context = context.borrow_mut();
                    context.residual_norms(vectors)
                },
            )
            .unwrap();
            assert!(metal_davidson.converged);
            let maximum_eigenvalue_error_mev = metal_davidson
                .eigenvalues
                .iter()
                .zip(&dense_eigenpairs.eigenvalues)
                .map(|(metal, cpu)| (metal - cpu).abs() * 13_605.693_122_994)
                .fold(0.0_f64, f64::max);
            eprintln!(
                "silicon Metal fixed-density occupied eigenvalue error: {maximum_eigenvalue_error_mev:e} meV"
            );
            assert!(maximum_eigenvalue_error_mev < 5.0);
            assert!(context.borrow().resident_projection_calls() > 2);
            assert!(context.borrow().norm_reduction_calls() > 1);
        }

        let ionic_charges = vec![valence_charge; input.atoms.len()];
        let ewald_a = ewald_energy_ry(&lattice, &input.atoms, &ionic_charges, 0.6).unwrap();
        let ewald_b = ewald_energy_ry(&lattice, &input.atoms, &ionic_charges, 0.8).unwrap();
        assert!((ewald_a - ewald_b).abs() < 1.0e-10, "Ewald eta dependence");
        assert!(
            (ewald_a - (-16.817_320_73)).abs() < 5.0e-8,
            "Ewald energy {ewald_a:.12} Ry differs from the QE fixture"
        );
    }

    #[test]
    fn uniform_orbital_conserves_charge_and_has_zero_hartree() {
        let lattice = Lattice::new([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]).unwrap();
        let fft = Fft3d::new([5, 5, 5]).unwrap();
        let g_vectors = [GVector { index: [0, 0, 0] }];
        let density = density_from_orbitals(
            &fft,
            &g_vectors,
            &[vec![Complex64::new(1.0, 0.0)]],
            &[2.0],
            lattice.volume,
        )
        .unwrap();
        let potential =
            hartree_xc_potential(&fft, &lattice, &density, XcFunctional::LdaPz).unwrap();
        assert!((potential.electron_count - 2.0).abs() < 1.0e-12);
        assert!(potential.hartree_energy_ry.abs() < 1.0e-14);
    }

    #[test]
    fn shifted_silicon_hamiltonian_is_hermitian_and_time_reversal_symmetric_when_available() {
        let input_path = Path::new("examples/si2-pbe.toml");
        if !input_path.exists() {
            return;
        }
        let input = SimulationInput::from_path(input_path).unwrap();
        if !input.pseudopotentials["Si"].exists() {
            return;
        }
        let pseudo = NormConservingPseudo::from_upf(&input.pseudopotentials["Si"]).unwrap();
        let pseudos = BTreeMap::from([("Si".to_owned(), pseudo)]);
        let lattice = Lattice::new(input.cell.lattice_bohr).unwrap();
        let k = [0.25, 0.125, 0.0];
        let plus_g = lattice.enumerate_g_vectors_at_k(input.cutoff_ry, k);
        let minus_k = k.map(|value| -value);
        let minus_g = lattice.enumerate_g_vectors_at_k(input.cutoff_ry, minus_k);
        assert_eq!(plus_g.len(), minus_g.len());
        let union: Vec<_> = plus_g.iter().chain(&minus_g).copied().collect();
        let grid_shape = Lattice::density_fft_grid(&union);
        let plus = CpuHamiltonian::new_at_k(
            &lattice,
            &plus_g,
            &input.atoms,
            &pseudos,
            input.cutoff_ry,
            k,
            grid_shape,
        )
        .unwrap();
        let minus = CpuHamiltonian::new_at_k(
            &lattice,
            &minus_g,
            &input.atoms,
            &pseudos,
            input.cutoff_ry,
            minus_k,
            grid_shape,
        )
        .unwrap();
        assert_eq!(plus.k_reduced(), k);

        for (g, &kinetic) in plus_g.iter().zip(plus.kinetic_ry()) {
            let mut q = lattice.cartesian_g(g.index);
            let k_cartesian = lattice.cartesian_k(k);
            for axis in 0..3 {
                q[axis] += k_cartesian[axis];
            }
            assert!((kinetic - norm_squared(q)).abs() < 1.0e-13);
        }

        let plus_matrix = plus.dense_matrix().unwrap();
        let scale = plus_matrix
            .iter()
            .map(|value| value.norm())
            .fold(0.0_f64, f64::max);
        let mut maximum_error = 0.0_f64;
        for row in 0..plus.dimension() {
            for column in 0..plus.dimension() {
                maximum_error = maximum_error.max(
                    (plus_matrix[row * plus.dimension() + column]
                        - plus_matrix[column * plus.dimension() + row].conj())
                    .norm(),
                );
            }
        }
        assert!(maximum_error / scale < 1.0e-12);

        let plus_eigenvalues = projected_hermitian_eigensolve(&plus_matrix, plus.dimension())
            .unwrap()
            .eigenvalues;
        let minus_matrix = minus.dense_matrix().unwrap();
        let minus_eigenvalues = projected_hermitian_eigensolve(&minus_matrix, minus.dimension())
            .unwrap()
            .eigenvalues;
        let maximum_spectral_error = plus_eigenvalues
            .iter()
            .zip(minus_eigenvalues)
            .map(|(plus, minus)| (plus - minus).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            maximum_spectral_error < 1.0e-10,
            "time-reversal eigenvalue error {maximum_spectral_error:e} Ry"
        );
    }

    #[test]
    fn mixed_ionic_charge_ewald_sum_is_eta_invariant() {
        let half_lattice = 3.992_424_385_254_755;
        let lattice = Lattice::new([
            [0.0, half_lattice, half_lattice],
            [half_lattice, 0.0, half_lattice],
            [half_lattice, half_lattice, 0.0],
        ])
        .unwrap();
        let atoms = vec![
            AtomInput {
                element: "Mg".into(),
                fractional: [0.0, 0.0, 0.0],
            },
            AtomInput {
                element: "O".into(),
                fractional: [0.5, 0.5, 0.5],
            },
        ];
        let charges = [10.0, 6.0];
        let ewald_a = ewald_energy_ry(&lattice, &atoms, &charges, 0.6).unwrap();
        let ewald_b = ewald_energy_ry(&lattice, &atoms, &charges, 0.8).unwrap();
        assert!((ewald_a - ewald_b).abs() < 1.0e-10, "Ewald eta dependence");
    }

    #[test]
    fn analytic_ewald_forces_match_energy_finite_differences() {
        let length = 8.0;
        let lattice =
            Lattice::new([[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]]).unwrap();
        let atoms = vec![
            AtomInput {
                element: "A".into(),
                fractional: [0.13, 0.22, 0.31],
            },
            AtomInput {
                element: "B".into(),
                fractional: [0.61, 0.47, 0.79],
            },
        ];
        let charges = [2.0, 3.0];
        let eta = 0.65;
        let analytic = ewald_forces_ry_bohr(&lattice, &atoms, &charges, eta).unwrap();
        let step_bohr = 1.0e-5;
        let mut maximum_error = 0.0_f64;
        for (atom, analytic_force) in analytic.iter().enumerate() {
            for (axis, &analytic_component) in analytic_force.iter().enumerate() {
                let mut plus = atoms.clone();
                let mut minus = atoms.clone();
                plus[atom].fractional[axis] += step_bohr / length;
                minus[atom].fractional[axis] -= step_bohr / length;
                let energy_plus = ewald_energy_ry(&lattice, &plus, &charges, eta).unwrap();
                let energy_minus = ewald_energy_ry(&lattice, &minus, &charges, eta).unwrap();
                let finite_difference = -(energy_plus - energy_minus) / (2.0 * step_bohr);
                maximum_error = maximum_error.max((analytic_component - finite_difference).abs());
            }
        }
        assert!(
            maximum_error < 2.0e-8,
            "Ewald force error {maximum_error:e}"
        );

        for axis in 0..3 {
            let total: f64 = analytic.iter().map(|force| force[axis]).sum();
            assert!(total.abs() < 1.0e-12, "net Ewald force {total:e}");
        }
        let alternate = ewald_forces_ry_bohr(&lattice, &atoms, &charges, 0.8).unwrap();
        let eta_error = analytic
            .iter()
            .flatten()
            .zip(alternate.iter().flatten())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f64, f64::max);
        assert!(eta_error < 1.0e-10, "Ewald force eta error {eta_error:e}");
    }

    #[test]
    fn local_ionic_forces_match_fixed_density_energy_derivative_when_available() {
        let input_path = Path::new("examples/si2.toml");
        if !input_path.exists() || !Path::new("pseudopotentials/Si.pz-vbc.UPF").exists() {
            return;
        }
        let input = SimulationInput::from_path(input_path).unwrap();
        let pseudo = NormConservingPseudo::from_upf(&input.pseudopotentials["Si"]).unwrap();
        let pseudos = BTreeMap::from([("Si".to_owned(), pseudo)]);
        let lattice = Lattice::new(input.cell.lattice_bohr).unwrap();
        let g_vectors = lattice.enumerate_g_vectors(input.cutoff_ry);
        let reference = CpuHamiltonian::new(
            &lattice,
            &g_vectors,
            &input.atoms,
            &pseudos,
            input.cutoff_ry,
        )
        .unwrap();
        let density: Vec<_> = (0..reference.fft().len())
            .map(|index| {
                0.12 + 0.013 * (index as f64 * 0.37).sin() + 0.009 * (index as f64 * 0.19).cos()
            })
            .collect();
        let analytic = reference.local_ionic_forces_ry_bohr(&density).unwrap();
        let cell_weight = lattice.volume / reference.fft().len() as f64;
        let local_energy = |atoms: &[AtomInput]| {
            let hamiltonian =
                CpuHamiltonian::new(&lattice, &g_vectors, atoms, &pseudos, input.cutoff_ry)
                    .unwrap();
            density
                .iter()
                .zip(hamiltonian.ionic_local_real_ry())
                .map(|(&rho, &potential)| rho * potential * cell_weight)
                .sum::<f64>()
        };

        let step_bohr = 1.0e-5;
        let mut maximum_error = 0.0_f64;
        for (atom, analytic_force) in analytic.iter().enumerate() {
            for (cartesian_axis, &analytic_component) in analytic_force.iter().enumerate() {
                let mut plus = input.atoms.clone();
                let mut minus = input.atoms.clone();
                for fractional_axis in 0..3 {
                    let delta_fractional =
                        step_bohr * lattice.reciprocal[fractional_axis][cartesian_axis] / TAU;
                    plus[atom].fractional[fractional_axis] += delta_fractional;
                    minus[atom].fractional[fractional_axis] -= delta_fractional;
                }
                let finite_difference =
                    -(local_energy(&plus) - local_energy(&minus)) / (2.0 * step_bohr);
                maximum_error = maximum_error.max((analytic_component - finite_difference).abs());
            }
        }
        assert!(
            maximum_error < 2.0e-7,
            "local force error {maximum_error:e}"
        );
    }

    #[test]
    fn nonlocal_ionic_forces_match_fixed_orbital_energy_derivative_when_available() {
        let input_path = Path::new("examples/si2.toml");
        if !input_path.exists() || !Path::new("pseudopotentials/Si.pz-vbc.UPF").exists() {
            return;
        }
        let input = SimulationInput::from_path(input_path).unwrap();
        let pseudo = NormConservingPseudo::from_upf(&input.pseudopotentials["Si"]).unwrap();
        let pseudos = BTreeMap::from([("Si".to_owned(), pseudo)]);
        let lattice = Lattice::new(input.cell.lattice_bohr).unwrap();
        let g_vectors = lattice.enumerate_g_vectors(input.cutoff_ry);
        let reference = CpuHamiltonian::new(
            &lattice,
            &g_vectors,
            &input.atoms,
            &pseudos,
            input.cutoff_ry,
        )
        .unwrap();
        let mut orbital: Vec<_> = (0..reference.dimension())
            .map(|index| {
                Complex64::new(
                    (index as f64 * 0.137 + 0.23).sin(),
                    (index as f64 * 0.091 - 0.41).cos(),
                )
            })
            .collect();
        let norm = orbital
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt();
        for coefficient in &mut orbital {
            *coefficient /= norm;
        }
        let occupation = 1.7;
        let analytic = reference
            .nonlocal_ionic_forces_ry_bohr(&[orbital.clone()], &[occupation])
            .unwrap();
        let nonlocal_energy = |atoms: &[AtomInput]| {
            let hamiltonian =
                CpuHamiltonian::new(&lattice, &g_vectors, atoms, &pseudos, input.cutoff_ry)
                    .unwrap();
            let mut applied = vec![Complex64::default(); hamiltonian.dimension()];
            hamiltonian.add_nonlocal(&orbital, &mut applied);
            occupation
                * orbital
                    .iter()
                    .zip(applied)
                    .map(|(left, right)| left.conj() * right)
                    .sum::<Complex64>()
                    .re
        };

        let step_bohr = 1.0e-5;
        let mut maximum_error = 0.0_f64;
        for (atom, analytic_force) in analytic.iter().enumerate() {
            for (cartesian_axis, &analytic_component) in analytic_force.iter().enumerate() {
                let mut plus = input.atoms.clone();
                let mut minus = input.atoms.clone();
                for fractional_axis in 0..3 {
                    let delta_fractional =
                        step_bohr * lattice.reciprocal[fractional_axis][cartesian_axis] / TAU;
                    plus[atom].fractional[fractional_axis] += delta_fractional;
                    minus[atom].fractional[fractional_axis] -= delta_fractional;
                }
                let finite_difference =
                    -(nonlocal_energy(&plus) - nonlocal_energy(&minus)) / (2.0 * step_bohr);
                maximum_error = maximum_error.max((analytic_component - finite_difference).abs());
            }
        }
        assert!(
            maximum_error < 2.0e-8,
            "nonlocal force error {maximum_error:e}"
        );
    }

    #[test]
    fn mgo_multi_species_hamiltonian_is_hermitian_when_available() {
        let input_path = Path::new("examples/mgo2-pbe.toml");
        let required = [
            Path::new("pseudopotentials/Mg_ONCV_PBE-1.0.UPF"),
            Path::new("pseudopotentials/O_ONCV_PBE-1.0.UPF"),
        ];
        if !input_path.exists() || required.iter().any(|path| !path.exists()) {
            return;
        }
        let input = SimulationInput::from_path(input_path).unwrap();
        let pseudos: BTreeMap<_, _> = input
            .pseudopotentials
            .iter()
            .map(|(element, path)| {
                (
                    element.clone(),
                    NormConservingPseudo::from_upf(path).unwrap(),
                )
            })
            .collect();
        let lattice = Lattice::new(input.cell.lattice_bohr).unwrap();
        let g_vectors = lattice.enumerate_g_vectors(input.cutoff_ry);
        let hamiltonian = CpuHamiltonian::new(
            &lattice,
            &g_vectors,
            &input.atoms,
            &pseudos,
            input.cutoff_ry,
        )
        .unwrap();
        let left: Vec<_> = (0..hamiltonian.dimension())
            .map(|index| Complex64::new((index as f64 * 0.037).sin(), (index as f64 * 0.051).cos()))
            .collect();
        let right: Vec<_> = (0..hamiltonian.dimension())
            .map(|index| Complex64::new((index as f64 * 0.071).cos(), (index as f64 * 0.029).sin()))
            .collect();
        let electronic_potential = vec![0.0; hamiltonian.fft().len()];
        let h_left = hamiltonian.apply_fft(&left, &electronic_potential).unwrap();
        let h_right = hamiltonian
            .apply_fft(&right, &electronic_potential)
            .unwrap();
        let left_h_right: Complex64 = left.iter().zip(&h_right).map(|(a, b)| a.conj() * b).sum();
        let h_left_right: Complex64 = h_left.iter().zip(&right).map(|(a, b)| a.conj() * b).sum();
        let scale = left_h_right.norm().max(h_left_right.norm()).max(1.0);
        assert!(
            (left_h_right - h_left_right).norm() / scale < 1.0e-11,
            "multi-species Hamiltonian is not Hermitian"
        );
    }
}
