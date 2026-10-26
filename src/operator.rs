//! CPU-FP64 Kohn-Sham Hamiltonian building blocks for the admitted silicon case.
//!
//! The radial transforms use the conventional plane-wave normalization
//! `exp(i G.r) / sqrt(volume)`. Energies and pseudopotentials are in Rydberg.

use crate::fft::Fft3d;
use crate::input::AtomInput;
use crate::lattice::{GVector, Lattice, Vec3};
use crate::pseudo::{NormConservingPseudo, Projector};
use crate::xc::lda_pz_unpolarized;
use anyhow::{Result, ensure};
#[cfg(all(feature = "metal", target_os = "macos"))]
use num_complex::Complex32;
use num_complex::Complex64;
use std::f64::consts::{PI, TAU};
use std::sync::OnceLock;

const FOUR_PI: f64 = 4.0 * PI;
const ELECTRON_CHARGE_SQUARED_RY: f64 = 2.0;

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
        _ => anyhow::bail!("only l=0 and l=1 silicon projectors are supported"),
    }
}

/// Fourier coefficient of one atom's local potential, including `1 / volume`.
pub fn local_form_factor_ry(pseudo: &NormConservingPseudo, q: f64, volume: f64) -> Result<f64> {
    ensure!(q.is_finite() && q >= 0.0, "invalid reciprocal magnitude");
    ensure!(volume.is_finite() && volume > 0.0, "invalid cell volume");
    let z_e2 = pseudo.valence_charge * ELECTRON_CHARGE_SQUARED_RY;
    let values: Vec<f64> = if q < 1.0e-12 {
        pseudo
            .radial_grid
            .iter()
            .zip(&pseudo.local_potential_ry)
            .map(|(&r, &v)| r * (r * v + z_e2))
            .collect()
    } else {
        pseudo
            .radial_grid
            .iter()
            .zip(&pseudo.local_potential_ry)
            .map(|(&r, &v)| (r * v + z_e2 * libm::erf(r)) * (q * r).sin() / q)
            .collect()
    };
    let short_range = FOUR_PI * weighted_simpson(&values, &pseudo.radial_weights)? / volume;
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
    let values: Vec<_> = pseudo
        .atomic_density
        .iter()
        .zip(&pseudo.radial_grid)
        .map(|(&density, &r)| density * spherical_bessel(0, q * r).unwrap_or_default())
        .collect();
    Ok(weighted_simpson(&values, &pseudo.radial_weights)? / volume)
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
}

#[derive(Clone)]
struct DenseReferenceData {
    lattice: Lattice,
    g_vectors: Vec<GVector>,
    atoms: Vec<AtomInput>,
    pseudo: NormConservingPseudo,
}

/// Dense-reference CPU Hamiltonian for operator validation and small silicon cells.
///
/// The local operator is deliberately formed directly in reciprocal space. This
/// is not the production FFT path; it provides an independent, transparent
/// reference against which the later FFT-resident CPU and Metal paths can be tested.
#[derive(Clone)]
pub struct CpuHamiltonian {
    kinetic_ry: Vec<f64>,
    local_ry: OnceLock<Vec<Complex64>>,
    dense_reference: DenseReferenceData,
    atom_projectors: Vec<AtomProjectors>,
    dij_ry: Vec<Vec<f64>>,
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
        pseudo: &NormConservingPseudo,
    ) -> Result<Self> {
        ensure!(!g_vectors.is_empty(), "Hamiltonian basis is empty");
        ensure!(
            atoms.iter().all(|atom| atom.element == pseudo.element),
            "the reference Hamiltonian currently supports one element"
        );
        let cartesian: Vec<_> = g_vectors
            .iter()
            .map(|g| lattice.cartesian_g(g.index))
            .collect();
        let kinetic_ry = cartesian.iter().map(|&g| norm_squared(g)).collect();
        let dimension = g_vectors.len();
        let grid_shape = Lattice::density_fft_grid(g_vectors);
        let fft = Fft3d::new(grid_shape)?;
        let basis_grid_indices = g_vectors
            .iter()
            .map(|g| flat_index_for_shape(g.index, grid_shape))
            .collect();

        let mut atom_projectors = Vec::with_capacity(atoms.len());
        for atom in atoms {
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
                                .zip(atom.fractional)
                                .map(|(&n, f)| n as f64 * f)
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
                    let q = norm_squared(lattice.cartesian_g(index)).sqrt();
                    let radial = local_form_factor_ry(pseudo, q, lattice.volume)?;
                    let structure: Complex64 = atoms
                        .iter()
                        .map(|atom| {
                            let phase = -TAU
                                * index
                                    .iter()
                                    .zip(atom.fractional)
                                    .map(|(&n, f)| n as f64 * f)
                                    .sum::<f64>();
                            Complex64::from_polar(1.0, phase)
                        })
                        .sum();
                    ionic_coefficients[(x * grid_shape[1] + y) * grid_shape[2] + z] =
                        radial * structure;
                }
            }
        }
        fft.inverse(&mut ionic_coefficients)?;
        let grid_points = fft.len() as f64;
        let maximum_imaginary = ionic_coefficients
            .iter()
            .map(|value| value.im.abs())
            .fold(0.0_f64, f64::max);
        ensure!(
            maximum_imaginary < 1.0e-10,
            "ionic local potential is not real ({maximum_imaginary:e})"
        );
        let ionic_local_real_ry = ionic_coefficients
            .iter()
            .map(|value| value.re * grid_points)
            .collect();
        Ok(Self {
            kinetic_ry,
            local_ry: OnceLock::new(),
            dense_reference: DenseReferenceData {
                lattice: lattice.clone(),
                g_vectors: g_vectors.to_vec(),
                atoms: atoms.to_vec(),
                pseudo: pseudo.clone(),
            },
            atom_projectors,
            dij_ry: pseudo.dij_ry.clone(),
            fft,
            basis_grid_indices,
            ionic_local_real_ry,
            nonlocal_dense_ry: OnceLock::new(),
        })
    }

    pub fn dimension(&self) -> usize {
        self.kinetic_ry.len()
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
                        weighted += self.dij_ry[atom.radial_projector[left]]
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
                let radial = local_form_factor_ry(&reference.pseudo, q, reference.lattice.volume)?;
                let structure: Complex64 = reference
                    .atoms
                    .iter()
                    .map(|atom| {
                        let phase = -TAU
                            * delta_index
                                .iter()
                                .zip(atom.fractional)
                                .map(|(&n, f)| n as f64 * f)
                                .sum::<f64>();
                        Complex64::from_polar(1.0, phase)
                    })
                    .sum();
                matrix[row * n + column] = radial * structure;
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
                        let coupling = self.dij_ry[atom.radial_projector[left]]
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
    ensure!(
        orbitals.len() == occupations.len(),
        "occupation count mismatch"
    );
    let indices: Vec<_> = g_vectors
        .iter()
        .map(|g| flat_index_for_shape(g.index, fft.shape()))
        .collect();
    let normalization = (fft.len() as f64).powi(2) / volume;
    let mut density = vec![0.0; fft.len()];
    for (orbital, &occupation) in orbitals.iter().zip(occupations) {
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
            *rho += occupation * normalization * value.norm_sqr();
        }
    }
    Ok(density)
}

pub fn atomic_starting_density(
    fft: &Fft3d,
    lattice: &Lattice,
    atoms: &[AtomInput],
    pseudo: &NormConservingPseudo,
    electron_count: f64,
) -> Result<Vec<f64>> {
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
                let q = norm_squared(lattice.cartesian_g(index)).sqrt();
                let radial = atomic_density_form_factor(pseudo, q, lattice.volume)?;
                let structure: Complex64 = atoms
                    .iter()
                    .map(|atom| {
                        let phase = -TAU
                            * index
                                .iter()
                                .zip(atom.fractional)
                                .map(|(&n, f)| n as f64 * f)
                                .sum::<f64>();
                        Complex64::from_polar(1.0, phase)
                    })
                    .sum();
                coefficients[(x * shape[1] + y) * shape[2] + z] = radial * structure;
            }
        }
    }
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

/// Periodic point-ion Ewald energy in Rydberg for the single-species scope.
/// `eta` is the Gaussian splitting parameter in inverse bohr.
pub fn ewald_energy_ry(
    lattice: &Lattice,
    atoms: &[AtomInput],
    ionic_charge: f64,
    eta: f64,
) -> Result<f64> {
    ensure!(eta.is_finite() && eta > 0.0, "Ewald eta must be positive");
    ensure!(!atoms.is_empty(), "Ewald sum requires atoms");
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
    for (i, &ri) in positions.iter().enumerate() {
        for (j, &rj) in positions.iter().enumerate() {
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
                            real_sum +=
                                ionic_charge * ionic_charge * libm::erfc(eta * distance) / distance;
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
                    .map(|&position| {
                        let phase = -(g[0] * position[0] + g[1] * position[1] + g[2] * position[2]);
                        ionic_charge * Complex64::from_polar(1.0, phase)
                    })
                    .sum();
                reciprocal_sum += (-g2 / (4.0 * eta * eta)).exp() * structure.norm_sqr() / g2;
            }
        }
    }
    reciprocal_sum *= 2.0 * PI / lattice.volume;
    let total_charge = ionic_charge * atoms.len() as f64;
    let self_energy = eta / PI.sqrt() * atoms.len() as f64 * ionic_charge * ionic_charge;
    let background = PI * total_charge * total_charge / (2.0 * eta * eta * lattice.volume);
    Ok(2.0 * (real_sum + reciprocal_sum - self_energy - background))
}

pub fn hartree_xc_potential(
    fft: &Fft3d,
    lattice: &Lattice,
    density: &[f64],
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
    let mut real_ry = Vec::with_capacity(fft.len());
    let mut xc_energy_ry = 0.0;
    let mut integral_rho_vxc_ry = 0.0;
    for (&rho, vh) in density.iter().zip(vh_g) {
        let nonnegative_density = rho.max(0.0);
        let (epsilon_xc_ha, v_xc_ha) = lda_pz_unpolarized(nonnegative_density);
        let epsilon_xc_ry = 2.0 * epsilon_xc_ha;
        let v_xc_ry = 2.0 * v_xc_ha;
        xc_energy_ry += nonnegative_density * epsilon_xc_ry * cell_weight;
        integral_rho_vxc_ry += nonnegative_density * v_xc_ry * cell_weight;
        real_ry.push(vh.re * grid_points + v_xc_ry);
    }
    Ok(ElectronicPotential {
        real_ry,
        hartree_energy_ry,
        xc_energy_ry,
        integral_rho_vxc_ry,
        electron_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(all(feature = "metal", target_os = "macos"))]
    use crate::eigensolver::block_davidson_batched_with_gram;
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
        let input_path = Path::new("examples/si2.toml");
        if !input_path.exists() {
            return;
        }
        let input = SimulationInput::from_path(input_path).unwrap();
        if !input.pseudopotentials["Si"].exists() {
            return;
        }
        let pseudo = NormConservingPseudo::from_upf(&input.pseudopotentials["Si"]).unwrap();
        let lattice = Lattice::new(input.cell.lattice_bohr).unwrap();
        let g_vectors = lattice.enumerate_g_vectors(input.cutoff_ry);
        let hamiltonian = CpuHamiltonian::new(&lattice, &g_vectors, &input.atoms, &pseudo).unwrap();
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

        let starting_density =
            atomic_starting_density(hamiltonian.fft(), &lattice, &input.atoms, &pseudo, 8.0)
                .unwrap();
        let integrated =
            starting_density.iter().sum::<f64>() * lattice.volume / starting_density.len() as f64;
        assert!((integrated - 8.0).abs() < 1.0e-12);

        let electronic =
            hartree_xc_potential(hamiltonian.fft(), &lattice, &starting_density).unwrap();
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
            let wavefunctions: Vec<Vec<Complex64>> = (0..3)
                .map(|state| {
                    let mut values: Vec<_> = (0..hamiltonian.dimension())
                        .map(|i| {
                            Complex64::new(
                                ((i + 3 * state) as f64 * 0.17).sin(),
                                ((2 * i + state) as f64 * 0.09).cos(),
                            )
                        })
                        .collect();
                    let norm = values.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
                    for value in &mut values {
                        *value /= norm;
                    }
                    values
                })
                .collect();
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
            assert_eq!(persistent_context.gram_reduction_calls(), 4);
            assert_eq!(
                persistent_context.gram_reduction_pairs(),
                2 * wavefunctions.len() * wavefunctions.len() + 2 * wavefunctions.len() + 4
            );

            persistent_context.begin_resident_davidson().unwrap();
            let context = std::cell::RefCell::new(&mut persistent_context);
            let metal_davidson = block_davidson_batched_with_gram(
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
        }

        let ewald_a = ewald_energy_ry(&lattice, &input.atoms, pseudo.valence_charge, 0.6).unwrap();
        let ewald_b = ewald_energy_ry(&lattice, &input.atoms, pseudo.valence_charge, 0.8).unwrap();
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
        let potential = hartree_xc_potential(&fft, &lattice, &density).unwrap();
        assert!((potential.electron_count - 2.0).abs() < 1.0e-12);
        assert!(potential.hartree_energy_ry.abs() < 1.0e-14);
    }
}
