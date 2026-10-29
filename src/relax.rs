use crate::backend::BackendKind;
use crate::input::SimulationInput;
use crate::lattice::Lattice;
use crate::result::ScfResult;
use crate::scf::run_input;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::f64::consts::TAU;

pub const RELAXATION_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RelaxationOptions {
    pub max_steps: usize,
    pub force_tolerance_ry_bohr: f64,
    pub initial_step_bohr: f64,
    pub maximum_step_bohr: f64,
    pub minimum_step_bohr: f64,
    pub line_search_reduction: f64,
    pub maximum_line_search_evaluations: usize,
    pub armijo_coefficient: f64,
}

impl Default for RelaxationOptions {
    fn default() -> Self {
        Self {
            max_steps: 40,
            force_tolerance_ry_bohr: 1.0e-4,
            initial_step_bohr: 0.1,
            maximum_step_bohr: 0.3,
            minimum_step_bohr: 1.0e-5,
            line_search_reduction: 0.5,
            maximum_line_search_evaluations: 12,
            armijo_coefficient: 1.0e-4,
        }
    }
}

impl RelaxationOptions {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.max_steps > 0, "relaxation max_steps must be positive");
        ensure!(
            self.force_tolerance_ry_bohr.is_finite() && self.force_tolerance_ry_bohr > 0.0,
            "relaxation force tolerance must be positive and finite"
        );
        ensure!(
            self.initial_step_bohr.is_finite()
                && self.maximum_step_bohr.is_finite()
                && self.minimum_step_bohr.is_finite()
                && self.minimum_step_bohr > 0.0
                && self.initial_step_bohr >= self.minimum_step_bohr
                && self.initial_step_bohr <= self.maximum_step_bohr,
            "relaxation displacement limits are invalid"
        );
        ensure!(
            self.line_search_reduction.is_finite()
                && (0.0..1.0).contains(&self.line_search_reduction),
            "line_search_reduction must be in (0, 1)"
        );
        ensure!(
            self.maximum_line_search_evaluations > 0,
            "maximum_line_search_evaluations must be positive"
        );
        ensure!(
            self.armijo_coefficient.is_finite() && (0.0..1.0).contains(&self.armijo_coefficient),
            "armijo_coefficient must be in (0, 1)"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RelaxationStep {
    pub step: usize,
    pub total_energy_ry: f64,
    pub maximum_force_norm_ry_bohr: f64,
    pub maximum_projected_force_norm_ry_bohr: f64,
    pub positions_fractional: Vec<[f64; 3]>,
    pub forces_ry_bohr: Vec<[f64; 3]>,
    pub scf_iterations: usize,
    pub scf_input_sha256: String,
    pub scf_evaluations: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accepted_maximum_displacement_bohr: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RelaxationResult {
    pub schema_version: u32,
    pub converged: bool,
    pub requested_backend: BackendKind,
    pub atom_count: usize,
    pub options: RelaxationOptions,
    pub total_scf_evaluations: usize,
    pub unaccepted_terminal_scf_evaluations: usize,
    pub steps: Vec<RelaxationStep>,
    pub final_scf: ScfResult,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
}

impl RelaxationResult {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == RELAXATION_SCHEMA_VERSION,
            "unsupported relaxation schema_version {}",
            self.schema_version
        );
        self.options.validate()?;
        ensure!(
            self.atom_count > 0,
            "relaxation atom_count must be positive"
        );
        ensure!(!self.steps.is_empty(), "relaxation history is empty");
        ensure!(
            self.total_scf_evaluations
                == self
                    .steps
                    .iter()
                    .map(|step| step.scf_evaluations)
                    .sum::<usize>()
                    + self.unaccepted_terminal_scf_evaluations,
            "relaxation SCF evaluation count is inconsistent"
        );
        for (index, step) in self.steps.iter().enumerate() {
            ensure!(step.step == index, "relaxation history is not sequential");
            ensure!(
                step.positions_fractional.len() == self.atom_count
                    && step.forces_ry_bohr.len() == self.atom_count
                    && step.total_energy_ry.is_finite()
                    && step.maximum_force_norm_ry_bohr.is_finite()
                    && step.maximum_force_norm_ry_bohr >= 0.0
                    && step.maximum_projected_force_norm_ry_bohr.is_finite()
                    && step.maximum_projected_force_norm_ry_bohr >= 0.0
                    && step
                        .positions_fractional
                        .iter()
                        .flatten()
                        .all(|value| value.is_finite())
                    && step
                        .forces_ry_bohr
                        .iter()
                        .flatten()
                        .all(|value| value.is_finite())
                    && step.scf_iterations > 0
                    && step.scf_evaluations > 0
                    && !step.scf_input_sha256.is_empty()
                    && step
                        .accepted_maximum_displacement_bohr
                        .is_none_or(|value| value.is_finite() && value > 0.0)
                    && (index == 0) == step.accepted_maximum_displacement_bohr.is_none(),
                "relaxation history contains invalid data"
            );
        }
        self.final_scf.validate()?;
        ensure!(
            self.final_scf.converged
                && self.final_scf.atom_count == self.atom_count
                && self.final_scf.provenance.requested_backend == self.requested_backend,
            "relaxation final SCF is inconsistent"
        );
        let final_step = self.steps.last().unwrap();
        ensure!(
            self.final_scf.total_energy_ry == Some(final_step.total_energy_ry)
                && self.final_scf.forces_ry_bohr.as_ref() == Some(&final_step.forces_ry_bohr),
            "relaxation final SCF differs from its history"
        );
        ensure!(
            self.converged
                == (final_step.maximum_force_norm_ry_bohr <= self.options.force_tolerance_ry_bohr),
            "relaxation convergence disagrees with its final force"
        );
        ensure!(
            self.converged == self.stop_reason.is_none(),
            "relaxation stop reason disagrees with convergence"
        );
        Ok(())
    }
}

fn dot(left: &[[f64; 3]], right: &[[f64; 3]]) -> f64 {
    left.iter()
        .flatten()
        .zip(right.iter().flatten())
        .map(|(a, b)| a * b)
        .sum()
}

fn maximum_norm(vectors: &[[f64; 3]]) -> f64 {
    vectors
        .iter()
        .map(|vector| vector.iter().map(|value| value * value).sum::<f64>().sqrt())
        .fold(0.0_f64, f64::max)
}

fn project_translation(forces: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let mut mean = [0.0; 3];
    for force in forces {
        for axis in 0..3 {
            mean[axis] += force[axis] / forces.len() as f64;
        }
    }
    forces
        .iter()
        .map(|force| {
            let mut projected = [0.0; 3];
            for axis in 0..3 {
                projected[axis] = force[axis] - mean[axis];
            }
            projected
        })
        .collect()
}

fn displaced_input(
    input: &SimulationInput,
    lattice: &Lattice,
    direction: &[[f64; 3]],
    maximum_displacement_bohr: f64,
) -> Result<(SimulationInput, f64, Vec<[f64; 3]>)> {
    ensure!(
        direction.len() == input.atoms.len(),
        "relaxation direction atom count differs"
    );
    let direction_norm = maximum_norm(direction);
    ensure!(
        direction_norm.is_finite() && direction_norm > 0.0,
        "relaxation direction is zero or invalid"
    );
    let scale = maximum_displacement_bohr / direction_norm;
    let displacements: Vec<_> = direction
        .iter()
        .map(|vector| {
            let mut value = [0.0; 3];
            for axis in 0..3 {
                value[axis] = scale * vector[axis];
            }
            value
        })
        .collect();
    let mut trial = input.clone();
    for (atom, displacement) in trial.atoms.iter_mut().zip(&displacements) {
        for fractional_axis in 0..3 {
            let delta = displacement
                .iter()
                .enumerate()
                .map(|(cartesian_axis, value)| {
                    value * lattice.reciprocal[fractional_axis][cartesian_axis] / TAU
                })
                .sum::<f64>();
            atom.fractional[fractional_axis] =
                (atom.fractional[fractional_axis] + delta).rem_euclid(1.0);
        }
    }
    Ok((trial, scale, displacements))
}

fn step_record(
    step: usize,
    input: &SimulationInput,
    result: &ScfResult,
    scf_evaluations: usize,
    accepted_displacement: Option<f64>,
) -> Result<RelaxationStep> {
    ensure!(
        result.converged,
        "relaxation requires a converged electronic SCF"
    );
    let forces = result
        .forces_ry_bohr
        .clone()
        .context("converged SCF has no analytic forces")?;
    let projected = project_translation(&forces);
    Ok(RelaxationStep {
        step,
        total_energy_ry: result
            .total_energy_ry
            .context("converged SCF has no total energy")?,
        maximum_force_norm_ry_bohr: maximum_norm(&forces),
        maximum_projected_force_norm_ry_bohr: maximum_norm(&projected),
        positions_fractional: input.atoms.iter().map(|atom| atom.fractional).collect(),
        forces_ry_bohr: forces,
        scf_iterations: result.iterations,
        scf_input_sha256: result.provenance.input_sha256.clone(),
        scf_evaluations,
        accepted_maximum_displacement_bohr: accepted_displacement,
    })
}

pub fn relax(
    input: &SimulationInput,
    requested_backend: BackendKind,
    options: &RelaxationOptions,
) -> Result<RelaxationResult> {
    input.validate()?;
    options.validate()?;
    ensure!(input.cell.fixed, "ionic relaxation requires a fixed cell");
    let lattice = Lattice::new(input.cell.lattice_bohr)?;
    let mut current_input = input.clone();
    let mut current_result = run_input(&current_input, requested_backend)?;
    ensure!(
        current_result.converged,
        "initial electronic SCF did not converge: {}",
        current_result
            .stop_reason
            .as_deref()
            .unwrap_or("unknown reason")
    );
    let mut total_scf_evaluations = 1_usize;
    let mut steps = vec![step_record(0, &current_input, &current_result, 1, None)?];
    let mut force = project_translation(current_result.forces_ry_bohr.as_ref().unwrap());
    let mut direction = force.clone();
    let mut proposed_step = options.initial_step_bohr;
    let mut stop_reason = None;
    let mut unaccepted_terminal_scf_evaluations = 0_usize;

    for step in 1..=options.max_steps {
        if maximum_norm(current_result.forces_ry_bohr.as_ref().unwrap())
            <= options.force_tolerance_ry_bohr
        {
            break;
        }
        if dot(&force, &direction) <= 0.0 {
            direction = force.clone();
        }
        let current_energy = current_result.total_energy_ry.unwrap();
        let mut displacement_limit = proposed_step.min(options.maximum_step_bohr);
        let mut accepted = None;
        let mut attempted_line_evaluations = 0_usize;
        for line_evaluation in 1..=options.maximum_line_search_evaluations {
            if displacement_limit < options.minimum_step_bohr {
                break;
            }
            let (trial_input, _, displacements) =
                displaced_input(&current_input, &lattice, &direction, displacement_limit)?;
            let directional_derivative = -dot(&force, &displacements);
            ensure!(
                directional_derivative.is_finite() && directional_derivative < 0.0,
                "relaxation direction is not an energy descent direction"
            );
            let trial_result = run_input(&trial_input, requested_backend)?;
            total_scf_evaluations += 1;
            attempted_line_evaluations += 1;
            if trial_result.converged
                && trial_result.total_energy_ry.unwrap()
                    <= current_energy + options.armijo_coefficient * directional_derivative
            {
                accepted = Some((
                    trial_input,
                    trial_result,
                    line_evaluation,
                    displacement_limit,
                ));
                break;
            }
            displacement_limit *= options.line_search_reduction;
        }
        let Some((trial_input, trial_result, line_evaluations, accepted_displacement)) = accepted
        else {
            unaccepted_terminal_scf_evaluations = attempted_line_evaluations;
            stop_reason = Some(format!(
                "line search failed at ionic step {step}; no step at or above {:.3e} bohr reduced the energy",
                options.minimum_step_bohr
            ));
            break;
        };

        let previous_force = force;
        let previous_direction = direction;
        current_input = trial_input;
        current_result = trial_result;
        force = project_translation(current_result.forces_ry_bohr.as_ref().unwrap());
        steps.push(step_record(
            step,
            &current_input,
            &current_result,
            line_evaluations,
            Some(accepted_displacement),
        )?);

        let denominator = dot(&previous_force, &previous_force);
        let difference: Vec<_> = force
            .iter()
            .zip(&previous_force)
            .map(|(current, previous)| {
                let mut value = [0.0; 3];
                for axis in 0..3 {
                    value[axis] = current[axis] - previous[axis];
                }
                value
            })
            .collect();
        let beta = if denominator > f64::MIN_POSITIVE {
            (dot(&force, &difference) / denominator).max(0.0)
        } else {
            0.0
        };
        direction = force
            .iter()
            .zip(&previous_direction)
            .map(|(current, previous)| {
                let mut value = [0.0; 3];
                for axis in 0..3 {
                    value[axis] = current[axis] + beta * previous[axis];
                }
                value
            })
            .collect();
        if dot(&force, &direction) <= 0.0 {
            direction = force.clone();
        }
        proposed_step = if line_evaluations == 1 {
            (accepted_displacement * 1.5).min(options.maximum_step_bohr)
        } else {
            accepted_displacement
        };
    }

    let final_raw_force = maximum_norm(current_result.forces_ry_bohr.as_ref().unwrap());
    let converged = final_raw_force <= options.force_tolerance_ry_bohr;
    if !converged && stop_reason.is_none() {
        stop_reason = Some(format!(
            "ionic relaxation did not converge in {} accepted steps (maximum force {:.3e} Ry/bohr, maximum projected force {:.3e} Ry/bohr; tolerance {:.3e})",
            options.max_steps,
            final_raw_force,
            maximum_norm(&force),
            options.force_tolerance_ry_bohr
        ));
    }
    let result = RelaxationResult {
        schema_version: RELAXATION_SCHEMA_VERSION,
        converged,
        requested_backend,
        atom_count: input.atoms.len(),
        options: options.clone(),
        total_scf_evaluations,
        unaccepted_terminal_scf_evaluations,
        steps,
        final_scf: current_result,
        stop_reason,
    };
    result.validate()?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn silicon_relaxation_options() -> RelaxationOptions {
        RelaxationOptions {
            max_steps: 12,
            force_tolerance_ry_bohr: 1.0e-4,
            initial_step_bohr: 0.05,
            maximum_step_bohr: 0.15,
            ..RelaxationOptions::default()
        }
    }

    fn assert_relaxed_silicon(result: &RelaxationResult) {
        result.validate().unwrap();
        assert!(result.converged);
        assert!(result.steps.len() >= 2);
        assert!(
            result.steps.last().unwrap().total_energy_ry
                < result.steps.first().unwrap().total_energy_ry - 5.0e-4
        );
        assert!(
            result.steps.last().unwrap().maximum_force_norm_ry_bohr
                <= result.options.force_tolerance_ry_bohr
        );
        let positions = &result.steps.last().unwrap().positions_fractional;
        for (&second, &first) in positions[1].iter().zip(&positions[0]) {
            let mut relative = second - first;
            relative -= relative.round();
            assert!(
                (relative - 0.25).abs() < 1.0e-7,
                "relative coordinate {relative:e}"
            );
        }
    }

    #[test]
    fn translation_projection_has_zero_sum() {
        let projected = project_translation(&[[1.0, -2.0, 0.5], [-0.5, 1.0, 0.25]]);
        for axis in 0..3 {
            assert!(projected.iter().map(|force| force[axis]).sum::<f64>().abs() < 1.0e-15);
        }
    }

    #[test]
    fn displacement_conversion_is_cartesian_for_a_skew_cell() {
        let input_path = std::path::Path::new("examples/si2.toml");
        if !input_path.exists() {
            return;
        }
        let input = SimulationInput::from_path(input_path).unwrap();
        let lattice = Lattice::new(input.cell.lattice_bohr).unwrap();
        let direction = [[0.0, 0.0, 0.0], [2.0, -1.0, 0.5]];
        let (trial, _, displacements) =
            displaced_input(&input, &lattice, &direction, 0.08).unwrap();
        let before = input.atoms[1].fractional;
        let after = trial.atoms[1].fractional;
        let mut recovered = [0.0; 3];
        for direct_axis in 0..3 {
            let mut delta = after[direct_axis] - before[direct_axis];
            delta -= delta.round();
            for (cartesian_axis, recovered_component) in recovered.iter_mut().enumerate() {
                *recovered_component += delta * lattice.direct[direct_axis][cartesian_axis];
            }
        }
        for axis in 0..3 {
            assert!((recovered[axis] - displacements[1][axis]).abs() < 1.0e-13);
        }
    }

    #[test]
    fn options_fail_closed() {
        let options = RelaxationOptions {
            line_search_reduction: 1.0,
            ..RelaxationOptions::default()
        };
        assert!(options.validate().is_err());
        let options = RelaxationOptions {
            initial_step_bohr: RelaxationOptions::default().maximum_step_bohr * 2.0,
            ..RelaxationOptions::default()
        };
        assert!(options.validate().is_err());
    }

    #[test]
    #[ignore = "multiple converged CPU SCFs; run explicitly in release mode"]
    fn cpu_relaxation_recovers_displaced_silicon_minimum() {
        let input_path = std::path::Path::new("examples/si2-displaced.toml");
        if !std::path::Path::new("pseudopotentials/Si.pz-vbc.UPF").exists() {
            return;
        }
        let input = SimulationInput::from_path(input_path).unwrap();
        let result = relax(&input, BackendKind::CpuF64, &silicon_relaxation_options()).unwrap();
        assert_relaxed_silicon(&result);
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    #[ignore = "multiple CPU and Metal SCFs; run explicitly in release mode"]
    fn metal_relaxation_matches_cpu_final_geometry() {
        let _device_guard = crate::metal::test_device_lock();
        let input_path = std::path::Path::new("examples/si2-displaced.toml");
        if !std::path::Path::new("pseudopotentials/Si.pz-vbc.UPF").exists() {
            return;
        }
        let input = SimulationInput::from_path(input_path).unwrap();
        let options = silicon_relaxation_options();
        let cpu = relax(&input, BackendKind::CpuF64, &options).unwrap();
        let metal = relax(&input, BackendKind::MetalMixed, &options).unwrap();
        assert_relaxed_silicon(&cpu);
        assert_relaxed_silicon(&metal);
        let cpu_positions = &cpu.steps.last().unwrap().positions_fractional;
        let metal_positions = &metal.steps.last().unwrap().positions_fractional;
        for (cpu_atom, metal_atom) in cpu_positions.iter().zip(metal_positions) {
            for axis in 0..3 {
                let mut difference = metal_atom[axis] - cpu_atom[axis];
                difference -= difference.round();
                assert!(difference.abs() < 2.0e-8, "position error {difference:e}");
            }
        }
    }
}
