use crate::backend::BackendKind;
use crate::input::SimulationInput;
use crate::lattice::Lattice;
use crate::provenance::{git_revision, sha256};
use crate::relax_checkpoint::{self, RELAXATION_CHECKPOINT_SCHEMA_VERSION, RelaxationCheckpoint};
use crate::result::ScfResult;
use crate::scf::{input_sha256, run_input};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::f64::consts::TAU;
use std::path::PathBuf;
use std::time::Instant;

pub const RELAXATION_SCHEMA_VERSION: u32 = 2;
const MINIMUM_RELAXATION_SCHEMA_VERSION: u32 = 1;

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RelaxationRestartProvenance {
    pub checkpoint_sha256: String,
    pub checkpoint_step: usize,
    pub writer_git_revision: String,
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
    #[serde(default)]
    pub elapsed_seconds: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub restart: Option<RelaxationRestartProvenance>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
}

impl RelaxationResult {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (MINIMUM_RELAXATION_SCHEMA_VERSION..=RELAXATION_SCHEMA_VERSION)
                .contains(&self.schema_version),
            "unsupported relaxation schema_version {}; expected {} through {}",
            self.schema_version,
            MINIMUM_RELAXATION_SCHEMA_VERSION,
            RELAXATION_SCHEMA_VERSION
        );
        ensure!(
            self.elapsed_seconds.is_finite() && self.elapsed_seconds >= 0.0,
            "relaxation elapsed time is invalid"
        );
        if self.schema_version == 1 {
            ensure!(
                self.restart.is_none(),
                "relaxation schema 1 cannot contain restart provenance"
            );
        }
        if let Some(restart) = &self.restart {
            ensure!(
                restart.checkpoint_sha256.len() == 64
                    && restart
                        .checkpoint_sha256
                        .bytes()
                        .all(|byte| byte.is_ascii_hexdigit())
                    && !restart.writer_git_revision.is_empty()
                    && restart.checkpoint_step < self.steps.len()
                    && restart.writer_git_revision == self.final_scf.provenance.git_revision,
                "relaxation restart provenance is invalid"
            );
        }
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

#[derive(Debug, Clone)]
pub struct RelaxationControl {
    pub resume_from: Option<PathBuf>,
    pub checkpoint_output: Option<PathBuf>,
    pub checkpoint_every: usize,
    pub stop_after_accepted_steps: Option<usize>,
}

fn pseudopotential_hashes(input: &SimulationInput) -> Result<BTreeMap<String, String>> {
    input
        .pseudopotentials
        .iter()
        .map(|(element, path)| Ok((element.clone(), sha256(path)?)))
        .collect()
}

fn vectors_are_finite(vectors: &[[f64; 3]]) -> bool {
    vectors.iter().flatten().all(|value| value.is_finite())
}

fn validate_checkpoint(
    checkpoint: &RelaxationCheckpoint,
    initial_input: &SimulationInput,
    initial_input_hash: &str,
    pseudopotential_sha256: &BTreeMap<String, String>,
    requested_backend: BackendKind,
    options: &RelaxationOptions,
) -> Result<()> {
    ensure!(
        checkpoint.schema_version == RELAXATION_CHECKPOINT_SCHEMA_VERSION,
        "unsupported relaxation checkpoint schema_version {}",
        checkpoint.schema_version
    );
    ensure!(
        checkpoint.writer_git_revision == git_revision(),
        "relaxation checkpoint was written by git revision {}, but the current revision is {}",
        checkpoint.writer_git_revision,
        git_revision()
    );
    ensure!(
        checkpoint.initial_input_sha256 == initial_input_hash,
        "relaxation checkpoint initial input hash differs"
    );
    ensure!(
        &checkpoint.pseudopotential_sha256 == pseudopotential_sha256,
        "relaxation checkpoint pseudopotential hashes differ"
    );
    ensure!(
        checkpoint.requested_backend == requested_backend,
        "relaxation checkpoint backend differs"
    );
    ensure!(
        &checkpoint.options == options,
        "relaxation checkpoint options differ"
    );
    ensure!(
        checkpoint.elapsed_seconds.is_finite() && checkpoint.elapsed_seconds >= 0.0,
        "relaxation checkpoint elapsed time is invalid"
    );
    ensure!(
        checkpoint.accepted_steps.checked_add(1) == Some(checkpoint.history.len())
            && checkpoint.accepted_steps <= options.max_steps,
        "relaxation checkpoint history length is inconsistent"
    );
    ensure!(
        checkpoint.total_scf_evaluations
            == checkpoint
                .history
                .iter()
                .map(|step| step.scf_evaluations)
                .sum::<usize>(),
        "relaxation checkpoint SCF evaluation count is inconsistent"
    );
    let atom_count = initial_input.atoms.len();
    ensure!(
        checkpoint.current_input.atoms.len() == atom_count
            && checkpoint.direction.len() == atom_count
            && checkpoint.projected_force.len() == atom_count
            && vectors_are_finite(&checkpoint.direction)
            && vectors_are_finite(&checkpoint.projected_force),
        "relaxation checkpoint vector dimensions or values are invalid"
    );
    ensure!(
        checkpoint.proposed_step_bohr.is_finite()
            && checkpoint.proposed_step_bohr >= options.minimum_step_bohr
            && checkpoint.proposed_step_bohr <= options.maximum_step_bohr,
        "relaxation checkpoint proposed step is invalid"
    );
    checkpoint.current_input.validate()?;
    checkpoint.current_scf.validate()?;
    ensure!(
        checkpoint.current_scf.converged,
        "relaxation checkpoint current SCF is not converged"
    );
    ensure!(
        checkpoint.current_scf.provenance.requested_backend == requested_backend,
        "relaxation checkpoint current SCF backend is inconsistent"
    );
    ensure!(
        checkpoint.current_scf.provenance.pseudopotential_sha256 == *pseudopotential_sha256,
        "relaxation checkpoint current SCF pseudopotential provenance is inconsistent"
    );
    let final_force = maximum_norm(
        checkpoint
            .current_scf
            .forces_ry_bohr
            .as_ref()
            .context("relaxation checkpoint current SCF has no forces")?,
    );
    let converged = final_force <= options.force_tolerance_ry_bohr;
    RelaxationResult {
        schema_version: RELAXATION_SCHEMA_VERSION,
        converged,
        requested_backend,
        atom_count,
        options: options.clone(),
        total_scf_evaluations: checkpoint.total_scf_evaluations,
        unaccepted_terminal_scf_evaluations: 0,
        steps: checkpoint.history.clone(),
        final_scf: checkpoint.current_scf.clone(),
        elapsed_seconds: checkpoint.elapsed_seconds,
        restart: None,
        stop_reason: (!converged).then(|| "checkpoint is not ionically converged".into()),
    }
    .validate()
    .context("relaxation checkpoint history is invalid")?;
    let mut reconstructed_initial = checkpoint.current_input.clone();
    reconstructed_initial.atoms = initial_input.atoms.clone();
    ensure!(
        serde_json::to_value(&reconstructed_initial)? == serde_json::to_value(initial_input)?,
        "relaxation checkpoint changed immutable simulation input fields"
    );
    let current_forces = checkpoint.current_scf.forces_ry_bohr.as_ref().unwrap();
    let expected_projected = project_translation(current_forces);
    ensure!(
        expected_projected
            .iter()
            .flatten()
            .zip(checkpoint.projected_force.iter().flatten())
            .all(|(expected, actual)| (expected - actual).abs() <= 1.0e-15),
        "relaxation checkpoint projected force is inconsistent"
    );
    for (index, step) in checkpoint.history.iter().enumerate() {
        ensure!(
            step.step == index,
            "relaxation checkpoint history is not sequential"
        );
    }
    let last = checkpoint
        .history
        .last()
        .context("relaxation checkpoint history is empty")?;
    ensure!(
        last.step == checkpoint.accepted_steps
            && last.positions_fractional
                == checkpoint
                    .current_input
                    .atoms
                    .iter()
                    .map(|atom| atom.fractional)
                    .collect::<Vec<_>>()
            && Some(&last.forces_ry_bohr) == checkpoint.current_scf.forces_ry_bohr.as_ref()
            && Some(last.total_energy_ry) == checkpoint.current_scf.total_energy_ry,
        "relaxation checkpoint current state differs from its history"
    );
    ensure!(
        last.scf_input_sha256 == checkpoint.current_scf.provenance.input_sha256,
        "relaxation checkpoint current SCF input hash differs from its history"
    );
    Ok(())
}

impl RelaxationControl {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.checkpoint_every > 0,
            "relaxation checkpoint interval must be positive"
        );
        ensure!(
            self.stop_after_accepted_steps.is_none_or(|steps| steps > 0),
            "relaxation stop-after count must be positive"
        );
        ensure!(
            self.stop_after_accepted_steps.is_none() || self.checkpoint_output.is_some(),
            "pausing relaxation requires a checkpoint output path"
        );
        Ok(())
    }
}

impl Default for RelaxationControl {
    fn default() -> Self {
        Self {
            resume_from: None,
            checkpoint_output: None,
            checkpoint_every: 1,
            stop_after_accepted_steps: None,
        }
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
    relax_with_control(
        input,
        requested_backend,
        options,
        &RelaxationControl::default(),
    )
}

pub fn relax_with_control(
    input: &SimulationInput,
    requested_backend: BackendKind,
    options: &RelaxationOptions,
    control: &RelaxationControl,
) -> Result<RelaxationResult> {
    input.validate()?;
    options.validate()?;
    control.validate()?;
    ensure!(input.cell.fixed, "ionic relaxation requires a fixed cell");
    let lattice = Lattice::new(input.cell.lattice_bohr)?;
    let initial_input_hash = input_sha256(input)?;
    let pseudopotential_sha256 = pseudopotential_hashes(input)?;
    let invocation_started = Instant::now();
    let mut restart = None;
    let (
        mut current_input,
        mut current_result,
        mut total_scf_evaluations,
        mut steps,
        mut force,
        mut direction,
        mut proposed_step,
        accepted_steps_at_start,
        prior_elapsed_seconds,
    ) = if let Some(path) = &control.resume_from {
        let loaded = relax_checkpoint::read(path)?;
        validate_checkpoint(
            &loaded.checkpoint,
            input,
            &initial_input_hash,
            &pseudopotential_sha256,
            requested_backend,
            options,
        )?;
        restart = Some(RelaxationRestartProvenance {
            checkpoint_sha256: loaded.file_sha256,
            checkpoint_step: loaded.checkpoint.accepted_steps,
            writer_git_revision: loaded.checkpoint.writer_git_revision.clone(),
        });
        let checkpoint = loaded.checkpoint;
        (
            checkpoint.current_input,
            checkpoint.current_scf,
            checkpoint.total_scf_evaluations,
            checkpoint.history,
            checkpoint.projected_force,
            checkpoint.direction,
            checkpoint.proposed_step_bohr,
            checkpoint.accepted_steps,
            checkpoint.elapsed_seconds,
        )
    } else {
        let current_input = input.clone();
        let current_result = run_input(&current_input, requested_backend)?;
        ensure!(
            current_result.converged,
            "initial electronic SCF did not converge: {}",
            current_result
                .stop_reason
                .as_deref()
                .unwrap_or("unknown reason")
        );
        let force = project_translation(current_result.forces_ry_bohr.as_ref().unwrap());
        let direction = force.clone();
        (
            current_input.clone(),
            current_result.clone(),
            1,
            vec![step_record(0, &current_input, &current_result, 1, None)?],
            force,
            direction,
            options.initial_step_bohr,
            0,
            0.0,
        )
    };
    let mut stop_reason = None;
    let mut unaccepted_terminal_scf_evaluations = 0_usize;
    let mut paused = false;

    for (accepted_step_index, step) in
        ((accepted_steps_at_start + 1)..=options.max_steps).enumerate()
    {
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
        let accepted_steps_this_run = accepted_step_index + 1;

        let current_is_converged = maximum_norm(current_result.forces_ry_bohr.as_ref().unwrap())
            <= options.force_tolerance_ry_bohr;
        let pause_requested = !current_is_converged
            && control
                .stop_after_accepted_steps
                .is_some_and(|limit| accepted_steps_this_run >= limit);
        let checkpoint_due = control.checkpoint_output.is_some()
            && (step % control.checkpoint_every == 0 || pause_requested);
        if checkpoint_due {
            let checkpoint = RelaxationCheckpoint {
                schema_version: RELAXATION_CHECKPOINT_SCHEMA_VERSION,
                writer_git_revision: git_revision(),
                initial_input_sha256: initial_input_hash.clone(),
                pseudopotential_sha256: pseudopotential_sha256.clone(),
                requested_backend,
                options: options.clone(),
                accepted_steps: step,
                current_input: current_input.clone(),
                current_scf: current_result.clone(),
                direction: direction.clone(),
                projected_force: force.clone(),
                proposed_step_bohr: proposed_step,
                total_scf_evaluations,
                history: steps.clone(),
                elapsed_seconds: prior_elapsed_seconds + invocation_started.elapsed().as_secs_f64(),
            };
            relax_checkpoint::write_atomic(
                control.checkpoint_output.as_deref().unwrap(),
                &checkpoint,
            )?;
        }
        if pause_requested {
            paused = true;
            stop_reason = Some(format!(
                "ionic relaxation paused after accepted step {step}; resume from the written checkpoint"
            ));
            break;
        }
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
        elapsed_seconds: prior_elapsed_seconds + invocation_started.elapsed().as_secs_f64(),
        restart,
        stop_reason,
    };
    ensure!(
        !paused || !result.converged,
        "a converged relaxation cannot be marked as paused"
    );
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

        let control = RelaxationControl {
            checkpoint_every: 0,
            ..RelaxationControl::default()
        };
        assert!(control.validate().is_err());
        let control = RelaxationControl {
            stop_after_accepted_steps: Some(1),
            ..RelaxationControl::default()
        };
        assert!(control.validate().is_err());
    }

    fn checkpoint_resume_options() -> RelaxationOptions {
        RelaxationOptions {
            max_steps: 12,
            force_tolerance_ry_bohr: 1.0e-4,
            initial_step_bohr: 0.01,
            maximum_step_bohr: 0.01,
            ..RelaxationOptions::default()
        }
    }

    fn assert_checkpoint_resume_equivalence(backend: BackendKind) {
        let input_path = std::path::Path::new("examples/si2-displaced.toml");
        if !std::path::Path::new("pseudopotentials/Si.pz-vbc.UPF").exists() {
            return;
        }
        let input = SimulationInput::from_path(input_path).unwrap();
        let options = checkpoint_resume_options();
        let uninterrupted = relax(&input, backend, &options).unwrap();
        assert!(uninterrupted.converged);
        assert!(uninterrupted.steps.len() > 2);

        let checkpoint_path = std::env::temp_dir().join(format!(
            "rust-pw-dft-relax-resume-{}-{backend:?}.bin",
            std::process::id()
        ));
        let paused = relax_with_control(
            &input,
            backend,
            &options,
            &RelaxationControl {
                checkpoint_output: Some(checkpoint_path.clone()),
                stop_after_accepted_steps: Some(1),
                ..RelaxationControl::default()
            },
        )
        .unwrap();
        assert!(!paused.converged);
        assert_eq!(paused.steps.len(), 2);
        assert!(checkpoint_path.exists());

        let mut modified_input = input.clone();
        modified_input.atoms[1].fractional[0] += 1.0e-3;
        let resume_control = RelaxationControl {
            resume_from: Some(checkpoint_path.clone()),
            checkpoint_output: Some(checkpoint_path.clone()),
            ..RelaxationControl::default()
        };
        assert!(
            relax_with_control(&modified_input, backend, &options, &resume_control)
                .unwrap_err()
                .to_string()
                .contains("initial input hash differs")
        );
        let modified_options = RelaxationOptions {
            max_steps: options.max_steps + 1,
            ..options.clone()
        };
        assert!(
            relax_with_control(&input, backend, &modified_options, &resume_control)
                .unwrap_err()
                .to_string()
                .contains("options differ")
        );

        let resumed = relax_with_control(&input, backend, &options, &resume_control).unwrap();
        assert!(resumed.converged);
        assert_eq!(resumed.steps, uninterrupted.steps);
        assert_eq!(
            resumed.total_scf_evaluations,
            uninterrupted.total_scf_evaluations
        );
        assert_eq!(
            resumed.final_scf.total_energy_ry,
            uninterrupted.final_scf.total_energy_ry
        );
        assert_eq!(
            resumed.final_scf.forces_ry_bohr,
            uninterrupted.final_scf.forces_ry_bohr
        );
        assert_eq!(resumed.restart.as_ref().unwrap().checkpoint_step, 1);
        std::fs::remove_file(checkpoint_path).unwrap();
    }

    #[test]
    #[ignore = "multiple converged CPU SCFs; run explicitly in release mode"]
    fn interrupted_cpu_relaxation_resumes_equivalently_and_rejects_mismatch() {
        assert_checkpoint_resume_equivalence(BackendKind::CpuF64);
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

        let strict_options = RelaxationOptions {
            max_steps: 1,
            force_tolerance_ry_bohr: 1.0e-8,
            ..silicon_relaxation_options()
        };
        let incomplete = relax(&input, BackendKind::CpuF64, &strict_options).unwrap();
        incomplete.validate().unwrap();
        assert!(!incomplete.converged);
        assert!(
            incomplete
                .stop_reason
                .as_deref()
                .unwrap()
                .contains("did not converge in 1 accepted steps")
        );
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

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    #[ignore = "multiple Metal SCFs; run explicitly in release mode"]
    fn interrupted_metal_relaxation_resumes_equivalently_and_rejects_mismatch() {
        let _device_guard = crate::metal::test_device_lock();
        assert_checkpoint_resume_equivalence(BackendKind::MetalMixed);
    }
}
