//! Numerically explicit building blocks for a small plane-wave DFT solver.
//!
//! Unsupported physics is rejected at input validation.  In particular, this
//! crate currently accepts only fixed-cell, spin-unpolarized, Gamma-point,
//! norm-conserving LDA-PZ silicon calculations.

pub mod backend;
pub mod checkpoint;
pub mod eigensolver;
pub mod fft;
pub mod fixtures;
pub mod input;
pub mod lattice;
pub mod metal;
pub mod operator;
pub mod provenance;
pub mod pseudo;
pub mod result;
pub mod scf;
pub mod xc;

pub use backend::{BackendKind, KohnShamBackend, PrecisionPolicy};
pub use input::SimulationInput;
pub use result::ScfResult;
pub use xc::XcFunctional;
