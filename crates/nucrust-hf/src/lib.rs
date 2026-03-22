//! Hauser-Feshbach statistical model for nuclear reaction cross sections.
//!
//! This crate provides the core Hauser-Feshbach (HF) computation, nuclear level
//! density (NLD) models, gamma-ray strength functions (GSF), width fluctuation
//! corrections (WFC), and multi-particle emission cascade calculations.

#![warn(missing_docs)]

pub mod cascade;
pub mod gsf;
pub mod hf;
pub mod nld;
pub mod wfc;

pub use gsf::{EnhancedGeneralizedLorentzian, QrpaTableInterp, StandardLorentzian};
pub use hf::{hauser_feshbach, DiscreteLevelInfo, DiscreteLevels, HfCalculation};
pub use nld::{BackShiftedFermiGas, ConstantTemperature, GilbertCameron, HfbTableInterp, Ignatyuk};
pub use wfc::{goe_wfc, moldauer_wfc, GoeResult, MoldauerResult};
