use crate::{Nuclide, Projectile};

/// Reaction channel (entrance or exit).
#[derive(Debug, Clone)]
pub struct Channel {
    /// Projectile (or ejectile) particle.
    pub projectile: Projectile,
    /// Target (or residual) nucleus.
    pub target: Nuclide,
    /// Q-value in MeV.
    pub q_value: f64,
}
