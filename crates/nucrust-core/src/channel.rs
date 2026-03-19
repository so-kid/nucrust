use crate::{Nuclide, Projectile};

/// Reaction channel (entrance or exit).
#[derive(Debug, Clone)]
pub struct Channel {
    pub projectile: Projectile,
    pub target: Nuclide,
    /// Q-value in MeV.
    pub q_value: f64,
}
