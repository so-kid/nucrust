# Nuclides and Channels

## Nuclide

A `Nuclide` represents a nucleus with proton number Z and mass number A:

```rust,ignore
use nucrust::nucrust_core::Nuclide;

let fe56 = Nuclide::new(26, 56).unwrap();
assert_eq!(fe56.z(), 26);
assert_eq!(fe56.a(), 56);
assert_eq!(fe56.n(), 30);  // neutron number
```

Valid ranges: Z = 0–118, A = 1–350, with N = A − Z ≥ 0.

## Projectile

The `Projectile` enum covers all supported incident particles:

| Variant | Particle | Spin | Charge |
|---------|----------|------|--------|
| `Neutron` | n | 1/2 | 0 |
| `Proton` | p | 1/2 | 1 |
| `Deuteron` | d | 1 | 1 |
| `Triton` | t | 1/2 | 1 |
| `Helion` | ³He | 1/2 | 2 |
| `Alpha` | α | 0 | 2 |
| `Gamma` | γ | 1 | 0 |

Each variant provides mass, spin, and charge via methods.

## Channel

A `Channel` combines a projectile with a target nuclide:

```rust,ignore
use nucrust::nucrust_core::{Channel, Nuclide, Projectile};

let channel = Channel {
    projectile: Projectile::Neutron,
    target: Nuclide::new(26, 56).unwrap(),
    q_value: 7.646,  // Q-value in MeV
};
```

## SpinParity

Quantum numbers use half-integer representation (2J stored as integer):

```rust,ignore
use nucrust::nucrust_core::{Parity, SpinParity};

let jp = SpinParity::new(5, Parity::Negative)?;  // J = 5/2⁻ (2J = 5)
assert_eq!(jp.two_j, 5);
let j = jp.two_j as f64 / 2.0;                   // 2.5
```

The associated function `SpinParity::triangle_condition(two_j1, two_j2, two_j3)`
checks angular momentum coupling validity.
