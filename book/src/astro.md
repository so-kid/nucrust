# Astrophysical Reaction Rates

The `nucrust-astro` crate computes quantities relevant to stellar nucleosynthesis.

## Maxwellian-Averaged Cross Section (MACS)

MACS is the reaction cross section averaged over a Maxwell-Boltzmann energy distribution at temperature T:

⟨σ⟩(kT) = (2/√π) · (1/kT)² · ∫₀^∞ σ(E) · E · exp(−E/kT) dE

```rust
use nucrust::nucrust_astro::compute_macs;

let macs = compute_macs(&energies, &cross_sections, kt)?;
// Returns MACS in millibarns
```

Uses 20-point Gauss-Laguerre quadrature with cubic spline interpolation of the cross section.

## Reaction Rate

The thermonuclear reaction rate NA⟨σv⟩ is computed as a function of temperature:

```rust
use nucrust::nucrust_astro::compute_reaction_rate;

let temperatures = vec![0.1, 0.5, 1.0, 3.0, 10.0]; // in GK (10⁹ K)
let rates = compute_reaction_rate(&energies, &cross_sections, &temperatures)?;
```

## S-factor

The astrophysical S-factor removes the Coulomb barrier penetration factor for charged-particle reactions:

S(E) = σ(E) · E · exp(2πη)

```rust
use nucrust::nucrust_astro::compute_s_factor;

let s_factors = compute_s_factor(&energies, &cross_sections, z1, z2, reduced_mass);
```

## Stellar Enhancement Factor (SEF)

Accounts for contributions from thermally excited nuclear states:

```rust
use nucrust::nucrust_astro::compute_sef;

let sef = compute_sef(&ground_state_rate, &stellar_rate, &temperatures);
```

## REACLIB Output

Reaction rates can be fitted to the 7-parameter REACLIB format for use in reaction network codes (SkyNet, WinNet, XNet):

```rust
use nucrust::nucrust_data::reaclib::fit_reaclib_params;

let params = fit_reaclib_params(&temperatures, &rates)?;
// params.a0 through params.a6
```
