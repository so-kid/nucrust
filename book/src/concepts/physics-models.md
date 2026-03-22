# Physics Models

nucrust uses trait-based abstractions for physics models, allowing different implementations to be swapped without changing the calculation logic.

## Optical Model Potentials

The `OpticalPotential` trait defines the nuclear+Coulomb potential for scattering calculations. Implementations:

- **Koning-Delaroche** — Global OMP for neutrons and protons (A ≥ 24, E ≤ 200 MeV)
- **McFadden-Satchler** — Alpha-particle OMP
- **Avrigeanu 2014** — Improved alpha-particle OMP

The optical potential determines transmission coefficients T_lj via Numerov integration of the radial Schrödinger equation.

## Nuclear Level Density (NLD)

The `LevelDensity` trait provides ρ(E, J, π) — the density of nuclear levels at excitation energy E with spin J and parity π. Models:

| Model | Description | Best for |
|-------|-------------|----------|
| Constant Temperature (CT) | Simple exponential | Quick estimates |
| Back-Shifted Fermi Gas (BSFG) | Fermi gas with pairing shift | Medium-mass nuclei |
| Gilbert-Cameron | CT at low E + BSFG at high E | General purpose |
| Ignatyuk | Energy-dependent *a* parameter | Near shell closures |
| HFB Table | Microscopic HFB tabulation | Exotic nuclei |

## Gamma-ray Strength Function (GSF)

The `GammaStrength` trait provides f_XL(E_γ) — the photon transmission strength. Models:

| Model | Description | Best for |
|-------|-------------|----------|
| Standard Lorentzian (SLO) | Single Lorentzian | E1 giant dipole resonance |
| Enhanced Generalized Lorentzian (EGLO) | Temperature-dependent width | General purpose |
| QRPA Table | Microscopic QRPA tabulation | Exotic nuclei |

## Choosing Models

For most applications with stable or near-stable nuclei:
- **OMP**: Koning-Delaroche (neutrons/protons) or Avrigeanu (alphas)
- **NLD**: Gilbert-Cameron
- **GSF**: EGLO

For exotic nuclei far from stability, microscopic models (HFB table, QRPA table) are recommended.
