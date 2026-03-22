# Python Bindings

nucrust provides Python bindings via PyO3 and rust-numpy, exposing core calculation functions to Python.

## Installation

```bash
pip install maturin
cd crates/nucrust-python
maturin develop --release
```

## Available Functions

### `calc_transmission_coeffs(z, a, projectile, energies)`

Compute optical model transmission coefficients.

```python
import nucrust
import numpy as np

energies = np.logspace(-3, 1.5, 200)
result = nucrust.calc_transmission_coeffs(
    z=26, a=56,
    projectile="n",
    energies=energies,
)
```

### `calc_hf_cross_section(z, a, projectile, energies, nld, gsf)`

Compute Hauser-Feshbach cross sections.

```python
xs = nucrust.calc_hf_cross_section(
    z=26, a=56,
    projectile="n",
    energies=energies,
    nld="gilbert-cameron",
    gsf="eglo",
)
print(xs["sigma_total"])   # numpy array
print(xs["sigma_reaction"])
```

### `calc_macs(energies, cross_sections, kt)`

Compute Maxwellian-Averaged Cross Section at temperature kT.

```python
macs = nucrust.calc_macs(
    energies=energies,
    cross_sections=sigma,
    kt=0.0253,  # kT = 30 keV in MeV
)
```

### `fit_reaclib(temperatures, rates)`

Fit reaction rates to REACLIB 7-parameter format.

```python
params = nucrust.fit_reaclib(
    temperatures=temps,
    rates=na_sigma_v,
)
# params is a dict with keys "a0" through "a6"
```

## NumPy Integration

All array inputs and outputs use NumPy arrays. Input arrays are zero-copy when possible.
