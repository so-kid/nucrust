# Python Bindings

nucrust provides Python bindings via PyO3 and rust-numpy, exposing core calculation functions to Python.

> **Note:** The current bindings are neutron-induced reactions only, with
> built-in default physics models (Koning-Delaroche optical potential,
> constant-temperature level density, standard Lorentzian γ-strength).
> Projectile and model selection will be exposed in a future release.

## Installation

```bash
pip install maturin
cd crates/nucrust-python
maturin develop --release
```

The importable module is named `nucrust_python`:

```python
import nucrust_python
```

## Classes

### `Nuclide(z, a)`

```python
nuc = nucrust_python.Nuclide(26, 56)
print(nuc.z, nuc.a, nuc.n)   # 26 56 30
```

### `CrossSection`

Returned by `calc_hf_cross_section`. Arrays are accessed via methods:

- `energies()` — energy grid (MeV) as a NumPy array
- `sigma_total()` — compound nucleus formation cross section (mb)
- `sigma_elastic()` — elastic cross section (mb)
- `sigma_reaction()` — reaction cross section (mb)
- `n_energies` — number of energy points (property)

### `ReactionRate`

Returned by `calc_macs`:

- `temperatures()` — temperature grid (GK) as a NumPy array
- `na_sigma_v()` — NA⟨σv⟩ values
- `macs()` — MACS values (mb), or `None` if unavailable

## Functions

### `calc_transmission_coeffs(z, a, e_min, e_max, n_energies, l_max=20)`

Compute neutron optical model transmission coefficients on a logarithmic
energy grid. Returns a tuple `(energies, t_l0)` of NumPy arrays, where
`t_l0` is the s-wave (l = 0) transmission coefficient.

```python
import nucrust_python

energies, t_l0 = nucrust_python.calc_transmission_coeffs(
    z=26, a=56,
    e_min=0.001, e_max=10.0,
    n_energies=200,
)
```

### `calc_hf_cross_section(z, a, e_min, e_max, n_energies, q_value=0.0)`

Compute Hauser-Feshbach cross sections for neutron capture. Returns a
`CrossSection` object.

```python
xs = nucrust_python.calc_hf_cross_section(
    z=26, a=56,
    e_min=0.001, e_max=1.0,
    n_energies=100,
    q_value=7.65,
)
print(xs.sigma_total())     # numpy array (mb)
print(xs.sigma_reaction())
```

### `calc_macs(z, a, e_min=0.001, e_max=1.0, n_energies=100, q_value=0.0)`

Compute the Maxwellian-averaged cross section and reaction rate. Internally
runs `calc_hf_cross_section` and integrates over the default temperature
grid. Returns a `ReactionRate` object.

```python
rate = nucrust_python.calc_macs(z=26, a=56, q_value=7.65)
print(rate.temperatures())  # GK
print(rate.macs())          # mb
```

### `fit_reaclib(temperatures, rates)`

Fit reaction rates to the REACLIB 7-parameter format. Returns a list of the
7 fitted coefficients `[a0, a1, ..., a6]`.

```python
coeffs = nucrust_python.fit_reaclib(
    temperatures=list(rate.temperatures()),
    rates=list(rate.na_sigma_v()),
)
a0, a1, a2, a3, a4, a5, a6 = coeffs
```

## NumPy Integration

All array outputs are NumPy arrays created via rust-numpy.
