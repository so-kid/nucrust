# Configuration

nucrust uses TOML configuration files for job specification.

## Full Configuration Reference

```toml
# Projectile: "n" (neutron), "p" (proton), "alpha"
projectile = "n"

# Target nuclide
[target]
z = 26   # proton number
a = 56   # mass number

# Energy grid
[energy]
min = 0.001      # minimum energy (MeV)
max = 20.0       # maximum energy (MeV)
points = 200     # number of energy points
spacing = "log"  # "log" or "linear"

# Physics models (all optional, shown with defaults)
[models]
omp = "koning-delaroche"   # optical model potential
nld = "gilbert-cameron"    # nuclear level density
gsf = "eglo"               # gamma-ray strength function
ripl3_path = "/path/to/ripl3"  # RIPL-3 data directory (optional)

# Output settings (optional)
[output]
format = ["json"]   # "json", "table", "hdf5", "reaclib"
path = "output"     # output directory

# Backend settings (optional)
[backend]
compute = "cpu"     # "cpu" or "gpu"
gpu_device = 0      # GPU device index (0-based)
```

## Model Options

### Optical Model Potentials (`omp`)

| Value | Description |
|-------|-------------|
| `"koning-delaroche"` | Koning-Delaroche global OMP for neutrons and protons (default) |
| `"mcfadden-satchler"` | McFadden-Satchler alpha-particle OMP |
| `"avrigeanu"` | Avrigeanu 2014 alpha-particle OMP |

### Nuclear Level Density (`nld`)

| Value | Description |
|-------|-------------|
| `"ct"` | Constant Temperature model |
| `"bsfg"` | Back-Shifted Fermi Gas model |
| `"gilbert-cameron"` | Gilbert-Cameron composite model (default) |
| `"ignatyuk"` | Ignatyuk energy-dependent model |
| `"hfb-table"` | HFB microscopic table interpolation |

### Gamma-ray Strength Function (`gsf`)

| Value | Description |
|-------|-------------|
| `"slo"` | Standard Lorentzian |
| `"eglo"` | Enhanced Generalized Lorentzian (default) |
| `"qrpa-table"` | QRPA microscopic table interpolation |

## Minimal Configuration

Only the target, projectile, and energy grid are required:

```toml
projectile = "n"

[target]
z = 6
a = 12

[energy]
min = 0.01
max = 10.0
points = 100
```

All other settings use sensible defaults.
