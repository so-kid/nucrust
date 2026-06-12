# Quick Start

## CLI: Calculating Transmission Coefficients

Create a TOML configuration file `fe56_neutron.toml`:

```toml
projectile = "n"

[target]
z = 26
a = 56

[energy]
min = 0.001
max = 20.0
points = 200
spacing = "log"

[models]
omp = "koning-delaroche"
nld = "gilbert-cameron"
gsf = "eglo"

[output]
format = ["json", "table"]
path = "results"
```

Run the calculation:

```bash
nucrust calc --config fe56_neutron.toml
```

This computes neutron transmission coefficients for ⁵⁶Fe and writes results to
the output directory (`output/` by default; override with `--output <DIR>`).

## Rust API: Basic Usage

Add nucrust to your `Cargo.toml`:

```toml
[dependencies]
nucrust = { path = "path/to/nucrust/crates/nucrust" }
```

This example is kept compiling as `crates/nucrust/examples/quickstart.rs`;
run it with `cargo run -p nucrust --example quickstart`.

```rust,ignore
use nucrust::nucrust_core::{Nuclide, Projectile, Channel, EnergyGrid};
use nucrust::nucrust_core::backend::NumerovConfig;
use nucrust::cpu_backend::cpu_transmission_coeffs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define the reaction: n + ⁵⁶Fe
    let target = Nuclide::new(26, 56)?;
    let channel = Channel {
        projectile: Projectile::Neutron,
        target,
        q_value: 0.0,
    };

    // Energy grid: 0.001 to 20 MeV, 100 points (log spacing)
    let energies = EnergyGrid::logarithmic(0.001, 20.0, 100)?;

    // Compute transmission coefficients
    let config = NumerovConfig::default();
    let tc = cpu_transmission_coeffs(&channel, &energies, &config)?;

    println!("l_max = {}", tc.l_max);
    println!("T at first grid point = {:.6e}", tc.data[0]);

    Ok(())
}
```

## Python API

```python
import nucrust_python

# Compute neutron transmission coefficients (log energy grid)
energies, t_l0 = nucrust_python.calc_transmission_coeffs(
    z=26, a=56,
    e_min=0.001, e_max=10.0,
    n_energies=100,
)
print(f"T_l0 at first grid point = {t_l0[0]:.6e}")
```

See [Python Bindings](./python.md) for the full API.
