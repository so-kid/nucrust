# REACLIB

REACLIB is the standard format for thermonuclear reaction rates used by reaction network codes in astrophysics.

## Format

Each REACLIB entry contains a 7-parameter fit:

λ(T₉) = exp(a₀ + a₁/T₉ + a₂/T₉^(1/3) + a₃·T₉^(1/3) + a₄·T₉ + a₅·T₉^(5/3) + a₆·ln(T₉))

where T₉ is the temperature in units of 10⁹ K.

## Reading REACLIB

```rust
use nucrust::nucrust_data::reaclib::{parse_reaclib, ReaclibEntry};

let text = std::fs::read_to_string("rates.dat")?;
let entries = parse_reaclib(&text)?;

for entry in &entries {
    // Evaluate rate at T9 = 1.0 GK
    let rate = entry.evaluate(1.0);
    println!("{}: NA<σv> = {:.3e}", entry.label, rate);
}
```

## Writing REACLIB

Fit computed reaction rates to REACLIB format:

```rust
use nucrust::nucrust_data::reaclib::fit_reaclib_params;

let temperatures = vec![0.1, 0.5, 1.0, 3.0, 10.0]; // GK
let rates = vec![1e5, 2e6, 5e7, 1e8, 3e8];          // NA<σv>

let fit = fit_reaclib_params(&temperatures, &rates)?;
```

The fitting uses QR decomposition (via the `faer` crate) for robust least-squares fitting.

## Compatibility

REACLIB output from nucrust is compatible with:

- **pynucastro** — Python nuclear reaction rate library
- **SkyNet** — Nuclear reaction network solver
- **WinNet** — r-process reaction network
- **XNet** — Thermonuclear reaction network
