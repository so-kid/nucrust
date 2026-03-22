# Energy Grids

## EnergyGrid

The `EnergyGrid` type ensures energy values are sorted and strictly positive:

```rust
use nucrust::nucrust_core::EnergyGrid;

// Logarithmic spacing (recommended for cross sections)
let log_grid = EnergyGrid::logarithmic(0.001, 20.0, 200)?;

// Linear spacing
let lin_grid = EnergyGrid::linear(0.1, 10.0, 100)?;

// Custom values
let custom = EnergyGrid::from_values(vec![0.01, 0.1, 1.0, 5.0, 10.0])?;
```

Logarithmic spacing is recommended for cross section calculations, as nuclear cross sections typically vary over many orders of magnitude.

## CubicSpline

For interpolation between grid points, nucrust provides natural cubic splines:

```rust
use nucrust::nucrust_core::spline::CubicSpline;

let x = vec![0.0, 1.0, 2.0, 3.0];
let y = vec![0.0, 1.0, 4.0, 9.0];
let spline = CubicSpline::new(&x, &y).unwrap();

let y_interp = spline.evaluate(1.5);
```

Splines are used throughout nucrust for interpolating transmission coefficients, level densities, and cross sections at arbitrary energies.
