# HDF5

HDF5 output is available with the `hdf5_io` feature flag. Requires HDF5 1.10.x.

## Cross Section Output

```
/energy           [n_e]    f64   Energy grid (MeV)
/sigma_total      [n_e]    f64   Total cross section (mb)
/sigma_elastic    [n_e]    f64   Elastic cross section (mb)
/sigma_reaction   [n_e]    f64   Reaction cross section (mb)
```

## Reaction Rate Output

```
/rate_0/
  /temperatures   [n_t]    f64   Temperature grid (GK)
  /na_sigma_v     [n_t]    f64   NA<σv> (cm³/s/mol)
  /macs           [n_t]    f64   MACS (mb) [optional]
  /s_factor       [n_t]    f64   S-factor (MeV·b) [optional]
  /sef            [n_t]    f64   Stellar enhancement factor [optional]
```

## Usage

```rust
use nucrust::nucrust_data::hdf5_io::{write_cross_section, read_cross_section};
use std::path::Path;

// Write
write_cross_section(Path::new("xs.h5"), &cross_section)?;

// Read
let xs = read_cross_section(Path::new("xs.h5"))?;
```

## Requirements

- HDF5 1.10.x (not 2.x due to `hdf5-metno` crate compatibility)
- Set `HDF5_DIR` environment variable if not in default path
