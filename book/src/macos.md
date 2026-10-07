# macOS / Apple Silicon Optimization

CUDA is not available on macOS, so on Mac the CPU path is the performance
path. nucrust provides two opt-in features that together use all CPU cores
and the NEON SIMD units of Apple M-series chips.

## TL;DR

```bash
# Fastest CPU configuration on an Apple Silicon Mac
cargo build --release -p nucrust --features "parallel simd"
cargo test  -p nucrust --features "parallel simd"
```

## `parallel` — multi-core via rayon

The `parallel` feature distributes the embarrassingly parallel hot loops
across all cores with [rayon](https://crates.io/crates/rayon):

| Crate | Parallelized work |
|-------|-------------------|
| `nucrust-optical` | Numerov integration per energy point (`compute_transmission_coeffs`) |
| `nucrust-hf` | Hauser-Feshbach J/π summation per energy point (`hauser_feshbach`) |
| `nucrust-special` | Batch Coulomb wave functions (`coulomb_wave_batch`) |

Each work item is independent, so results are bit-identical to the
sequential path. On a 10-core M-series chip, transmission-coefficient grids
and HF excitation functions speed up roughly in proportion to the number of
performance cores.

rayon sizes its thread pool from `std::thread::available_parallelism`,
which on macOS counts both performance and efficiency cores. To restrict
work to a fixed number of threads (e.g. only P-cores), set:

```bash
RAYON_NUM_THREADS=8 cargo run --release -p nucrust --features parallel -- calc ...
```

## `simd` — NEON vectorization

The `simd` feature batches Coulomb continued-fraction evaluation with
[`wide`](https://crates.io/crates/wide)'s `f64x4`. On `aarch64-apple-darwin`
this lowers to pairs of 128-bit NEON `float64x2` operations; no
Apple-specific code is required.

## Codegen tuning

`.cargo/config.toml` is reserved for local, uncommitted configuration in
this project (see [Installation](./installation.md)). To tune codegen for
Apple Silicon, add to your local `.cargo/config.toml`:

```toml
[target.aarch64-apple-darwin]
rustflags = ["-C", "target-cpu=apple-m1"]
```

`apple-m1` is the baseline for every Apple Silicon Mac and keeps
cross-compilation from other hosts working. If you build only for your own
machine, `-C target-cpu=native` selects the newest CPU you have:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release -p nucrust --features "parallel simd"
```

## HDF5 on macOS

The optional `hdf5_io` feature of `nucrust-data` needs HDF5 1.10.x. See
[Installation](./installation.md) for the Homebrew setup. All other
features, including `parallel` and `simd`, have no system dependencies.

## CI

The `Test (macOS Apple Silicon)` job in `.github/workflows/ci.yml` runs the
full test suite plus the `parallel`/`simd` feature combinations on GitHub's
`macos-latest` (arm64) runners on every push.
