# Installation

## Requirements

- **Rust** 1.75 or later (2021 edition)
- **HDF5 1.10.x** (optional, for HDF5 I/O feature)
- **CUDA Toolkit** (optional, for GPU acceleration)

## Building from Source

```bash
git clone https://github.com/so-kid/nucrust.git
cd nucrust
cargo build --workspace --release
```

The CLI binary will be at `target/release/nucrust`.

## Feature Flags

Features are defined on individual crates (the aggregate `nucrust` crate has
no feature flags of its own):

| Feature | Crate | Description | Default |
|---------|-------|-------------|---------|
| `parallel` | `nucrust-core` | rayon-based CPU parallelism | **on** |
| `cuda` | `nucrust-gpu` | GPU acceleration (requires CUDA) | off |
| `hdf5_io` | `nucrust-data` | HDF5 file I/O | off |
| `simd` | `nucrust-special` | SIMD-optimized Coulomb functions | off |

Build with specific features by targeting the crate that defines them:

```bash
cargo build --release -p nucrust-gpu --features cuda
cargo build --release -p nucrust-data --features hdf5_io
cargo test --workspace --all-features    # everything at once
```

## HDF5 Setup

nucrust requires HDF5 1.10.x (not 2.x). On macOS with Homebrew:

```bash
brew install hdf5@1.10
```

Set the `HDF5_DIR` environment variable, or add to `.cargo/config.toml` (not committed):

```toml
[env]
HDF5_DIR = "/opt/homebrew/opt/hdf5@1.10"
```

## GPU Setup (CUDA)

Requires CUDA Toolkit 12.0+ and an NVIDIA GPU with compute capability 7.0+.

```bash
cargo build --release --features cuda
```

## Python Bindings

Requires Python 3.8+ and [maturin](https://www.maturin.rs/):

```bash
pip install maturin
cd crates/nucrust-python
maturin develop --release
```

## Verifying Installation

```bash
cargo test --workspace
cargo test --workspace --all-features  # full validation
```
