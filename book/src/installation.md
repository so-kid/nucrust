# Installation

## Requirements

- **Rust** 1.75 or later (2024 edition)
- **HDF5 1.10.x** (optional, for HDF5 I/O feature)
- **CUDA Toolkit** (optional, for GPU acceleration)

## Building from Source

```bash
git clone https://github.com/nucrust/nucrust.git
cd nucrust
cargo build --workspace --release
```

The CLI binary will be at `target/release/nucrust`.

## Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `cuda` | Enable GPU acceleration (requires CUDA) | off |
| `hdf5_io` | Enable HDF5 file I/O | off |
| `simd` | Enable SIMD-optimized Coulomb functions | off |

Build with specific features:

```bash
cargo build --release --features cuda
cargo build --release --features hdf5_io
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
