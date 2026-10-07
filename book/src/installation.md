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

Features are defined on individual crates. The aggregate `nucrust` crate
forwards the CPU acceleration features (`parallel`, `simd`) to its
dependencies:

| Feature | Crate | Description | Default |
|---------|-------|-------------|---------|
| `parallel` | `nucrust-core` | rayon-based CPU parallelism | **on** |
| `parallel` | `nucrust-special`, `nucrust-optical`, `nucrust-hf`, `nucrust` | Energy-parallel hot loops (see [macOS / Apple Silicon](./macos.md)) | off |
| `simd` | `nucrust-special`, `nucrust` | SIMD-optimized Coulomb functions | off |
| `cuda` | `nucrust-gpu` | GPU acceleration (requires CUDA Toolkit / `nvcc` at build time) | off |
| `hdf5_io` | `nucrust-data` | HDF5 file I/O (requires HDF5 1.10.x) | off |

Build with specific features by targeting the crate that defines them:

```bash
cargo build --release -p nucrust --features "parallel simd"
cargo build --release -p nucrust-gpu --features cuda
cargo build --release -p nucrust-data --features hdf5_io
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
cargo build --release -p nucrust-gpu --features cuda
```

## Python Bindings

Requires Python 3.9+ and [maturin](https://www.maturin.rs/). The extension
is built against the stable ABI (`abi3`), so one wheel works on every
CPython ≥ 3.9. With [uv](https://docs.astral.sh/uv/):

```bash
cd crates/nucrust-python
uv venv && source .venv/bin/activate
uv pip install maturin numpy
maturin develop --release
```

## Verifying Installation

```bash
cargo test --workspace
# All features except `cuda` (which needs nvcc); this is what CI runs
cargo test --workspace --features nucrust-data/hdf5_io,nucrust/parallel,nucrust/simd
```
