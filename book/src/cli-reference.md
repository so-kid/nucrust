# CLI Reference

## Synopsis

```
nucrust <COMMAND> [OPTIONS]
```

## Commands

### `calc`

Calculate cross sections and reaction rates for a single target.

```bash
nucrust calc --config <PATH> [--backend cpu|gpu] [--format json,table] [--output <DIR>]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-c, --config` | (required) | Path to TOML configuration file |
| `--backend` | `cpu` | Compute backend |
| `--format` | `json` | Output formats (comma-separated) |
| `-o, --output` | `output` | Output directory |

### `batch`

Batch calculation across multiple nuclides.

```bash
nucrust batch --config <PATH> [--backend cpu|gpu]
```

### `fit`

R-matrix parameter fitting.

```bash
nucrust fit --config <PATH> [--method lm|mcmc]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--method` | `lm` | Fitting method: Levenberg-Marquardt or MCMC |

### `info`

Display nuclide information.

```bash
nucrust info <NUCLIDE> [--ripl3 <PATH>]
```

Example: `nucrust info Fe56`

### `export`

Convert between output formats.

```bash
nucrust export --input <PATH> --format <FORMAT> [--output <PATH>]
```

Supported formats: `json`, `table`, `hdf5`, `reaclib`.
