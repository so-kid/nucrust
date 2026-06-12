# CLI Reference

## Synopsis

```text
nucrust <COMMAND> [OPTIONS]
```

## Implementation Status

| Command | Status |
|---------|--------|
| `calc` | Implemented (transmission coefficients; `json` / `table` output) |
| `batch` | Defined, not yet implemented |
| `fit` | Defined, not yet implemented |
| `info` | Defined, not yet implemented |
| `export` | Defined, not yet implemented |

## Commands

### `calc`

Calculate cross sections and reaction rates for a single target. Currently
computes transmission coefficients on the configured energy grid and writes
them to the output directory.

```bash
nucrust calc --config <PATH> [--backend cpu|gpu] [--format json,table] [--output <DIR>]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-c, --config` | (required) | Path to TOML configuration file |
| `--backend` | `cpu` | Compute backend (`cpu` or `gpu`) |
| `--format` | `json` | Output formats, comma-separated. Implemented: `json`, `table`. `reaclib` is accepted but reports "not yet supported". |
| `-o, --output` | `output` | Output directory |

Output files: `result.json` (summary) and `transmission.tsv` (energy vs. T_{l=0}).

### `batch`

Batch calculation across multiple nuclides. **Not yet implemented** — the
command parses its arguments and exits.

```bash
nucrust batch --config <PATH> [--backend cpu|gpu]
```

### `fit`

R-matrix parameter fitting. **Not yet implemented** — the command parses its
arguments and exits.

```bash
nucrust fit --config <PATH> [--method lm|mcmc]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--method` | `lm` | Fitting method: Levenberg-Marquardt (`lm`) or `mcmc` |

### `info`

Display nuclide information. **Not yet implemented** — nuclide lookup is
pending.

```bash
nucrust info <NUCLIDE> [--ripl3 <PATH>]
```

Example: `nucrust info Fe56`

### `export`

Convert between output formats. **Not yet implemented** — the command parses
its arguments and exits. Planned formats: `json`, `table`, `reaclib`, `hdf5`.

```bash
nucrust export --input <PATH> --format <FORMAT> [--output <PATH>]
```
