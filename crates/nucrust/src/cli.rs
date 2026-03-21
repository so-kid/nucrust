//! Command-line interface for nucrust.
//!
//! Subcommands:
//! - `calc`:   Single nuclide cross section / reaction rate calculation
//! - `batch`:  Batch calculation across nuclide chart
//! - `fit`:    R-matrix parameter fitting
//! - `info`:   Display nuclide information
//! - `export`: Format conversion (HDF5 → REACLIB, etc.)

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "nucrust",
    version,
    about = "GPU-accelerated nuclear reaction rate calculation framework"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Calculate cross sections and reaction rates for a single target.
    Calc {
        /// Path to TOML configuration file.
        #[arg(short, long)]
        config: PathBuf,

        /// Compute backend: "cpu" or "gpu".
        #[arg(long, default_value = "cpu")]
        backend: String,

        /// Output formats (comma-separated): json, table, reaclib.
        #[arg(long, value_delimiter = ',', default_value = "json")]
        format: Vec<String>,

        /// Output directory.
        #[arg(short, long, default_value = "output")]
        output: PathBuf,
    },

    /// Batch calculation across multiple nuclides.
    Batch {
        /// Path to batch configuration file.
        #[arg(short, long)]
        config: PathBuf,

        /// Compute backend.
        #[arg(long, default_value = "cpu")]
        backend: String,
    },

    /// R-matrix parameter fitting.
    Fit {
        /// Path to fit configuration file.
        #[arg(short, long)]
        config: PathBuf,

        /// Fitting method: "lm" (Levenberg-Marquardt) or "mcmc".
        #[arg(long, default_value = "lm")]
        method: String,
    },

    /// Display nuclide information.
    Info {
        /// Nuclide name (e.g., "Fe56", "U238").
        nuclide: String,

        /// Path to RIPL-3 data directory.
        #[arg(long)]
        ripl3: Option<PathBuf>,
    },

    /// Export / convert output formats.
    Export {
        /// Input file path.
        #[arg(short, long)]
        input: PathBuf,

        /// Output format: json, reaclib, table.
        #[arg(long)]
        format: String,

        /// Output file path.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}
