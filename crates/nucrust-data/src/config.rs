use serde::Deserialize;
use std::path::PathBuf;

use nucrust_core::CoreError;

/// Job configuration (deserialized from TOML).
#[derive(Debug, Deserialize)]
pub struct JobConfig {
    /// Target nuclide specification (Z, A).
    pub target: TargetConfig,
    /// Projectile identifier (e.g., `"n"`, `"p"`, `"a"`).
    pub projectile: String,
    /// Energy grid specification.
    pub energy: EnergyConfig,
    /// Physics model selections.
    #[serde(default)]
    pub models: ModelConfig,
    /// Output format and path settings.
    #[serde(default)]
    pub output: OutputConfig,
    /// Compute backend settings (CPU/GPU).
    #[serde(default)]
    pub backend: BackendConfig,
}

/// Target nuclide specification.
#[derive(Debug, Deserialize)]
pub struct TargetConfig {
    /// Proton number.
    pub z: u16,
    /// Mass number.
    pub a: u16,
}

/// Energy grid configuration.
#[derive(Debug, Deserialize)]
pub struct EnergyConfig {
    /// Minimum energy (MeV).
    pub min: f64,
    /// Maximum energy (MeV).
    pub max: f64,
    /// Number of energy points.
    pub points: usize,
    /// Grid spacing type (`"log"` or `"linear"`). Defaults to `"log"`.
    #[serde(default = "default_log")]
    pub spacing: String,
}

fn default_log() -> String {
    "log".to_string()
}

/// Physics model selections.
#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    /// Optical model potential name (default: `"koning-delaroche"`).
    #[serde(default = "default_omp")]
    pub omp: String,
    /// Nuclear level density model name (default: `"gilbert-cameron"`).
    #[serde(default = "default_nld")]
    pub nld: String,
    /// Gamma-ray strength function model name (default: `"eglo"`).
    #[serde(default = "default_gsf")]
    pub gsf: String,
    /// Path to the RIPL-3 data directory (optional).
    pub ripl3_path: Option<PathBuf>,
}

fn default_omp() -> String {
    "koning-delaroche".to_string()
}
fn default_nld() -> String {
    "gilbert-cameron".to_string()
}
fn default_gsf() -> String {
    "eglo".to_string()
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            omp: default_omp(),
            nld: default_nld(),
            gsf: default_gsf(),
            ripl3_path: None,
        }
    }
}

/// Output configuration.
#[derive(Debug, Deserialize)]
pub struct OutputConfig {
    /// Output format(s) (default: `["json"]`). Supported: `"json"`, `"table"`, `"hdf5"`.
    #[serde(default = "default_formats")]
    pub format: Vec<String>,
    /// Output directory path (default: `"output"`).
    #[serde(default = "default_output_path")]
    pub path: PathBuf,
}

fn default_formats() -> Vec<String> {
    vec!["json".to_string()]
}
fn default_output_path() -> PathBuf {
    PathBuf::from("output")
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            format: default_formats(),
            path: default_output_path(),
        }
    }
}

/// Compute backend configuration.
#[derive(Debug, Deserialize)]
pub struct BackendConfig {
    /// Compute backend (`"cpu"` or `"gpu"`). Defaults to `"cpu"`.
    #[serde(default = "default_cpu")]
    pub compute: String,
    /// GPU device index (0-based). Defaults to `0`.
    #[serde(default)]
    pub gpu_device: u32,
}

fn default_cpu() -> String {
    "cpu".to_string()
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            compute: default_cpu(),
            gpu_device: 0,
        }
    }
}

/// Parse a TOML configuration file.
pub fn parse_config(input: &str) -> Result<JobConfig, CoreError> {
    toml::from_str(input).map_err(|e| CoreError::ParseError {
        file: "config.toml".to_string(),
        line: 0,
        message: e.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_CONFIG: &str = r#"
projectile = "n"

[target]
z = 26
a = 56

[energy]
min = 0.001
max = 20.0
points = 200
spacing = "log"

[models]
omp = "koning-delaroche"
nld = "gilbert-cameron"
gsf = "eglo"

[output]
format = ["json", "table"]
path = "results"

[backend]
compute = "cpu"
"#;

    #[test]
    fn parse_full_config() {
        let config = parse_config(SAMPLE_CONFIG).unwrap();
        assert_eq!(config.target.z, 26);
        assert_eq!(config.target.a, 56);
        assert_eq!(config.projectile, "n");
        assert_eq!(config.energy.points, 200);
        assert!((config.energy.min - 0.001).abs() < 1e-15);
        assert_eq!(config.models.omp, "koning-delaroche");
        assert_eq!(config.output.format, vec!["json", "table"]);
        assert_eq!(config.backend.compute, "cpu");
    }

    const MINIMAL_CONFIG: &str = r#"
projectile = "p"

[target]
z = 6
a = 12

[energy]
min = 0.01
max = 10.0
points = 100
"#;

    #[test]
    fn parse_minimal_config_with_defaults() {
        let config = parse_config(MINIMAL_CONFIG).unwrap();
        assert_eq!(config.target.z, 6);
        assert_eq!(config.projectile, "p");
        assert_eq!(config.models.omp, "koning-delaroche");
        assert_eq!(config.models.nld, "gilbert-cameron");
        assert_eq!(config.backend.compute, "cpu");
        assert_eq!(config.output.format, vec!["json"]);
    }

    #[test]
    fn invalid_config_returns_error() {
        let bad = "this is not valid toml [[[";
        assert!(parse_config(bad).is_err());
    }
}
