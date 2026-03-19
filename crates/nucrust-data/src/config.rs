use serde::Deserialize;
use std::path::PathBuf;

use nucrust_core::CoreError;

/// Job configuration (deserialized from TOML).
#[derive(Debug, Deserialize)]
pub struct JobConfig {
    pub target: TargetConfig,
    pub projectile: String,
    pub energy: EnergyConfig,
    #[serde(default)]
    pub models: ModelConfig,
    #[serde(default)]
    pub output: OutputConfig,
    #[serde(default)]
    pub backend: BackendConfig,
}

#[derive(Debug, Deserialize)]
pub struct TargetConfig {
    pub z: u16,
    pub a: u16,
}

#[derive(Debug, Deserialize)]
pub struct EnergyConfig {
    pub min: f64,
    pub max: f64,
    pub points: usize,
    #[serde(default = "default_log")]
    pub spacing: String,
}

fn default_log() -> String {
    "log".to_string()
}

#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    #[serde(default = "default_omp")]
    pub omp: String,
    #[serde(default = "default_nld")]
    pub nld: String,
    #[serde(default = "default_gsf")]
    pub gsf: String,
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

#[derive(Debug, Deserialize)]
pub struct OutputConfig {
    #[serde(default = "default_formats")]
    pub format: Vec<String>,
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

#[derive(Debug, Deserialize)]
pub struct BackendConfig {
    #[serde(default = "default_cpu")]
    pub compute: String,
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
