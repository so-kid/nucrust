//! HDF5 input/output for cross sections and reaction rates.
//!
//! Requires the `hdf5_io` feature flag.
//! Uses the `hdf5-metno` crate for thread-safe HDF5 access.

#[cfg(feature = "hdf5_io")]
use hdf5::File as H5File;
use nucrust_core::{CoreError, CrossSection, EnergyGrid, ReactionRate};
use std::path::Path;

/// Write a CrossSection to an HDF5 file.
///
/// Structure:
/// ```text
/// /energy           [n_e]       f64
/// /sigma_total      [n_e]       f64
/// /sigma_elastic    [n_e]       f64
/// /sigma_reaction   [n_e]       f64
/// ```
#[cfg(feature = "hdf5_io")]
pub fn write_cross_section(path: &Path, xs: &CrossSection) -> Result<(), CoreError> {
    let file =
        H5File::create(path).map_err(|e| CoreError::Io(std::io::Error::other(e.to_string())))?;

    let energies = xs.energy.as_slice();
    file.new_dataset::<f64>()
        .shape([energies.len()])
        .create("energy")
        .map_err(|e| CoreError::Io(std::io::Error::other(e.to_string())))?
        .write(energies)
        .map_err(|e| CoreError::Io(std::io::Error::other(e.to_string())))?;

    write_f64_dataset(&file, "sigma_total", &xs.sigma_total)?;
    write_f64_dataset(&file, "sigma_elastic", &xs.sigma_elastic)?;
    write_f64_dataset(&file, "sigma_reaction", &xs.sigma_reaction)?;

    Ok(())
}

/// Read a CrossSection from an HDF5 file.
#[cfg(feature = "hdf5_io")]
pub fn read_cross_section(path: &Path) -> Result<CrossSection, CoreError> {
    let file =
        H5File::open(path).map_err(|e| CoreError::Io(std::io::Error::other(e.to_string())))?;

    let energy_data: Vec<f64> = file
        .dataset("energy")
        .map_err(|e| CoreError::Io(std::io::Error::other(e.to_string())))?
        .read_1d()
        .map_err(|e| CoreError::Io(std::io::Error::other(e.to_string())))?
        .to_vec();

    let energy = EnergyGrid::from_values(energy_data)?;
    let sigma_total = read_f64_dataset(&file, "sigma_total")?;
    let sigma_elastic = read_f64_dataset(&file, "sigma_elastic")?;
    let sigma_reaction = read_f64_dataset(&file, "sigma_reaction")?;

    Ok(CrossSection {
        energy,
        sigma_total,
        sigma_elastic,
        sigma_reaction,
        partial: vec![],
    })
}

/// Write reaction rates to an HDF5 file.
///
/// Structure:
/// ```text
/// /temperatures     [n_t]       f64
/// /na_sigma_v       [n_t]       f64
/// /macs             [n_t]       f64  (optional)
/// ```
#[cfg(feature = "hdf5_io")]
pub fn write_reaction_rates(path: &Path, rates: &[ReactionRate]) -> Result<(), CoreError> {
    let file =
        H5File::create(path).map_err(|e| CoreError::Io(std::io::Error::other(e.to_string())))?;

    for (i, rate) in rates.iter().enumerate() {
        let group_name = format!("rate_{}", i);
        let group = file
            .create_group(&group_name)
            .map_err(|e| CoreError::Io(std::io::Error::other(e.to_string())))?;

        write_f64_dataset_in(&group, "temperatures", &rate.temperatures)?;
        write_f64_dataset_in(&group, "na_sigma_v", &rate.na_sigma_v)?;

        if let Some(ref macs) = rate.macs {
            write_f64_dataset_in(&group, "macs", macs)?;
        }
        if let Some(ref sf) = rate.s_factor {
            write_f64_dataset_in(&group, "s_factor", sf)?;
        }
        if let Some(ref sef) = rate.sef {
            write_f64_dataset_in(&group, "sef", sef)?;
        }
    }

    Ok(())
}

#[cfg(feature = "hdf5_io")]
fn write_f64_dataset(file: &H5File, name: &str, data: &[f64]) -> Result<(), CoreError> {
    file.new_dataset::<f64>()
        .shape([data.len()])
        .create(name)
        .map_err(|e| CoreError::Io(std::io::Error::other(e.to_string())))?
        .write(data)
        .map_err(|e| CoreError::Io(std::io::Error::other(e.to_string())))?;
    Ok(())
}

#[cfg(feature = "hdf5_io")]
fn write_f64_dataset_in(group: &hdf5::Group, name: &str, data: &[f64]) -> Result<(), CoreError> {
    group
        .new_dataset::<f64>()
        .shape([data.len()])
        .create(name)
        .map_err(|e| CoreError::Io(std::io::Error::other(e.to_string())))?
        .write(data)
        .map_err(|e| CoreError::Io(std::io::Error::other(e.to_string())))?;
    Ok(())
}

#[cfg(feature = "hdf5_io")]
fn read_f64_dataset(file: &H5File, name: &str) -> Result<Vec<f64>, CoreError> {
    Ok(file
        .dataset(name)
        .map_err(|e| CoreError::Io(std::io::Error::other(e.to_string())))?
        .read_1d()
        .map_err(|e| CoreError::Io(std::io::Error::other(e.to_string())))?
        .to_vec())
}
