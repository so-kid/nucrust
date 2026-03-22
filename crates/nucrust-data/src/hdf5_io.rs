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

#[cfg(all(test, feature = "hdf5_io"))]
mod tests {
    use super::*;
    use nucrust_core::{CrossSection, EnergyGrid, ReactionRate};
    use std::path::PathBuf;

    fn temp_h5_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("nucrust_hdf5_test");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    #[test]
    fn cross_section_roundtrip() {
        let path = temp_h5_path("xs_roundtrip.h5");
        let energies = EnergyGrid::from_values(vec![0.1, 0.5, 1.0, 5.0, 10.0]).unwrap();
        let xs = CrossSection {
            energy: energies,
            sigma_total: vec![100.0, 50.0, 30.0, 10.0, 5.0],
            sigma_elastic: vec![60.0, 30.0, 18.0, 6.0, 3.0],
            sigma_reaction: vec![40.0, 20.0, 12.0, 4.0, 2.0],
            partial: vec![],
        };

        write_cross_section(&path, &xs).unwrap();
        let loaded = read_cross_section(&path).unwrap();

        assert_eq!(loaded.energy.len(), 5);
        for i in 0..5 {
            assert!(
                (loaded.sigma_total[i] - xs.sigma_total[i]).abs() < 1e-12,
                "sigma_total[{}]: {} vs {}",
                i,
                loaded.sigma_total[i],
                xs.sigma_total[i]
            );
            assert!(
                (loaded.sigma_elastic[i] - xs.sigma_elastic[i]).abs() < 1e-12,
                "sigma_elastic[{}]",
                i
            );
            assert!(
                (loaded.sigma_reaction[i] - xs.sigma_reaction[i]).abs() < 1e-12,
                "sigma_reaction[{}]",
                i
            );
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn reaction_rates_roundtrip() {
        let path = temp_h5_path("rates_roundtrip.h5");
        let rates = vec![ReactionRate {
            temperatures: vec![0.1, 0.5, 1.0, 3.0],
            na_sigma_v: vec![1.0e5, 2.0e6, 5.0e7, 1.0e8],
            macs: Some(vec![100.0, 200.0, 300.0, 400.0]),
            s_factor: None,
            sef: Some(vec![1.0, 1.01, 1.05, 1.1]),
        }];

        write_reaction_rates(&path, &rates).unwrap();

        // Verify file exists and is readable
        let file = H5File::open(&path).unwrap();
        let group = file.group("rate_0").unwrap();
        let temps: Vec<f64> = group
            .dataset("temperatures")
            .unwrap()
            .read_1d()
            .unwrap()
            .to_vec();
        assert_eq!(temps.len(), 4);
        assert!((temps[0] - 0.1).abs() < 1e-12);

        let na_sv: Vec<f64> = group
            .dataset("na_sigma_v")
            .unwrap()
            .read_1d()
            .unwrap()
            .to_vec();
        assert!((na_sv[2] - 5.0e7).abs() < 1e-3);

        // MACS should be present
        let macs: Vec<f64> = group.dataset("macs").unwrap().read_1d().unwrap().to_vec();
        assert_eq!(macs.len(), 4);

        // SEF should be present
        let sef: Vec<f64> = group.dataset("sef").unwrap().read_1d().unwrap().to_vec();
        assert!((sef[3] - 1.1).abs() < 1e-12);

        // s_factor should NOT be present
        assert!(group.dataset("s_factor").is_err());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn cross_section_overwrite() {
        let path = temp_h5_path("xs_overwrite.h5");
        let energies = EnergyGrid::from_values(vec![1.0, 2.0]).unwrap();
        let xs1 = CrossSection {
            energy: energies.clone(),
            sigma_total: vec![10.0, 20.0],
            sigma_elastic: vec![5.0, 10.0],
            sigma_reaction: vec![5.0, 10.0],
            partial: vec![],
        };
        write_cross_section(&path, &xs1).unwrap();

        let xs2 = CrossSection {
            energy: energies,
            sigma_total: vec![99.0, 88.0],
            sigma_elastic: vec![1.0, 2.0],
            sigma_reaction: vec![98.0, 86.0],
            partial: vec![],
        };
        write_cross_section(&path, &xs2).unwrap();

        let loaded = read_cross_section(&path).unwrap();
        assert!((loaded.sigma_total[0] - 99.0).abs() < 1e-12);

        std::fs::remove_file(&path).ok();
    }
}
