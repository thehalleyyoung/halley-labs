//! HDF5 format support for simulation trajectory data.
//!
//! Provides a lightweight HDF5-like format for storing simulation trajectories,
//! conservation quantities, and violation reports. Uses a simple hierarchical
//! binary layout that can interoperate with HDF5 tools via the JSON metadata
//! sidecar approach.
//!
//! For full HDF5 support, enable the `hdf5` feature (requires libhdf5).
//! This module provides the format abstraction layer.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{self, Write, Read, BufWriter, BufReader};
use std::path::Path;

/// Metadata for an HDF5-compatible dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMeta {
    /// Dataset name/path (e.g., "/particles/positions")
    pub name: String,
    /// Shape of the dataset
    pub shape: Vec<usize>,
    /// Data type description
    pub dtype: String,
    /// Optional attributes
    pub attributes: HashMap<String, String>,
}

/// A simulation trajectory stored in HDF5-compatible format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hdf5Trajectory {
    /// Metadata about the simulation
    pub metadata: Hdf5Metadata,
    /// Time values for each frame
    pub times: Vec<f64>,
    /// Positions at each frame: [frame][particle][xyz]
    pub positions: Vec<Vec<[f64; 3]>>,
    /// Velocities at each frame: [frame][particle][xyz]
    pub velocities: Vec<Vec<[f64; 3]>>,
    /// Conserved quantity time series: name → values
    pub conservation_data: HashMap<String, Vec<f64>>,
}

/// Metadata header for HDF5 trajectory files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hdf5Metadata {
    pub format_version: String,
    pub creator: String,
    pub num_particles: usize,
    pub num_frames: usize,
    pub dt: f64,
    pub integrator: String,
    pub conservation_laws: Vec<String>,
    pub attributes: HashMap<String, String>,
}

/// Writer for HDF5-compatible trajectory files.
///
/// Uses a JSON-based serialization format that captures the hierarchical
/// structure of HDF5 files. For native HDF5 I/O, use the `hdf5` crate.
pub struct Hdf5Writer;

impl Hdf5Writer {
    /// Write a trajectory to a JSON-based HDF5-compatible file.
    pub fn write_trajectory<P: AsRef<Path>>(
        path: P,
        trajectory: &Hdf5Trajectory,
    ) -> io::Result<()> {
        let file = std::fs::File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, trajectory)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
    }

    /// Write conservation time-series data to a compact binary format.
    pub fn write_conservation_binary<P: AsRef<Path>>(
        path: P,
        law_names: &[&str],
        data: &[Vec<f64>],
    ) -> io::Result<()> {
        let file = std::fs::File::create(path)?;
        let mut w = BufWriter::new(file);

        // Header: magic, version, num_laws, num_steps
        w.write_all(b"CLNT")?; // magic
        w.write_all(&1u32.to_le_bytes())?; // version
        w.write_all(&(law_names.len() as u32).to_le_bytes())?;
        if data.is_empty() {
            w.write_all(&0u32.to_le_bytes())?;
            return Ok(());
        }
        w.write_all(&(data[0].len() as u32).to_le_bytes())?;

        // Law names (length-prefixed strings)
        for name in law_names {
            let bytes = name.as_bytes();
            w.write_all(&(bytes.len() as u32).to_le_bytes())?;
            w.write_all(bytes)?;
        }

        // Data: interleaved [step0_law0, step0_law1, ..., step1_law0, ...]
        for series in data {
            for &val in series {
                w.write_all(&val.to_le_bytes())?;
            }
        }

        Ok(())
    }
}

/// Reader for HDF5-compatible trajectory files.
pub struct Hdf5Reader;

impl Hdf5Reader {
    /// Read a trajectory from a JSON-based HDF5-compatible file.
    pub fn read_trajectory<P: AsRef<Path>>(path: P) -> io::Result<Hdf5Trajectory> {
        let file = std::fs::File::open(path)?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
    }

    /// Read conservation time-series from compact binary format.
    pub fn read_conservation_binary<P: AsRef<Path>>(
        path: P,
    ) -> io::Result<(Vec<String>, Vec<Vec<f64>>)> {
        let file = std::fs::File::open(path)?;
        let mut r = BufReader::new(file);

        // Read header
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != b"CLNT" {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid magic bytes"));
        }

        let mut buf4 = [0u8; 4];
        r.read_exact(&mut buf4)?;
        let _version = u32::from_le_bytes(buf4);

        r.read_exact(&mut buf4)?;
        let num_laws = u32::from_le_bytes(buf4) as usize;

        r.read_exact(&mut buf4)?;
        let num_steps = u32::from_le_bytes(buf4) as usize;

        // Read law names
        let mut names = Vec::with_capacity(num_laws);
        for _ in 0..num_laws {
            r.read_exact(&mut buf4)?;
            let len = u32::from_le_bytes(buf4) as usize;
            let mut name_bytes = vec![0u8; len];
            r.read_exact(&mut name_bytes)?;
            names.push(String::from_utf8_lossy(&name_bytes).to_string());
        }

        // Read data
        let mut data = Vec::with_capacity(num_laws);
        let mut buf8 = [0u8; 8];
        for _ in 0..num_laws {
            let mut series = Vec::with_capacity(num_steps);
            for _ in 0..num_steps {
                r.read_exact(&mut buf8)?;
                series.push(f64::from_le_bytes(buf8));
            }
            data.push(series);
        }

        Ok((names, data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trajectory_roundtrip() {
        let traj = Hdf5Trajectory {
            metadata: Hdf5Metadata {
                format_version: "1.0".into(),
                creator: "conservation-lint".into(),
                num_particles: 2,
                num_frames: 2,
                dt: 0.001,
                integrator: "velocity-verlet".into(),
                conservation_laws: vec!["energy".into(), "momentum".into()],
                attributes: HashMap::new(),
            },
            times: vec![0.0, 0.001],
            positions: vec![
                vec![[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
                vec![[1.001, 0.0, 0.0], [-1.001, 0.0, 0.0]],
            ],
            velocities: vec![
                vec![[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
                vec![[0.0, 0.999, 0.0], [0.0, -0.999, 0.0]],
            ],
            conservation_data: {
                let mut m = HashMap::new();
                m.insert("energy".into(), vec![-0.5, -0.5]);
                m.insert("momentum".into(), vec![0.0, 0.0]);
                m
            },
        };

        let tmp = std::env::temp_dir().join("test_traj.h5.json");
        Hdf5Writer::write_trajectory(&tmp, &traj).unwrap();
        let read_traj = Hdf5Reader::read_trajectory(&tmp).unwrap();
        assert_eq!(read_traj.metadata.num_particles, 2);
        assert_eq!(read_traj.times.len(), 2);
        assert_eq!(read_traj.positions.len(), 2);
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn binary_conservation_roundtrip() {
        let names = vec!["energy", "momentum_x"];
        let data = vec![
            vec![-0.5, -0.500001, -0.500003],
            vec![0.0, 1e-15, 2e-15],
        ];

        let tmp = std::env::temp_dir().join("test_conservation.clnt");
        Hdf5Writer::write_conservation_binary(&tmp, &names, &data).unwrap();
        let (read_names, read_data) = Hdf5Reader::read_conservation_binary(&tmp).unwrap();
        assert_eq!(read_names, vec!["energy", "momentum_x"]);
        assert_eq!(read_data.len(), 2);
        assert_eq!(read_data[0].len(), 3);
        assert!((read_data[0][0] - (-0.5)).abs() < 1e-15);
        std::fs::remove_file(&tmp).ok();
    }
}
