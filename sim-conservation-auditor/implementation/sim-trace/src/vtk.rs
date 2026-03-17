//! VTK (Visualization Toolkit) format support for simulation data.
//!
//! Provides reading and writing of VTK Legacy and XML formats for
//! particle data and field data visualization in tools like ParaView.

use serde::{Deserialize, Serialize};
use std::io::{self, Write, BufWriter, BufRead};
use std::path::Path;

/// VTK file format variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VtkFormat {
    /// Legacy ASCII format (.vtk)
    LegacyAscii,
    /// Legacy binary format (.vtk)
    LegacyBinary,
    /// XML PolyData format (.vtp)
    XmlPolyData,
    /// XML UnstructuredGrid format (.vtu)
    XmlUnstructuredGrid,
}

/// Writer for VTK format files.
pub struct VtkWriter {
    format: VtkFormat,
}

/// Particle data for VTK export.
#[derive(Debug, Clone)]
pub struct VtkParticleData {
    pub positions: Vec<[f64; 3]>,
    pub velocities: Vec<[f64; 3]>,
    pub masses: Vec<f64>,
    pub scalar_fields: Vec<(String, Vec<f64>)>,
    pub vector_fields: Vec<(String, Vec<[f64; 3]>)>,
}

/// Field data on a structured grid for VTK export.
#[derive(Debug, Clone)]
pub struct VtkFieldData {
    pub dimensions: [usize; 3],
    pub origin: [f64; 3],
    pub spacing: [f64; 3],
    pub scalar_fields: Vec<(String, Vec<f64>)>,
    pub vector_fields: Vec<(String, Vec<[f64; 3]>)>,
}

impl VtkWriter {
    pub fn new(format: VtkFormat) -> Self {
        Self { format }
    }

    /// Write particle data to a VTK Legacy ASCII file.
    pub fn write_particles<P: AsRef<Path>>(
        &self,
        path: P,
        data: &VtkParticleData,
    ) -> io::Result<()> {
        let file = std::fs::File::create(path)?;
        let mut w = BufWriter::new(file);

        let n = data.positions.len();
        writeln!(w, "# vtk DataFile Version 3.0")?;
        writeln!(w, "ConservationLint particle data")?;
        writeln!(w, "ASCII")?;
        writeln!(w, "DATASET POLYDATA")?;
        writeln!(w, "POINTS {} double", n)?;
        for p in &data.positions {
            writeln!(w, "{:.12e} {:.12e} {:.12e}", p[0], p[1], p[2])?;
        }

        // Vertices connectivity
        writeln!(w, "VERTICES {} {}", n, 2 * n)?;
        for i in 0..n {
            writeln!(w, "1 {}", i)?;
        }

        writeln!(w, "POINT_DATA {}", n)?;

        // Velocities
        if !data.velocities.is_empty() {
            writeln!(w, "VECTORS velocity double")?;
            for v in &data.velocities {
                writeln!(w, "{:.12e} {:.12e} {:.12e}", v[0], v[1], v[2])?;
            }
        }

        // Masses
        if !data.masses.is_empty() {
            writeln!(w, "SCALARS mass double 1")?;
            writeln!(w, "LOOKUP_TABLE default")?;
            for m in &data.masses {
                writeln!(w, "{:.12e}", m)?;
            }
        }

        // Additional scalar fields
        for (name, values) in &data.scalar_fields {
            writeln!(w, "SCALARS {} double 1", name)?;
            writeln!(w, "LOOKUP_TABLE default")?;
            for v in values {
                writeln!(w, "{:.12e}", v)?;
            }
        }

        // Additional vector fields
        for (name, values) in &data.vector_fields {
            writeln!(w, "VECTORS {} double", name)?;
            for v in values {
                writeln!(w, "{:.12e} {:.12e} {:.12e}", v[0], v[1], v[2])?;
            }
        }

        Ok(())
    }

    /// Write structured grid field data to VTK Legacy ASCII format.
    pub fn write_structured_grid<P: AsRef<Path>>(
        &self,
        path: P,
        data: &VtkFieldData,
    ) -> io::Result<()> {
        let file = std::fs::File::create(path)?;
        let mut w = BufWriter::new(file);

        let [nx, ny, nz] = data.dimensions;
        let total = nx * ny * nz;

        writeln!(w, "# vtk DataFile Version 3.0")?;
        writeln!(w, "ConservationLint field data")?;
        writeln!(w, "ASCII")?;
        writeln!(w, "DATASET STRUCTURED_POINTS")?;
        writeln!(w, "DIMENSIONS {} {} {}", nx, ny, nz)?;
        writeln!(w, "ORIGIN {:.12e} {:.12e} {:.12e}", data.origin[0], data.origin[1], data.origin[2])?;
        writeln!(w, "SPACING {:.12e} {:.12e} {:.12e}", data.spacing[0], data.spacing[1], data.spacing[2])?;
        writeln!(w, "POINT_DATA {}", total)?;

        for (name, values) in &data.scalar_fields {
            writeln!(w, "SCALARS {} double 1", name)?;
            writeln!(w, "LOOKUP_TABLE default")?;
            for v in values {
                writeln!(w, "{:.12e}", v)?;
            }
        }

        for (name, values) in &data.vector_fields {
            writeln!(w, "VECTORS {} double", name)?;
            for v in values {
                writeln!(w, "{:.12e} {:.12e} {:.12e}", v[0], v[1], v[2])?;
            }
        }

        Ok(())
    }
}

/// Reader for VTK Legacy ASCII files.
pub struct VtkReader;

impl VtkReader {
    /// Read particle positions from a VTK Legacy ASCII file.
    pub fn read_particles<P: AsRef<Path>>(path: P) -> io::Result<VtkParticleData> {
        let file = std::fs::File::open(path)?;
        let reader = io::BufReader::new(file);
        let mut positions = Vec::new();
        let mut velocities = Vec::new();
        let mut masses = Vec::new();
        let mut reading_points = false;
        let mut reading_velocities = false;
        let mut reading_masses = false;
        let mut points_remaining = 0usize;

        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();

            if trimmed.starts_with("POINTS") {
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 2 {
                    points_remaining = parts[1].parse().unwrap_or(0);
                    reading_points = true;
                    reading_velocities = false;
                    reading_masses = false;
                }
                continue;
            }

            if trimmed.starts_with("VECTORS velocity") {
                reading_points = false;
                reading_velocities = true;
                reading_masses = false;
                continue;
            }

            if trimmed.starts_with("SCALARS mass") {
                reading_points = false;
                reading_velocities = false;
                reading_masses = true;
                continue;
            }

            if trimmed.starts_with("LOOKUP_TABLE") {
                continue;
            }

            if reading_points && points_remaining > 0 {
                let parts: Vec<f64> = trimmed.split_whitespace()
                    .filter_map(|s| s.parse().ok())
                    .collect();
                if parts.len() == 3 {
                    positions.push([parts[0], parts[1], parts[2]]);
                    points_remaining -= 1;
                }
            } else if reading_velocities {
                let parts: Vec<f64> = trimmed.split_whitespace()
                    .filter_map(|s| s.parse().ok())
                    .collect();
                if parts.len() == 3 {
                    velocities.push([parts[0], parts[1], parts[2]]);
                }
            } else if reading_masses {
                if let Ok(m) = trimmed.parse::<f64>() {
                    masses.push(m);
                }
            }
        }

        Ok(VtkParticleData {
            positions,
            velocities,
            masses,
            scalar_fields: Vec::new(),
            vector_fields: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn write_and_read_particles() {
        let data = VtkParticleData {
            positions: vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            velocities: vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            masses: vec![1.0, 2.0],
            scalar_fields: vec![],
            vector_fields: vec![],
        };

        let tmp = std::env::temp_dir().join("test_conservation_lint.vtk");
        let writer = VtkWriter::new(VtkFormat::LegacyAscii);
        writer.write_particles(&tmp, &data).unwrap();

        // Verify file was written
        let mut content = String::new();
        std::fs::File::open(&tmp).unwrap().read_to_string(&mut content).unwrap();
        assert!(content.contains("POINTS 2 double"));
        assert!(content.contains("VECTORS velocity double"));
        assert!(content.contains("SCALARS mass double"));

        // Read back
        let read_data = VtkReader::read_particles(&tmp).unwrap();
        assert_eq!(read_data.positions.len(), 2);
        assert!((read_data.positions[0][0] - 1.0).abs() < 1e-10);
        assert_eq!(read_data.velocities.len(), 2);
        assert_eq!(read_data.masses.len(), 2);

        std::fs::remove_file(&tmp).ok();
    }
}
