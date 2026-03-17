//! Output writing utilities.
//!
//! Provides functions to write formatted output to files or stdout,
//! manage output directories, and apply file naming conventions.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Core write
// ---------------------------------------------------------------------------

/// Write output to a file path, or to stdout if `path` is `None`.
pub fn write_output(path: Option<&Path>, content: &str) -> Result<(), Box<dyn std::error::Error>> {
    match path {
        Some(p) => {
            if let Some(parent) = p.parent() {
                if !parent.exists() {
                    fs::create_dir_all(parent)
                        .map_err(|e| format!("Cannot create directory {:?}: {}", parent, e))?;
                }
            }
            fs::write(p, content)
                .map_err(|e| format!("Cannot write to {:?}: {}", p, e))?;
            Ok(())
        }
        None => {
            let stdout = std::io::stdout();
            let mut handle = stdout.lock();
            handle.write_all(content.as_bytes())
                .map_err(|e| format!("Cannot write to stdout: {}", e))?;
            Ok(())
        }
    }
}

/// Write binary content to a file.
pub fn write_binary(path: &Path, content: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Cannot create directory {:?}: {}", parent, e))?;
        }
    }
    fs::write(path, content)
        .map_err(|e| format!("Cannot write binary to {:?}: {}", path, e))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Directory management
// ---------------------------------------------------------------------------

/// Ensure a directory exists, creating it and its parents if needed.
pub fn ensure_dir(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if !path.exists() {
        fs::create_dir_all(path)
            .map_err(|e| format!("Cannot create directory {:?}: {}", path, e))?;
    } else if !path.is_dir() {
        return Err(format!("{:?} exists but is not a directory", path).into());
    }
    Ok(())
}

/// List files in a directory with a given extension.
pub fn list_files_with_extension(dir: &Path, ext: &str) -> Vec<PathBuf> {
    let mut results = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_file() {
                if let Some(file_ext) = p.extension() {
                    if file_ext == ext {
                        results.push(p);
                    }
                }
            }
        }
    }
    results.sort();
    results
}

// ---------------------------------------------------------------------------
// File naming conventions
// ---------------------------------------------------------------------------

/// Generate an output file name based on a base name, a tag, and an extension.
///
/// Example: `output_filename("workload", "analysis", "json")` => `"workload_analysis.json"`
pub fn output_filename(base: &str, tag: &str, ext: &str) -> String {
    let sanitised_base = sanitise_name(base);
    let sanitised_tag = sanitise_name(tag);
    if sanitised_tag.is_empty() {
        format!("{}.{}", sanitised_base, ext)
    } else {
        format!("{}_{}.{}", sanitised_base, sanitised_tag, ext)
    }
}

/// Sanitise a name for use in file paths: lowercase, replace non-alphanumeric
/// characters with underscores, collapse runs.
pub fn sanitise_name(name: &str) -> String {
    let mut result = String::with_capacity(name.len());
    let mut last_was_underscore = false;
    for c in name.chars() {
        if c.is_alphanumeric() {
            result.push(c.to_ascii_lowercase());
            last_was_underscore = false;
        } else if !last_was_underscore && !result.is_empty() {
            result.push('_');
            last_was_underscore = true;
        }
    }
    // Trim trailing underscore
    if result.ends_with('_') {
        result.pop();
    }
    result
}

/// Build a path inside an output directory.
pub fn output_path(dir: &Path, base: &str, tag: &str, ext: &str) -> PathBuf {
    dir.join(output_filename(base, tag, ext))
}

/// Generate a timestamped file name.
pub fn timestamped_filename(base: &str, ext: &str) -> String {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{}_{}.{}", sanitise_name(base), ts, ext)
}

// ---------------------------------------------------------------------------
// Append mode
// ---------------------------------------------------------------------------

/// Append content to a file, creating it if it doesn't exist.
pub fn append_to_file(path: &Path, content: &str) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs::OpenOptions;
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Cannot create directory {:?}: {}", parent, e))?;
        }
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("Cannot open {:?} for append: {}", path, e))?;
    file.write_all(content.as_bytes())
        .map_err(|e| format!("Cannot append to {:?}: {}", path, e))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_sanitise_name() {
        assert_eq!(sanitise_name("Hello World!"), "hello_world");
        assert_eq!(sanitise_name("TPC-C Workload"), "tpc_c_workload");
        assert_eq!(sanitise_name("simple"), "simple");
        assert_eq!(sanitise_name("a--b--c"), "a_b_c");
        assert_eq!(sanitise_name(""), "");
    }

    #[test]
    fn test_output_filename() {
        assert_eq!(output_filename("workload", "analysis", "json"), "workload_analysis.json");
        assert_eq!(output_filename("test", "", "txt"), "test.txt");
    }

    #[test]
    fn test_output_path() {
        let dir = PathBuf::from("/tmp/results");
        let p = output_path(&dir, "bench", "report", "csv");
        assert_eq!(p, PathBuf::from("/tmp/results/bench_report.csv"));
    }

    #[test]
    fn test_timestamped_filename() {
        let name = timestamped_filename("run", "json");
        assert!(name.starts_with("run_"));
        assert!(name.ends_with(".json"));
    }

    #[test]
    fn test_write_output_stdout() {
        // Writing to stdout should succeed
        let result = write_output(None, "test output\n");
        assert!(result.is_ok());
    }
}
