//! Scene loading utilities.
//!
//! Detects format, parses scene files, resolves paths, and validates
//! loaded scenes with position-aware error reporting.

use std::path::{Path, PathBuf};

use xr_scene::parser::{SceneBuilder, SceneFormat, SceneParser};
use xr_types::device::DeviceConfig;
use xr_types::error::{Diagnostic, Severity, VerifierError, VerifierResult};
use xr_types::geometry::{BoundingBox, Volume};
use xr_types::scene::{
    DependencyType, FeedbackType, InteractableElement, InteractionType, SceneModel,
};

/// Scene loader with format detection and validation.
pub struct SceneLoader {
    parser: SceneParser,
    resolve_relative: bool,
}

impl SceneLoader {
    pub fn new() -> Self {
        Self {
            parser: SceneParser::new(),
            resolve_relative: true,
        }
    }

    pub fn strict() -> Self {
        Self {
            parser: SceneParser::new().strict(),
            resolve_relative: true,
        }
    }

    /// Load a scene from a file path.
    pub fn load(&self, path: &Path) -> VerifierResult<SceneModel> {
        let path = self.resolve_path(path)?;
        tracing::debug!("Loading scene from: {}", path.display());

        let content = std::fs::read_to_string(&path).map_err(|e| {
            VerifierError::SceneParsing(format!(
                "Failed to read scene file '{}': {}",
                path.display(),
                e
            ))
        })?;

        let format = SceneParser::detect_format(&content);
        tracing::debug!("Detected format: {:?}", format);

        match format {
            SceneFormat::NativeJson => {
                self.parser.parse_and_build(&content)
            }
            SceneFormat::UnityYaml => {
                Err(VerifierError::SceneParsing(
                    "Unity YAML format is not yet fully supported. \
                     Please export to JSON using the Unity adapter."
                        .into(),
                ))
            }
            SceneFormat::Gltf => {
                Err(VerifierError::SceneParsing(
                    "glTF format is not yet supported.".into(),
                ))
            }
            SceneFormat::Usd => {
                Err(VerifierError::SceneParsing(
                    "USD format is not yet supported.".into(),
                ))
            }
            SceneFormat::Unknown => {
                // Try JSON parsing anyway
                tracing::warn!(
                    "Unknown file format for '{}', attempting JSON parse",
                    path.display()
                );
                self.parser.parse_and_build(&content)
            }
        }
    }

    /// Load a scene with additional validation diagnostics.
    pub fn load_with_validation(
        &self,
        path: &Path,
    ) -> VerifierResult<(SceneModel, Vec<Diagnostic>)> {
        let path = self.resolve_path(path)?;
        let content = std::fs::read_to_string(&path).map_err(|e| {
            VerifierError::SceneParsing(format!(
                "Failed to read scene file '{}': {}",
                path.display(),
                e
            ))
        })?;

        let format = SceneParser::detect_format(&content);

        // Parse
        let parsed = self.parser.parse_json(&content)?;
        let mut diagnostics = Vec::new();

        // Report parse warnings
        for warning in &parsed.warnings {
            diagnostics.push(Diagnostic::warning("PARSE", warning));
        }

        // Structural validation
        let validation_errors = parsed.validate();
        for err in &validation_errors {
            diagnostics.push(Diagnostic::error("STRUCTURE", err));
        }

        // Build scene model
        let scene = self.parser.build_scene_model(&parsed)?;

        // Semantic validation
        let semantic_issues = scene.validate();
        for issue in &semantic_issues {
            diagnostics.push(Diagnostic::warning("SEMANTIC", issue));
        }

        // Position-specific checks
        self.check_positions(&scene, &mut diagnostics);

        // Format-specific warnings
        if format == SceneFormat::Unknown {
            diagnostics.push(Diagnostic::info(
                "FORMAT",
                "File format could not be auto-detected; parsed as JSON",
            ));
        }

        Ok((scene, diagnostics))
    }

    /// Resolve relative file paths.
    fn resolve_path(&self, path: &Path) -> VerifierResult<PathBuf> {
        if self.resolve_relative && path.is_relative() {
            let cwd = std::env::current_dir().map_err(|e| {
                VerifierError::SceneParsing(format!("Cannot determine current directory: {}", e))
            })?;
            Ok(cwd.join(path))
        } else {
            Ok(path.to_path_buf())
        }
    }

    /// Check element positions for common issues.
    fn check_positions(&self, scene: &SceneModel, diagnostics: &mut Vec<Diagnostic>) {
        for element in &scene.elements {
            // Underground elements
            if element.position[1] < 0.0 {
                diagnostics.push(
                    Diagnostic::warning(
                        "POS001",
                        &format!(
                            "Element '{}' is below ground plane (y={:.3}m)",
                            element.name, element.position[1]
                        ),
                    )
                    .with_element(element.id)
                    .with_suggestion("Check element position; y < 0 is below floor level"),
                );
            }

            // Very high elements
            if element.position[1] > 3.0 {
                diagnostics.push(
                    Diagnostic::warning(
                        "POS002",
                        &format!(
                            "Element '{}' is very high (y={:.3}m); may be unreachable",
                            element.name, element.position[1]
                        ),
                    )
                    .with_element(element.id),
                );
            }

            // Very far from origin
            let dist = (element.position[0].powi(2)
                + element.position[1].powi(2)
                + element.position[2].powi(2))
            .sqrt();
            if dist > 5.0 {
                diagnostics.push(
                    Diagnostic::info(
                        "POS003",
                        &format!(
                            "Element '{}' is {:.2}m from origin",
                            element.name, dist
                        ),
                    )
                    .with_element(element.id),
                );
            }

            // Check for NaN/Inf
            for (i, &coord) in element.position.iter().enumerate() {
                if coord.is_nan() || coord.is_infinite() {
                    let axis = ["x", "y", "z"][i];
                    diagnostics.push(
                        Diagnostic::error(
                            "POS004",
                            &format!(
                                "Element '{}' has invalid {} coordinate: {}",
                                element.name, axis, coord
                            ),
                        )
                        .with_element(element.id),
                    );
                }
            }
        }

        // Check for duplicate positions
        for i in 0..scene.elements.len() {
            for j in (i + 1)..scene.elements.len() {
                let a = &scene.elements[i];
                let b = &scene.elements[j];
                let dx = (a.position[0] - b.position[0]).abs();
                let dy = (a.position[1] - b.position[1]).abs();
                let dz = (a.position[2] - b.position[2]).abs();
                if dx < 1e-6 && dy < 1e-6 && dz < 1e-6 {
                    diagnostics.push(
                        Diagnostic::warning(
                            "POS005",
                            &format!(
                                "Elements '{}' and '{}' are at the same position",
                                a.name, b.name
                            ),
                        )
                        .with_element(a.id),
                    );
                }
            }
        }
    }

    /// Detect the format of a scene file without fully parsing it.
    pub fn detect_format(path: &Path) -> VerifierResult<SceneFormat> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            VerifierError::SceneParsing(format!(
                "Failed to read file '{}': {}",
                path.display(),
                e
            ))
        })?;
        Ok(SceneParser::detect_format(&content))
    }

    /// Get file size and basic info without parsing.
    pub fn file_info(path: &Path) -> VerifierResult<SceneFileInfo> {
        let metadata = std::fs::metadata(path).map_err(|e| {
            VerifierError::SceneParsing(format!(
                "Cannot stat file '{}': {}",
                path.display(),
                e
            ))
        })?;

        let content = std::fs::read_to_string(path).map_err(|e| {
            VerifierError::SceneParsing(format!(
                "Cannot read file '{}': {}",
                path.display(),
                e
            ))
        })?;

        let format = SceneParser::detect_format(&content);
        let line_count = content.lines().count();

        Ok(SceneFileInfo {
            path: path.to_path_buf(),
            size_bytes: metadata.len(),
            line_count,
            format,
        })
    }
}

impl Default for SceneLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Basic file information about a scene file.
#[derive(Debug, Clone)]
pub struct SceneFileInfo {
    pub path: PathBuf,
    pub size_bytes: u64,
    pub line_count: usize,
    pub format: SceneFormat,
}

impl std::fmt::Display for SceneFileInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} ({:?}, {} bytes, {} lines)",
            self.path.display(),
            self.format,
            self.size_bytes,
            self.line_count,
        )
    }
}

/// Generate a demo scene for testing the loader.
pub fn generate_demo_scene_json() -> String {
    let scene = SceneBuilder::new("demo_loader_test")
        .description("Auto-generated demo scene for loader testing")
        .author("xr-cli")
        .quest_3();

    let (scene, idx0) = scene.add_button("test_button", [0.0, 1.2, -0.5], 0.05);
    let (scene, idx1) = scene.add_button("test_button_2", [0.2, 1.2, -0.5], 0.05);
    let scene = scene.sequential(idx0, idx1);

    let model = scene.build();
    xr_scene::parser::scene_to_json(&model).unwrap_or_else(|_| "{}".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp_scene(content: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("xr_cli_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("test_scene_{}.json", uuid::Uuid::new_v4()));
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        path
    }

    #[test]
    fn test_scene_loader_new() {
        let loader = SceneLoader::new();
        assert!(loader.resolve_relative);
    }

    #[test]
    fn test_scene_loader_strict() {
        let _loader = SceneLoader::strict();
    }

    #[test]
    fn test_generate_demo_scene_json() {
        let json = generate_demo_scene_json();
        assert!(json.contains("demo_loader_test"));
        assert!(json.contains("test_button"));
    }

    #[test]
    fn test_load_demo_scene() {
        let json = generate_demo_scene_json();
        let path = write_temp_scene(&json);
        let loader = SceneLoader::new();
        let scene = loader.load(&path);
        assert!(scene.is_ok(), "Failed to load demo scene: {:?}", scene.err());
        let scene = scene.unwrap();
        assert_eq!(scene.name, "demo_loader_test");
        assert!(scene.elements.len() >= 2);

        // Clean up
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_with_validation() {
        let json = generate_demo_scene_json();
        let path = write_temp_scene(&json);
        let loader = SceneLoader::new();
        let (scene, diagnostics) = loader.load_with_validation(&path).unwrap();
        assert_eq!(scene.name, "demo_loader_test");
        // There may or may not be diagnostics, but the call should succeed
        let _ = diagnostics;

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_nonexistent_file() {
        let loader = SceneLoader::new();
        let result = loader.load(Path::new("/nonexistent/path/to/scene.json"));
        assert!(result.is_err());
    }

    #[test]
    fn test_load_invalid_json() {
        let path = write_temp_scene("this is not json {{{");
        let loader = SceneLoader::new();
        let result = loader.load(&path);
        assert!(result.is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_file_info() {
        let json = generate_demo_scene_json();
        let path = write_temp_scene(&json);
        let info = SceneLoader::file_info(&path).unwrap();
        assert!(info.size_bytes > 0);
        assert!(info.line_count > 0);
        assert_eq!(info.format, SceneFormat::NativeJson);
        let display = format!("{}", info);
        assert!(display.contains("NativeJson"));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_detect_format() {
        let json = generate_demo_scene_json();
        let path = write_temp_scene(&json);
        let format = SceneLoader::detect_format(&path).unwrap();
        assert_eq!(format, SceneFormat::NativeJson);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_check_positions_underground() {
        let mut scene = SceneModel::new("test");
        let mut elem = InteractableElement::new("underground", [0.0, -1.0, 0.0], InteractionType::Click);
        elem.visual.label = Some("Underground".into());
        scene.add_element(elem);

        let loader = SceneLoader::new();
        let mut diagnostics = Vec::new();
        loader.check_positions(&scene, &mut diagnostics);

        assert!(
            diagnostics.iter().any(|d| d.code == "POS001"),
            "Expected POS001 diagnostic for underground element"
        );
    }

    #[test]
    fn test_check_positions_very_high() {
        let mut scene = SceneModel::new("test");
        let mut elem = InteractableElement::new("high", [0.0, 5.0, 0.0], InteractionType::Click);
        elem.visual.label = Some("High".into());
        scene.add_element(elem);

        let loader = SceneLoader::new();
        let mut diagnostics = Vec::new();
        loader.check_positions(&scene, &mut diagnostics);

        assert!(
            diagnostics.iter().any(|d| d.code == "POS002"),
            "Expected POS002 diagnostic for very high element"
        );
    }

    #[test]
    fn test_check_positions_duplicate() {
        let mut scene = SceneModel::new("test");
        let mut e1 = InteractableElement::new("a", [1.0, 1.0, 1.0], InteractionType::Click);
        e1.visual.label = Some("A".into());
        scene.add_element(e1);
        let mut e2 = InteractableElement::new("b", [1.0, 1.0, 1.0], InteractionType::Click);
        e2.visual.label = Some("B".into());
        scene.add_element(e2);

        let loader = SceneLoader::new();
        let mut diagnostics = Vec::new();
        loader.check_positions(&scene, &mut diagnostics);

        assert!(
            diagnostics.iter().any(|d| d.code == "POS005"),
            "Expected POS005 diagnostic for duplicate positions"
        );
    }
}
