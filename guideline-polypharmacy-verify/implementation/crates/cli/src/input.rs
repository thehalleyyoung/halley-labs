//! Input loading and validation for GuardPharma CLI.
//!
//! Handles loading patient profiles, medication lists, and guideline documents
//! from JSON, YAML, and plain-text files. Includes validation, discovery, and
//! builder utilities.

use anyhow::{bail, Context, Result};
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use guardpharma_types::{
    AscitesGrade, CypEnzyme, DrugId, DrugRoute, PatientInfo, RenalFunction, Severity, Sex,
};

// ─────────────────────────── Domain Types ────────────────────────────────

/// A clinical condition attached to a patient.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClinicalCondition {
    pub code: String,
    pub name: String,
    pub active: bool,
}

impl ClinicalCondition {
    pub fn new(code: &str, name: &str) -> Self {
        ClinicalCondition {
            code: code.to_string(),
            name: name.to_string(),
            active: true,
        }
    }

    pub fn inactive(code: &str, name: &str) -> Self {
        ClinicalCondition {
            code: code.to_string(),
            name: name.to_string(),
            active: false,
        }
    }

    pub fn code_starts_with(&self, prefix: &str) -> bool {
        self.code.starts_with(prefix)
    }
}

impl fmt::Display for ClinicalCondition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = if self.active { "active" } else { "resolved" };
        write!(f, "{} ({}, {})", self.name, self.code, status)
    }
}

/// A medication the patient is currently taking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveMedication {
    pub drug_id: DrugId,
    pub name: String,
    pub dose_mg: f64,
    pub frequency_hours: f64,
    pub route: DrugRoute,
    pub drug_class: String,
    pub start_date: Option<String>,
    pub prescriber: Option<String>,
    pub indication: Option<String>,
}

impl ActiveMedication {
    pub fn new(name: &str, dose_mg: f64) -> Self {
        ActiveMedication {
            drug_id: DrugId::new(name),
            name: name.to_string(),
            dose_mg,
            frequency_hours: 24.0,
            route: DrugRoute::Oral,
            drug_class: String::new(),
            start_date: None,
            prescriber: None,
            indication: None,
        }
    }

    pub fn with_frequency(mut self, hours: f64) -> Self {
        self.frequency_hours = hours;
        self
    }

    pub fn with_route(mut self, route: DrugRoute) -> Self {
        self.route = route;
        self
    }

    pub fn with_class(mut self, class: &str) -> Self {
        self.drug_class = class.to_string();
        self
    }

    pub fn with_start_date(mut self, date: &str) -> Self {
        self.start_date = Some(date.to_string());
        self
    }

    pub fn with_prescriber(mut self, prescriber: &str) -> Self {
        self.prescriber = Some(prescriber.to_string());
        self
    }

    pub fn with_indication(mut self, indication: &str) -> Self {
        self.indication = Some(indication.to_string());
        self
    }

    pub fn canonical_name(&self) -> String {
        self.name.to_lowercase()
    }
}

impl fmt::Display for ActiveMedication {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}mg q{}h {:?}", self.name, self.dose_mg, self.frequency_hours, self.route)
    }
}

/// Full patient profile for verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatientProfile {
    pub id: Option<String>,
    pub info: PatientInfo,
    pub conditions: Vec<ClinicalCondition>,
    pub medications: Vec<ActiveMedication>,
    pub egfr: Option<f64>,
    pub allergies: Vec<String>,
    pub notes: Option<String>,
}

impl PatientProfile {
    pub fn new(info: PatientInfo) -> Self {
        PatientProfile {
            id: None,
            info,
            conditions: Vec::new(),
            medications: Vec::new(),
            egfr: None,
            allergies: Vec::new(),
            notes: None,
        }
    }

    pub fn with_conditions(mut self, conditions: Vec<ClinicalCondition>) -> Self {
        self.conditions = conditions;
        self
    }

    pub fn with_medications(mut self, medications: Vec<ActiveMedication>) -> Self {
        self.medications = medications;
        self
    }

    pub fn with_egfr(mut self, egfr: f64) -> Self {
        self.egfr = Some(egfr);
        self
    }

    pub fn age(&self) -> f64 {
        self.info.age_years
    }

    pub fn sex(&self) -> Sex {
        self.info.sex
    }

    pub fn renal_function(&self) -> RenalFunction {
        match self.egfr {
            Some(val) if val >= 90.0 => RenalFunction::Normal,
            Some(val) if val >= 60.0 => RenalFunction::MildImpairment,
            Some(val) if val >= 30.0 => RenalFunction::ModerateImpairment,
            Some(val) if val >= 15.0 => RenalFunction::SevereImpairment,
            Some(_) => RenalFunction::EndStage,
            None => RenalFunction::Normal,
        }
    }

    pub fn has_condition_code(&self, prefix: &str) -> bool {
        self.conditions
            .iter()
            .any(|c| c.active && c.code_starts_with(prefix))
    }

    pub fn active_conditions(&self) -> Vec<&ClinicalCondition> {
        self.conditions.iter().filter(|c| c.active).collect()
    }

    pub fn medication_count(&self) -> usize {
        self.medications.len()
    }

    pub fn is_elderly(&self) -> bool {
        self.info.age_years >= 65.0
    }

    pub fn drug_pairs(&self) -> Vec<(&ActiveMedication, &ActiveMedication)> {
        let mut pairs = Vec::new();
        for i in 0..self.medications.len() {
            for j in (i + 1)..self.medications.len() {
                pairs.push((&self.medications[i], &self.medications[j]));
            }
        }
        pairs
    }
}

impl Default for PatientProfile {
    fn default() -> Self {
        PatientProfile::new(PatientInfo::default())
    }
}

/// A parsed guideline document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuidelineDocument {
    pub id: String,
    pub name: String,
    pub version: String,
    pub source: String,
    pub description: String,
    pub rules: Vec<GuidelineRule>,
    pub metadata: GuidelineMetadata,
}

impl GuidelineDocument {
    pub fn new(name: &str) -> Self {
        GuidelineDocument {
            id: name.to_lowercase().replace(' ', "-"),
            name: name.to_string(),
            version: "1.0".to_string(),
            source: String::new(),
            description: String::new(),
            rules: Vec::new(),
            metadata: GuidelineMetadata::default(),
        }
    }

    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    pub fn rules_for_severity(&self, min_severity: Severity) -> Vec<&GuidelineRule> {
        self.rules
            .iter()
            .filter(|r| r.severity >= min_severity)
            .collect()
    }

    pub fn rules_for_drug(&self, drug_id: &DrugId) -> Vec<&GuidelineRule> {
        let name = drug_id.as_str().to_lowercase();
        self.rules
            .iter()
            .filter(|r| {
                r.affected_drugs
                    .iter()
                    .any(|d| d.to_lowercase() == name)
            })
            .collect()
    }
}

impl fmt::Display for GuidelineDocument {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} v{} ({} rules)", self.name, self.version, self.rules.len())
    }
}

/// A single rule within a guideline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuidelineRule {
    pub id: String,
    pub description: String,
    pub condition: String,
    pub action: String,
    pub severity: Severity,
    pub evidence_level: String,
    pub affected_drugs: Vec<String>,
    pub affected_conditions: Vec<String>,
    pub contraindications: Vec<String>,
    pub monitoring_required: bool,
    pub dose_adjustment: Option<String>,
}

impl GuidelineRule {
    pub fn new(id: &str, description: &str, severity: Severity) -> Self {
        GuidelineRule {
            id: id.to_string(),
            description: description.to_string(),
            condition: String::new(),
            action: String::new(),
            severity,
            evidence_level: "B".to_string(),
            affected_drugs: Vec::new(),
            affected_conditions: Vec::new(),
            contraindications: Vec::new(),
            monitoring_required: false,
            dose_adjustment: None,
        }
    }
}

/// Metadata for a guideline document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuidelineMetadata {
    pub authors: Vec<String>,
    pub publication_date: Option<String>,
    pub last_updated: Option<String>,
    pub organization: Option<String>,
    pub url: Option<String>,
    pub doi: Option<String>,
    pub category: Option<String>,
    pub tags: Vec<String>,
}

impl Default for GuidelineMetadata {
    fn default() -> Self {
        GuidelineMetadata {
            authors: Vec::new(),
            publication_date: None,
            last_updated: None,
            organization: None,
            url: None,
            doi: None,
            category: None,
            tags: Vec::new(),
        }
    }
}

// ─────────────────────────── Error Types ─────────────────────────────────

/// Errors encountered during input loading.
#[derive(Debug, Clone, thiserror::Error)]
pub enum InputError {
    #[error("File not found: {path}")]
    FileNotFound { path: String },

    #[error("Parse error in {path}: {message}")]
    ParseError { path: String, message: String },

    #[error("Validation error: {message}")]
    ValidationError { message: String },

    #[error("Incompatible guideline versions: {details}")]
    IncompatibleVersions { details: String },

    #[error("Unsupported file format: {extension}")]
    UnsupportedFormat { extension: String },

    #[error("Empty input: {description}")]
    EmptyInput { description: String },
}

// ──────────────────────── File Loading Functions ─────────────────────────

/// Load guideline documents from a list of file/directory paths.
pub fn load_guidelines(paths: &[PathBuf]) -> Result<Vec<GuidelineDocument>> {
    let mut documents = Vec::new();
    let mut seen_ids = HashSet::new();

    for path in paths {
        if path.is_dir() {
            let discovery = GuidelineDiscovery::new(path);
            let found = discovery.discover()?;
            info!("Discovered {} guideline files in {}", found.len(), path.display());

            for file_path in found {
                let doc = load_single_guideline(&file_path)
                    .with_context(|| format!("Failed to load guideline: {}", file_path.display()))?;
                if seen_ids.insert(doc.id.clone()) {
                    documents.push(doc);
                } else {
                    warn!("Duplicate guideline ID '{}', skipping {}", doc.id, file_path.display());
                }
            }
        } else if path.is_file() {
            let doc = load_single_guideline(path)
                .with_context(|| format!("Failed to load guideline: {}", path.display()))?;
            if seen_ids.insert(doc.id.clone()) {
                documents.push(doc);
            } else {
                warn!("Duplicate guideline ID '{}'", doc.id);
            }
        } else {
            bail!(InputError::FileNotFound {
                path: path.display().to_string(),
            });
        }
    }

    if documents.is_empty() {
        bail!(InputError::EmptyInput {
            description: "No guideline documents found".to_string(),
        });
    }

    info!("Loaded {} guideline documents total", documents.len());
    Ok(documents)
}

/// Load a single guideline file.
fn load_single_guideline(path: &Path) -> Result<GuidelineDocument> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Cannot read file: {}", path.display()))?;

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "json" => {
            let doc: GuidelineDocument = serde_json::from_str(&content).map_err(|e| {
                InputError::ParseError {
                    path: path.display().to_string(),
                    message: e.to_string(),
                }
            })?;
            debug!("Loaded JSON guideline: {} ({} rules)", doc.name, doc.rules.len());
            Ok(doc)
        }
        "yaml" | "yml" => {
            let doc: GuidelineDocument = serde_yaml::from_str(&content).map_err(|e| {
                InputError::ParseError {
                    path: path.display().to_string(),
                    message: e.to_string(),
                }
            })?;
            debug!("Loaded YAML guideline: {} ({} rules)", doc.name, doc.rules.len());
            Ok(doc)
        }
        "toml" => {
            let doc: GuidelineDocument = toml::from_str(&content).map_err(|e| {
                InputError::ParseError {
                    path: path.display().to_string(),
                    message: e.to_string(),
                }
            })?;
            debug!("Loaded TOML guideline: {} ({} rules)", doc.name, doc.rules.len());
            Ok(doc)
        }
        _ => bail!(InputError::UnsupportedFormat {
            extension: ext,
        }),
    }
}

/// Load a patient profile from a JSON or YAML file.
pub fn load_patient(path: &Path) -> Result<PatientProfile> {
    if !path.exists() {
        bail!(InputError::FileNotFound {
            path: path.display().to_string(),
        });
    }

    let content = fs::read_to_string(path)
        .with_context(|| format!("Cannot read patient file: {}", path.display()))?;

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let profile: PatientProfile = match ext.as_str() {
        "json" => serde_json::from_str(&content).map_err(|e| InputError::ParseError {
            path: path.display().to_string(),
            message: e.to_string(),
        })?,
        "yaml" | "yml" => serde_yaml::from_str(&content).map_err(|e| InputError::ParseError {
            path: path.display().to_string(),
            message: e.to_string(),
        })?,
        _ => bail!(InputError::UnsupportedFormat { extension: ext }),
    };

    validate_patient_profile(&profile)?;
    info!(
        "Loaded patient profile: age={}, meds={}, conditions={}",
        profile.age(),
        profile.medication_count(),
        profile.conditions.len()
    );
    Ok(profile)
}

/// Load a medication list from a file.
pub fn load_medication_list(path: &Path) -> Result<Vec<ActiveMedication>> {
    if !path.exists() {
        bail!(InputError::FileNotFound {
            path: path.display().to_string(),
        });
    }

    let content = fs::read_to_string(path)
        .with_context(|| format!("Cannot read medication file: {}", path.display()))?;

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let medications: Vec<ActiveMedication> = match ext.as_str() {
        "json" => serde_json::from_str(&content).map_err(|e| InputError::ParseError {
            path: path.display().to_string(),
            message: e.to_string(),
        })?,
        "yaml" | "yml" => serde_yaml::from_str(&content).map_err(|e| InputError::ParseError {
            path: path.display().to_string(),
            message: e.to_string(),
        })?,
        "txt" | "csv" => {
            let parser = MedicationListParser::new();
            parser.parse_text(&content)?
        }
        _ => bail!(InputError::UnsupportedFormat { extension: ext }),
    };

    validate_medications(&medications)?;
    info!("Loaded {} medications", medications.len());
    Ok(medications)
}

/// Parse an inline JSON patient profile from a command-line string.
pub fn parse_inline_patient(json_str: &str) -> Result<PatientProfile> {
    let profile: PatientProfile = serde_json::from_str(json_str).map_err(|e| {
        InputError::ParseError {
            path: "<inline>".to_string(),
            message: e.to_string(),
        }
    })?;
    validate_patient_profile(&profile)?;
    Ok(profile)
}

/// Validate a set of input file paths and return any errors found.
pub fn validate_input_files(paths: &[PathBuf]) -> Vec<InputError> {
    let mut errors = Vec::new();

    for path in paths {
        if !path.exists() {
            errors.push(InputError::FileNotFound {
                path: path.display().to_string(),
            });
            continue;
        }

        if path.is_file() {
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();

            match ext.as_str() {
                "json" | "yaml" | "yml" | "toml" | "txt" | "csv" => {}
                _ => {
                    errors.push(InputError::UnsupportedFormat {
                        extension: ext,
                    });
                }
            }

            // Check file is readable and non-empty
            match fs::metadata(path) {
                Ok(meta) => {
                    if meta.len() == 0 {
                        errors.push(InputError::EmptyInput {
                            description: format!("File is empty: {}", path.display()),
                        });
                    }
                }
                Err(e) => {
                    errors.push(InputError::ParseError {
                        path: path.display().to_string(),
                        message: format!("Cannot read metadata: {}", e),
                    });
                }
            }
        }
    }

    errors
}

// ──────────────────────── Validation Functions ───────────────────────────

fn validate_patient_profile(profile: &PatientProfile) -> Result<()> {
    if profile.info.age_years < 0.0 || profile.info.age_years > 150.0 {
        bail!(InputError::ValidationError {
            message: format!("Invalid age: {}", profile.info.age_years),
        });
    }

    if profile.info.weight_kg <= 0.0 || profile.info.weight_kg > 500.0 {
        bail!(InputError::ValidationError {
            message: format!("Invalid weight: {} kg", profile.info.weight_kg),
        });
    }

    if profile.info.height_cm <= 0.0 || profile.info.height_cm > 300.0 {
        bail!(InputError::ValidationError {
            message: format!("Invalid height: {} cm", profile.info.height_cm),
        });
    }

    if profile.info.serum_creatinine < 0.0 {
        bail!(InputError::ValidationError {
            message: format!("Invalid serum creatinine: {}", profile.info.serum_creatinine),
        });
    }

    if let Some(egfr) = profile.egfr {
        if egfr < 0.0 || egfr > 200.0 {
            bail!(InputError::ValidationError {
                message: format!("Invalid eGFR: {}", egfr),
            });
        }
    }

    // Check for duplicate medications
    let mut seen_drugs = HashSet::new();
    for med in &profile.medications {
        let key = med.canonical_name();
        if !seen_drugs.insert(key.clone()) {
            warn!("Duplicate medication: {}", med.name);
        }
    }

    for med in &profile.medications {
        if med.dose_mg <= 0.0 {
            bail!(InputError::ValidationError {
                message: format!("Invalid dose for {}: {} mg", med.name, med.dose_mg),
            });
        }
        if med.frequency_hours <= 0.0 {
            bail!(InputError::ValidationError {
                message: format!(
                    "Invalid frequency for {}: {} hours",
                    med.name, med.frequency_hours
                ),
            });
        }
    }

    Ok(())
}

fn validate_medications(medications: &[ActiveMedication]) -> Result<()> {
    if medications.is_empty() {
        bail!(InputError::EmptyInput {
            description: "Medication list is empty".to_string(),
        });
    }

    for med in medications {
        if med.dose_mg <= 0.0 {
            bail!(InputError::ValidationError {
                message: format!("Invalid dose for {}: {} mg", med.name, med.dose_mg),
            });
        }
        if med.frequency_hours <= 0.0 {
            bail!(InputError::ValidationError {
                message: format!("Invalid frequency for {}: {} hours", med.name, med.frequency_hours),
            });
        }
        if med.name.is_empty() {
            bail!(InputError::ValidationError {
                message: "Medication name cannot be empty".to_string(),
            });
        }
    }

    Ok(())
}

// ────────────────────── Guideline Discovery ──────────────────────────────

/// Discovers guideline files in a directory tree.
pub struct GuidelineDiscovery {
    root: PathBuf,
    extensions: Vec<String>,
    max_depth: usize,
}

impl GuidelineDiscovery {
    pub fn new(root: &Path) -> Self {
        GuidelineDiscovery {
            root: root.to_path_buf(),
            extensions: vec![
                "json".to_string(),
                "yaml".to_string(),
                "yml".to_string(),
                "toml".to_string(),
            ],
            max_depth: 5,
        }
    }

    pub fn with_extensions(mut self, exts: Vec<String>) -> Self {
        self.extensions = exts;
        self
    }

    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Discover all guideline files under the root directory.
    pub fn discover(&self) -> Result<Vec<PathBuf>> {
        let mut results = Vec::new();
        self.walk_dir(&self.root, 0, &mut results)?;
        results.sort();
        Ok(results)
    }

    fn walk_dir(&self, dir: &Path, depth: usize, results: &mut Vec<PathBuf>) -> Result<()> {
        if depth > self.max_depth {
            return Ok(());
        }

        let entries = fs::read_dir(dir)
            .with_context(|| format!("Cannot read directory: {}", dir.display()))?;

        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                // Skip hidden directories
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if !name.starts_with('.') {
                        self.walk_dir(&path, depth + 1, results)?;
                    }
                }
            } else if path.is_file() {
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    if self.extensions.contains(&ext.to_lowercase()) {
                        results.push(path);
                    }
                }
            }
        }

        Ok(())
    }
}

// ──────────────────── Patient Profile Builder ────────────────────────────

/// Builds a patient profile from individual CLI flags.
pub struct PatientProfileBuilder {
    age: Option<f64>,
    sex: Option<Sex>,
    weight: Option<f64>,
    height: Option<f64>,
    creatinine: Option<f64>,
    egfr: Option<f64>,
    medications: Vec<ActiveMedication>,
    conditions: Vec<ClinicalCondition>,
    allergies: Vec<String>,
    albumin: Option<f64>,
    bilirubin: Option<f64>,
    inr: Option<f64>,
}

impl PatientProfileBuilder {
    pub fn new() -> Self {
        PatientProfileBuilder {
            age: None,
            sex: None,
            weight: None,
            height: None,
            creatinine: None,
            egfr: None,
            medications: Vec::new(),
            conditions: Vec::new(),
            allergies: Vec::new(),
            albumin: None,
            bilirubin: None,
            inr: None,
        }
    }

    pub fn age(mut self, age: f64) -> Self {
        self.age = Some(age);
        self
    }

    pub fn sex(mut self, sex: Sex) -> Self {
        self.sex = Some(sex);
        self
    }

    pub fn weight(mut self, weight_kg: f64) -> Self {
        self.weight = Some(weight_kg);
        self
    }

    pub fn height(mut self, height_cm: f64) -> Self {
        self.height = Some(height_cm);
        self
    }

    pub fn creatinine(mut self, scr: f64) -> Self {
        self.creatinine = Some(scr);
        self
    }

    pub fn egfr(mut self, egfr: f64) -> Self {
        self.egfr = Some(egfr);
        self
    }

    pub fn albumin(mut self, albumin: f64) -> Self {
        self.albumin = Some(albumin);
        self
    }

    pub fn bilirubin(mut self, bilirubin: f64) -> Self {
        self.bilirubin = Some(bilirubin);
        self
    }

    pub fn inr(mut self, inr: f64) -> Self {
        self.inr = Some(inr);
        self
    }

    pub fn add_medication(mut self, med: ActiveMedication) -> Self {
        self.medications.push(med);
        self
    }

    pub fn add_condition(mut self, condition: ClinicalCondition) -> Self {
        self.conditions.push(condition);
        self
    }

    pub fn add_allergy(mut self, allergy: String) -> Self {
        self.allergies.push(allergy);
        self
    }

    /// Parse a sex string like "male", "female", "m", "f".
    pub fn parse_sex(s: &str) -> Result<Sex> {
        match s.to_lowercase().as_str() {
            "male" | "m" => Ok(Sex::Male),
            "female" | "f" => Ok(Sex::Female),
            _ => bail!(InputError::ValidationError {
                message: format!("Invalid sex: '{}'. Use 'male' or 'female'", s),
            }),
        }
    }

    /// Parse a route string like "oral", "iv", "sc".
    pub fn parse_route(s: &str) -> Result<DrugRoute> {
        match s.to_lowercase().as_str() {
            "oral" | "po" => Ok(DrugRoute::Oral),
            "iv" | "intravenous" => Ok(DrugRoute::Intravenous),
            "sc" | "subcutaneous" | "subq" => Ok(DrugRoute::Subcutaneous),
            "im" | "intramuscular" => Ok(DrugRoute::Intramuscular),
            "td" | "transdermal" | "patch" => Ok(DrugRoute::Transdermal),
            "sl" | "sublingual" => Ok(DrugRoute::Sublingual),
            "pr" | "rectal" => Ok(DrugRoute::Rectal),
            "inh" | "inhalation" | "inhaled" => Ok(DrugRoute::Inhalation),
            _ => bail!(InputError::ValidationError {
                message: format!("Unknown route: '{}'. Use oral, iv, sc, im, td, sl, pr, inh", s),
            }),
        }
    }

    /// Build the patient profile, using defaults where values weren't provided.
    pub fn build(self) -> Result<PatientProfile> {
        let info = PatientInfo {
            age_years: self.age.unwrap_or(70.0),
            weight_kg: self.weight.unwrap_or(70.0),
            height_cm: self.height.unwrap_or(170.0),
            sex: self.sex.unwrap_or(Sex::Male),
            serum_creatinine: self.creatinine.unwrap_or(1.0),
            albumin: self.albumin.or(Some(4.0)),
            bilirubin: self.bilirubin.or(Some(1.0)),
            inr: self.inr.or(Some(1.0)),
            encephalopathy_grade: Some(0),
            ascites: Some(AscitesGrade::None),
        };

        let mut profile = PatientProfile::new(info);
        profile.medications = self.medications;
        profile.conditions = self.conditions;
        profile.egfr = self.egfr;
        profile.allergies = self.allergies;

        validate_patient_profile(&profile)?;
        Ok(profile)
    }
}

impl Default for PatientProfileBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────── Medication List Parser ─────────────────────────────

/// Parses medication lists from simple text/CSV format.
///
/// Supported formats:
/// - `name dose_mg` (one per line)
/// - `name,dose_mg,frequency_hours,route` (CSV)
/// - `name dose_mg route` (space-separated)
pub struct MedicationListParser {
    separator: char,
    has_header: bool,
}

impl MedicationListParser {
    pub fn new() -> Self {
        MedicationListParser {
            separator: ',',
            has_header: true,
        }
    }

    pub fn with_separator(mut self, sep: char) -> Self {
        self.separator = sep;
        self
    }

    pub fn with_header(mut self, has_header: bool) -> Self {
        self.has_header = has_header;
        self
    }

    /// Parse a text string into a list of medications.
    pub fn parse_text(&self, text: &str) -> Result<Vec<ActiveMedication>> {
        let mut medications = Vec::new();
        let lines: Vec<&str> = text.lines().collect();

        if lines.is_empty() {
            bail!(InputError::EmptyInput {
                description: "Medication text is empty".to_string(),
            });
        }

        let start_line = if self.has_header && lines.len() > 1 { 1 } else { 0 };

        for (i, line) in lines.iter().enumerate().skip(start_line) {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let med = self.parse_line(line, i + 1)?;
            medications.push(med);
        }

        Ok(medications)
    }

    fn parse_line(&self, line: &str, line_num: usize) -> Result<ActiveMedication> {
        // Try CSV format first
        let fields: Vec<&str> = if self.separator == ',' {
            line.split(',').map(|s| s.trim()).collect()
        } else {
            line.split_whitespace().collect()
        };

        if fields.is_empty() {
            bail!(InputError::ParseError {
                path: format!("line {}", line_num),
                message: "Empty line".to_string(),
            });
        }

        let name = fields[0].trim().to_string();
        if name.is_empty() {
            bail!(InputError::ParseError {
                path: format!("line {}", line_num),
                message: "Missing medication name".to_string(),
            });
        }

        let dose_mg = if fields.len() > 1 {
            fields[1].parse::<f64>().map_err(|_| InputError::ParseError {
                path: format!("line {}", line_num),
                message: format!("Invalid dose: '{}'", fields[1]),
            })?
        } else {
            bail!(InputError::ParseError {
                path: format!("line {}", line_num),
                message: "Missing dose value".to_string(),
            });
        };

        let mut med = ActiveMedication::new(&name, dose_mg);

        if fields.len() > 2 {
            if let Ok(freq) = fields[2].parse::<f64>() {
                med.frequency_hours = freq;
            }
        }

        if fields.len() > 3 {
            if let Ok(route) = PatientProfileBuilder::parse_route(fields[3]) {
                med.route = route;
            }
        }

        if fields.len() > 4 {
            med.drug_class = fields[4].to_string();
        }

        Ok(med)
    }
}

impl Default for MedicationListParser {
    fn default() -> Self {
        Self::new()
    }
}

// ───────────────────────── Utility Functions ─────────────────────────────

/// Merge a separately loaded medication list into a patient profile.
pub fn merge_medications(
    profile: &mut PatientProfile,
    additional: Vec<ActiveMedication>,
) {
    let existing: HashSet<String> = profile
        .medications
        .iter()
        .map(|m| m.canonical_name())
        .collect();

    for med in additional {
        if !existing.contains(&med.canonical_name()) {
            profile.medications.push(med);
        } else {
            debug!("Skipping duplicate medication: {}", med.name);
        }
    }
}

/// List of supported file extensions.
pub fn supported_extensions() -> Vec<&'static str> {
    vec!["json", "yaml", "yml", "toml", "txt", "csv"]
}

/// Check if a path has a supported extension.
pub fn is_supported_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| supported_extensions().contains(&e.to_lowercase().as_str()))
        .unwrap_or(false)
}

// ────────────────────────────── Tests ────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_clinical_condition_new() {
        let c = ClinicalCondition::new("I10", "Hypertension");
        assert_eq!(c.code, "I10");
        assert!(c.active);
        assert!(c.code_starts_with("I"));
    }

    #[test]
    fn test_clinical_condition_inactive() {
        let c = ClinicalCondition::inactive("I10", "Hypertension");
        assert!(!c.active);
    }

    #[test]
    fn test_active_medication_new() {
        let m = ActiveMedication::new("Warfarin", 5.0);
        assert_eq!(m.drug_id, DrugId::new("Warfarin"));
        assert_eq!(m.dose_mg, 5.0);
        assert_eq!(m.frequency_hours, 24.0);
        assert_eq!(m.route, DrugRoute::Oral);
    }

    #[test]
    fn test_active_medication_builder() {
        let m = ActiveMedication::new("Metformin", 500.0)
            .with_frequency(12.0)
            .with_route(DrugRoute::Oral)
            .with_class("biguanide")
            .with_start_date("2024-01-01")
            .with_indication("Type 2 Diabetes");
        assert_eq!(m.frequency_hours, 12.0);
        assert_eq!(m.drug_class, "biguanide");
        assert!(m.indication.is_some());
    }

    #[test]
    fn test_patient_profile_default() {
        let p = PatientProfile::default();
        assert_eq!(p.age(), 70.0);
        assert!(p.is_elderly());
        assert_eq!(p.medication_count(), 0);
    }

    #[test]
    fn test_patient_profile_conditions() {
        let p = PatientProfile::default().with_conditions(vec![
            ClinicalCondition::new("I10", "Hypertension"),
            ClinicalCondition::new("E11", "Type 2 Diabetes"),
        ]);
        assert!(p.has_condition_code("I10"));
        assert!(p.has_condition_code("E11"));
        assert!(!p.has_condition_code("N18"));
    }

    #[test]
    fn test_patient_drug_pairs() {
        let p = PatientProfile::default().with_medications(vec![
            ActiveMedication::new("DrugA", 10.0),
            ActiveMedication::new("DrugB", 20.0),
            ActiveMedication::new("DrugC", 30.0),
        ]);
        let pairs = p.drug_pairs();
        assert_eq!(pairs.len(), 3); // C(3,2) = 3
    }

    #[test]
    fn test_guideline_document_new() {
        let g = GuidelineDocument::new("Test Guideline");
        assert_eq!(g.name, "Test Guideline");
        assert_eq!(g.id, "test-guideline");
        assert_eq!(g.rule_count(), 0);
    }

    #[test]
    fn test_guideline_rule_new() {
        let r = GuidelineRule::new("R001", "Avoid combination", Severity::Major);
        assert_eq!(r.id, "R001");
        assert_eq!(r.severity, Severity::Major);
    }

    #[test]
    fn test_guideline_rules_for_severity() {
        let mut g = GuidelineDocument::new("Test");
        g.rules.push(GuidelineRule::new("R1", "Minor rule", Severity::Minor));
        g.rules.push(GuidelineRule::new("R2", "Major rule", Severity::Major));
        g.rules.push(GuidelineRule::new("R3", "Contra rule", Severity::Contraindicated));

        let major_plus = g.rules_for_severity(Severity::Major);
        assert_eq!(major_plus.len(), 2);
    }

    #[test]
    fn test_patient_profile_builder() {
        let profile = PatientProfileBuilder::new()
            .age(75.0)
            .sex(Sex::Female)
            .weight(65.0)
            .height(160.0)
            .creatinine(1.2)
            .egfr(55.0)
            .add_medication(ActiveMedication::new("Warfarin", 5.0))
            .add_condition(ClinicalCondition::new("I10", "Hypertension"))
            .build()
            .unwrap();

        assert_eq!(profile.age(), 75.0);
        assert_eq!(profile.sex(), Sex::Female);
        assert_eq!(profile.medication_count(), 1);
        assert_eq!(profile.renal_function(), RenalFunction::Moderate);
    }

    #[test]
    fn test_patient_profile_builder_defaults() {
        let profile = PatientProfileBuilder::new().build().unwrap();
        assert_eq!(profile.age(), 70.0);
        assert_eq!(profile.sex(), Sex::Male);
    }

    #[test]
    fn test_parse_sex() {
        assert_eq!(PatientProfileBuilder::parse_sex("male").unwrap(), Sex::Male);
        assert_eq!(PatientProfileBuilder::parse_sex("f").unwrap(), Sex::Female);
        assert_eq!(PatientProfileBuilder::parse_sex("Female").unwrap(), Sex::Female);
        assert!(PatientProfileBuilder::parse_sex("unknown").is_err());
    }

    #[test]
    fn test_parse_route() {
        assert_eq!(PatientProfileBuilder::parse_route("oral").unwrap(), DrugRoute::Oral);
        assert_eq!(PatientProfileBuilder::parse_route("iv").unwrap(), DrugRoute::Intravenous);
        assert_eq!(PatientProfileBuilder::parse_route("SC").unwrap(), DrugRoute::Subcutaneous);
        assert!(PatientProfileBuilder::parse_route("xyz").is_err());
    }

    #[test]
    fn test_medication_list_parser_csv() {
        let parser = MedicationListParser::new();
        let text = "name,dose,frequency,route\nWarfarin,5,24,oral\nMetformin,500,12,oral\n";
        let meds = parser.parse_text(text).unwrap();
        assert_eq!(meds.len(), 2);
        assert_eq!(meds[0].name, "Warfarin");
        assert_eq!(meds[0].dose_mg, 5.0);
        assert_eq!(meds[1].name, "Metformin");
        assert_eq!(meds[1].dose_mg, 500.0);
    }

    #[test]
    fn test_medication_list_parser_with_comments() {
        let parser = MedicationListParser::new().with_header(false);
        let text = "# This is a comment\nAspirin,81\n\n# Another comment\nLisinopril,10\n";
        let meds = parser.parse_text(text).unwrap();
        assert_eq!(meds.len(), 2);
    }

    #[test]
    fn test_merge_medications() {
        let mut profile = PatientProfile::default().with_medications(vec![
            ActiveMedication::new("Warfarin", 5.0),
        ]);
        let additional = vec![
            ActiveMedication::new("Warfarin", 5.0), // duplicate
            ActiveMedication::new("Aspirin", 81.0),  // new
        ];
        merge_medications(&mut profile, additional);
        assert_eq!(profile.medication_count(), 2);
    }

    #[test]
    fn test_validate_input_files() {
        let paths = vec![
            PathBuf::from("/nonexistent/file.json"),
            PathBuf::from("/nonexistent/file.xyz"),
        ];
        let errors = validate_input_files(&paths);
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_supported_extensions() {
        let exts = supported_extensions();
        assert!(exts.contains(&"json"));
        assert!(exts.contains(&"yaml"));
        assert!(exts.contains(&"toml"));
    }

    #[test]
    fn test_is_supported_file() {
        assert!(is_supported_file(Path::new("test.json")));
        assert!(is_supported_file(Path::new("test.yaml")));
        assert!(!is_supported_file(Path::new("test.exe")));
    }

    #[test]
    fn test_guideline_discovery_new() {
        let discovery = GuidelineDiscovery::new(Path::new("/tmp"));
        assert_eq!(discovery.max_depth, 5);
        assert_eq!(discovery.extensions.len(), 4);
    }

    #[test]
    fn test_guideline_discovery_with_extensions() {
        let discovery = GuidelineDiscovery::new(Path::new("/tmp"))
            .with_extensions(vec!["json".to_string()])
            .with_max_depth(2);
        assert_eq!(discovery.extensions.len(), 1);
        assert_eq!(discovery.max_depth, 2);
    }

    #[test]
    fn test_load_patient_json() {
        let dir = std::env::temp_dir().join("guardpharma_test_input");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("patient.json");

        let profile = PatientProfile::default().with_medications(vec![
            ActiveMedication::new("Warfarin", 5.0),
        ]);
        let json = serde_json::to_string_pretty(&profile).unwrap();
        fs::write(&path, &json).unwrap();

        let loaded = load_patient(&path).unwrap();
        assert_eq!(loaded.medication_count(), 1);

        let _ = fs::remove_file(&path);
        let _ = fs::remove_dir(&dir);
    }

    #[test]
    fn test_load_patient_nonexistent() {
        assert!(load_patient(Path::new("/nonexistent/patient.json")).is_err());
    }

    #[test]
    fn test_parse_inline_patient() {
        let json = r#"{"info":{"age_years":65,"weight_kg":80,"height_cm":175,"sex":"Male","serum_creatinine":1.0},"conditions":[],"medications":[]}"#;
        let profile = parse_inline_patient(json).unwrap();
        assert_eq!(profile.age(), 65.0);
    }

    #[test]
    fn test_patient_renal_function() {
        let profile = PatientProfile::default().with_egfr(45.0);
        assert_eq!(profile.renal_function(), RenalFunction::Moderate);

        let profile2 = PatientProfile::default();
        assert_eq!(profile2.renal_function(), RenalFunction::Normal);
    }

    #[test]
    fn test_validate_patient_invalid_age() {
        let mut profile = PatientProfile::default();
        profile.info.age_years = -5.0;
        assert!(validate_patient_profile(&profile).is_err());
    }

    #[test]
    fn test_validate_patient_invalid_weight() {
        let mut profile = PatientProfile::default();
        profile.info.weight_kg = 0.0;
        assert!(validate_patient_profile(&profile).is_err());
    }

    #[test]
    fn test_validate_medications_empty() {
        assert!(validate_medications(&[]).is_err());
    }

    #[test]
    fn test_validate_medications_invalid_dose() {
        let meds = vec![ActiveMedication::new("Test", -5.0)];
        assert!(validate_medications(&meds).is_err());
    }

    #[test]
    fn test_clinical_condition_display() {
        let c = ClinicalCondition::new("I10", "Hypertension");
        let s = format!("{}", c);
        assert!(s.contains("Hypertension"));
        assert!(s.contains("I10"));
        assert!(s.contains("active"));
    }

    #[test]
    fn test_active_medication_display() {
        let m = ActiveMedication::new("Warfarin", 5.0);
        let s = format!("{}", m);
        assert!(s.contains("Warfarin"));
        assert!(s.contains("5"));
    }

    #[test]
    fn test_guideline_document_display() {
        let mut g = GuidelineDocument::new("Anticoag Guide");
        g.rules.push(GuidelineRule::new("R1", "Rule 1", Severity::Major));
        let s = format!("{}", g);
        assert!(s.contains("Anticoag Guide"));
        assert!(s.contains("1 rules"));
    }
}
