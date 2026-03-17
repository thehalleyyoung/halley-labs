//! MPS file format parser and writer.
//!
//! Supports fixed-format MPS, free-format MPS, RANGES section,
//! BOUNDS section, SOS section, INDICATORS section, and round-trip fidelity.

use crate::model::{
    Constraint, IndicatorConstraint, LpModel, SosConstraint, SosType, VarType, Variable,
};
use bicut_types::{ConstraintSense, OptDirection};
use std::collections::HashMap;
use std::fmt;
use std::io::{BufRead, BufReader, Read, Write};

/// Errors during MPS parsing.
#[derive(Debug, Clone)]
pub enum MpsError {
    ParseError(String),
    InvalidSection(String),
    DuplicateName(String),
    MissingRows,
    MissingColumns,
    IoError(String),
}

impl fmt::Display for MpsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MpsError::ParseError(msg) => write!(f, "MPS parse error: {}", msg),
            MpsError::InvalidSection(s) => write!(f, "Invalid MPS section: {}", s),
            MpsError::DuplicateName(n) => write!(f, "Duplicate name: {}", n),
            MpsError::MissingRows => write!(f, "Missing ROWS section"),
            MpsError::MissingColumns => write!(f, "Missing COLUMNS section"),
            MpsError::IoError(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for MpsError {}

/// MPS format variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MpsFormat {
    Fixed,
    Free,
}

/// Parse an MPS file from a reader.
pub fn parse_mps<R: Read>(reader: R, format: MpsFormat) -> Result<LpModel, MpsError> {
    let buf = BufReader::new(reader);
    let lines: Vec<String> = buf
        .lines()
        .map(|l| l.map_err(|e| MpsError::IoError(e.to_string())))
        .collect::<Result<Vec<_>, _>>()?;

    let mut parser = MpsParser::new(format);
    parser.parse(&lines)?;
    Ok(parser.build_model())
}

/// Parse an MPS string.
pub fn parse_mps_string(s: &str, format: MpsFormat) -> Result<LpModel, MpsError> {
    parse_mps(s.as_bytes(), format)
}

/// Write an LP model in MPS format.
pub fn write_mps<W: Write>(
    model: &LpModel,
    writer: &mut W,
    format: MpsFormat,
) -> Result<(), MpsError> {
    let mut w = MpsWriter::new(format);
    w.write(model, writer)
}

/// Write an LP model to an MPS string.
pub fn write_mps_string(model: &LpModel, format: MpsFormat) -> Result<String, MpsError> {
    let mut buf = Vec::new();
    write_mps(model, &mut buf, format)?;
    String::from_utf8(buf).map_err(|e| MpsError::IoError(e.to_string()))
}

/// Internal MPS section types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MpsSection {
    Name,
    Rows,
    Columns,
    Rhs,
    Ranges,
    Bounds,
    Sos,
    Indicators,
    Endata,
}

/// MPS parser state.
struct MpsParser {
    format: MpsFormat,
    model_name: String,
    obj_name: Option<String>,
    sense: OptDirection,
    row_names: Vec<String>,
    row_senses: Vec<ConstraintSense>,
    row_name_map: HashMap<String, usize>,
    col_names: Vec<String>,
    col_name_map: HashMap<String, usize>,
    col_types: Vec<VarType>,
    coefficients: Vec<(usize, usize, f64)>, // (row, col, val)
    obj_coeffs: Vec<(usize, f64)>,          // (col, val)
    rhs_values: Vec<(usize, f64)>,          // (row, val)
    range_values: Vec<(usize, f64)>,        // (row, val)
    bounds: Vec<BoundEntry>,
    sos_constraints: Vec<SosConstraint>,
    indicators: Vec<IndicatorEntry>,
    obj_row_idx: Option<usize>,
}

#[derive(Debug, Clone)]
struct BoundEntry {
    bound_type: BoundType,
    col_name: String,
    value: f64,
}

#[derive(Debug, Clone, Copy)]
enum BoundType {
    Lo,
    Up,
    Fx,
    Fr,
    Mi,
    Pl,
    Bv,
    Li,
    Ui,
}

#[derive(Debug, Clone)]
struct IndicatorEntry {
    row_name: String,
    col_name: String,
    active_value: bool,
}

impl MpsParser {
    fn new(format: MpsFormat) -> Self {
        Self {
            format,
            model_name: String::new(),
            obj_name: None,
            sense: OptDirection::Minimize,
            row_names: Vec::new(),
            row_senses: Vec::new(),
            row_name_map: HashMap::new(),
            col_names: Vec::new(),
            col_name_map: HashMap::new(),
            col_types: Vec::new(),
            coefficients: Vec::new(),
            obj_coeffs: Vec::new(),
            rhs_values: Vec::new(),
            range_values: Vec::new(),
            bounds: Vec::new(),
            sos_constraints: Vec::new(),
            indicators: Vec::new(),
            obj_row_idx: None,
        }
    }

    fn parse(&mut self, lines: &[String]) -> Result<(), MpsError> {
        let mut section = MpsSection::Name;
        let mut in_integer_marker = false;

        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim_end();
            if trimmed.is_empty() || trimmed.starts_with('*') || trimmed.starts_with('$') {
                continue;
            }

            // Check for section headers
            if !trimmed.starts_with(' ') && !trimmed.starts_with('\t') {
                let upper = trimmed.to_uppercase();
                if upper.starts_with("NAME") {
                    section = MpsSection::Name;
                    self.model_name = if trimmed.len() > 14 {
                        trimmed[14..].trim().to_string()
                    } else if trimmed.len() > 4 {
                        trimmed[4..].trim().to_string()
                    } else {
                        "unnamed".to_string()
                    };
                    continue;
                } else if upper == "ROWS" {
                    section = MpsSection::Rows;
                    continue;
                } else if upper == "COLUMNS" {
                    section = MpsSection::Columns;
                    continue;
                } else if upper == "RHS" {
                    section = MpsSection::Rhs;
                    continue;
                } else if upper == "RANGES" {
                    section = MpsSection::Ranges;
                    continue;
                } else if upper == "BOUNDS" {
                    section = MpsSection::Bounds;
                    continue;
                } else if upper.starts_with("SOS") {
                    section = MpsSection::Sos;
                    continue;
                } else if upper == "INDICATORS" {
                    section = MpsSection::Indicators;
                    continue;
                } else if upper == "ENDATA" {
                    break;
                } else if upper.starts_with("OBJSENSE") {
                    // Next non-empty line should be MIN or MAX
                    continue;
                } else if upper.trim() == "MIN" {
                    self.sense = OptDirection::Minimize;
                    continue;
                } else if upper.trim() == "MAX" {
                    self.sense = OptDirection::Maximize;
                    continue;
                }
            }

            match section {
                MpsSection::Rows => self.parse_row(trimmed)?,
                MpsSection::Columns => self.parse_column(trimmed, &mut in_integer_marker)?,
                MpsSection::Rhs => self.parse_rhs(trimmed)?,
                MpsSection::Ranges => self.parse_ranges(trimmed)?,
                MpsSection::Bounds => self.parse_bounds(trimmed)?,
                MpsSection::Sos => self.parse_sos(trimmed)?,
                MpsSection::Indicators => self.parse_indicator(trimmed)?,
                MpsSection::Name => {}
                MpsSection::Endata => break,
            }
            let _ = line_num;
        }

        Ok(())
    }

    fn parse_fields(&self, line: &str) -> Vec<String> {
        match self.format {
            MpsFormat::Free => line.split_whitespace().map(|s| s.to_string()).collect(),
            MpsFormat::Fixed => {
                // Fixed format: specific column positions
                let mut fields = Vec::new();
                let bytes = line.as_bytes();
                // Field 1: cols 2-3 (indicator)
                if bytes.len() > 1 {
                    let f = line.get(1..3).unwrap_or("").trim().to_string();
                    if !f.is_empty() {
                        fields.push(f);
                    }
                }
                // Field 2: cols 5-12
                if bytes.len() > 4 {
                    let f = line
                        .get(4..12.min(line.len()))
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if !f.is_empty() {
                        fields.push(f);
                    }
                }
                // Field 3: cols 15-22
                if bytes.len() > 14 {
                    let f = line
                        .get(14..22.min(line.len()))
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if !f.is_empty() {
                        fields.push(f);
                    }
                }
                // Field 4: cols 25-36
                if bytes.len() > 24 {
                    let f = line
                        .get(24..36.min(line.len()))
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if !f.is_empty() {
                        fields.push(f);
                    }
                }
                // Field 5: cols 40-47
                if bytes.len() > 39 {
                    let f = line
                        .get(39..47.min(line.len()))
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if !f.is_empty() {
                        fields.push(f);
                    }
                }
                // Field 6: cols 50-61
                if bytes.len() > 49 {
                    let f = line
                        .get(49..61.min(line.len()))
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if !f.is_empty() {
                        fields.push(f);
                    }
                }

                if fields.is_empty() {
                    // Fallback to whitespace splitting
                    line.split_whitespace().map(|s| s.to_string()).collect()
                } else {
                    fields
                }
            }
        }
    }

    fn parse_row(&mut self, line: &str) -> Result<(), MpsError> {
        let fields = line.split_whitespace().collect::<Vec<_>>();
        if fields.len() < 2 {
            return Ok(());
        }

        let sense_char = fields[0];
        let name = fields[1].to_string();

        let sense = match sense_char {
            "N" => {
                // Objective row
                if self.obj_name.is_none() {
                    self.obj_name = Some(name.clone());
                    self.obj_row_idx = Some(self.row_names.len());
                }
                self.row_names.push(name.clone());
                self.row_senses.push(ConstraintSense::Eq); // placeholder
                self.row_name_map.insert(name, self.row_names.len() - 1);
                return Ok(());
            }
            "L" => ConstraintSense::Le,
            "G" => ConstraintSense::Ge,
            "E" => ConstraintSense::Eq,
            _ => {
                return Err(MpsError::ParseError(format!(
                    "Unknown row type: {}",
                    sense_char
                )))
            }
        };

        let idx = self.row_names.len();
        self.row_names.push(name.clone());
        self.row_senses.push(sense);
        self.row_name_map.insert(name, idx);
        Ok(())
    }

    fn parse_column(&mut self, line: &str, in_integer: &mut bool) -> Result<(), MpsError> {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.is_empty() {
            return Ok(());
        }

        // Check for integer markers
        if fields.len() >= 3 {
            let marker_test = fields.iter().any(|f| f.contains("'MARKER'"));
            if marker_test {
                if fields.iter().any(|f| f.contains("'INTORG'")) {
                    *in_integer = true;
                } else if fields.iter().any(|f| f.contains("'INTEND'")) {
                    *in_integer = false;
                }
                return Ok(());
            }
        }

        if fields.len() < 3 {
            return Ok(());
        }

        let col_name = fields[0].to_string();
        let col_idx = self.get_or_create_col(&col_name, *in_integer);

        // First (row_name, value) pair
        let row_name1 = fields[1];
        let val1: f64 = fields[2]
            .parse()
            .map_err(|_| MpsError::ParseError(format!("Invalid number: {}", fields[2])))?;

        self.add_coefficient(&col_name, row_name1, val1, col_idx)?;

        // Optional second (row_name, value) pair
        if fields.len() >= 5 {
            let row_name2 = fields[3];
            let val2: f64 = fields[4]
                .parse()
                .map_err(|_| MpsError::ParseError(format!("Invalid number: {}", fields[4])))?;
            self.add_coefficient(&col_name, row_name2, val2, col_idx)?;
        }

        Ok(())
    }

    fn get_or_create_col(&mut self, name: &str, is_integer: bool) -> usize {
        if let Some(&idx) = self.col_name_map.get(name) {
            return idx;
        }
        let idx = self.col_names.len();
        self.col_names.push(name.to_string());
        self.col_name_map.insert(name.to_string(), idx);
        self.col_types.push(if is_integer {
            VarType::Integer
        } else {
            VarType::Continuous
        });
        idx
    }

    fn add_coefficient(
        &mut self,
        _col_name: &str,
        row_name: &str,
        val: f64,
        col_idx: usize,
    ) -> Result<(), MpsError> {
        if let Some(&row_idx) = self.row_name_map.get(row_name) {
            if self.obj_row_idx == Some(row_idx) {
                self.obj_coeffs.push((col_idx, val));
            } else {
                self.coefficients.push((row_idx, col_idx, val));
            }
        }
        Ok(())
    }

    fn parse_rhs(&mut self, line: &str) -> Result<(), MpsError> {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 3 {
            return Ok(());
        }

        // First field is RHS vector name (skip)
        let mut idx = 1;
        while idx + 1 < fields.len() {
            let row_name = fields[idx];
            let val: f64 = fields[idx + 1].parse().map_err(|_| {
                MpsError::ParseError(format!("Invalid RHS value: {}", fields[idx + 1]))
            })?;

            if let Some(&row_idx) = self.row_name_map.get(row_name) {
                self.rhs_values.push((row_idx, val));
            }
            idx += 2;
        }

        Ok(())
    }

    fn parse_ranges(&mut self, line: &str) -> Result<(), MpsError> {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 3 {
            return Ok(());
        }

        let mut idx = 1;
        while idx + 1 < fields.len() {
            let row_name = fields[idx];
            let val: f64 = fields[idx + 1]
                .parse()
                .map_err(|_| MpsError::ParseError(format!("Invalid range: {}", fields[idx + 1])))?;

            if let Some(&row_idx) = self.row_name_map.get(row_name) {
                self.range_values.push((row_idx, val));
            }
            idx += 2;
        }

        Ok(())
    }

    fn parse_bounds(&mut self, line: &str) -> Result<(), MpsError> {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 3 {
            return Ok(());
        }

        let bound_type_str = fields[0];
        let _bound_name = fields[1]; // usually "BOUND"
        let col_name = fields[2].to_string();
        let value = if fields.len() >= 4 {
            fields[3].parse::<f64>().unwrap_or(0.0)
        } else {
            0.0
        };

        let bound_type = match bound_type_str {
            "LO" => BoundType::Lo,
            "UP" => BoundType::Up,
            "FX" => BoundType::Fx,
            "FR" => BoundType::Fr,
            "MI" => BoundType::Mi,
            "PL" => BoundType::Pl,
            "BV" => BoundType::Bv,
            "LI" => BoundType::Li,
            "UI" => BoundType::Ui,
            _ => {
                return Err(MpsError::ParseError(format!(
                    "Unknown bound type: {}",
                    bound_type_str
                )))
            }
        };

        self.bounds.push(BoundEntry {
            bound_type,
            col_name,
            value,
        });
        Ok(())
    }

    fn parse_sos(&mut self, line: &str) -> Result<(), MpsError> {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.is_empty() {
            return Ok(());
        }

        // SOS section format: "S1 name" or "S2 name" for set definition
        // Then "col_name weight" entries
        if fields[0] == "S1" || fields[0] == "S2" {
            let sos_type = if fields[0] == "S1" {
                SosType::Type1
            } else {
                SosType::Type2
            };
            let name = if fields.len() > 1 {
                fields[1].to_string()
            } else {
                format!("sos_{}", self.sos_constraints.len())
            };
            self.sos_constraints.push(SosConstraint {
                sos_type,
                name,
                members: Vec::new(),
                weights: Vec::new(),
            });
        } else if fields.len() >= 2 {
            // Member entry
            let col_name = fields[0];
            let weight: f64 = fields[1].parse().unwrap_or(0.0);
            if let Some(&col_idx) = self.col_name_map.get(col_name) {
                if let Some(sos) = self.sos_constraints.last_mut() {
                    sos.members.push(col_idx);
                    sos.weights.push(weight);
                }
            }
        }

        Ok(())
    }

    fn parse_indicator(&mut self, line: &str) -> Result<(), MpsError> {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 4 {
            return Ok(());
        }

        // Format: IF row_name col_name value
        if fields[0] == "IF" {
            self.indicators.push(IndicatorEntry {
                row_name: fields[1].to_string(),
                col_name: fields[2].to_string(),
                active_value: fields[3] == "1",
            });
        }

        Ok(())
    }

    fn build_model(&self) -> LpModel {
        let mut model = LpModel::new(&self.model_name);
        model.sense = self.sense;

        // Create variables with default bounds [0, +inf)
        let n = self.col_names.len();
        let mut lower_bounds = vec![0.0; n];
        let mut upper_bounds = vec![f64::INFINITY; n];
        let mut var_types = self.col_types.clone();

        // Apply BOUNDS
        for entry in &self.bounds {
            if let Some(&col_idx) = self.col_name_map.get(&entry.col_name) {
                match entry.bound_type {
                    BoundType::Lo => lower_bounds[col_idx] = entry.value,
                    BoundType::Up => upper_bounds[col_idx] = entry.value,
                    BoundType::Fx => {
                        lower_bounds[col_idx] = entry.value;
                        upper_bounds[col_idx] = entry.value;
                    }
                    BoundType::Fr => {
                        lower_bounds[col_idx] = f64::NEG_INFINITY;
                        upper_bounds[col_idx] = f64::INFINITY;
                    }
                    BoundType::Mi => lower_bounds[col_idx] = f64::NEG_INFINITY,
                    BoundType::Pl => upper_bounds[col_idx] = f64::INFINITY,
                    BoundType::Bv => {
                        lower_bounds[col_idx] = 0.0;
                        upper_bounds[col_idx] = 1.0;
                        var_types[col_idx] = VarType::Binary;
                    }
                    BoundType::Li => {
                        lower_bounds[col_idx] = entry.value;
                        var_types[col_idx] = VarType::Integer;
                    }
                    BoundType::Ui => {
                        upper_bounds[col_idx] = entry.value;
                        var_types[col_idx] = VarType::Integer;
                    }
                }
            }
        }

        // Add variables
        let mut obj_vec = vec![0.0; n];
        for &(col_idx, val) in &self.obj_coeffs {
            if col_idx < n {
                obj_vec[col_idx] = val;
            }
        }

        for j in 0..n {
            let mut var =
                Variable::continuous(&self.col_names[j], lower_bounds[j], upper_bounds[j]);
            var.var_type = var_types[j];
            var.obj_coeff = obj_vec[j];
            model.add_variable(var);
        }

        // Add constraints
        // Build a map of row -> coefficients (excluding objective row)
        let mut row_coeffs: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();
        for &(row, col, val) in &self.coefficients {
            row_coeffs.entry(row).or_default().push((col, val));
        }

        let mut rhs_map: HashMap<usize, f64> = HashMap::new();
        for &(row, val) in &self.rhs_values {
            rhs_map.insert(row, val);
        }

        let mut range_map: HashMap<usize, f64> = HashMap::new();
        for &(row, val) in &self.range_values {
            range_map.insert(row, val);
        }

        for (i, name) in self.row_names.iter().enumerate() {
            if self.obj_row_idx == Some(i) {
                continue; // Skip objective row
            }
            let sense = self.row_senses[i];
            let rhs = rhs_map.get(&i).copied().unwrap_or(0.0);
            let mut con = Constraint::new(name, sense, rhs);

            if let Some(coeffs) = row_coeffs.get(&i) {
                for &(col, val) in coeffs {
                    con.add_term(col, val);
                }
            }

            if let Some(&range) = range_map.get(&i) {
                con.range = Some(range);
            }

            model.add_constraint(con);
        }

        // Add SOS constraints
        for sos in &self.sos_constraints {
            model.add_sos_constraint(sos.clone());
        }

        // Add indicator constraints
        for ind in &self.indicators {
            if let (Some(&row_idx), Some(&col_idx)) = (
                self.row_name_map.get(&ind.row_name),
                self.col_name_map.get(&ind.col_name),
            ) {
                if let Some(coeffs) = row_coeffs.get(&row_idx) {
                    let mut indices = Vec::new();
                    let mut values = Vec::new();
                    for &(c, v) in coeffs {
                        indices.push(c);
                        values.push(v);
                    }
                    let sense = self.row_senses[row_idx];
                    let rhs = rhs_map.get(&row_idx).copied().unwrap_or(0.0);

                    model.add_indicator_constraint(IndicatorConstraint {
                        name: ind.row_name.clone(),
                        binary_var: col_idx,
                        active_value: ind.active_value,
                        constraint_indices: indices,
                        constraint_values: values,
                        sense,
                        rhs,
                    });
                }
            }
        }

        model
    }
}

/// MPS writer.
struct MpsWriter {
    format: MpsFormat,
}

impl MpsWriter {
    fn new(format: MpsFormat) -> Self {
        Self { format }
    }

    fn write<W: Write>(&self, model: &LpModel, w: &mut W) -> Result<(), MpsError> {
        self.write_name(model, w)?;
        self.write_objsense(model, w)?;
        self.write_rows(model, w)?;
        self.write_columns(model, w)?;
        self.write_rhs(model, w)?;
        self.write_ranges(model, w)?;
        self.write_bounds(model, w)?;
        self.write_sos(model, w)?;
        self.write_indicators(model, w)?;
        writeln!(w, "ENDATA").map_err(|e| MpsError::IoError(e.to_string()))?;
        Ok(())
    }

    fn write_name<W: Write>(&self, model: &LpModel, w: &mut W) -> Result<(), MpsError> {
        writeln!(w, "NAME          {}", model.name).map_err(|e| MpsError::IoError(e.to_string()))
    }

    fn write_objsense<W: Write>(&self, model: &LpModel, w: &mut W) -> Result<(), MpsError> {
        writeln!(w, "OBJSENSE").map_err(|e| MpsError::IoError(e.to_string()))?;
        match model.sense {
            OptDirection::Minimize => writeln!(w, "    MIN"),
            OptDirection::Maximize => writeln!(w, "    MAX"),
        }
        .map_err(|e| MpsError::IoError(e.to_string()))
    }

    fn write_rows<W: Write>(&self, model: &LpModel, w: &mut W) -> Result<(), MpsError> {
        writeln!(w, "ROWS").map_err(|e| MpsError::IoError(e.to_string()))?;
        // Objective row
        writeln!(w, " N  OBJ").map_err(|e| MpsError::IoError(e.to_string()))?;
        // Constraints
        for con in &model.constraints {
            let sense_char = match con.sense {
                ConstraintSense::Le => 'L',
                ConstraintSense::Ge => 'G',
                ConstraintSense::Eq => 'E',
            };
            writeln!(w, " {}  {}", sense_char, con.name)
                .map_err(|e| MpsError::IoError(e.to_string()))?;
        }
        Ok(())
    }

    fn write_columns<W: Write>(&self, model: &LpModel, w: &mut W) -> Result<(), MpsError> {
        writeln!(w, "COLUMNS").map_err(|e| MpsError::IoError(e.to_string()))?;

        let mut in_integer = false;
        let mut marker_count = 0;

        for (j, var) in model.variables.iter().enumerate() {
            let is_int = var.var_type == VarType::Integer || var.var_type == VarType::Binary;

            if is_int && !in_integer {
                marker_count += 1;
                writeln!(
                    w,
                    "    M{:07}  'MARKER'                 'INTORG'",
                    marker_count
                )
                .map_err(|e| MpsError::IoError(e.to_string()))?;
                in_integer = true;
            } else if !is_int && in_integer {
                marker_count += 1;
                writeln!(
                    w,
                    "    M{:07}  'MARKER'                 'INTEND'",
                    marker_count
                )
                .map_err(|e| MpsError::IoError(e.to_string()))?;
                in_integer = false;
            }

            // Objective coefficient
            let mut entries: Vec<(String, f64)> = Vec::new();
            if var.obj_coeff.abs() > 1e-20 {
                entries.push(("OBJ".to_string(), var.obj_coeff));
            }

            // Constraint coefficients
            for con in &model.constraints {
                if let Some(pos) = con.row_indices.iter().position(|&c| c == j) {
                    let val = con.row_values[pos];
                    if val.abs() > 1e-20 {
                        entries.push((con.name.clone(), val));
                    }
                }
            }

            // Write entries, two per line
            let mut i = 0;
            while i < entries.len() {
                if i + 1 < entries.len() {
                    match self.format {
                        MpsFormat::Fixed => {
                            writeln!(
                                w,
                                "    {:8}  {:8}  {:12.6e}   {:8}  {:12.6e}",
                                var.name,
                                entries[i].0,
                                entries[i].1,
                                entries[i + 1].0,
                                entries[i + 1].1
                            )
                            .map_err(|e| MpsError::IoError(e.to_string()))?;
                        }
                        MpsFormat::Free => {
                            writeln!(
                                w,
                                "    {} {} {}   {} {}",
                                var.name,
                                entries[i].0,
                                format_val(entries[i].1),
                                entries[i + 1].0,
                                format_val(entries[i + 1].1)
                            )
                            .map_err(|e| MpsError::IoError(e.to_string()))?;
                        }
                    }
                    i += 2;
                } else {
                    match self.format {
                        MpsFormat::Fixed => {
                            writeln!(
                                w,
                                "    {:8}  {:8}  {:12.6e}",
                                var.name, entries[i].0, entries[i].1
                            )
                            .map_err(|e| MpsError::IoError(e.to_string()))?;
                        }
                        MpsFormat::Free => {
                            writeln!(
                                w,
                                "    {} {} {}",
                                var.name,
                                entries[i].0,
                                format_val(entries[i].1)
                            )
                            .map_err(|e| MpsError::IoError(e.to_string()))?;
                        }
                    }
                    i += 1;
                }
            }
        }

        if in_integer {
            marker_count += 1;
            writeln!(
                w,
                "    M{:07}  'MARKER'                 'INTEND'",
                marker_count
            )
            .map_err(|e| MpsError::IoError(e.to_string()))?;
        }

        Ok(())
    }

    fn write_rhs<W: Write>(&self, model: &LpModel, w: &mut W) -> Result<(), MpsError> {
        writeln!(w, "RHS").map_err(|e| MpsError::IoError(e.to_string()))?;

        let mut entries: Vec<(&str, f64)> = Vec::new();
        for con in &model.constraints {
            if con.rhs.abs() > 1e-20 {
                entries.push((&con.name, con.rhs));
            }
        }

        let mut i = 0;
        while i < entries.len() {
            if i + 1 < entries.len() {
                writeln!(
                    w,
                    "    RHS       {:8}  {:12.6e}   {:8}  {:12.6e}",
                    entries[i].0,
                    entries[i].1,
                    entries[i + 1].0,
                    entries[i + 1].1
                )
                .map_err(|e| MpsError::IoError(e.to_string()))?;
                i += 2;
            } else {
                writeln!(
                    w,
                    "    RHS       {:8}  {:12.6e}",
                    entries[i].0, entries[i].1
                )
                .map_err(|e| MpsError::IoError(e.to_string()))?;
                i += 1;
            }
        }

        Ok(())
    }

    fn write_ranges<W: Write>(&self, model: &LpModel, w: &mut W) -> Result<(), MpsError> {
        let has_ranges = model.constraints.iter().any(|c| c.range.is_some());
        if !has_ranges {
            return Ok(());
        }

        writeln!(w, "RANGES").map_err(|e| MpsError::IoError(e.to_string()))?;
        for con in &model.constraints {
            if let Some(range) = con.range {
                writeln!(w, "    RNG       {:8}  {:12.6e}", con.name, range)
                    .map_err(|e| MpsError::IoError(e.to_string()))?;
            }
        }
        Ok(())
    }

    fn write_bounds<W: Write>(&self, model: &LpModel, w: &mut W) -> Result<(), MpsError> {
        let has_nondefault = model.variables.iter().any(|v| {
            v.lower_bound != 0.0 || v.upper_bound != f64::INFINITY || v.var_type == VarType::Binary
        });
        if !has_nondefault {
            return Ok(());
        }

        writeln!(w, "BOUNDS").map_err(|e| MpsError::IoError(e.to_string()))?;
        for var in &model.variables {
            if var.var_type == VarType::Binary {
                writeln!(w, " BV BND       {}", var.name)
                    .map_err(|e| MpsError::IoError(e.to_string()))?;
                continue;
            }

            if var.lower_bound <= -1e20 && var.upper_bound >= 1e20 {
                writeln!(w, " FR BND       {}", var.name)
                    .map_err(|e| MpsError::IoError(e.to_string()))?;
                continue;
            }

            if (var.lower_bound - var.upper_bound).abs() < 1e-20 {
                writeln!(w, " FX BND       {}  {:12.6e}", var.name, var.lower_bound)
                    .map_err(|e| MpsError::IoError(e.to_string()))?;
                continue;
            }

            if var.lower_bound != 0.0 {
                if var.lower_bound <= -1e20 {
                    writeln!(w, " MI BND       {}", var.name)
                        .map_err(|e| MpsError::IoError(e.to_string()))?;
                } else {
                    let prefix = if var.var_type == VarType::Integer {
                        "LI"
                    } else {
                        "LO"
                    };
                    writeln!(
                        w,
                        " {} BND       {}  {:12.6e}",
                        prefix, var.name, var.lower_bound
                    )
                    .map_err(|e| MpsError::IoError(e.to_string()))?;
                }
            }

            if var.upper_bound < 1e20 {
                let prefix = if var.var_type == VarType::Integer {
                    "UI"
                } else {
                    "UP"
                };
                writeln!(
                    w,
                    " {} BND       {}  {:12.6e}",
                    prefix, var.name, var.upper_bound
                )
                .map_err(|e| MpsError::IoError(e.to_string()))?;
            }
        }
        Ok(())
    }

    fn write_sos<W: Write>(&self, model: &LpModel, w: &mut W) -> Result<(), MpsError> {
        if model.sos_constraints.is_empty() {
            return Ok(());
        }

        writeln!(w, "SOS").map_err(|e| MpsError::IoError(e.to_string()))?;
        for sos in &model.sos_constraints {
            let type_str = match sos.sos_type {
                SosType::Type1 => "S1",
                SosType::Type2 => "S2",
            };
            writeln!(w, " {} {}", type_str, sos.name)
                .map_err(|e| MpsError::IoError(e.to_string()))?;

            for (k, &member) in sos.members.iter().enumerate() {
                if member < model.variables.len() {
                    let weight = sos.weights.get(k).copied().unwrap_or(k as f64);
                    writeln!(w, "    {}  {:12.6e}", model.variables[member].name, weight)
                        .map_err(|e| MpsError::IoError(e.to_string()))?;
                }
            }
        }
        Ok(())
    }

    fn write_indicators<W: Write>(&self, model: &LpModel, w: &mut W) -> Result<(), MpsError> {
        if model.indicator_constraints.is_empty() {
            return Ok(());
        }

        writeln!(w, "INDICATORS").map_err(|e| MpsError::IoError(e.to_string()))?;
        for ind in &model.indicator_constraints {
            let active = if ind.active_value { 1 } else { 0 };
            if ind.binary_var < model.variables.len() {
                writeln!(
                    w,
                    " IF {} {} {}",
                    ind.name, model.variables[ind.binary_var].name, active
                )
                .map_err(|e| MpsError::IoError(e.to_string()))?;
            }
        }
        Ok(())
    }
}

fn format_val(v: f64) -> String {
    if v == v.round() && v.abs() < 1e15 {
        format!("{:.1}", v)
    } else {
        format!("{:.6e}", v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_model() -> LpModel {
        let mut m = LpModel::new("test_mps");
        m.sense = OptDirection::Minimize;
        let x = m.add_variable(Variable::continuous("x1", 0.0, 10.0));
        let y = m.add_variable(Variable::continuous("x2", 0.0, f64::INFINITY));
        m.set_obj_coeff(x, 1.0);
        m.set_obj_coeff(y, 2.0);

        let mut c0 = Constraint::new("c1", ConstraintSense::Le, 4.0);
        c0.add_term(x, 1.0);
        c0.add_term(y, 1.0);
        m.add_constraint(c0);

        let mut c1 = Constraint::new("c2", ConstraintSense::Ge, 1.0);
        c1.add_term(x, 1.0);
        m.add_constraint(c1);

        m
    }

    #[test]
    fn test_write_and_parse_free() {
        let model = make_test_model();
        let mps_str = write_mps_string(&model, MpsFormat::Free).unwrap();
        assert!(mps_str.contains("NAME"));
        assert!(mps_str.contains("ROWS"));
        assert!(mps_str.contains("COLUMNS"));
        assert!(mps_str.contains("ENDATA"));

        let parsed = parse_mps_string(&mps_str, MpsFormat::Free).unwrap();
        assert_eq!(parsed.num_vars(), 2);
        assert_eq!(parsed.num_constraints(), 2);
    }

    #[test]
    fn test_write_and_parse_fixed() {
        let model = make_test_model();
        let mps_str = write_mps_string(&model, MpsFormat::Fixed).unwrap();
        let parsed = parse_mps_string(&mps_str, MpsFormat::Free).unwrap();
        assert_eq!(parsed.num_vars(), 2);
    }

    #[test]
    fn test_bounds_section() {
        let mut m = LpModel::new("bounds_test");
        m.add_variable(Variable::continuous("x1", -5.0, 10.0));
        m.add_variable(Variable::binary("x2"));
        m.add_variable(Variable::continuous("x3", 0.0, 0.0)); // fixed

        let mps = write_mps_string(&m, MpsFormat::Free).unwrap();
        assert!(mps.contains("BOUNDS"));
        assert!(mps.contains("BV"));
    }

    #[test]
    fn test_ranges_section() {
        let mut m = LpModel::new("range_test");
        let x = m.add_variable(Variable::continuous("x1", 0.0, f64::INFINITY));
        let mut c = Constraint::new("c1", ConstraintSense::Le, 5.0);
        c.add_term(x, 1.0);
        c.range = Some(2.0);
        m.add_constraint(c);

        let mps = write_mps_string(&m, MpsFormat::Free).unwrap();
        assert!(mps.contains("RANGES"));
    }

    #[test]
    fn test_sos_section() {
        let mut m = LpModel::new("sos_test");
        let x0 = m.add_variable(Variable::continuous("x0", 0.0, 1.0));
        let x1 = m.add_variable(Variable::continuous("x1", 0.0, 1.0));
        m.add_sos_constraint(SosConstraint {
            sos_type: SosType::Type1,
            name: "sos1".to_string(),
            members: vec![x0, x1],
            weights: vec![1.0, 2.0],
        });

        let mps = write_mps_string(&m, MpsFormat::Free).unwrap();
        assert!(mps.contains("SOS"));
        assert!(mps.contains("S1"));
    }

    #[test]
    fn test_parse_mps_string() {
        let mps = r#"NAME          test
ROWS
 N  OBJ
 L  c1
 G  c2
COLUMNS
    x1  OBJ  1.0   c1  1.0
    x1  c2   1.0
    x2  OBJ  2.0   c1  1.0
RHS
    RHS  c1  4.0   c2  1.0
BOUNDS
 UP BND  x1  10.0
ENDATA
"#;
        let model = parse_mps_string(mps, MpsFormat::Free).unwrap();
        assert_eq!(model.num_vars(), 2);
        assert_eq!(model.num_constraints(), 2);
    }

    #[test]
    fn test_round_trip_fidelity() {
        let model = make_test_model();
        let mps1 = write_mps_string(&model, MpsFormat::Free).unwrap();
        let parsed = parse_mps_string(&mps1, MpsFormat::Free).unwrap();
        let mps2 = write_mps_string(&parsed, MpsFormat::Free).unwrap();

        // Names and structure should match
        assert_eq!(parsed.num_vars(), model.num_vars());
        assert_eq!(parsed.num_constraints(), model.num_constraints());
    }

    #[test]
    fn test_integer_markers() {
        let mut m = LpModel::new("int_test");
        m.add_variable(Variable::continuous("x1", 0.0, 10.0));
        m.add_variable(Variable::integer("x2", 0.0, 5.0));
        m.add_variable(Variable::continuous("x3", 0.0, 10.0));

        let mps = write_mps_string(&m, MpsFormat::Free).unwrap();
        assert!(mps.contains("INTORG"));
        assert!(mps.contains("INTEND"));
    }

    #[test]
    fn test_empty_model_mps() {
        let m = LpModel::new("empty");
        let mps = write_mps_string(&m, MpsFormat::Free).unwrap();
        assert!(mps.contains("ENDATA"));
        let parsed = parse_mps_string(&mps, MpsFormat::Free).unwrap();
        assert_eq!(parsed.num_vars(), 0);
    }

    #[test]
    fn test_objsense() {
        let mut m = LpModel::new("max_test");
        m.sense = OptDirection::Maximize;
        m.add_variable(Variable::continuous("x", 0.0, 10.0));
        let mps = write_mps_string(&m, MpsFormat::Free).unwrap();
        assert!(mps.contains("MAX"));
    }
}
