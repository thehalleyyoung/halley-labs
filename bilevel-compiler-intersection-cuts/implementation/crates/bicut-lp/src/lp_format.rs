//! CPLEX-style LP file format parser and writer.
//!
//! Supports Minimize/Maximize objectives, constraint sections (<=, >=, =),
//! Bounds, Generals, Binaries, and End sections. Round-trip fidelity with
//! the [`LpModel`] type.

use crate::model::{Constraint, LpModel, VarType, Variable};
use bicut_types::{ConstraintSense, OptDirection};
use std::collections::HashMap;
use std::fmt;
use std::io::{BufRead, BufReader, Read, Write};

/// Errors during LP format parsing.
#[derive(Debug, Clone)]
pub enum LpFormatError {
    ParseError(String),
    InvalidSection(String),
    MissingObjective,
    IoError(String),
}

impl fmt::Display for LpFormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LpFormatError::ParseError(msg) => write!(f, "LP parse error: {}", msg),
            LpFormatError::InvalidSection(s) => write!(f, "Invalid LP section: {}", s),
            LpFormatError::MissingObjective => write!(f, "Missing objective section"),
            LpFormatError::IoError(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for LpFormatError {}

/// Parse an LP format file from a reader.
pub fn parse_lp<R: Read>(reader: R) -> Result<LpModel, LpFormatError> {
    let buf = BufReader::new(reader);
    let lines: Vec<String> = buf
        .lines()
        .map(|l| l.map_err(|e| LpFormatError::IoError(e.to_string())))
        .collect::<Result<Vec<_>, _>>()?;

    let mut parser = LpParser::new();
    parser.parse(&lines)?;
    Ok(parser.build_model())
}

/// Parse an LP format string.
pub fn parse_lp_string(s: &str) -> Result<LpModel, LpFormatError> {
    parse_lp(s.as_bytes())
}

/// Write an LP model in LP format.
pub fn write_lp<W: Write>(model: &LpModel, writer: &mut W) -> Result<(), LpFormatError> {
    let w = LpWriter;
    w.write(model, writer)
}

/// Write an LP model to an LP format string.
pub fn write_lp_string(model: &LpModel) -> Result<String, LpFormatError> {
    let mut buf = Vec::new();
    write_lp(model, &mut buf)?;
    String::from_utf8(buf).map_err(|e| LpFormatError::IoError(e.to_string()))
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LpSection {
    Objective,
    SubjectTo,
    Bounds,
    Generals,
    Binaries,
    End,
}

struct LpParser {
    sense: OptDirection,
    model_name: String,
    obj_terms: Vec<(String, f64)>,
    constraints: Vec<RawConstraint>,
    bounds: HashMap<String, (f64, f64)>,
    generals: Vec<String>,
    binaries: Vec<String>,
}

struct RawConstraint {
    name: String,
    terms: Vec<(String, f64)>,
    sense: ConstraintSense,
    rhs: f64,
}

impl LpParser {
    fn new() -> Self {
        Self {
            sense: OptDirection::Minimize,
            model_name: String::new(),
            obj_terms: Vec::new(),
            constraints: Vec::new(),
            bounds: HashMap::new(),
            generals: Vec::new(),
            binaries: Vec::new(),
        }
    }

    fn parse(&mut self, lines: &[String]) -> Result<(), LpFormatError> {
        // Join lines into one big string, then re-split on section keywords.
        // LP format allows expressions to span multiple lines, so we accumulate
        // tokens until we hit a section keyword or end.
        let mut section: Option<LpSection> = None;
        let mut accum = String::new();
        let mut constraint_counter = 0usize;

        for raw_line in lines {
            let line = strip_comment(raw_line);
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if let Some(new_section) = detect_section(trimmed) {
                // Flush accumulated text for the previous section.
                self.flush_section(section, &accum, &mut constraint_counter)?;
                accum.clear();

                if new_section == LpSection::Objective {
                    let upper = trimmed.to_uppercase();
                    if upper.starts_with("MIN") {
                        self.sense = OptDirection::Minimize;
                    } else {
                        self.sense = OptDirection::Maximize;
                    }
                }
                section = Some(new_section);
                continue;
            }

            accum.push(' ');
            accum.push_str(trimmed);
        }

        // Flush final section.
        self.flush_section(section, &accum, &mut constraint_counter)?;
        Ok(())
    }

    fn flush_section(
        &mut self,
        section: Option<LpSection>,
        text: &str,
        constraint_counter: &mut usize,
    ) -> Result<(), LpFormatError> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(());
        }
        match section {
            Some(LpSection::Objective) => self.parse_objective(text)?,
            Some(LpSection::SubjectTo) => self.parse_constraints(text, constraint_counter)?,
            Some(LpSection::Bounds) => self.parse_bounds(text)?,
            Some(LpSection::Generals) => self.parse_generals(text),
            Some(LpSection::Binaries) => self.parse_binaries(text),
            Some(LpSection::End) | None => {}
        }
        Ok(())
    }

    // -- Objective ----------------------------------------------------------

    fn parse_objective(&mut self, text: &str) -> Result<(), LpFormatError> {
        // Possible leading label: "obj:" or similar
        let text = strip_label(text);
        self.obj_terms = parse_linear_expr(text)?;
        Ok(())
    }

    // -- Constraints --------------------------------------------------------

    fn parse_constraints(&mut self, text: &str, counter: &mut usize) -> Result<(), LpFormatError> {
        // Constraints are separated by sense operators (<= >= =).
        // We split on newlines that start new constraints (indicated by a
        // label followed by ':' or a new expression).  Because we already
        // joined lines, constraints are separated by their sense operators.
        // Strategy: tokenise, then walk tokens splitting at sense operators.

        // Split text at points that begin new constraint rows.  A new row
        // starts when we see a name followed by a colon, or when we reach a
        // second sense operator after the RHS value.
        let rows = split_constraint_rows(text);
        for row in rows {
            let row = row.trim();
            if row.is_empty() {
                continue;
            }
            self.parse_single_constraint(row, counter)?;
        }
        Ok(())
    }

    fn parse_single_constraint(
        &mut self,
        row: &str,
        counter: &mut usize,
    ) -> Result<(), LpFormatError> {
        // Possible label prefix.
        let (name, expr) = extract_label(row);
        let name = name.unwrap_or_else(|| {
            *counter += 1;
            format!("c{}", counter)
        });

        // Find the sense operator and split into LHS / RHS.
        let (lhs, sense, rhs) = split_sense(expr)?;
        let terms = parse_linear_expr(lhs)?;
        let rhs_val: f64 = rhs
            .trim()
            .parse()
            .map_err(|_| LpFormatError::ParseError(format!("Invalid RHS: '{}'", rhs.trim())))?;

        self.constraints.push(RawConstraint {
            name,
            terms,
            sense,
            rhs: rhs_val,
        });
        Ok(())
    }

    // -- Bounds -------------------------------------------------------------

    fn parse_bounds(&mut self, text: &str) -> Result<(), LpFormatError> {
        // Each bound is on its own logical line.  Because we collapsed lines,
        // we rely on patterns: "<val> <= <var> <= <val>", "<var> free", etc.
        for piece in split_bound_lines(text) {
            let piece = piece.trim();
            if piece.is_empty() {
                continue;
            }
            self.parse_single_bound(piece)?;
        }
        Ok(())
    }

    fn parse_single_bound(&mut self, text: &str) -> Result<(), LpFormatError> {
        let upper = text.to_uppercase();

        // "<var> free"
        if upper.ends_with("FREE") {
            let var = text[..text.len() - 4].trim();
            self.bounds
                .insert(var.to_string(), (f64::NEG_INFINITY, f64::INFINITY));
            return Ok(());
        }

        // Try double-bounded: "lb <= var <= ub"
        let parts: Vec<&str> = text.splitn(5, "<=").collect();
        if parts.len() == 3 {
            let lb: f64 = parts[0].trim().parse().map_err(|_| {
                LpFormatError::ParseError(format!("Bad bound lb: '{}'", parts[0].trim()))
            })?;
            let var = parts[1].trim().to_string();
            let ub: f64 = parts[2].trim().parse().map_err(|_| {
                LpFormatError::ParseError(format!("Bad bound ub: '{}'", parts[2].trim()))
            })?;
            self.bounds.insert(var, (lb, ub));
            return Ok(());
        }

        // Try ">="-style double bound: "ub >= var >= lb"
        let parts_ge: Vec<&str> = text.splitn(5, ">=").collect();
        if parts_ge.len() == 3 {
            let ub: f64 = parts_ge[0].trim().parse().map_err(|_| {
                LpFormatError::ParseError(format!("Bad bound ub: '{}'", parts_ge[0].trim()))
            })?;
            let var = parts_ge[1].trim().to_string();
            let lb: f64 = parts_ge[2].trim().parse().map_err(|_| {
                LpFormatError::ParseError(format!("Bad bound lb: '{}'", parts_ge[2].trim()))
            })?;
            self.bounds.insert(var, (lb, ub));
            return Ok(());
        }

        // Single bound: "var <= ub" or "var >= lb" or "lb <= var" etc.
        if let Some(pos) = text.find("<=") {
            let lhs = text[..pos].trim();
            let rhs = text[pos + 2..].trim();
            if let Ok(val) = lhs.parse::<f64>() {
                // val <= var
                let entry = self
                    .bounds
                    .entry(rhs.to_string())
                    .or_insert((0.0, f64::INFINITY));
                entry.0 = val;
            } else if let Ok(val) = rhs.parse::<f64>() {
                // var <= val
                let entry = self
                    .bounds
                    .entry(lhs.to_string())
                    .or_insert((0.0, f64::INFINITY));
                entry.1 = val;
            }
            return Ok(());
        }

        if let Some(pos) = text.find(">=") {
            let lhs = text[..pos].trim();
            let rhs = text[pos + 2..].trim();
            if let Ok(val) = rhs.parse::<f64>() {
                // var >= val
                let entry = self
                    .bounds
                    .entry(lhs.to_string())
                    .or_insert((0.0, f64::INFINITY));
                entry.0 = val;
            } else if let Ok(val) = lhs.parse::<f64>() {
                // val >= var
                let entry = self
                    .bounds
                    .entry(rhs.to_string())
                    .or_insert((0.0, f64::INFINITY));
                entry.1 = val;
            }
            return Ok(());
        }

        if let Some(pos) = text.find('=') {
            let lhs = text[..pos].trim();
            let rhs = text[pos + 1..].trim();
            if let Ok(val) = rhs.parse::<f64>() {
                self.bounds.insert(lhs.to_string(), (val, val));
            }
            return Ok(());
        }

        Ok(())
    }

    // -- Generals / Binaries ------------------------------------------------

    fn parse_generals(&mut self, text: &str) {
        for tok in text.split_whitespace() {
            self.generals.push(tok.to_string());
        }
    }

    fn parse_binaries(&mut self, text: &str) {
        for tok in text.split_whitespace() {
            self.binaries.push(tok.to_string());
        }
    }

    // -- Build model --------------------------------------------------------

    fn build_model(&self) -> LpModel {
        let mut model = LpModel::new(&self.model_name);
        model.sense = self.sense;

        // Collect all variable names in order of first appearance.
        let mut var_order: Vec<String> = Vec::new();
        let mut var_set: HashMap<String, usize> = HashMap::new();

        let ensure_var =
            |name: &str, order: &mut Vec<String>, set: &mut HashMap<String, usize>| -> usize {
                if let Some(&idx) = set.get(name) {
                    idx
                } else {
                    let idx = order.len();
                    order.push(name.to_string());
                    set.insert(name.to_string(), idx);
                    idx
                }
            };

        for (name, _) in &self.obj_terms {
            ensure_var(name, &mut var_order, &mut var_set);
        }
        for rc in &self.constraints {
            for (name, _) in &rc.terms {
                ensure_var(name, &mut var_order, &mut var_set);
            }
        }
        for name in self.bounds.keys() {
            ensure_var(name, &mut var_order, &mut var_set);
        }
        for name in &self.generals {
            ensure_var(name, &mut var_order, &mut var_set);
        }
        for name in &self.binaries {
            ensure_var(name, &mut var_order, &mut var_set);
        }

        // Determine types.
        let general_set: std::collections::HashSet<&str> =
            self.generals.iter().map(|s| s.as_str()).collect();
        let binary_set: std::collections::HashSet<&str> =
            self.binaries.iter().map(|s| s.as_str()).collect();

        // Build objective coefficient map.
        let mut obj_map: HashMap<usize, f64> = HashMap::new();
        for (name, coeff) in &self.obj_terms {
            let idx = var_set[name.as_str()];
            *obj_map.entry(idx).or_insert(0.0) += coeff;
        }

        // Add variables.
        for (j, name) in var_order.iter().enumerate() {
            let (lb, ub) = if binary_set.contains(name.as_str()) {
                (0.0, 1.0)
            } else if let Some(&(l, u)) = self.bounds.get(name) {
                (l, u)
            } else {
                (0.0, f64::INFINITY)
            };

            let var_type = if binary_set.contains(name.as_str()) {
                VarType::Binary
            } else if general_set.contains(name.as_str()) {
                VarType::Integer
            } else {
                VarType::Continuous
            };

            let mut var = Variable::continuous(name, lb, ub);
            var.var_type = var_type;
            var.obj_coeff = obj_map.get(&j).copied().unwrap_or(0.0);
            model.add_variable(var);
        }

        // Add constraints.
        for rc in &self.constraints {
            let mut con = Constraint::new(&rc.name, rc.sense, rc.rhs);
            for (name, coeff) in &rc.terms {
                let idx = var_set[name.as_str()];
                con.add_term(idx, *coeff);
            }
            model.add_constraint(con);
        }

        model
    }
}

// ---------------------------------------------------------------------------
// Helper functions for parsing
// ---------------------------------------------------------------------------

/// Strip an inline `\` comment (CPLEX LP convention).
fn strip_comment(line: &str) -> &str {
    if let Some(pos) = line.find('\\') {
        &line[..pos]
    } else {
        line
    }
}

/// Detect whether a trimmed line is a section header keyword.
fn detect_section(trimmed: &str) -> Option<LpSection> {
    let upper = trimmed.to_uppercase();
    // Section keywords must start at the beginning of the line.
    if upper.starts_with("MIN") {
        Some(LpSection::Objective)
    } else if upper.starts_with("MAX") {
        Some(LpSection::Objective)
    } else if upper.starts_with("SUBJECT TO")
        || upper.starts_with("SUCH THAT")
        || upper.starts_with("S.T.")
        || upper == "ST"
        || upper.starts_with("ST ")
    {
        Some(LpSection::SubjectTo)
    } else if upper.starts_with("BOUND") {
        Some(LpSection::Bounds)
    } else if upper.starts_with("GENERAL") || upper.starts_with("GEN ") || upper == "GEN" {
        Some(LpSection::Generals)
    } else if upper.starts_with("BINAR") || upper.starts_with("BIN ") || upper == "BIN" {
        Some(LpSection::Binaries)
    } else if upper.starts_with("END") {
        Some(LpSection::End)
    } else {
        None
    }
}

/// Strip an optional label prefix (e.g. "obj:") from the start of an expression.
fn strip_label(text: &str) -> &str {
    if let Some(pos) = text.find(':') {
        text[pos + 1..].trim()
    } else {
        text.trim()
    }
}

/// Extract an optional label from a constraint line.  Returns (Some(label), rest)
/// or (None, text) if no label is found.
fn extract_label(text: &str) -> (Option<String>, &str) {
    // A label ends with ':' and the label itself is an identifier (no operators).
    if let Some(pos) = text.find(':') {
        let candidate = text[..pos].trim();
        // Ensure the candidate looks like an identifier (no <=, >=, +, - etc.).
        if !candidate.is_empty()
            && !candidate.contains('<')
            && !candidate.contains('>')
            && !candidate.contains('+')
            && !candidate.contains('-')
        {
            return (Some(candidate.to_string()), text[pos + 1..].trim());
        }
    }
    (None, text)
}

/// Split a constraint section text into individual constraint rows.  We look
/// for label patterns (word followed by colon) to delimit rows.  If no labels
/// are found we fall back to splitting at sense operators that follow a number
/// (RHS value) and precede a new expression.
fn split_constraint_rows(text: &str) -> Vec<String> {
    let mut rows: Vec<String> = Vec::new();
    let mut current = String::new();

    // Tokenise into whitespace-separated chunks, tracking sense operators
    // to detect row boundaries.
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let mut i = 0;
    while i < tokens.len() {
        let tok = tokens[i];

        // If token looks like "label:" at the start of a new row.
        if tok.ends_with(':') && !current.trim().is_empty() {
            rows.push(current.trim().to_string());
            current = String::new();
            current.push_str(tok);
            current.push(' ');
            i += 1;
            continue;
        }

        // If this token contains a colon but is not an operator (e.g. "c1:"),
        // and it's not the very first token of the current buffer, start new row.
        if tok.contains(':')
            && !tok.starts_with('+')
            && !tok.starts_with('-')
            && !current.trim().is_empty()
        {
            // Check if text before colon is an identifier.
            if let Some(colon_pos) = tok.find(':') {
                let before = &tok[..colon_pos];
                if !before.is_empty() && before.chars().all(|c| c.is_alphanumeric() || c == '_') {
                    rows.push(current.trim().to_string());
                    current = String::new();
                }
            }
        }

        current.push_str(tok);
        current.push(' ');
        i += 1;
    }
    if !current.trim().is_empty() {
        rows.push(current.trim().to_string());
    }
    rows
}

/// Split expression at sense operator, returning (lhs, sense, rhs).
fn split_sense(expr: &str) -> Result<(&str, ConstraintSense, &str), LpFormatError> {
    // Order matters: check '<=' / '>=' before '='.
    if let Some(pos) = expr.find("<=") {
        return Ok((&expr[..pos], ConstraintSense::Le, &expr[pos + 2..]));
    }
    if let Some(pos) = expr.find(">=") {
        return Ok((&expr[..pos], ConstraintSense::Ge, &expr[pos + 2..]));
    }
    if let Some(pos) = expr.find('=') {
        return Ok((&expr[..pos], ConstraintSense::Eq, &expr[pos + 1..]));
    }
    Err(LpFormatError::ParseError(format!(
        "No sense operator found in constraint: '{}'",
        expr
    )))
}

/// Parse a linear expression string like "2 x1 + 3.5 x2 - x3" into
/// a list of (variable_name, coefficient) pairs.
fn parse_linear_expr(text: &str) -> Result<Vec<(String, f64)>, LpFormatError> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(Vec::new());
    }

    let mut terms: Vec<(String, f64)> = Vec::new();
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let mut i = 0;
    let mut sign: f64 = 1.0;

    while i < tokens.len() {
        let tok = tokens[i];

        if tok == "+" {
            sign = 1.0;
            i += 1;
            continue;
        }
        if tok == "-" {
            sign = -1.0;
            i += 1;
            continue;
        }

        // Token could be: a number, a variable, or a combined "+3x", "-x", "3.5x" etc.
        // Also handle "[ ... ] / 2" (quadratic markers) by skipping them.
        if tok == "[" || tok == "]" || tok == "/" {
            i += 1;
            continue;
        }

        // Try to parse as number.  If the next token is a variable, pair them.
        if let Some((coeff, var_name)) = try_parse_coeff_var(tok) {
            terms.push((var_name, sign * coeff));
            sign = 1.0;
            i += 1;
            continue;
        }

        if let Ok(val) = tok.parse::<f64>() {
            // Next token should be a variable name.
            if i + 1 < tokens.len() {
                let next = tokens[i + 1];
                if next != "+" && next != "-" && next != "<=" && next != ">=" && next != "=" {
                    terms.push((next.to_string(), sign * val));
                    sign = 1.0;
                    i += 2;
                    continue;
                }
            }
            // Bare number with no following variable -- might be an isolated constant; skip.
            i += 1;
            continue;
        }

        // If it starts with +/-, split the sign off.
        if tok.starts_with('+') || tok.starts_with('-') {
            let tok_sign: f64 = if tok.starts_with('-') { -1.0 } else { 1.0 };
            let rest = &tok[1..];
            if rest.is_empty() {
                sign = tok_sign;
                i += 1;
                continue;
            }
            if let Some((coeff, var_name)) = try_parse_coeff_var(rest) {
                terms.push((var_name, sign * tok_sign * coeff));
                sign = 1.0;
                i += 1;
                continue;
            }
            if let Ok(val) = rest.parse::<f64>() {
                // Number with sign, next should be var.
                if i + 1 < tokens.len() {
                    let next = tokens[i + 1];
                    if next != "+" && next != "-" && next != "<=" && next != ">=" && next != "=" {
                        terms.push((next.to_string(), sign * tok_sign * val));
                        sign = 1.0;
                        i += 2;
                        continue;
                    }
                }
                i += 1;
                continue;
            }
            // Must be a variable with sign prefix.
            terms.push((rest.to_string(), sign * tok_sign));
            sign = 1.0;
            i += 1;
            continue;
        }

        // Must be a bare variable name (coefficient = 1).
        terms.push((tok.to_string(), sign * 1.0));
        sign = 1.0;
        i += 1;
    }

    Ok(terms)
}

/// Try to split a single token into (coefficient, variable_name).
/// Handles forms like "3.5x1", "2x", where digits run into a letter.
fn try_parse_coeff_var(tok: &str) -> Option<(f64, String)> {
    // Find the boundary where numeric part ends and identifier starts.
    let mut boundary = 0;
    let chars: Vec<char> = tok.chars().collect();
    // Skip optional leading sign inside token.
    if boundary < chars.len() && (chars[boundary] == '+' || chars[boundary] == '-') {
        boundary += 1;
    }
    // Walk digits/dots/e/E.
    while boundary < chars.len() {
        let c = chars[boundary];
        if c.is_ascii_digit() || c == '.' || c == 'e' || c == 'E' {
            boundary += 1;
        } else if (c == '+' || c == '-')
            && boundary > 0
            && (chars[boundary - 1] == 'e' || chars[boundary - 1] == 'E')
        {
            boundary += 1;
        } else {
            break;
        }
    }

    if boundary == 0 || boundary >= tok.len() {
        return None;
    }

    let num_part = &tok[..boundary];
    let var_part = &tok[boundary..];

    // var_part must start with a letter or underscore.
    if var_part.is_empty() {
        return None;
    }
    let first = var_part.chars().next()?;
    if !first.is_alphabetic() && first != '_' {
        return None;
    }

    let coeff: f64 = num_part.parse().ok()?;
    Some((coeff, var_part.to_string()))
}

/// Split bounds section text into individual bound lines.  In the original file
/// each bound is on its own line, but after joining they're separated by implicit
/// patterns.  We re-split whenever we see a token that starts a new bound
/// expression (a number or variable that follows a previous complete bound).
fn split_bound_lines(text: &str) -> Vec<String> {
    // Heuristic: split on tokens where a new bound starts.  A new bound
    // starts when we see either "<var> free", "<val> <= <var>", or
    // "<var> <= <val>".  Since these were originally separate lines, we
    // look for sequences where after a complete bound we see something that
    // cannot be part of the current bound.
    //
    // Simple approach: bounds never contain '+' or '-' as binary operators,
    // so each "free" keyword or each pair of <= / >= resets a line.
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut sense_count = 0u32;
    let mut saw_free = false;

    for tok in &tokens {
        let upper = tok.to_uppercase();
        if upper == "FREE" {
            current.push_str(tok);
            current.push(' ');
            lines.push(current.trim().to_string());
            current.clear();
            sense_count = 0;
            saw_free = true;
            continue;
        }
        saw_free = false;

        if *tok == "<=" || *tok == ">=" || *tok == "=" {
            sense_count += 1;
        }

        // If we already completed a double-bounded expression (2 senses)
        // or a single-bounded expression (1 sense) followed by what looks
        // like a new value/var token, start a new line.
        if sense_count > 2 {
            lines.push(current.trim().to_string());
            current.clear();
            sense_count = if *tok == "<=" || *tok == ">=" || *tok == "=" {
                1
            } else {
                0
            };
        }

        current.push_str(tok);
        current.push(' ');
    }
    if !current.trim().is_empty() {
        lines.push(current.trim().to_string());
    }
    let _ = saw_free;
    lines
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

struct LpWriter;

impl LpWriter {
    fn write<W: Write>(&self, model: &LpModel, w: &mut W) -> Result<(), LpFormatError> {
        self.write_objective(model, w)?;
        self.write_constraints(model, w)?;
        self.write_bounds(model, w)?;
        self.write_generals(model, w)?;
        self.write_binaries(model, w)?;
        writeln!(w, "End").map_err(|e| LpFormatError::IoError(e.to_string()))?;
        Ok(())
    }

    fn write_objective<W: Write>(&self, model: &LpModel, w: &mut W) -> Result<(), LpFormatError> {
        let keyword = match model.sense {
            OptDirection::Minimize => "Minimize",
            OptDirection::Maximize => "Maximize",
        };
        writeln!(w, "{}", keyword).map_err(io_err)?;
        write!(w, "  obj:").map_err(io_err)?;

        let mut first = true;
        for var in &model.variables {
            if var.obj_coeff.abs() < 1e-20 {
                continue;
            }
            if first {
                write!(w, " {}", format_term(var.obj_coeff, &var.name, true)).map_err(io_err)?;
                first = false;
            } else {
                write!(w, " {}", format_term(var.obj_coeff, &var.name, false)).map_err(io_err)?;
            }
        }
        if first {
            // No objective terms – write a zero.
            write!(
                w,
                " 0 {}",
                model.variables.first().map_or("x0", |v| &v.name)
            )
            .map_err(io_err)?;
        }
        writeln!(w).map_err(io_err)?;
        Ok(())
    }

    fn write_constraints<W: Write>(&self, model: &LpModel, w: &mut W) -> Result<(), LpFormatError> {
        if model.constraints.is_empty() {
            return Ok(());
        }
        writeln!(w, "Subject To").map_err(io_err)?;
        for con in &model.constraints {
            write!(w, "  {}:", con.name).map_err(io_err)?;
            let mut first = true;
            for (&col, &val) in con.row_indices.iter().zip(con.row_values.iter()) {
                if val.abs() < 1e-20 {
                    continue;
                }
                let var_name = &model.variables[col].name;
                if first {
                    write!(w, " {}", format_term(val, var_name, true)).map_err(io_err)?;
                    first = false;
                } else {
                    write!(w, " {}", format_term(val, var_name, false)).map_err(io_err)?;
                }
            }
            let sense_str = match con.sense {
                ConstraintSense::Le => "<=",
                ConstraintSense::Ge => ">=",
                ConstraintSense::Eq => "=",
            };
            writeln!(w, " {} {}", sense_str, format_val(con.rhs)).map_err(io_err)?;
        }
        Ok(())
    }

    fn write_bounds<W: Write>(&self, model: &LpModel, w: &mut W) -> Result<(), LpFormatError> {
        let has_nondefault = model.variables.iter().any(|v| {
            v.lower_bound != 0.0 || v.upper_bound != f64::INFINITY || v.var_type == VarType::Binary
        });
        if !has_nondefault {
            return Ok(());
        }

        writeln!(w, "Bounds").map_err(io_err)?;
        for var in &model.variables {
            if var.var_type == VarType::Binary {
                writeln!(w, "  0 <= {} <= 1", var.name).map_err(io_err)?;
                continue;
            }

            let lb_is_default = var.lower_bound == 0.0;
            let ub_is_default = var.upper_bound == f64::INFINITY || var.upper_bound >= 1e20;
            if lb_is_default && ub_is_default {
                continue;
            }

            let lb_neg_inf = var.lower_bound <= -1e20 || var.lower_bound == f64::NEG_INFINITY;
            let ub_pos_inf = var.upper_bound >= 1e20 || var.upper_bound == f64::INFINITY;

            if lb_neg_inf && ub_pos_inf {
                writeln!(w, "  {} free", var.name).map_err(io_err)?;
            } else if lb_neg_inf {
                writeln!(
                    w,
                    "  -inf <= {} <= {}",
                    var.name,
                    format_val(var.upper_bound)
                )
                .map_err(io_err)?;
            } else if ub_pos_inf {
                writeln!(
                    w,
                    "  {} <= {} <= +inf",
                    format_val(var.lower_bound),
                    var.name
                )
                .map_err(io_err)?;
            } else {
                writeln!(
                    w,
                    "  {} <= {} <= {}",
                    format_val(var.lower_bound),
                    var.name,
                    format_val(var.upper_bound)
                )
                .map_err(io_err)?;
            }
        }
        Ok(())
    }

    fn write_generals<W: Write>(&self, model: &LpModel, w: &mut W) -> Result<(), LpFormatError> {
        let ints: Vec<&str> = model
            .variables
            .iter()
            .filter(|v| v.var_type == VarType::Integer)
            .map(|v| v.name.as_str())
            .collect();
        if ints.is_empty() {
            return Ok(());
        }
        writeln!(w, "Generals").map_err(io_err)?;
        writeln!(w, "  {}", ints.join(" ")).map_err(io_err)?;
        Ok(())
    }

    fn write_binaries<W: Write>(&self, model: &LpModel, w: &mut W) -> Result<(), LpFormatError> {
        let bins: Vec<&str> = model
            .variables
            .iter()
            .filter(|v| v.var_type == VarType::Binary)
            .map(|v| v.name.as_str())
            .collect();
        if bins.is_empty() {
            return Ok(());
        }
        writeln!(w, "Binaries").map_err(io_err)?;
        writeln!(w, "  {}", bins.join(" ")).map_err(io_err)?;
        Ok(())
    }
}

fn io_err(e: std::io::Error) -> LpFormatError {
    LpFormatError::IoError(e.to_string())
}

fn format_val(v: f64) -> String {
    if v == v.round() && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{}", v)
    }
}

/// Format a single "coeff * var" term for LP output.
/// `is_first` controls whether a leading '+' is suppressed.
fn format_term(coeff: f64, var_name: &str, is_first: bool) -> String {
    let abs = coeff.abs();
    let sign_neg = coeff < 0.0;
    let coeff_str = if (abs - 1.0).abs() < 1e-20 {
        String::new()
    } else if abs == abs.round() && abs < 1e15 {
        format!("{} ", abs as i64)
    } else {
        format!("{} ", abs)
    };

    if is_first {
        if sign_neg {
            format!("- {}{}", coeff_str, var_name)
        } else {
            format!("{}{}", coeff_str, var_name)
        }
    } else if sign_neg {
        format!("- {}{}", coeff_str, var_name)
    } else {
        format!("+ {}{}", coeff_str, var_name)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_model() -> LpModel {
        let mut m = LpModel::new("test_lp");
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
    fn test_write_and_parse() {
        let model = make_test_model();
        let lp_str = write_lp_string(&model).unwrap();
        assert!(lp_str.contains("Minimize"));
        assert!(lp_str.contains("Subject To"));
        assert!(lp_str.contains("End"));

        let parsed = parse_lp_string(&lp_str).unwrap();
        assert_eq!(parsed.num_vars(), 2);
        assert_eq!(parsed.num_constraints(), 2);
        assert_eq!(parsed.sense, OptDirection::Minimize);
    }

    #[test]
    fn test_round_trip() {
        let model = make_test_model();
        let lp1 = write_lp_string(&model).unwrap();
        let parsed = parse_lp_string(&lp1).unwrap();
        let lp2 = write_lp_string(&parsed).unwrap();

        assert_eq!(parsed.num_vars(), model.num_vars());
        assert_eq!(parsed.num_constraints(), model.num_constraints());

        let re_parsed = parse_lp_string(&lp2).unwrap();
        assert_eq!(re_parsed.num_vars(), model.num_vars());
    }

    #[test]
    fn test_parse_lp_string_basic() {
        let lp = r#"
\ Simple LP
Minimize
  obj: x1 + 2 x2
Subject To
  c1: x1 + x2 <= 4
  c2: x1 >= 1
Bounds
  0 <= x1 <= 10
End
"#;
        let model = parse_lp_string(lp).unwrap();
        assert_eq!(model.num_vars(), 2);
        assert_eq!(model.num_constraints(), 2);
        assert_eq!(model.sense, OptDirection::Minimize);
        assert!((model.variables[0].obj_coeff - 1.0).abs() < 1e-10);
        assert!((model.variables[1].obj_coeff - 2.0).abs() < 1e-10);
        assert!((model.constraints[0].rhs - 4.0).abs() < 1e-10);
        assert_eq!(model.constraints[0].sense, ConstraintSense::Le);
        assert_eq!(model.constraints[1].sense, ConstraintSense::Ge);
    }

    #[test]
    fn test_maximize() {
        let lp = r#"
Maximize
  obj: 3 x1 + 5 x2
Subject To
  c1: x1 + x2 <= 10
End
"#;
        let model = parse_lp_string(lp).unwrap();
        assert_eq!(model.sense, OptDirection::Maximize);
        assert!((model.variables[0].obj_coeff - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_generals_and_binaries() {
        let lp = r#"
Minimize
  obj: x1 + x2 + x3
Subject To
  c1: x1 + x2 + x3 <= 10
Generals
  x1
Binaries
  x3
End
"#;
        let model = parse_lp_string(lp).unwrap();
        assert_eq!(model.variables[0].var_type, VarType::Integer);
        assert_eq!(model.variables[1].var_type, VarType::Continuous);
        assert_eq!(model.variables[2].var_type, VarType::Binary);
    }

    #[test]
    fn test_equality_constraint() {
        let lp = r#"
Minimize
  obj: x1
Subject To
  eq1: x1 + x2 = 5
End
"#;
        let model = parse_lp_string(lp).unwrap();
        assert_eq!(model.constraints[0].sense, ConstraintSense::Eq);
        assert!((model.constraints[0].rhs - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_free_variable() {
        let lp = r#"
Minimize
  obj: x1
Subject To
  c1: x1 <= 5
Bounds
  x1 free
End
"#;
        let model = parse_lp_string(lp).unwrap();
        assert!(model.variables[0].lower_bound <= -1e20);
        assert!(model.variables[0].upper_bound >= 1e20);
    }

    #[test]
    fn test_negative_coefficients() {
        let lp = r#"
Minimize
  obj: - x1 + 2 x2 - 3 x3
Subject To
  c1: x1 - x2 + x3 <= 0
End
"#;
        let model = parse_lp_string(lp).unwrap();
        assert!((model.variables[0].obj_coeff - (-1.0)).abs() < 1e-10);
        assert!((model.variables[1].obj_coeff - 2.0).abs() < 1e-10);
        assert!((model.variables[2].obj_coeff - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_write_bounds() {
        let mut m = LpModel::new("bounds_test");
        m.add_variable(Variable::continuous("x1", -5.0, 10.0));
        m.add_variable(Variable::binary("x2"));
        m.add_variable(Variable::continuous("x3", 0.0, 0.0)); // fixed

        let lp = write_lp_string(&m).unwrap();
        assert!(lp.contains("Bounds"));
        assert!(lp.contains("x1"));
    }

    #[test]
    fn test_write_generals_binaries() {
        let mut m = LpModel::new("intbin");
        m.add_variable(Variable::integer("y1", 0.0, 10.0));
        m.add_variable(Variable::binary("b1"));

        let lp = write_lp_string(&m).unwrap();
        assert!(lp.contains("Generals"));
        assert!(lp.contains("y1"));
        assert!(lp.contains("Binaries"));
        assert!(lp.contains("b1"));
    }
}
