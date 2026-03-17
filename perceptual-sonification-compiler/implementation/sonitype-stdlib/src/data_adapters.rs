//! Data source adapters for SoniType.
//!
//! Read and normalise data from CSV, JSON, in-memory arrays, and real-time
//! streaming sources before feeding them into sonification pipelines.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// DataValue
// ---------------------------------------------------------------------------

/// A dynamically-typed data value.
#[derive(Debug, Clone, PartialEq)]
pub enum DataValue {
    Float(f64),
    Int(i64),
    Str(String),
    Bool(bool),
    Null,
}

impl DataValue {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            Self::Int(v) => Some(*v as f64),
            Self::Bool(v) => Some(if *v { 1.0 } else { 0.0 }),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn is_numeric(&self) -> bool {
        matches!(self, Self::Float(_) | Self::Int(_))
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }
}

impl std::fmt::Display for DataValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Float(v) => write!(f, "{}", v),
            Self::Int(v) => write!(f, "{}", v),
            Self::Str(v) => write!(f, "{}", v),
            Self::Bool(v) => write!(f, "{}", v),
            Self::Null => write!(f, "null"),
        }
    }
}

// ---------------------------------------------------------------------------
// DataFieldType
// ---------------------------------------------------------------------------

/// Inferred type of a data field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataFieldType {
    Numeric,
    Categorical,
    Boolean,
    Unknown,
}

// ---------------------------------------------------------------------------
// DataDistribution
// ---------------------------------------------------------------------------

/// Statistics about a numeric data column.
#[derive(Debug, Clone)]
pub struct DataDistribution {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub median: f64,
    pub count: usize,
    pub q1: f64,
    pub q3: f64,
}

impl DataDistribution {
    /// Compute distribution from a slice of f64 values.
    pub fn from_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self {
                min: 0.0, max: 0.0, mean: 0.0, std_dev: 0.0,
                median: 0.0, count: 0, q1: 0.0, q3: 0.0,
            };
        }
        let n = values.len();
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min = sorted[0];
        let max = sorted[n - 1];
        let mean = values.iter().sum::<f64>() / n as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();
        let median = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };
        let q1 = sorted[n / 4];
        let q3 = sorted[3 * n / 4];
        Self { min, max, mean, std_dev, median, count: n, q1, q3 }
    }

    /// Inter-quartile range.
    pub fn iqr(&self) -> f64 {
        self.q3 - self.q1
    }

    /// Data range.
    pub fn range(&self) -> f64 {
        self.max - self.min
    }
}

// ---------------------------------------------------------------------------
// CsvDataSource
// ---------------------------------------------------------------------------

/// Read CSV data as a data source.
#[derive(Debug, Clone)]
pub struct CsvDataSource {
    pub headers: Vec<String>,
    pub field_types: Vec<DataFieldType>,
    pub rows: Vec<Vec<DataValue>>,
}

impl CsvDataSource {
    /// Parse CSV content from a string.
    pub fn from_str(content: &str) -> Result<Self, String> {
        let mut lines = content.lines();
        let header_line = lines.next().ok_or("Empty CSV")?;
        let headers: Vec<String> = header_line.split(',')
            .map(|h| h.trim().to_string())
            .collect();
        let num_cols = headers.len();
        let mut rows: Vec<Vec<DataValue>> = Vec::new();
        for (line_num, line) in lines.enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split(',').collect();
            if fields.len() != num_cols {
                return Err(format!(
                    "Row {} has {} fields, expected {}",
                    line_num + 2, fields.len(), num_cols
                ));
            }
            let row: Vec<DataValue> = fields.iter().map(|f| Self::parse_field(f.trim())).collect();
            rows.push(row);
        }
        let field_types = Self::infer_types(&headers, &rows);
        Ok(Self { headers, field_types, rows })
    }

    fn parse_field(s: &str) -> DataValue {
        if s.is_empty() || s.eq_ignore_ascii_case("null") || s.eq_ignore_ascii_case("na")
            || s.eq_ignore_ascii_case("nan") || s == "?" {
            return DataValue::Null;
        }
        if let Ok(i) = s.parse::<i64>() {
            return DataValue::Int(i);
        }
        if let Ok(f) = s.parse::<f64>() {
            return DataValue::Float(f);
        }
        if s.eq_ignore_ascii_case("true") {
            return DataValue::Bool(true);
        }
        if s.eq_ignore_ascii_case("false") {
            return DataValue::Bool(false);
        }
        DataValue::Str(s.to_string())
    }

    fn infer_types(headers: &[String], rows: &[Vec<DataValue>]) -> Vec<DataFieldType> {
        let num_cols = headers.len();
        let mut types = vec![DataFieldType::Unknown; num_cols];
        for col in 0..num_cols {
            let mut has_numeric = false;
            let mut has_string = false;
            let mut has_bool = false;
            for row in rows {
                match &row[col] {
                    DataValue::Float(_) | DataValue::Int(_) => has_numeric = true,
                    DataValue::Str(_) => has_string = true,
                    DataValue::Bool(_) => has_bool = true,
                    DataValue::Null => {}
                }
            }
            types[col] = if has_string {
                DataFieldType::Categorical
            } else if has_bool && !has_numeric {
                DataFieldType::Boolean
            } else if has_numeric {
                DataFieldType::Numeric
            } else {
                DataFieldType::Unknown
            };
        }
        types
    }

    /// Get column index by name.
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.headers.iter().position(|h| h == name)
    }

    /// Extract a numeric column as Vec<f64>, skipping nulls.
    pub fn numeric_column(&self, name: &str) -> Option<Vec<f64>> {
        let idx = self.column_index(name)?;
        let vals: Vec<f64> = self.rows.iter()
            .filter_map(|row| row.get(idx).and_then(|v| v.as_f64()))
            .collect();
        Some(vals)
    }

    /// Compute distribution for a named column.
    pub fn column_distribution(&self, name: &str) -> Option<DataDistribution> {
        let vals = self.numeric_column(name)?;
        if vals.is_empty() {
            return None;
        }
        Some(DataDistribution::from_values(&vals))
    }

    /// Iterate rows as (index, row) for streaming.
    pub fn stream_rows(&self) -> impl Iterator<Item = (usize, &Vec<DataValue>)> {
        self.rows.iter().enumerate()
    }

    /// Get unique categories for a string/categorical column.
    pub fn unique_categories(&self, name: &str) -> Option<Vec<String>> {
        let idx = self.column_index(name)?;
        let mut cats: Vec<String> = self.rows.iter()
            .filter_map(|row| row.get(idx).and_then(|v| v.as_str()).map(|s| s.to_string()))
            .collect();
        cats.sort();
        cats.dedup();
        Some(cats)
    }

    /// Number of rows.
    pub fn num_rows(&self) -> usize {
        self.rows.len()
    }

    /// Number of columns.
    pub fn num_cols(&self) -> usize {
        self.headers.len()
    }

    /// Handle missing values by replacing with column mean (for numeric columns).
    pub fn fill_missing_with_mean(&mut self) {
        for col in 0..self.headers.len() {
            if self.field_types[col] != DataFieldType::Numeric {
                continue;
            }
            let vals: Vec<f64> = self.rows.iter()
                .filter_map(|row| row.get(col).and_then(|v| v.as_f64()))
                .collect();
            if vals.is_empty() {
                continue;
            }
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            for row in &mut self.rows {
                if row[col].is_null() {
                    row[col] = DataValue::Float(mean);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// JsonDataSource
// ---------------------------------------------------------------------------

/// Read JSON data as a data source.
#[derive(Debug, Clone)]
pub struct JsonDataSource {
    pub root: JsonValue,
}

/// Simplified JSON value type.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Object(HashMap<String, JsonValue>),
    Array(Vec<JsonValue>),
    Number(f64),
    Str(String),
    Bool(bool),
    Null,
}

impl JsonDataSource {
    /// Create from a JsonValue root.
    pub fn new(root: JsonValue) -> Self {
        Self { root }
    }

    /// Navigate to a nested path (e.g., "data.items.0.value").
    pub fn navigate(&self, path: &str) -> Option<&JsonValue> {
        let parts: Vec<&str> = path.split('.').collect();
        let mut current = &self.root;
        for part in parts {
            current = match current {
                JsonValue::Object(map) => map.get(part)?,
                JsonValue::Array(arr) => {
                    let idx: usize = part.parse().ok()?;
                    arr.get(idx)?
                }
                _ => return None,
            };
        }
        Some(current)
    }

    /// Iterate over array items at a path.
    pub fn iterate_array(&self, path: &str) -> Vec<&JsonValue> {
        match self.navigate(path) {
            Some(JsonValue::Array(arr)) => arr.iter().collect(),
            _ => Vec::new(),
        }
    }

    /// Infer the type of a JsonValue.
    pub fn infer_type(value: &JsonValue) -> DataFieldType {
        match value {
            JsonValue::Number(_) => DataFieldType::Numeric,
            JsonValue::Str(_) => DataFieldType::Categorical,
            JsonValue::Bool(_) => DataFieldType::Boolean,
            _ => DataFieldType::Unknown,
        }
    }

    /// Convert a JsonValue to a DataValue.
    pub fn to_data_value(value: &JsonValue) -> DataValue {
        match value {
            JsonValue::Number(n) => DataValue::Float(*n),
            JsonValue::Str(s) => DataValue::Str(s.clone()),
            JsonValue::Bool(b) => DataValue::Bool(*b),
            JsonValue::Null => DataValue::Null,
            _ => DataValue::Null,
        }
    }

    /// Extract numeric values from an array at a path.
    pub fn numeric_array(&self, path: &str) -> Vec<f64> {
        self.iterate_array(path).iter()
            .filter_map(|v| match v {
                JsonValue::Number(n) => Some(*n),
                _ => None,
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// ArrayDataSource
// ---------------------------------------------------------------------------

/// In-memory array data source.
#[derive(Debug, Clone)]
pub struct ArrayDataSource {
    /// Single-variable data.
    pub single: Option<Vec<f64>>,
    /// Multi-variable data (each inner Vec is a variable/column).
    pub multi: Option<Vec<Vec<f64>>>,
    /// Named-field data.
    pub named: Option<Vec<HashMap<String, DataValue>>>,
}

impl ArrayDataSource {
    /// Create from a single variable.
    pub fn from_single(data: Vec<f64>) -> Self {
        Self { single: Some(data), multi: None, named: None }
    }

    /// Create from multiple variables (column-major).
    pub fn from_multi(data: Vec<Vec<f64>>) -> Self {
        Self { single: None, multi: Some(data), named: None }
    }

    /// Create from named-field records.
    pub fn from_named(data: Vec<HashMap<String, DataValue>>) -> Self {
        Self { single: None, multi: None, named: Some(data) }
    }

    /// Get distribution for single-variable data.
    pub fn distribution(&self) -> Option<DataDistribution> {
        self.single.as_ref().map(|v| DataDistribution::from_values(v))
    }

    /// Get distribution for a specific variable in multi-variable data.
    pub fn multi_distribution(&self, index: usize) -> Option<DataDistribution> {
        self.multi.as_ref()
            .and_then(|m| m.get(index))
            .map(|v| DataDistribution::from_values(v))
    }

    /// Get distribution for a named numeric field.
    pub fn named_distribution(&self, field: &str) -> Option<DataDistribution> {
        let vals: Vec<f64> = self.named.as_ref()?
            .iter()
            .filter_map(|row| row.get(field).and_then(|v| v.as_f64()))
            .collect();
        if vals.is_empty() { return None; }
        Some(DataDistribution::from_values(&vals))
    }

    /// Number of data points.
    pub fn len(&self) -> usize {
        if let Some(ref s) = self.single {
            s.len()
        } else if let Some(ref m) = self.multi {
            m.first().map(|v| v.len()).unwrap_or(0)
        } else if let Some(ref n) = self.named {
            n.len()
        } else {
            0
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of variables/dimensions.
    pub fn num_variables(&self) -> usize {
        if self.single.is_some() {
            1
        } else if let Some(ref m) = self.multi {
            m.len()
        } else if let Some(ref n) = self.named {
            n.first().map(|r| r.len()).unwrap_or(0)
        } else {
            0
        }
    }
}

// ---------------------------------------------------------------------------
// StreamingDataSource
// ---------------------------------------------------------------------------

/// Real-time push-based streaming data source.
#[derive(Debug, Clone)]
pub struct StreamingDataSource {
    /// Internal ring buffer.
    buffer: Vec<f64>,
    /// Maximum buffer (window) size.
    pub window_size: usize,
    /// Write pointer.
    write_pos: usize,
    /// Total number of samples received.
    total_count: usize,
    /// Running sum for efficient mean calculation.
    running_sum: f64,
    /// Running sum of squares for efficient variance.
    running_sum_sq: f64,
}

impl StreamingDataSource {
    pub fn new(window_size: usize) -> Self {
        assert!(window_size > 0);
        Self {
            buffer: vec![0.0; window_size],
            window_size,
            write_pos: 0,
            total_count: 0,
            running_sum: 0.0,
            running_sum_sq: 0.0,
        }
    }

    /// Push a new data value.
    pub fn push(&mut self, value: f64) {
        // If buffer is full, subtract the old value from running sums.
        if self.total_count >= self.window_size {
            let old = self.buffer[self.write_pos];
            self.running_sum -= old;
            self.running_sum_sq -= old * old;
        }
        self.buffer[self.write_pos] = value;
        self.running_sum += value;
        self.running_sum_sq += value * value;
        self.write_pos = (self.write_pos + 1) % self.window_size;
        self.total_count += 1;
    }

    /// Push multiple values.
    pub fn push_batch(&mut self, values: &[f64]) {
        for &v in values {
            self.push(v);
        }
    }

    /// Number of values currently in the window.
    pub fn current_count(&self) -> usize {
        self.total_count.min(self.window_size)
    }

    /// Get the windowed data as a slice (ordered oldest → newest).
    pub fn window_data(&self) -> Vec<f64> {
        let n = self.current_count();
        if n < self.window_size {
            self.buffer[..n].to_vec()
        } else {
            let mut result = Vec::with_capacity(self.window_size);
            for i in 0..self.window_size {
                result.push(self.buffer[(self.write_pos + i) % self.window_size]);
            }
            result
        }
    }

    /// Windowed mean.
    pub fn mean(&self) -> f64 {
        let n = self.current_count();
        if n == 0 { 0.0 } else { self.running_sum / n as f64 }
    }

    /// Windowed variance.
    pub fn variance(&self) -> f64 {
        let n = self.current_count();
        if n == 0 { return 0.0; }
        let mean = self.mean();
        self.running_sum_sq / n as f64 - mean * mean
    }

    /// Windowed standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.variance().max(0.0).sqrt()
    }

    /// Windowed min.
    pub fn min(&self) -> f64 {
        let data = self.window_data();
        data.iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// Windowed max.
    pub fn max(&self) -> f64 {
        let data = self.window_data();
        data.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Full distribution of the current window.
    pub fn distribution(&self) -> DataDistribution {
        DataDistribution::from_values(&self.window_data())
    }

    /// Total number of values received (including overwritten).
    pub fn total_received(&self) -> usize {
        self.total_count
    }

    /// Get the most recent value.
    pub fn latest(&self) -> Option<f64> {
        if self.total_count == 0 {
            None
        } else {
            let idx = if self.write_pos == 0 { self.window_size - 1 } else { self.write_pos - 1 };
            Some(self.buffer[idx])
        }
    }
}

// ---------------------------------------------------------------------------
// DataNormalizer
// ---------------------------------------------------------------------------

/// Normalisation strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationMethod {
    /// Scale to `[0, 1]` using min and max.
    MinMax,
    /// Standardise to mean=0, std=1.
    ZScore,
    /// Robust scaling using IQR.
    Robust,
    /// Apply log transform then min-max.
    LogTransform,
}

/// Normalise data values to a standard range.
#[derive(Debug, Clone)]
pub struct DataNormalizer {
    pub method: NormalizationMethod,
    pub distribution: DataDistribution,
}

impl DataNormalizer {
    /// Fit a normalizer to a data slice.
    pub fn fit(method: NormalizationMethod, values: &[f64]) -> Self {
        let distribution = DataDistribution::from_values(values);
        Self { method, distribution }
    }

    /// Transform a single value.
    pub fn transform(&self, value: f64) -> f64 {
        match self.method {
            NormalizationMethod::MinMax => {
                let range = self.distribution.range();
                if range.abs() < f64::EPSILON {
                    0.5
                } else {
                    (value - self.distribution.min) / range
                }
            }
            NormalizationMethod::ZScore => {
                if self.distribution.std_dev < f64::EPSILON {
                    0.0
                } else {
                    (value - self.distribution.mean) / self.distribution.std_dev
                }
            }
            NormalizationMethod::Robust => {
                let iqr = self.distribution.iqr();
                if iqr.abs() < f64::EPSILON {
                    0.0
                } else {
                    (value - self.distribution.median) / iqr
                }
            }
            NormalizationMethod::LogTransform => {
                let log_val = if value > 0.0 { value.ln() } else { 0.0 };
                let log_min = if self.distribution.min > 0.0 {
                    self.distribution.min.ln()
                } else {
                    0.0
                };
                let log_max = if self.distribution.max > 0.0 {
                    self.distribution.max.ln()
                } else {
                    1.0
                };
                let range = log_max - log_min;
                if range.abs() < f64::EPSILON {
                    0.5
                } else {
                    (log_val - log_min) / range
                }
            }
        }
    }

    /// Transform a batch of values.
    pub fn transform_batch(&self, values: &[f64]) -> Vec<f64> {
        values.iter().map(|v| self.transform(*v)).collect()
    }

    /// Inverse transform (only for MinMax).
    pub fn inverse_transform(&self, normalised: f64) -> f64 {
        match self.method {
            NormalizationMethod::MinMax => {
                normalised * self.distribution.range() + self.distribution.min
            }
            NormalizationMethod::ZScore => {
                normalised * self.distribution.std_dev + self.distribution.mean
            }
            NormalizationMethod::Robust => {
                normalised * self.distribution.iqr() + self.distribution.median
            }
            NormalizationMethod::LogTransform => {
                let log_min = if self.distribution.min > 0.0 {
                    self.distribution.min.ln()
                } else {
                    0.0
                };
                let log_max = if self.distribution.max > 0.0 {
                    self.distribution.max.ln()
                } else {
                    1.0
                };
                let log_val = normalised * (log_max - log_min) + log_min;
                log_val.exp()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_value_as_f64() {
        assert_eq!(DataValue::Float(3.14).as_f64(), Some(3.14));
        assert_eq!(DataValue::Int(42).as_f64(), Some(42.0));
        assert_eq!(DataValue::Str("hello".into()).as_f64(), None);
    }

    #[test]
    fn test_data_distribution_basic() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let d = DataDistribution::from_values(&vals);
        assert!((d.min - 1.0).abs() < 1e-10);
        assert!((d.max - 5.0).abs() < 1e-10);
        assert!((d.mean - 3.0).abs() < 1e-10);
        assert!((d.median - 3.0).abs() < 1e-10);
        assert_eq!(d.count, 5);
    }

    #[test]
    fn test_csv_parse() {
        let csv = "name,value\nalice,10\nbob,20\ncharlie,30";
        let src = CsvDataSource::from_str(csv).unwrap();
        assert_eq!(src.num_rows(), 3);
        assert_eq!(src.num_cols(), 2);
        assert_eq!(src.headers, vec!["name", "value"]);
    }

    #[test]
    fn test_csv_numeric_column() {
        let csv = "x,y\n1,10\n2,20\n3,30";
        let src = CsvDataSource::from_str(csv).unwrap();
        let col = src.numeric_column("y").unwrap();
        assert_eq!(col, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_csv_column_distribution() {
        let csv = "val\n10\n20\n30\n40\n50";
        let src = CsvDataSource::from_str(csv).unwrap();
        let d = src.column_distribution("val").unwrap();
        assert!((d.mean - 30.0).abs() < 1e-6);
    }

    #[test]
    fn test_csv_missing_values() {
        let csv = "x\n1\nna\n3\nnull\n5";
        let mut src = CsvDataSource::from_str(csv).unwrap();
        assert_eq!(src.num_rows(), 5);
        src.fill_missing_with_mean();
        let col = src.numeric_column("x").unwrap();
        assert_eq!(col.len(), 5); // Nulls replaced with mean
    }

    #[test]
    fn test_csv_categories() {
        let csv = "cat,val\nA,1\nB,2\nA,3\nC,4";
        let src = CsvDataSource::from_str(csv).unwrap();
        let cats = src.unique_categories("cat").unwrap();
        assert_eq!(cats, vec!["A", "B", "C"]);
    }

    #[test]
    fn test_csv_field_type_inference() {
        let csv = "name,value,flag\nalice,10,true\nbob,20,false";
        let src = CsvDataSource::from_str(csv).unwrap();
        assert_eq!(src.field_types[0], DataFieldType::Categorical);
        assert_eq!(src.field_types[1], DataFieldType::Numeric);
        assert_eq!(src.field_types[2], DataFieldType::Boolean);
    }

    #[test]
    fn test_json_navigate() {
        let mut obj = HashMap::new();
        let mut inner = HashMap::new();
        inner.insert("value".to_string(), JsonValue::Number(42.0));
        obj.insert("data".to_string(), JsonValue::Object(inner));
        let src = JsonDataSource::new(JsonValue::Object(obj));
        let val = src.navigate("data.value").unwrap();
        assert_eq!(*val, JsonValue::Number(42.0));
    }

    #[test]
    fn test_json_iterate_array() {
        let arr = JsonValue::Array(vec![
            JsonValue::Number(1.0),
            JsonValue::Number(2.0),
            JsonValue::Number(3.0),
        ]);
        let mut root = HashMap::new();
        root.insert("items".to_string(), arr);
        let src = JsonDataSource::new(JsonValue::Object(root));
        let items = src.iterate_array("items");
        assert_eq!(items.len(), 3);
    }

    #[test]
    fn test_json_numeric_array() {
        let arr = JsonValue::Array(vec![
            JsonValue::Number(10.0),
            JsonValue::Number(20.0),
        ]);
        let mut root = HashMap::new();
        root.insert("vals".to_string(), arr);
        let src = JsonDataSource::new(JsonValue::Object(root));
        let nums = src.numeric_array("vals");
        assert_eq!(nums, vec![10.0, 20.0]);
    }

    #[test]
    fn test_array_single() {
        let src = ArrayDataSource::from_single(vec![1.0, 2.0, 3.0]);
        assert_eq!(src.len(), 3);
        assert_eq!(src.num_variables(), 1);
        let d = src.distribution().unwrap();
        assert!((d.mean - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_array_multi() {
        let src = ArrayDataSource::from_multi(vec![
            vec![1.0, 2.0, 3.0],
            vec![10.0, 20.0, 30.0],
        ]);
        assert_eq!(src.num_variables(), 2);
        let d = src.multi_distribution(1).unwrap();
        assert!((d.mean - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_streaming_push_and_mean() {
        let mut s = StreamingDataSource::new(5);
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            s.push(v);
        }
        assert!((s.mean() - 3.0).abs() < 1e-10);
        assert_eq!(s.current_count(), 5);
    }

    #[test]
    fn test_streaming_window_overflow() {
        let mut s = StreamingDataSource::new(3);
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            s.push(v);
        }
        // Window should contain 3, 4, 5
        let data = s.window_data();
        assert_eq!(data.len(), 3);
        assert!((s.mean() - 4.0).abs() < 1e-10);
        assert_eq!(s.total_received(), 5);
    }

    #[test]
    fn test_streaming_latest() {
        let mut s = StreamingDataSource::new(10);
        s.push(42.0);
        s.push(99.0);
        assert_eq!(s.latest(), Some(99.0));
    }

    #[test]
    fn test_streaming_distribution() {
        let mut s = StreamingDataSource::new(100);
        for v in [10.0, 20.0, 30.0, 40.0, 50.0] {
            s.push(v);
        }
        let d = s.distribution();
        assert!((d.mean - 30.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalizer_minmax() {
        let vals = vec![0.0, 50.0, 100.0];
        let n = DataNormalizer::fit(NormalizationMethod::MinMax, &vals);
        assert!((n.transform(0.0) - 0.0).abs() < 1e-10);
        assert!((n.transform(50.0) - 0.5).abs() < 1e-10);
        assert!((n.transform(100.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalizer_zscore() {
        let vals = vec![10.0, 20.0, 30.0];
        let n = DataNormalizer::fit(NormalizationMethod::ZScore, &vals);
        let z_mean = n.transform(20.0);
        assert!((z_mean - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalizer_inverse_minmax() {
        let vals = vec![10.0, 20.0, 30.0];
        let n = DataNormalizer::fit(NormalizationMethod::MinMax, &vals);
        let normalised = n.transform(25.0);
        let recovered = n.inverse_transform(normalised);
        assert!((recovered - 25.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalizer_batch() {
        let vals = vec![0.0, 50.0, 100.0];
        let n = DataNormalizer::fit(NormalizationMethod::MinMax, &vals);
        let batch = n.transform_batch(&[0.0, 25.0, 50.0, 75.0, 100.0]);
        assert_eq!(batch.len(), 5);
        assert!((batch[0] - 0.0).abs() < 1e-10);
        assert!((batch[4] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalizer_log_transform() {
        let vals = vec![1.0, 10.0, 100.0, 1000.0];
        let n = DataNormalizer::fit(NormalizationMethod::LogTransform, &vals);
        let t1 = n.transform(1.0);
        let t1000 = n.transform(1000.0);
        assert!((t1 - 0.0).abs() < 1e-6);
        assert!((t1000 - 1.0).abs() < 1e-6);
    }
}
