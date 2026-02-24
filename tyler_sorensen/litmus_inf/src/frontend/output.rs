//! Output formatting for verification results.

use std::fmt;

/// Format verification results for display.
pub struct OutputFormatter;

impl OutputFormatter {
    /// Format a simple summary.
    pub fn format_summary(test_name: &str, consistent: bool) -> String {
        format!("{}: {}", test_name,
            if consistent { "PASS" } else { "FAIL" })
    }
}
