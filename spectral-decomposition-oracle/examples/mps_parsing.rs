//! Example: MPS and LP file parsing
//!
//! Demonstrates loading MIP instances from standard file formats:
//! - MPS (Mathematical Programming System) format
//! - LP (CPLEX LP) format
//!
//! Run: cargo run --example mps_parsing

use spectral_types::mip::{read_mps, read_lp, MipInstance};

fn main() {
    println!("=== SpectralOracle — MPS/LP Parsing Example ===\n");

    // ---------------------------------------------------------------
    // Example 1: Parse a small MPS instance
    // ---------------------------------------------------------------
    let mps_content = r#"NAME          example
ROWS
 N  obj
 L  c1
 L  c2
 G  c3
COLUMNS
    x1        obj           -1.0        c1            1.0
    x1        c2            2.0         c3            1.0
    x2        obj           -2.0        c1            1.0
    x2        c2            1.0         c3            3.0
    x3        obj           -1.5        c1            3.0
    x3        c3            1.0
    MARKER    'MARKER'      'INTORG'
    y1        obj           -3.0        c2            4.0
    y1        c3            2.0
    MARKER    'MARKER'      'INTEND'
RHS
    rhs       c1            4.0
    rhs       c2            6.0
    rhs       c3            2.0
BOUNDS
 UP bound     x1            10.0
 UP bound     x2            10.0
 UP bound     x3            10.0
 BV bound     y1
ENDATA
"#;

    match read_mps(mps_content) {
        Ok(instance) => {
            print_instance_info(&instance);
        }
        Err(e) => {
            eprintln!("MPS parse error: {}", e);
        }
    }

    // ---------------------------------------------------------------
    // Example 2: Parse a small LP instance
    // ---------------------------------------------------------------
    println!("\n--- LP Format ---\n");

    let lp_content = r#"\ Example LP file
Minimize
  obj: x1 + 2 x2 + 3 x3
Subject To
  c1: x1 + x2 <= 10
  c2: x2 + x3 >= 3
  c3: x1 + x2 + x3 = 8
Bounds
  0 <= x1 <= 5
  0 <= x2 <= 5
  0 <= x3 <= 5
End
"#;

    match read_lp(lp_content) {
        Ok(instance) => {
            print_instance_info(&instance);
        }
        Err(e) => {
            eprintln!("LP parse error: {}", e);
        }
    }

    println!("\n=== Done ===");
}

fn print_instance_info(inst: &MipInstance) {
    println!("Instance: {}", inst.name);
    println!("  Variables:   {} (bin={}, int={}, cont={})",
        inst.num_variables, inst.num_binary(),
        inst.num_integer(), inst.num_continuous());
    println!("  Constraints: {} (eq={})",
        inst.num_constraints, inst.num_equality_constraints());
    println!("  Nonzeros:    {}", inst.nnz());
    println!("  Density:     {:.4e}", inst.density());
    println!("  Is MIP:      {}", inst.is_mip());

    let stats = inst.statistics();
    println!("  Coeff range: {:.2e}", stats.coeff_range);
    println!("  Avg row nnz: {:.1}", stats.avg_row_nnz);
}
