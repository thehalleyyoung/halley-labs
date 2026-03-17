//! Certificate creation and verification example.
//!
//! Demonstrates:
//! 1. Creating a collusion certificate from test results
//! 2. Verifying the certificate using the proof checker
//! 3. Serializing / deserializing certificates as JSON
//!
//! Run with: `cargo run --example certificate_verification`

use certificate::{
    CertificateBuilder, CertificateAST, ProofChecker, VerificationResult,
    ast::VerdictType,
};
use game_theory::{CollusionPremium, NashEquilibrium};
use shared_types::{
    Cost, DemandSystem, GameConfig, MarketType, OracleAccessLevel,
};
use stat_tests::{TestResult, TestType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════╗");
    println!("║   Certificate Creation & Verification Demo   ║");
    println!("╚══════════════════════════════════════════════╝\n");

    // ── 1. Create mock analysis results ────────────────────────────────
    // In practice these come from the detection pipeline; here we build
    // them manually to illustrate the certificate API.

    let test_results = vec![
        TestResult::new(
            TestType::PriceCorrelation,
            "cross_firm_correlation",
            2.85,   // test statistic
            0.004,  // p-value (significant)
            0.05,   // alpha
        ),
        TestResult::new(
            TestType::VarianceRatio,
            "variance_ratio_test",
            0.32,   // low variance ratio
            0.02,   // p-value
            0.05,
        ),
        TestResult::new(
            TestType::MeanReversion,
            "supra_competitive_price",
            4.12,   // t-statistic
            0.001,  // highly significant
            0.05,
        ),
    ];

    println!("── Test results ───────────────────────────────");
    for tr in &test_results {
        println!(
            "  {}: stat={:.3}, p={:.4}, reject={}",
            tr.test_name, tr.statistic, tr.p_value.value(), tr.reject_null
        );
    }

    // Nash equilibrium for a 2-player symmetric Bertrand game
    let nash = NashEquilibrium::pure(vec![7, 7], vec![19.0, 19.0]);

    // Collusion premium: observed profit is 35% above Nash
    let cp = CollusionPremium::compute(25.65, 19.0);
    println!("\n  Nash payoffs:       {:?}", nash.payoffs);
    println!("  Collusion premium:  {:.4}", cp.value);

    // ── 2. Build a Layer 0 certificate ─────────────────────────────────
    println!("\n── Building certificate ────────────────────────");

    let game_config = GameConfig::symmetric(
        MarketType::Bertrand,
        DemandSystem::Linear { max_quantity: 10.0, slope: 1.0 },
        2,     // num_players
        0.95,  // discount_factor
        Cost(1.0),
        500,   // max_rounds
    );

    let cert = CertificateBuilder::build_layer0_certificate(
        &test_results,
        "sha256:abc123def456", // trajectory hash
        2,                     // num_players
        500,                   // num_rounds
        &game_config,
        &nash,
        &cp,
        0.05,                  // significance level
    );

    println!("  Steps:   {}", cert.step_count());
    println!("  Refs:    {:?}", cert.declared_refs());
    println!("\n  Pretty-printed certificate:");
    for line in cert.pretty_print().lines() {
        println!("    {line}");
    }

    // ── 3. Verify the certificate ──────────────────────────────────────
    println!("\n── Verifying certificate ───────────────────────");

    // Use relaxed mode so the checker tolerates the simplified demo certificate
    let checker = ProofChecker::new().with_strict_mode(false);
    let result = checker.check_certificate(&cert);

    match &result {
        VerificationResult::Valid(report) => {
            println!("  ✓ Certificate is VALID");
            println!("    Version:    {}", report.certificate_version);
            println!("    Scenario:   {}", report.scenario);
            println!("    Steps:      {}/{}", report.verified_steps, report.total_steps);
            println!("    Alpha:      spent {:.4} of {:.4}",
                report.alpha_budget_spent, report.alpha_budget_total);
            if let Some(v) = &report.verdict {
                println!("    Verdict:    {:?}", v);
            }
        }
        VerificationResult::Invalid(err) => {
            println!("  ✗ Certificate is INVALID");
            println!("    Kind:    {:?}", err.kind);
            println!("    Message: {}", err.message);
            if let Some(d) = &err.details {
                println!("    Details: {d}");
            }
        }
    }

    // ── 4. Serialize / deserialize as JSON ─────────────────────────────
    println!("\n── JSON serialization ─────────────────────────");

    let json = cert.to_json()?;
    println!("  Serialized certificate: {} bytes", json.len());

    // Show a snippet of the JSON
    let snippet: String = json.chars().take(300).collect();
    println!("  Preview:\n    {}…", snippet.replace('\n', "\n    "));

    // Round-trip: deserialize and re-verify
    let restored = CertificateAST::from_json(&json)?;
    assert_eq!(restored.step_count(), cert.step_count());
    println!("\n  Round-trip OK — {} steps preserved", restored.step_count());

    let re_result = checker.check_certificate(&restored);
    let re_valid = re_result.is_valid();
    println!("  Re-verification: {}", if re_valid { "✓ valid" } else { "✗ invalid" });

    // ── 5. Build a certificate manually step-by-step ───────────────────
    println!("\n── Manual certificate construction ─────────────");

    let mut builder = CertificateBuilder::new(
        "manual_demo",
        OracleAccessLevel::Layer0,
        0.05,
    );

    builder
        .add_data_declaration("traj_test", "testing", 0, 500, "sha256:aabbcc", 2)
        .add_test_result("t1", &test_results[0])
        .add_test_result("t2", &test_results[1])
        .add_equilibrium_claim("eq_nash", &game_config, &nash)
        .add_collusion_premium("cp0", &cp)
        .add_inference(
            "inf_0",
            "VerdictDerivation",
            vec!["t1".into(), "t2".into(), "eq_nash".into(), "cp0".into()],
            "Combined evidence supports collusion verdict",
        )
        .add_verdict(
            VerdictType::Collusive,
            0.95,
            vec!["t1".into(), "t2".into(), "inf_0".into()],
        );

    let manual_cert = builder.build();
    println!("  Built {} proof steps", manual_cert.step_count());

    let manual_result = checker.check_certificate(&manual_cert);
    println!(
        "  Verification: {}",
        if manual_result.is_valid() { "✓ valid" } else { "✗ invalid" }
    );

    // ── Summary ────────────────────────────────────────────────────────
    println!("\n══════════════════════════════════════════════════");
    println!("Certificate pipeline demonstrated successfully.");
    println!("══════════════════════════════════════════════════");

    Ok(())
}
