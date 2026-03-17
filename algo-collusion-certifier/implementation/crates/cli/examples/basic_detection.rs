//! Basic collusion detection example.
//!
//! Demonstrates the full pipeline:
//! 1. Set up a 2-player Bertrand market with linear demand
//! 2. Run Q-learning agents that may learn to collude
//! 3. Run the composite hypothesis test
//! 4. Check if collusion is detected
//!
//! Run with: `cargo run --example basic_detection`

use collusion_core::{
    DecaySchedule, DetectionConfig, DetectionPipeline, PricingAlgorithm, QLearningAgent,
    QLearningConfig,
};
use market_sim::{BertrandMarket, LinearDemand};
use shared_types::{
    AlgorithmType, MarketOutcome as StMarketOutcome, MarketType, OracleAccessLevel,
    PlayerId, Price, PriceTrajectory, Profit, Quantity, RoundNumber,
};
use stat_tests::{SupraCompetitivePriceTest, PriceParallelismTest};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════╗");
    println!("║   Algorithmic Collusion Detection Example    ║");
    println!("╚══════════════════════════════════════════════╝\n");

    // ── 1. Set up a Bertrand market ────────────────────────────────────
    // Linear demand: Q_i = 10 - 1.0*p_i + 0.5*p_j (differentiated products)
    let demand = LinearDemand::new(10.0, 1.0, 0.5, 2)?;
    let market = BertrandMarket::with_constant_costs(
        Box::new(demand),
        &[1.0, 1.0], // symmetric marginal costs
        0.0,          // price min
        10.0,         // price max
        15,           // grid size
    )?;

    // Analytical Nash and monopoly prices for this demand system:
    //   Nash:     p* = (a + b*mc) / (2b - c) = (10 + 1) / 1.5 ≈ 7.33
    //   Monopoly: p_m = (a + (b-c)*mc) / (2(b-c)) = 10.5 / 1.0 = 10.5
    let nash_price = 7.33;
    let monopoly_price = 10.5;
    let competitive_price = 1.0; // marginal cost

    println!("Market configuration:");
    println!("  Players:          2");
    println!("  Demand:           Linear (a=10, b=1, c=0.5)");
    println!("  Marginal cost:    1.0");
    println!("  Nash price:       {nash_price:.2}");
    println!("  Monopoly price:   {monopoly_price:.2}");

    // ── 2. Create Q-learning agents ────────────────────────────────────
    let base_config = QLearningConfig {
        player_id: PlayerId(0),
        num_price_levels: 15,
        price_min: Price(0.0),
        price_max: Price(10.0),
        learning_rate: 0.15,
        discount_factor: 0.95,
        epsilon_start: 1.0,
        epsilon_end: 0.02,
        epsilon_decay: 0.998,
        decay_schedule: DecaySchedule::Exponential,
        ..Default::default()
    };

    let config1 = QLearningConfig {
        player_id: PlayerId(1),
        ..base_config.clone()
    };

    let mut agent0 = QLearningAgent::new(base_config);
    let mut agent1 = QLearningAgent::new(config1);

    println!("\n── Running simulation ──────────────────────────");

    // ── 3. Simulate the market ─────────────────────────────────────────
    let num_rounds = 500;
    let mut outcomes: Vec<StMarketOutcome> = Vec::with_capacity(num_rounds);

    for round in 0..num_rounds {
        let rn = RoundNumber(round);

        // Agents choose prices
        let action0 = agent0.act(rn);
        let action1 = agent1.act(rn);
        let p0 = action0.action_value();
        let p1 = action1.action_value();

        // Simulate the market round using market-sim
        let ms_actions = vec![
            market_sim::PlayerAction::new(0, p0),
            market_sim::PlayerAction::new(1, p1),
        ];
        let ms_outcome = market.simulate_round(&ms_actions, round as u64)?;

        // Convert to shared-types MarketOutcome for the agents
        let st_outcome = StMarketOutcome::new(
            rn,
            vec![action0, action1],
            ms_outcome.prices.iter().map(|&p| Price(p)).collect(),
            ms_outcome.quantities.iter().map(|&q| Quantity(q)).collect(),
            ms_outcome.profits.iter().map(|&p| Profit(p)).collect(),
        );

        // Agents observe the outcome
        agent0.observe(&st_outcome);
        agent1.observe(&st_outcome);

        outcomes.push(st_outcome);
    }

    // Build a PriceTrajectory for analysis
    let trajectory = PriceTrajectory::new(
        outcomes,
        MarketType::Bertrand,
        2,
        AlgorithmType::QLearning,
        42,
    );

    // Print summary statistics
    let final_mean = trajectory.mean_price();
    println!("  Rounds simulated: {}", trajectory.len());
    println!("  Final mean price: {:.4}", final_mean.0);
    println!(
        "  Position:         {:.1}% of Nash→Monopoly range",
        ((final_mean.0 - nash_price) / (monopoly_price - nash_price) * 100.0)
            .clamp(0.0, 100.0)
    );

    // ── 4. Run detection pipeline ──────────────────────────────────────
    println!("\n── Detection pipeline ──────────────────────────");

    let detection_config = DetectionConfig {
        significance_level: 0.05,
        max_oracle_level: OracleAccessLevel::Layer0,
        nash_price: Price(nash_price),
        monopoly_price: Price(monopoly_price),
        competitive_price: Price(competitive_price),
        cp_threshold: 0.3,
        early_termination: true,
        ..Default::default()
    };

    let pipeline = DetectionPipeline::new(detection_config);
    let report = pipeline.run(&trajectory)?;

    println!("  Verdict:          {}", report.result.verdict);
    println!("  Confidence:       {:.1}%", report.result.confidence * 100.0);
    println!(
        "  Collusion premium: {:.4}",
        report.result.collusion_premium_estimate
    );
    println!("  Oracle layer:     {:?}", report.result.layer_reached);
    println!("  Trajectory len:   {}", report.trajectory_length);

    // ── 5. Run individual statistical tests ────────────────────────────
    println!("\n── Statistical tests ──────────────────────────");

    // Extract per-player price series as raw f64 for the test functions
    let prices_p0: Vec<f64> = trajectory
        .prices_for_player(PlayerId(0))
        .iter()
        .map(|p| p.0)
        .collect();
    let prices_p1: Vec<f64> = trajectory
        .prices_for_player(PlayerId(1))
        .iter()
        .map(|p| p.0)
        .collect();

    // Supra-competitive price test: are prices significantly above Nash?
    let supra_test = SupraCompetitivePriceTest::new(Price(nash_price));
    if let Ok(result) = supra_test.test(&prices_p0) {
        println!(
            "  Supra-competitive: t={:.4}, p={:.4}, supra={}",
            result.t_statistic,
            result.parametric_p_value.value(),
            result.is_supra_competitive
        );
    }

    // Price parallelism test: do firms move prices in lockstep?
    let parallelism_test = PriceParallelismTest::new(5);
    if let Ok(result) = parallelism_test.test(&prices_p0, &prices_p1) {
        println!(
            "  Price parallelism: corr={:.4}, p={:.4}, parallel={}",
            result.level_correlation,
            result.p_value.value(),
            result.is_parallel
        );
    }

    // ── Summary ────────────────────────────────────────────────────────
    println!("\n══════════════════════════════════════════════════");
    if report.result.verdict.is_collusive() {
        println!("⚠  COLLUSION DETECTED — agents learned supracompetitive pricing");
    } else if report.result.verdict.is_competitive() {
        println!("✓  COMPETITIVE — no evidence of collusion");
    } else {
        println!("?  INCONCLUSIVE — mixed signals from statistical tests");
    }
    println!("══════════════════════════════════════════════════");

    Ok(())
}
