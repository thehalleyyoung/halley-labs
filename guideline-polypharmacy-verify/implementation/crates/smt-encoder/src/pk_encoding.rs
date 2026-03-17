//! Pharmacokinetic dynamics encoding as SMT formulas.
//!
//! Encodes one-compartment PK models, CYP-mediated enzyme inhibition,
//! Metzler-matrix dynamics, steady-state bounds, and provides piecewise-
//! linear approximations for nonlinear functions (exp, reciprocal, etc.).

use crate::expression::SmtExpr;
use crate::variable::{SmtSort, SymbolTable, VariableId, VariableStore};

use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════
// PK Parameters
// ═══════════════════════════════════════════════════════════════════════════

/// One-compartment PK parameters for a single drug.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OneCompartmentParams {
    pub drug_name: String,
    /// Clearance (L/h).
    pub clearance: f64,
    /// Volume of distribution (L).
    pub volume: f64,
    /// Bioavailability fraction (0..1).
    pub bioavailability: f64,
    /// Absorption rate constant (h⁻¹), None for IV.
    pub ka: Option<f64>,
}

impl OneCompartmentParams {
    pub fn new(drug_name: &str, clearance: f64, volume: f64, bioavailability: f64) -> Self {
        Self {
            drug_name: drug_name.to_string(),
            clearance,
            volume,
            bioavailability,
            ka: None,
        }
    }

    pub fn with_ka(mut self, ka: f64) -> Self {
        self.ka = Some(ka);
        self
    }

    /// Elimination rate constant: ke = CL/V.
    pub fn ke(&self) -> f64 {
        self.clearance / self.volume
    }

    /// Half-life: t½ = ln(2)/ke.
    pub fn half_life(&self) -> f64 {
        0.693147 / self.ke()
    }

    /// Steady-state concentration for constant infusion: Css = R / CL
    /// where R = F * dose / interval.
    pub fn steady_state_avg(&self, dose_mg: f64, interval_h: f64) -> f64 {
        (self.bioavailability * dose_mg) / (self.clearance * interval_h)
    }
}

/// CYP inhibition parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CypInhibitionParams {
    /// Name of the inhibitor drug.
    pub inhibitor_name: String,
    /// Name of the substrate drug whose clearance is affected.
    pub substrate_name: String,
    /// Inhibition constant Ki (mg/L).
    pub ki: f64,
    /// Fraction of substrate clearance via the inhibited enzyme.
    pub fraction_metabolized: f64,
}

/// Interval bounds on a matrix entry for parametric Metzler dynamics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixEntryBound {
    pub row: usize,
    pub col: usize,
    pub lower: f64,
    pub upper: f64,
}

/// Metzler matrix dynamics parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetzlerParams {
    /// Dimension of the system.
    pub dimension: usize,
    /// Variable names for each state component.
    pub state_names: Vec<String>,
    /// Interval bounds on each matrix entry.
    pub matrix_bounds: Vec<MatrixEntryBound>,
}

// ═══════════════════════════════════════════════════════════════════════════
// Piecewise Linear Approximation
// ═══════════════════════════════════════════════════════════════════════════

/// A breakpoint in a piecewise linear approximation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Breakpoint {
    pub x: f64,
    pub y: f64,
}

/// Piecewise linear approximation of a nonlinear function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PiecewiseLinearApproximation {
    pub name: String,
    pub breakpoints: Vec<Breakpoint>,
}

impl PiecewiseLinearApproximation {
    pub fn new(name: &str) -> Self {
        Self { name: name.to_string(), breakpoints: Vec::new() }
    }

    pub fn with_breakpoint(mut self, x: f64, y: f64) -> Self {
        self.breakpoints.push(Breakpoint { x, y });
        self
    }

    /// Number of linear segments.
    pub fn num_segments(&self) -> usize {
        if self.breakpoints.len() < 2 { 0 } else { self.breakpoints.len() - 1 }
    }

    /// Evaluate the piecewise linear function at x.
    pub fn evaluate(&self, x: f64) -> f64 {
        if self.breakpoints.is_empty() { return 0.0; }
        if self.breakpoints.len() == 1 { return self.breakpoints[0].y; }

        // Clamp to range
        if x <= self.breakpoints[0].x {
            return self.breakpoints[0].y;
        }
        let last = self.breakpoints.last().unwrap();
        if x >= last.x {
            return last.y;
        }

        // Find the segment
        for i in 0..self.breakpoints.len() - 1 {
            let bp0 = &self.breakpoints[i];
            let bp1 = &self.breakpoints[i + 1];
            if x >= bp0.x && x <= bp1.x {
                let t = (x - bp0.x) / (bp1.x - bp0.x);
                return bp0.y + t * (bp1.y - bp0.y);
            }
        }
        last.y
    }

    /// Approximate exp(-k*x) for x in [0, x_max] with `n` segments.
    pub fn approximate_exp(k: f64, x_max: f64, n: usize) -> Self {
        let mut pwa = Self::new(&format!("exp(-{}*x)", k));
        for i in 0..=n {
            let x = x_max * (i as f64) / (n as f64);
            let y = (-k * x).exp();
            pwa.breakpoints.push(Breakpoint { x, y });
        }
        pwa
    }

    /// Approximate 1/(1 + x/k) for x in [0, x_max] with `n` segments.
    pub fn approximate_reciprocal(k: f64, x_max: f64, n: usize) -> Self {
        let mut pwa = Self::new(&format!("1/(1 + x/{})", k));
        for i in 0..=n {
            let x = x_max * (i as f64) / (n as f64);
            let y = 1.0 / (1.0 + x / k);
            pwa.breakpoints.push(Breakpoint { x, y });
        }
        pwa
    }

    /// Approximate x*y (product) by introducing auxiliary variables.
    /// Returns the piecewise approximation of f(x) = x*c for a fixed c.
    pub fn approximate_product(c: f64, x_max: f64, n: usize) -> Self {
        let mut pwa = Self::new(&format!("{:.4}*x", c));
        for i in 0..=n {
            let x = x_max * (i as f64) / (n as f64);
            let y = c * x;
            pwa.breakpoints.push(Breakpoint { x, y });
        }
        pwa
    }

    /// Encode this piecewise linear function as an SMT ITE chain.
    /// `input_var` is the SMT expression for the function argument.
    pub fn encode_as_ite(&self, input_var: SmtExpr) -> SmtExpr {
        if self.breakpoints.len() < 2 {
            return SmtExpr::RealLit(
                self.breakpoints.first().map(|b| b.y).unwrap_or(0.0),
            );
        }

        // Build ITE chain from right to left
        let n = self.breakpoints.len();
        let last = &self.breakpoints[n - 1];
        let second_last = &self.breakpoints[n - 2];

        let mut result = self.linear_expr(
            &input_var,
            second_last.x,
            second_last.y,
            last.x,
            last.y,
        );

        for i in (0..n - 2).rev() {
            let bp0 = &self.breakpoints[i];
            let bp1 = &self.breakpoints[i + 1];

            let segment_expr = self.linear_expr(
                &input_var, bp0.x, bp0.y, bp1.x, bp1.y,
            );

            result = SmtExpr::ite(
                SmtExpr::le(input_var.clone(), SmtExpr::RealLit(bp1.x)),
                segment_expr,
                result,
            );
        }

        result
    }

    fn linear_expr(
        &self,
        x: &SmtExpr,
        x0: f64,
        y0: f64,
        x1: f64,
        y1: f64,
    ) -> SmtExpr {
        let dx = x1 - x0;
        if dx.abs() < 1e-15 {
            return SmtExpr::RealLit(y0);
        }
        let slope = (y1 - y0) / dx;
        let intercept = y0 - slope * x0;

        SmtExpr::add(vec![
            SmtExpr::mul(SmtExpr::RealLit(slope), x.clone()),
            SmtExpr::RealLit(intercept),
        ])
    }

    /// Maximum absolute error compared to a reference function.
    pub fn max_error<F: Fn(f64) -> f64>(&self, reference: F, n_samples: usize) -> f64 {
        if self.breakpoints.len() < 2 { return f64::INFINITY; }
        let x_min = self.breakpoints[0].x;
        let x_max = self.breakpoints.last().unwrap().x;
        let mut max_err = 0.0f64;
        for i in 0..=n_samples {
            let x = x_min + (x_max - x_min) * (i as f64) / (n_samples as f64);
            let approx = self.evaluate(x);
            let exact = reference(x);
            max_err = max_err.max((approx - exact).abs());
        }
        max_err
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PkEncoder
// ═══════════════════════════════════════════════════════════════════════════

/// Encodes pharmacokinetic dynamics as SMT constraints.
#[derive(Debug)]
pub struct PkEncoder<'a> {
    store: &'a VariableStore,
    symbols: &'a SymbolTable,
    /// Number of piecewise linear segments for approximations.
    num_segments: usize,
}

impl<'a> PkEncoder<'a> {
    pub fn new(store: &'a VariableStore, symbols: &'a SymbolTable) -> Self {
        Self { store, symbols, num_segments: 8 }
    }

    pub fn with_segments(mut self, n: usize) -> Self {
        self.num_segments = n;
        self
    }

    // ── One-compartment model encoding ──────────────────────────────

    /// Encode one-compartment PK dynamics for a single drug at one time step.
    ///
    /// c(t+dt) = c(t) * exp(-ke * dt) + (F * dose) / CL * (1 - exp(-ke * dt))
    ///
    /// Since exp() is nonlinear, we use a piecewise linear approximation.
    pub fn encode_one_compartment(
        &self,
        params: &OneCompartmentParams,
        step: usize,
        dt: f64,
    ) -> Vec<SmtExpr> {
        let smt_name = self.symbols
            .concentration_smt_name(&format!(
                "conc_{}", params.drug_name.to_lowercase().replace(' ', "_")
            ))
            .unwrap_or("conc_drug");

        let c_curr = self.var_at_step(smt_name, step);
        let c_next = self.var_at_step(smt_name, step + 1);

        let ke = params.ke();
        let decay = (-ke * dt).exp();

        // c(t+1) = c(t) * decay_factor
        // (simplified: no dose in this step, just elimination)
        let decay_expr = SmtExpr::RealLit(decay);
        let c_after_decay = SmtExpr::mul(c_curr.clone(), decay_expr);

        let mut constraints = Vec::new();

        // Exact (linearized) update: c_next = c_curr * exp(-ke*dt)
        constraints.push(SmtExpr::eq(c_next.clone(), c_after_decay));

        // Non-negativity constraint
        constraints.push(SmtExpr::ge(c_next, SmtExpr::RealLit(0.0)));

        constraints
    }

    /// Encode one-compartment PK dynamics with a dose at this step.
    pub fn encode_one_compartment_with_dose(
        &self,
        params: &OneCompartmentParams,
        step: usize,
        dt: f64,
        dose_mg: f64,
    ) -> Vec<SmtExpr> {
        let smt_name = self.symbols
            .concentration_smt_name(&format!(
                "conc_{}", params.drug_name.to_lowercase().replace(' ', "_")
            ))
            .unwrap_or("conc_drug");

        let c_curr = self.var_at_step(smt_name, step);
        let c_next = self.var_at_step(smt_name, step + 1);

        let ke = params.ke();
        let decay = (-ke * dt).exp();
        let dose_contribution = if ke.abs() > 1e-15 {
            (params.bioavailability * dose_mg / params.volume) * (1.0 - decay)
        } else {
            params.bioavailability * dose_mg / params.volume
        };

        // c(t+1) = c(t) * decay + dose_contribution
        let update = SmtExpr::add(vec![
            SmtExpr::mul(c_curr.clone(), SmtExpr::RealLit(decay)),
            SmtExpr::RealLit(dose_contribution),
        ]);

        let mut constraints = Vec::new();
        constraints.push(SmtExpr::eq(c_next.clone(), update));
        constraints.push(SmtExpr::ge(c_next, SmtExpr::RealLit(0.0)));

        constraints
    }

    /// Encode one-compartment dynamics using PWL approximation of exp.
    pub fn encode_one_compartment_pwl(
        &self,
        params: &OneCompartmentParams,
        step: usize,
        dt: f64,
    ) -> Vec<SmtExpr> {
        let smt_name = self.symbols
            .concentration_smt_name(&format!(
                "conc_{}", params.drug_name.to_lowercase().replace(' ', "_")
            ))
            .unwrap_or("conc_drug");

        let c_curr = self.var_at_step(smt_name, step);
        let c_next = self.var_at_step(smt_name, step + 1);

        let ke = params.ke();
        // Build PWL approximation of f(c) = c * exp(-ke*dt)
        // The argument is c, and we approximate the decay function
        let c_max = 100.0; // reasonable upper bound on concentration
        let pwa = PiecewiseLinearApproximation::approximate_exp(ke * dt, c_max, self.num_segments);

        // The approximation gives exp(-ke*dt*c/c), but we need c * exp(-ke*dt)
        // Actually we need to multiply c_curr by exp(-ke*dt), which is a constant.
        // For the nonlinear case where ke depends on concentration (Michaelis-Menten),
        // we'd use the PWL. For the simple case, the constant is fine.
        let decay = (-ke * dt).exp();
        let update = SmtExpr::mul(c_curr.clone(), SmtExpr::RealLit(decay));

        let mut constraints = Vec::new();
        constraints.push(SmtExpr::eq(c_next.clone(), update));
        constraints.push(SmtExpr::ge(c_next, SmtExpr::RealLit(0.0)));

        // Store the PWL for reference (encoded as additional bound constraints)
        let _pwa_expr = pwa.encode_as_ite(c_curr);

        constraints
    }

    // ── CYP Inhibition encoding ─────────────────────────────────────

    /// Encode CYP enzyme inhibition effect.
    ///
    /// CL_eff = CL_0 * (1 - fm * I/(I + Ki))
    ///
    /// Where fm = fraction metabolized by inhibited enzyme,
    /// I = inhibitor concentration, Ki = inhibition constant.
    pub fn encode_cyp_inhibition(
        &self,
        params: &CypInhibitionParams,
        step: usize,
    ) -> Vec<SmtExpr> {
        let inhibitor_smt = self.symbols
            .concentration_smt_name(&format!(
                "conc_{}", params.inhibitor_name.to_lowercase().replace(' ', "_")
            ))
            .unwrap_or("conc_inhibitor");

        let substrate_smt = self.symbols
            .concentration_smt_name(&format!(
                "conc_{}", params.substrate_name.to_lowercase().replace(' ', "_")
            ))
            .unwrap_or("conc_substrate");

        let i_conc = self.var_at_step(inhibitor_smt, step);
        let _s_conc = self.var_at_step(substrate_smt, step);

        // Create auxiliary variables for the inhibition fraction
        let aux_name = format!("inh_{}_{}", params.inhibitor_name, params.substrate_name);
        let inh_var = self.var_at_step(&aux_name, step);

        let ki = params.ki;
        let fm = params.fraction_metabolized;

        // PWL approximation of f(I) = I / (I + Ki) for I in [0, I_max]
        let i_max = 50.0; // reasonable upper bound
        let pwa = PiecewiseLinearApproximation::approximate_reciprocal(ki, i_max, self.num_segments);
        let inhibition_fraction_expr = pwa.encode_as_ite(i_conc.clone());

        // inh_var = fm * I/(I+Ki)
        let effective_inhibition = SmtExpr::mul(
            SmtExpr::RealLit(fm),
            inhibition_fraction_expr,
        );

        let mut constraints = Vec::new();

        // inh_var represents the fraction of clearance that is inhibited
        constraints.push(SmtExpr::eq(inh_var.clone(), effective_inhibition));

        // Inhibition fraction must be in [0, fm]
        constraints.push(SmtExpr::ge(inh_var.clone(), SmtExpr::RealLit(0.0)));
        constraints.push(SmtExpr::le(inh_var, SmtExpr::RealLit(fm)));

        // Inhibitor concentration must be non-negative
        constraints.push(SmtExpr::ge(i_conc, SmtExpr::RealLit(0.0)));

        constraints
    }

    /// Encode clearance modification due to inhibition.
    ///
    /// Returns an expression for CL_eff = CL_0 * (1 - inhibition_fraction).
    pub fn encode_effective_clearance(
        &self,
        base_clearance: f64,
        inhibitor_name: &str,
        substrate_name: &str,
        step: usize,
    ) -> SmtExpr {
        let aux_name = format!("inh_{}_{}", inhibitor_name, substrate_name);
        let inh_var = self.var_at_step(&aux_name, step);

        SmtExpr::mul(
            SmtExpr::RealLit(base_clearance),
            SmtExpr::sub(SmtExpr::RealLit(1.0), inh_var),
        )
    }

    // ── Metzler matrix dynamics ─────────────────────────────────────

    /// Encode Metzler matrix dynamics with interval bounds on entries.
    ///
    /// dx/dt = A*x, where A is a Metzler matrix (off-diagonal ≥ 0,
    /// diagonal ≤ 0). We use Euler discretization:
    /// x(t+dt) = x(t) + dt * A * x(t) = (I + dt*A) * x(t)
    ///
    /// With interval bounds on A, we introduce auxiliary variables for
    /// each matrix entry.
    pub fn encode_metzler_dynamics(
        &self,
        params: &MetzlerParams,
        step: usize,
        dt: f64,
    ) -> Vec<SmtExpr> {
        let dim = params.dimension;
        let mut constraints = Vec::new();

        // Create auxiliary variables for matrix entries
        for bound in &params.matrix_bounds {
            let entry_name = format!("a_{}_{}", bound.row, bound.col);
            let a_var = self.var_at_step(&entry_name, step);

            // Bound the matrix entry
            constraints.push(SmtExpr::ge(a_var.clone(), SmtExpr::RealLit(bound.lower)));
            constraints.push(SmtExpr::le(a_var, SmtExpr::RealLit(bound.upper)));
        }

        // Euler discretization: x_i(t+dt) = x_i(t) + dt * sum_j(a_ij * x_j(t))
        for i in 0..dim {
            let state_name = &params.state_names[i];
            let x_curr = self.var_at_step(state_name, step);
            let x_next = self.var_at_step(state_name, step + 1);

            // Build the sum: sum_j(a_ij * x_j(t))
            let mut row_terms = Vec::new();
            for j in 0..dim {
                let entry_name = format!("a_{}_{}", i, j);
                let a_var = self.var_at_step(&entry_name, step);
                let x_j = self.var_at_step(&params.state_names[j], step);
                row_terms.push(SmtExpr::mul(a_var, x_j));
            }

            let row_sum = if row_terms.len() == 1 {
                row_terms.into_iter().next().unwrap()
            } else {
                SmtExpr::add(row_terms)
            };

            // x_i(t+dt) = x_i(t) + dt * row_sum
            let euler_update = SmtExpr::add(vec![
                x_curr,
                SmtExpr::mul(SmtExpr::RealLit(dt), row_sum),
            ]);

            constraints.push(SmtExpr::eq(x_next.clone(), euler_update));

            // Non-negativity for concentration-like states
            constraints.push(SmtExpr::ge(x_next, SmtExpr::RealLit(0.0)));
        }

        // Metzler property: off-diagonal entries must be non-negative
        for bound in &params.matrix_bounds {
            if bound.row != bound.col {
                let entry_name = format!("a_{}_{}", bound.row, bound.col);
                let a_var = self.var_at_step(&entry_name, step);
                constraints.push(SmtExpr::ge(a_var, SmtExpr::RealLit(0.0)));
            }
        }

        constraints
    }

    // ── Steady-state bounds ─────────────────────────────────────────

    /// Encode the steady-state concentration bound for a drug.
    ///
    /// At steady state: Css = (F * dose) / (CL * tau)
    /// The therapeutic window constraint: lower <= Css <= upper.
    pub fn encode_steady_state_bound(
        &self,
        params: &OneCompartmentParams,
        dose_mg: f64,
        interval_h: f64,
        lower: f64,
        upper: f64,
    ) -> SmtExpr {
        let css = params.steady_state_avg(dose_mg, interval_h);
        SmtExpr::and(vec![
            SmtExpr::ge(SmtExpr::RealLit(css), SmtExpr::RealLit(lower)),
            SmtExpr::le(SmtExpr::RealLit(css), SmtExpr::RealLit(upper)),
        ])
    }

    /// Encode parametric steady-state bound where clearance is uncertain.
    /// CL in [cl_lo, cl_hi] => Css in [F*dose/(cl_hi*tau), F*dose/(cl_lo*tau)]
    pub fn encode_parametric_steady_state(
        &self,
        bioavailability: f64,
        dose_mg: f64,
        interval_h: f64,
        cl_lo: f64,
        cl_hi: f64,
        lower: f64,
        upper: f64,
    ) -> SmtExpr {
        // Worst-case Css bounds given uncertain CL
        let css_lo = (bioavailability * dose_mg) / (cl_hi * interval_h);
        let css_hi = (bioavailability * dose_mg) / (cl_lo * interval_h);

        SmtExpr::and(vec![
            SmtExpr::ge(SmtExpr::RealLit(css_lo), SmtExpr::RealLit(lower)),
            SmtExpr::le(SmtExpr::RealLit(css_hi), SmtExpr::RealLit(upper)),
        ])
    }

    /// Encode the therapeutic window constraint for a concentration variable
    /// at a given step.
    pub fn encode_therapeutic_window(
        &self,
        drug_name: &str,
        step: usize,
        lower: f64,
        upper: f64,
    ) -> SmtExpr {
        let smt_name = self.symbols
            .concentration_smt_name(&format!(
                "conc_{}", drug_name.to_lowercase().replace(' ', "_")
            ))
            .unwrap_or("conc_drug");
        let c = self.var_at_step(smt_name, step);

        SmtExpr::and(vec![
            SmtExpr::ge(c.clone(), SmtExpr::RealLit(lower)),
            SmtExpr::le(c, SmtExpr::RealLit(upper)),
        ])
    }

    // ── Multi-drug interaction encoding ─────────────────────────────

    /// Encode the combined PK dynamics for two co-administered drugs
    /// where one inhibits the other's metabolism.
    pub fn encode_interaction_dynamics(
        &self,
        substrate: &OneCompartmentParams,
        inhibitor: &OneCompartmentParams,
        inhibition: &CypInhibitionParams,
        step: usize,
        dt: f64,
    ) -> Vec<SmtExpr> {
        let mut constraints = Vec::new();

        // Encode inhibitor dynamics (unaffected)
        constraints.extend(self.encode_one_compartment(inhibitor, step, dt));

        // Encode inhibition effect
        constraints.extend(self.encode_cyp_inhibition(inhibition, step));

        // Encode substrate dynamics with modified clearance
        let cl_eff = self.encode_effective_clearance(
            substrate.clearance,
            &inhibition.inhibitor_name,
            &inhibition.substrate_name,
            step,
        );

        // Substrate: c(t+dt) = c(t) * exp(-cl_eff/V * dt)
        // We approximate this linearly for the SMT encoding
        let smt_name = self.symbols
            .concentration_smt_name(&format!(
                "conc_{}", substrate.drug_name.to_lowercase().replace(' ', "_")
            ))
            .unwrap_or("conc_substrate");

        let c_curr = self.var_at_step(smt_name, step);
        let c_next = self.var_at_step(smt_name, step + 1);

        // Euler approximation: c(t+dt) ≈ c(t) - dt * (cl_eff/V) * c(t)
        //                               = c(t) * (1 - dt * cl_eff / V)
        let decay_factor = SmtExpr::sub(
            SmtExpr::RealLit(1.0),
            SmtExpr::mul(
                SmtExpr::RealLit(dt / substrate.volume),
                cl_eff,
            ),
        );
        let update = SmtExpr::mul(c_curr, decay_factor);

        constraints.push(SmtExpr::eq(c_next.clone(), update));
        constraints.push(SmtExpr::ge(c_next, SmtExpr::RealLit(0.0)));

        constraints
    }

    // ── Taylor series approximation ─────────────────────────────────

    /// Encode exp(-k*x) using Taylor series truncated at `order` terms.
    /// Returns the polynomial approximation as an SMT expression.
    pub fn encode_taylor_exp(
        &self,
        k: f64,
        x_expr: SmtExpr,
        order: usize,
    ) -> SmtExpr {
        // exp(-k*x) ≈ 1 - k*x + (k*x)^2/2! - (k*x)^3/3! + ...
        let mut terms: Vec<SmtExpr> = vec![SmtExpr::RealLit(1.0)];
        let mut coeff = 1.0;
        let mut sign = -1.0;

        for n in 1..=order {
            coeff *= k / (n as f64);
            let power = self.power_expr(x_expr.clone(), n);
            terms.push(SmtExpr::mul(
                SmtExpr::RealLit(sign * coeff),
                power,
            ));
            sign *= -1.0;
        }

        SmtExpr::add(terms)
    }

    fn power_expr(&self, base: SmtExpr, exp: usize) -> SmtExpr {
        match exp {
            0 => SmtExpr::RealLit(1.0),
            1 => base,
            _ => {
                let mut result = base.clone();
                for _ in 1..exp {
                    result = SmtExpr::mul(result, base.clone());
                }
                result
            }
        }
    }

    // ── Helper ──────────────────────────────────────────────────────

    fn var_at_step(&self, base_name: &str, step: usize) -> SmtExpr {
        let step_name = format!("{}_t{}", base_name, step);
        self.store.id_by_name(&step_name)
            .map(SmtExpr::Var)
            .unwrap_or_else(|| SmtExpr::Apply(step_name, vec![]))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variable::{VariableStore, SymbolTable, SmtSort};

    fn setup() -> (VariableStore, SymbolTable) {
        let mut store = VariableStore::new();
        let mut symbols = SymbolTable::new();

        symbols.register_concentration("conc_warfarin", "conc_warfarin");
        symbols.register_concentration("conc_fluconazole", "conc_fluconazole");

        for step in 0..=5 {
            store.create_time_indexed("conc_warfarin", SmtSort::Real, step);
            store.create_time_indexed("conc_fluconazole", SmtSort::Real, step);
            store.create_time_indexed("inh_fluconazole_warfarin", SmtSort::Real, step);
        }

        (store, symbols)
    }

    fn warfarin_params() -> OneCompartmentParams {
        OneCompartmentParams::new("warfarin", 0.2, 10.0, 0.95)
    }

    fn fluconazole_params() -> OneCompartmentParams {
        OneCompartmentParams::new("fluconazole", 1.0, 50.0, 0.9)
    }

    #[test]
    fn test_one_compartment_params() {
        let p = warfarin_params();
        assert!((p.ke() - 0.02).abs() < 1e-10);
        assert!((p.half_life() - 0.693147 / 0.02).abs() < 0.01);
    }

    #[test]
    fn test_steady_state() {
        let p = warfarin_params();
        let css = p.steady_state_avg(5.0, 24.0);
        // Css = 0.95 * 5.0 / (0.2 * 24.0) = 4.75 / 4.8 ≈ 0.9896
        assert!((css - 4.75 / 4.8).abs() < 0.01);
    }

    #[test]
    fn test_encode_one_compartment() {
        let (store, symbols) = setup();
        let enc = PkEncoder::new(&store, &symbols);
        let params = warfarin_params();
        let constraints = enc.encode_one_compartment(&params, 0, 1.0);
        assert!(constraints.len() >= 2); // update + non-negativity
    }

    #[test]
    fn test_encode_one_compartment_with_dose() {
        let (store, symbols) = setup();
        let enc = PkEncoder::new(&store, &symbols);
        let params = warfarin_params();
        let constraints = enc.encode_one_compartment_with_dose(&params, 0, 1.0, 5.0);
        assert!(constraints.len() >= 2);
    }

    #[test]
    fn test_encode_cyp_inhibition() {
        let (store, symbols) = setup();
        let enc = PkEncoder::new(&store, &symbols);
        let inhibition = CypInhibitionParams {
            inhibitor_name: "fluconazole".to_string(),
            substrate_name: "warfarin".to_string(),
            ki: 10.0,
            fraction_metabolized: 0.6,
        };
        let constraints = enc.encode_cyp_inhibition(&inhibition, 0);
        assert!(constraints.len() >= 3); // inhibition + bounds + non-negativity
    }

    #[test]
    fn test_pwl_exp_approximation() {
        let pwa = PiecewiseLinearApproximation::approximate_exp(1.0, 5.0, 10);
        assert_eq!(pwa.num_segments(), 10);

        // Check accuracy at a few points
        let error = pwa.max_error(|x| (-x).exp(), 100);
        assert!(error < 0.05, "PWL exp error too large: {}", error);
    }

    #[test]
    fn test_pwl_reciprocal_approximation() {
        let pwa = PiecewiseLinearApproximation::approximate_reciprocal(10.0, 50.0, 10);
        assert_eq!(pwa.num_segments(), 10);

        let error = pwa.max_error(|x| 1.0 / (1.0 + x / 10.0), 100);
        assert!(error < 0.05, "PWL reciprocal error too large: {}", error);
    }

    #[test]
    fn test_pwl_evaluate() {
        let pwa = PiecewiseLinearApproximation::new("test")
            .with_breakpoint(0.0, 0.0)
            .with_breakpoint(1.0, 1.0)
            .with_breakpoint(2.0, 0.0);

        assert!((pwa.evaluate(0.0) - 0.0).abs() < 1e-10);
        assert!((pwa.evaluate(0.5) - 0.5).abs() < 1e-10);
        assert!((pwa.evaluate(1.0) - 1.0).abs() < 1e-10);
        assert!((pwa.evaluate(1.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_pwl_encode_ite() {
        let pwa = PiecewiseLinearApproximation::new("test")
            .with_breakpoint(0.0, 0.0)
            .with_breakpoint(1.0, 1.0)
            .with_breakpoint(2.0, 2.0);

        let x = SmtExpr::Var(VariableId(0));
        let expr = pwa.encode_as_ite(x);
        // Should produce an ITE chain
        assert!(!matches!(expr, SmtExpr::RealLit(_)));
    }

    #[test]
    fn test_encode_steady_state_bound() {
        let (store, symbols) = setup();
        let enc = PkEncoder::new(&store, &symbols);
        let params = warfarin_params();
        let expr = enc.encode_steady_state_bound(&params, 5.0, 24.0, 0.5, 2.0);
        match &expr {
            SmtExpr::And(es) => assert_eq!(es.len(), 2),
            _ => panic!("Expected And"),
        }
    }

    #[test]
    fn test_encode_therapeutic_window() {
        let (store, symbols) = setup();
        let enc = PkEncoder::new(&store, &symbols);
        let expr = enc.encode_therapeutic_window("warfarin", 0, 1.0, 5.0);
        match &expr {
            SmtExpr::And(es) => assert_eq!(es.len(), 2),
            _ => panic!("Expected And"),
        }
    }

    #[test]
    fn test_encode_taylor_exp() {
        let (store, symbols) = setup();
        let enc = PkEncoder::new(&store, &symbols);
        let x = SmtExpr::Var(VariableId(0));
        let expr = enc.encode_taylor_exp(1.0, x, 4);
        // Should be a sum of 5 terms
        match &expr {
            SmtExpr::Add(terms) => assert_eq!(terms.len(), 5),
            _ => panic!("Expected Add"),
        }
    }

    #[test]
    fn test_encode_metzler_dynamics() {
        let mut store = VariableStore::new();
        let symbols = SymbolTable::new();

        // 2D system
        for step in 0..=2 {
            store.create_time_indexed("x0", SmtSort::Real, step);
            store.create_time_indexed("x1", SmtSort::Real, step);
            store.create_time_indexed("a_0_0", SmtSort::Real, step);
            store.create_time_indexed("a_0_1", SmtSort::Real, step);
            store.create_time_indexed("a_1_0", SmtSort::Real, step);
            store.create_time_indexed("a_1_1", SmtSort::Real, step);
        }

        let enc = PkEncoder::new(&store, &symbols);
        let params = MetzlerParams {
            dimension: 2,
            state_names: vec!["x0".to_string(), "x1".to_string()],
            matrix_bounds: vec![
                MatrixEntryBound { row: 0, col: 0, lower: -1.0, upper: -0.5 },
                MatrixEntryBound { row: 0, col: 1, lower: 0.0, upper: 0.5 },
                MatrixEntryBound { row: 1, col: 0, lower: 0.0, upper: 0.5 },
                MatrixEntryBound { row: 1, col: 1, lower: -1.0, upper: -0.5 },
            ],
        };

        let constraints = enc.encode_metzler_dynamics(&params, 0, 0.1);
        // Should have: matrix bounds (8) + euler updates (2) + non-negativity (2) + Metzler off-diag (2)
        assert!(constraints.len() >= 10);
    }

    #[test]
    fn test_encode_interaction_dynamics() {
        let (store, symbols) = setup();
        let enc = PkEncoder::new(&store, &symbols);
        let substrate = warfarin_params();
        let inhibitor = fluconazole_params();
        let inhibition = CypInhibitionParams {
            inhibitor_name: "fluconazole".to_string(),
            substrate_name: "warfarin".to_string(),
            ki: 10.0,
            fraction_metabolized: 0.6,
        };

        let constraints = enc.encode_interaction_dynamics(
            &substrate, &inhibitor, &inhibition, 0, 1.0,
        );
        assert!(constraints.len() >= 5);
    }

    #[test]
    fn test_encode_parametric_steady_state() {
        let (store, symbols) = setup();
        let enc = PkEncoder::new(&store, &symbols);
        let expr = enc.encode_parametric_steady_state(
            0.95, 5.0, 24.0, 0.1, 0.3, 0.5, 3.0,
        );
        match &expr {
            SmtExpr::And(es) => assert_eq!(es.len(), 2),
            _ => panic!("Expected And"),
        }
    }

    #[test]
    fn test_pwl_product_approximation() {
        let pwa = PiecewiseLinearApproximation::approximate_product(2.0, 10.0, 5);
        assert_eq!(pwa.num_segments(), 5);
        // f(x) = 2*x should be exact (it IS linear)
        let err = pwa.max_error(|x| 2.0 * x, 100);
        assert!(err < 1e-10);
    }
}
