//! Leakage-contract encoding into SMT formulas.
//!
//! [`LeakageEncoder`] translates a high-level leakage contract (specifying
//! which observations a program may produce) into SMT-LIB2 assertions that
//! can be checked by the solver layer.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::expr::{Expr, ExprId, ExprPool, Sort, Value};
use crate::smtlib::{Script, SmtCommand};
use crate::theories::CacheTheory;

// ---------------------------------------------------------------------------
// LeakageEncoder
// ---------------------------------------------------------------------------

/// Translates leakage contracts and program traces into SMT queries.
///
/// The encoder builds an SMT script that, when checked, determines whether
/// a program's observable side-channel behaviour is consistent with its
/// declared leakage contract.
#[derive(Debug)]
pub struct LeakageEncoder {
    /// The expression pool shared across the encoding.
    pub pool: ExprPool,
    /// Cache theory used for microarchitectural modelling.
    pub cache_theory: CacheTheory,
    /// Accumulated top-level assertions.
    assertions: Vec<ExprId>,
    /// Fresh variable counter for generating unique names.
    fresh_counter: u64,
    /// Map from user-visible names to their declared SMT variable ids.
    variables: HashMap<String, ExprId>,
}

impl LeakageEncoder {
    /// Create a new encoder with the given cache theory.
    pub fn new(cache_theory: CacheTheory) -> Self {
        Self {
            pool: ExprPool::new(),
            cache_theory,
            assertions: Vec::new(),
            fresh_counter: 0,
            variables: HashMap::new(),
        }
    }

    /// Create an encoder with default L1 cache geometry.
    pub fn with_defaults() -> Self {
        Self::new(CacheTheory::l1_default())
    }

    /// Declare a fresh bitvector variable of the given width.
    pub fn declare_bv(&mut self, name: &str, width: u32) -> ExprId {
        let id = self.pool.intern(Expr::Var(name.to_string(), Sort::BitVec(width)));
        self.variables.insert(name.to_string(), id);
        id
    }

    /// Declare a fresh Boolean variable.
    pub fn declare_bool(&mut self, name: &str) -> ExprId {
        let id = self.pool.intern(Expr::Var(name.to_string(), Sort::Bool));
        self.variables.insert(name.to_string(), id);
        id
    }

    /// Generate a fresh variable name with an optional prefix.
    pub fn fresh_name(&mut self, prefix: &str) -> String {
        let name = format!("{}_{}", prefix, self.fresh_counter);
        self.fresh_counter += 1;
        name
    }

    /// Add a top-level assertion.
    pub fn assert(&mut self, expr: ExprId) {
        self.assertions.push(expr);
    }

    /// Encode the non-interference condition: two runs that agree on public
    /// inputs must produce identical observations.
    pub fn encode_noninterference(
        &mut self,
        _public_vars: &[ExprId],
        _observation_exprs: &[ExprId],
    ) -> ExprId {
        // TODO: build ∀ secret . (agree_public ⇒ agree_obs) negation
        log::debug!("LeakageEncoder::encode_noninterference stub");
        self.pool.intern(Expr::Const(Value::Bool(true)))
    }

    /// Build the complete SMT-LIB2 script for the current encoding.
    pub fn build_script(&self) -> Script {
        let mut script = Script::new();
        // Set logic
        script.push(SmtCommand::SetLogic("QF_ABV".to_string()));

        // Emit variable declarations
        for (name, id) in &self.variables {
            if let Some(Expr::Var(_, ref sort)) = self.pool.get(*id) {
                script.push(SmtCommand::DeclareConst {
                    name: name.clone(),
                    sort: sort.clone(),
                });
            }
        }

        // Emit cache theory declarations
        // (stubbed in CacheTheory::emit_declarations)

        // Emit assertions
        for &a in &self.assertions {
            script.push(SmtCommand::Assert(a));
        }

        script.push(SmtCommand::CheckSat);
        script
    }

    /// Number of assertions accumulated so far.
    pub fn num_assertions(&self) -> usize {
        self.assertions.len()
    }

    /// Access the underlying expression pool.
    pub fn pool(&self) -> &ExprPool {
        &self.pool
    }

    /// Mutable access to the underlying expression pool.
    pub fn pool_mut(&mut self) -> &mut ExprPool {
        &mut self.pool
    }
}
