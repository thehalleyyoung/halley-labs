//! Feature unification engine.
//!
//! Provides a rule-based unification engine that checks and propagates
//! linguistic features through a parse tree.  The engine carries ~80 default
//! English grammar constraints that cover the 15 metamorphic transformations
//! in the NLP pipeline.

use crate::features::{
    Feature, FeatureBundle, FeatureConflict, FeatureStructure,
    NumberValue, PersonValue, TenseValue, AspectValue, VoiceValue,
    MoodValue, CaseValue, GenderValue, DefinitenessValue, AnimacyValue,
    TransitivityValue, FinitenessValue,
};
use shared_types::ParseTree;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── GrammarConstraint ───────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GrammarConstraint {
    /// Subject and verb must agree on `feature_name`.
    Agreement(String),
    /// A head with `head_feature` selects a dependent with `dependent_feature`.
    Selection(String, String),
    /// A governor of `governor_category` governs `governed_case` on its complement.
    Government(String, String),
    /// Antecedent and pronoun must share the listed features.
    Binding(Vec<String>, Vec<String>),
}

impl GrammarConstraint {
    pub fn name(&self) -> String {
        match self {
            GrammarConstraint::Agreement(f) => format!("Agreement({f})"),
            GrammarConstraint::Selection(h, d) => format!("Selection({h},{d})"),
            GrammarConstraint::Government(g, c) => format!("Government({g},{c})"),
            GrammarConstraint::Binding(a, p) => format!("Binding({},{})", a.join("+"), p.join("+")),
        }
    }
}

impl fmt::Display for GrammarConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name())
    }
}

// ── ConstraintViolation ─────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConstraintViolation {
    pub constraint: String,
    pub node1: String,
    pub node2: String,
    pub explanation: String,
}

impl ConstraintViolation {
    pub fn new(
        constraint: impl Into<String>,
        node1: impl Into<String>,
        node2: impl Into<String>,
        explanation: impl Into<String>,
    ) -> Self {
        Self {
            constraint: constraint.into(),
            node1: node1.into(),
            node2: node2.into(),
            explanation: explanation.into(),
        }
    }
}

impl fmt::Display for ConstraintViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {} ↔ {}: {}",
            self.constraint, self.node1, self.node2, self.explanation
        )
    }
}

// ── UnificationRule ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct UnificationRule {
    pub name: String,
    /// Category pattern for the left-hand child (e.g. "NP").
    pub lhs_pattern: String,
    /// Category pattern for the right-hand child (e.g. "VP").
    pub rhs_pattern: String,
    /// Features to propagate into the result.
    pub result_features: Vec<String>,
    /// The constraint that this rule enforces.
    pub constraint: GrammarConstraint,
}

impl UnificationRule {
    pub fn new(
        name: impl Into<String>,
        lhs: impl Into<String>,
        rhs: impl Into<String>,
        features: Vec<&str>,
        constraint: GrammarConstraint,
    ) -> Self {
        Self {
            name: name.into(),
            lhs_pattern: lhs.into(),
            rhs_pattern: rhs.into(),
            result_features: features.into_iter().map(String::from).collect(),
            constraint,
        }
    }

    /// Check whether this rule matches a pair of parse-node labels.
    pub fn matches(&self, lhs_label: &str, rhs_label: &str) -> bool {
        label_matches(&self.lhs_pattern, lhs_label)
            && label_matches(&self.rhs_pattern, rhs_label)
    }
}

fn label_matches(pattern: &str, label: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    pattern == label
}

// ── UnificationResult ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum UnificationResult {
    Success(FeatureBundle),
    Failure(Vec<FeatureConflict>),
}

impl UnificationResult {
    pub fn is_success(&self) -> bool {
        matches!(self, UnificationResult::Success(_))
    }

    pub fn is_failure(&self) -> bool {
        matches!(self, UnificationResult::Failure(_))
    }

    pub fn conflicts(&self) -> Vec<&FeatureConflict> {
        match self {
            UnificationResult::Success(_) => vec![],
            UnificationResult::Failure(cs) => cs.iter().collect(),
        }
    }
}

// ── UnificationEngine ───────────────────────────────────────────────────────

/// The core unification engine that applies grammar rules to parse trees.
#[derive(Debug, Clone)]
pub struct UnificationEngine {
    pub rules: Vec<UnificationRule>,
    pub feature_defaults: HashMap<String, Feature>,
}

impl UnificationEngine {
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            feature_defaults: HashMap::new(),
        }
    }

    /// Build an engine pre-loaded with the default ~80 English constraints.
    pub fn with_defaults() -> Self {
        let mut engine = Self::new();
        engine.rules = DefaultConstraints::all_rules();
        engine.feature_defaults = default_feature_map();
        engine
    }

    pub fn add_rule(&mut self, rule: UnificationRule) {
        self.rules.push(rule);
    }

    /// Attempt to unify two feature structures according to the loaded rules.
    pub fn unify(
        &self,
        lhs: &FeatureStructure,
        rhs: &FeatureStructure,
    ) -> UnificationResult {
        // Gather applicable rules
        let applicable: Vec<&UnificationRule> = self
            .rules
            .iter()
            .filter(|r| r.matches(&lhs.category, &rhs.category))
            .collect();

        if applicable.is_empty() {
            // No rule applies — trivially succeed with merged features.
            match lhs.features.unify_with(&rhs.features) {
                Ok(merged) => UnificationResult::Success(merged),
                Err(conflicts) => UnificationResult::Failure(conflicts),
            }
        } else {
            let mut all_conflicts = Vec::new();
            for rule in &applicable {
                for feat_name in &rule.result_features {
                    if let (Some(lf), Some(rf)) =
                        (lhs.features.get(feat_name), rhs.features.get(feat_name))
                    {
                        if !lf.is_compatible(rf) {
                            all_conflicts.push(FeatureConflict::new(
                                feat_name,
                                format!("{lf:?}"),
                                format!("{rf:?}"),
                                format!("Rule '{}' requires matching {feat_name}", rule.name),
                            ));
                        }
                    }
                }
            }
            if all_conflicts.is_empty() {
                match lhs.features.unify_with(&rhs.features) {
                    Ok(merged) => UnificationResult::Success(merged),
                    Err(conflicts) => UnificationResult::Failure(conflicts),
                }
            } else {
                UnificationResult::Failure(all_conflicts)
            }
        }
    }

    /// Apply a single rule to two feature structures.
    pub fn apply_rule(
        &self,
        rule: &UnificationRule,
        lhs: &FeatureStructure,
        rhs: &FeatureStructure,
    ) -> UnificationResult {
        if !rule.matches(&lhs.category, &rhs.category) {
            return UnificationResult::Success(FeatureBundle::new());
        }
        let mut conflicts = Vec::new();
        for feat in &rule.result_features {
            if let (Some(lf), Some(rf)) = (lhs.features.get(feat), rhs.features.get(feat)) {
                if !lf.is_compatible(rf) {
                    conflicts.push(FeatureConflict::new(
                        feat,
                        format!("{lf:?}"),
                        format!("{rf:?}"),
                        format!("Rule '{}' failed on feature {feat}", rule.name),
                    ));
                }
            }
        }
        if conflicts.is_empty() {
            match lhs.features.unify_with(&rhs.features) {
                Ok(merged) => UnificationResult::Success(merged),
                Err(c) => UnificationResult::Failure(c),
            }
        } else {
            UnificationResult::Failure(conflicts)
        }
    }

    /// Propagate features up through a parse tree (bottom-up).
    pub fn propagate_features(
        &self,
        tree: &ParseTree,
    ) -> HashMap<usize, FeatureBundle> {
        let mut bundles: HashMap<usize, FeatureBundle> = HashMap::new();

        // Bottom-up traversal: process children before parents.
        let order = bottom_up_order(tree);
        for &idx in &order {
            let node = &tree.nodes[idx];
            let mut fb = FeatureBundle::new();

            // Initialise from node's own features map.
            for (k, v) in &node.features {
                if let Some(feat) = parse_feature_string(k, v) {
                    fb.set(k.clone(), feat);
                }
            }

            // Merge children's features (head-child propagation).
            if !node.children.is_empty() {
                // Use first child as head (simplification).
                if let Some(child_fb) = bundles.get(&node.children[0]) {
                    fb.merge(child_fb);
                }
            }

            bundles.insert(idx, fb);
        }
        bundles
    }

    /// Check a single grammar constraint against two feature bundles.
    pub fn check_constraint(
        &self,
        constraint: &GrammarConstraint,
        fb1: &FeatureBundle,
        fb2: &FeatureBundle,
    ) -> Option<ConstraintViolation> {
        match constraint {
            GrammarConstraint::Agreement(feat_name) => {
                if let (Some(f1), Some(f2)) = (fb1.get(feat_name), fb2.get(feat_name)) {
                    if !f1.is_compatible(f2) {
                        return Some(ConstraintViolation::new(
                            constraint.name(),
                            format!("{f1}"),
                            format!("{f2}"),
                            format!("{feat_name} values disagree"),
                        ));
                    }
                }
                None
            }
            GrammarConstraint::Selection(head_feat, dep_feat) => {
                if let Some(hf) = fb1.get(head_feat) {
                    if let Some(df) = fb2.get(dep_feat) {
                        if !hf.is_compatible(df) {
                            return Some(ConstraintViolation::new(
                                constraint.name(),
                                format!("{hf}"),
                                format!("{df}"),
                                format!("Head feature {head_feat} selects incompatible {dep_feat}"),
                            ));
                        }
                    }
                }
                None
            }
            GrammarConstraint::Government(gov_cat, governed_case) => {
                // Government: check that complement has the governed case.
                if let Some(cf) = fb2.get("Case") {
                    let expected_case_str = governed_case.as_str();
                    let actual_str = format!("{cf:?}");
                    if !actual_str.contains(expected_case_str) {
                        return Some(ConstraintViolation::new(
                            constraint.name(),
                            gov_cat.clone(),
                            actual_str,
                            format!("Expected case {governed_case}"),
                        ));
                    }
                }
                None
            }
            GrammarConstraint::Binding(ante_feats, pro_feats) => {
                for (af, pf) in ante_feats.iter().zip(pro_feats.iter()) {
                    if let (Some(a), Some(p)) = (fb1.get(af), fb2.get(pf)) {
                        if !a.is_compatible(p) {
                            return Some(ConstraintViolation::new(
                                constraint.name(),
                                format!("{a}"),
                                format!("{p}"),
                                format!("Binding features {af}/{pf} disagree"),
                            ));
                        }
                    }
                }
                None
            }
        }
    }

    /// Check all loaded constraints against a sentence's parse tree.
    pub fn check_all_constraints(
        &self,
        tree: &ParseTree,
    ) -> Vec<ConstraintViolation> {
        let bundles = self.propagate_features(tree);
        let mut violations = Vec::new();

        // For each internal node, check rules between its children.
        for node in &tree.nodes {
            if node.children.len() >= 2 {
                let lhs_fb = bundles.get(&node.children[0]).cloned().unwrap_or_default();
                let rhs_fb = bundles.get(&node.children[1]).cloned().unwrap_or_default();
                let lhs_label = &tree.nodes[node.children[0]].label;
                let rhs_label = &tree.nodes[node.children[1]].label;

                for rule in &self.rules {
                    if rule.matches(lhs_label, rhs_label) {
                        if let Some(v) = self.check_constraint(&rule.constraint, &lhs_fb, &rhs_fb)
                        {
                            violations.push(v);
                        }
                    }
                }
            }
        }
        violations
    }

    /// Occurs-check helper: detect if a feature name would create a cycle.
    pub fn occurs_check(
        &self,
        _name: &str,
        _bundle: &FeatureBundle,
    ) -> bool {
        // With atomic features only, cycles are impossible.
        // This is a placeholder for more complex feature structures.
        false
    }
}

impl Default for UnificationEngine {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ── DefaultConstraints ──────────────────────────────────────────────────────

/// Built-in set of ~80 English grammar constraints.
pub struct DefaultConstraints;

impl DefaultConstraints {
    pub fn all_rules() -> Vec<UnificationRule> {
        let mut rules = Vec::new();

        // ── 1–10: Subject-verb agreement ────────────────────────────────
        rules.push(UnificationRule::new(
            "SV-Number", "NP", "VP", vec!["Number"],
            GrammarConstraint::Agreement("Number".into()),
        ));
        rules.push(UnificationRule::new(
            "SV-Person", "NP", "VP", vec!["Person"],
            GrammarConstraint::Agreement("Person".into()),
        ));
        rules.push(UnificationRule::new(
            "SV-Number-S", "NP", "S", vec!["Number"],
            GrammarConstraint::Agreement("Number".into()),
        ));
        rules.push(UnificationRule::new(
            "SV-Person-S", "NP", "S", vec!["Person"],
            GrammarConstraint::Agreement("Person".into()),
        ));
        rules.push(UnificationRule::new(
            "Aux-Main-Tense", "VP", "VP", vec!["Tense"],
            GrammarConstraint::Agreement("Tense".into()),
        ));
        rules.push(UnificationRule::new(
            "Copula-Pred-Number", "NP", "AP", vec!["Number"],
            GrammarConstraint::Agreement("Number".into()),
        ));
        rules.push(UnificationRule::new(
            "Copula-NP-Pred", "NP", "NP", vec!["Number"],
            GrammarConstraint::Agreement("Number".into()),
        ));
        rules.push(UnificationRule::new(
            "Existential-Number", "NP", "VP", vec!["Number"],
            GrammarConstraint::Agreement("Number".into()),
        ));
        rules.push(UnificationRule::new(
            "Relative-Pron-Animacy", "NP", "CP", vec!["Animacy"],
            GrammarConstraint::Agreement("Animacy".into()),
        ));
        rules.push(UnificationRule::new(
            "Det-Noun-Number", "NP", "NP", vec!["Number"],
            GrammarConstraint::Agreement("Number".into()),
        ));

        // ── 11–20: Case constraints ─────────────────────────────────────
        rules.push(UnificationRule::new(
            "Subj-Nominative", "NP", "VP", vec!["Case"],
            GrammarConstraint::Government("VP".into(), "Nominative".into()),
        ));
        rules.push(UnificationRule::new(
            "Obj-Accusative", "VP", "NP", vec!["Case"],
            GrammarConstraint::Government("VP".into(), "Accusative".into()),
        ));
        rules.push(UnificationRule::new(
            "Prep-Obj-Accusative", "PP", "NP", vec!["Case"],
            GrammarConstraint::Government("PP".into(), "Accusative".into()),
        ));
        rules.push(UnificationRule::new(
            "Poss-Genitive", "NP", "NP", vec!["Case"],
            GrammarConstraint::Government("NP".into(), "Genitive".into()),
        ));
        rules.push(UnificationRule::new(
            "IO-Dative", "VP", "NP", vec!["Case"],
            GrammarConstraint::Government("VP".into(), "Dative".into()),
        ));
        rules.push(UnificationRule::new(
            "For-Obj", "PP", "NP", vec!["Case"],
            GrammarConstraint::Government("PP".into(), "Accusative".into()),
        ));
        rules.push(UnificationRule::new(
            "Pred-Nominative", "VP", "NP", vec!["Case"],
            GrammarConstraint::Government("VP".into(), "Nominative".into()),
        ));
        rules.push(UnificationRule::new(
            "Inf-Subj-Acc", "VP", "NP", vec!["Case"],
            GrammarConstraint::Government("VP".into(), "Accusative".into()),
        ));
        rules.push(UnificationRule::new(
            "Gerund-Poss", "VP", "NP", vec!["Case"],
            GrammarConstraint::Government("VP".into(), "Genitive".into()),
        ));
        rules.push(UnificationRule::new(
            "Appos-Case", "NP", "NP", vec!["Case"],
            GrammarConstraint::Agreement("Case".into()),
        ));

        // ── 21–30: Selection / c-selection ──────────────────────────────
        rules.push(UnificationRule::new(
            "V-selects-finite", "VP", "CP", vec!["Finiteness"],
            GrammarConstraint::Selection("Finiteness".into(), "Finiteness".into()),
        ));
        rules.push(UnificationRule::new(
            "V-selects-that-clause", "VP", "CP", vec!["Mood"],
            GrammarConstraint::Selection("Mood".into(), "Mood".into()),
        ));
        rules.push(UnificationRule::new(
            "Adj-selects-inf", "AP", "VP", vec!["Finiteness"],
            GrammarConstraint::Selection("Finiteness".into(), "Finiteness".into()),
        ));
        rules.push(UnificationRule::new(
            "Modal-selects-bare", "VP", "VP", vec!["Finiteness"],
            GrammarConstraint::Selection("Finiteness".into(), "Finiteness".into()),
        ));
        rules.push(UnificationRule::new(
            "Perf-selects-participle", "VP", "VP", vec!["Finiteness"],
            GrammarConstraint::Selection("Finiteness".into(), "Finiteness".into()),
        ));
        rules.push(UnificationRule::new(
            "Prog-selects-gerund", "VP", "VP", vec!["Finiteness"],
            GrammarConstraint::Selection("Finiteness".into(), "Finiteness".into()),
        ));
        rules.push(UnificationRule::new(
            "Pass-selects-participle", "VP", "VP", vec!["Finiteness"],
            GrammarConstraint::Selection("Finiteness".into(), "Finiteness".into()),
        ));
        rules.push(UnificationRule::new(
            "Comp-selects-finite", "CP", "S", vec!["Finiteness"],
            GrammarConstraint::Selection("Finiteness".into(), "Finiteness".into()),
        ));
        rules.push(UnificationRule::new(
            "N-selects-PP", "NP", "PP", vec![],
            GrammarConstraint::Selection("Transitivity".into(), "Transitivity".into()),
        ));
        rules.push(UnificationRule::new(
            "V-selects-NP-animacy", "VP", "NP", vec!["Animacy"],
            GrammarConstraint::Selection("Animacy".into(), "Animacy".into()),
        ));

        // ── 31–40: Binding constraints ──────────────────────────────────
        rules.push(UnificationRule::new(
            "Reflexive-Number", "NP", "NP", vec!["Number"],
            GrammarConstraint::Binding(vec!["Number".into()], vec!["Number".into()]),
        ));
        rules.push(UnificationRule::new(
            "Reflexive-Person", "NP", "NP", vec!["Person"],
            GrammarConstraint::Binding(vec!["Person".into()], vec!["Person".into()]),
        ));
        rules.push(UnificationRule::new(
            "Reflexive-Gender", "NP", "NP", vec!["Gender"],
            GrammarConstraint::Binding(vec!["Gender".into()], vec!["Gender".into()]),
        ));
        rules.push(UnificationRule::new(
            "Pronoun-Number", "NP", "NP", vec!["Number"],
            GrammarConstraint::Binding(vec!["Number".into()], vec!["Number".into()]),
        ));
        rules.push(UnificationRule::new(
            "Pronoun-Gender", "NP", "NP", vec!["Gender"],
            GrammarConstraint::Binding(vec!["Gender".into()], vec!["Gender".into()]),
        ));
        rules.push(UnificationRule::new(
            "Pronoun-Person", "NP", "NP", vec!["Person"],
            GrammarConstraint::Binding(vec!["Person".into()], vec!["Person".into()]),
        ));
        rules.push(UnificationRule::new(
            "Pronoun-Animacy", "NP", "NP", vec!["Animacy"],
            GrammarConstraint::Binding(vec!["Animacy".into()], vec!["Animacy".into()]),
        ));
        rules.push(UnificationRule::new(
            "RelPron-Animacy", "NP", "CP", vec!["Animacy"],
            GrammarConstraint::Binding(vec!["Animacy".into()], vec!["Animacy".into()]),
        ));
        rules.push(UnificationRule::new(
            "Reciprocal-Number", "NP", "NP", vec!["Number"],
            GrammarConstraint::Binding(vec!["Number".into()], vec!["Number".into()]),
        ));
        rules.push(UnificationRule::new(
            "Poss-Binding", "NP", "NP", vec!["Person", "Number"],
            GrammarConstraint::Binding(
                vec!["Person".into(), "Number".into()],
                vec!["Person".into(), "Number".into()],
            ),
        ));

        // ── 41–50: Tense / aspect / mood ────────────────────────────────
        rules.push(UnificationRule::new(
            "Tense-Sequence", "S", "CP", vec!["Tense"],
            GrammarConstraint::Selection("Tense".into(), "Tense".into()),
        ));
        rules.push(UnificationRule::new(
            "Aspect-Consistency", "VP", "VP", vec!["Aspect"],
            GrammarConstraint::Agreement("Aspect".into()),
        ));
        rules.push(UnificationRule::new(
            "Voice-Propagation", "VP", "VP", vec!["Voice"],
            GrammarConstraint::Agreement("Voice".into()),
        ));
        rules.push(UnificationRule::new(
            "Mood-Clause", "CP", "S", vec!["Mood"],
            GrammarConstraint::Agreement("Mood".into()),
        ));
        rules.push(UnificationRule::new(
            "Subjunctive-Tense", "VP", "VP", vec!["Tense", "Mood"],
            GrammarConstraint::Agreement("Mood".into()),
        ));
        rules.push(UnificationRule::new(
            "Imperative-Person", "VP", "NP", vec!["Person"],
            GrammarConstraint::Agreement("Person".into()),
        ));
        rules.push(UnificationRule::new(
            "Conditional-Tense", "S", "S", vec!["Tense"],
            GrammarConstraint::Agreement("Tense".into()),
        ));
        rules.push(UnificationRule::new(
            "Perf-Tense-Consistent", "VP", "VP", vec!["Tense"],
            GrammarConstraint::Agreement("Tense".into()),
        ));
        rules.push(UnificationRule::new(
            "Prog-Aspect", "VP", "VP", vec!["Aspect"],
            GrammarConstraint::Agreement("Aspect".into()),
        ));
        rules.push(UnificationRule::new(
            "Future-Modal", "VP", "VP", vec!["Tense"],
            GrammarConstraint::Agreement("Tense".into()),
        ));

        // ── 51–60: Definiteness & quantifier constraints ────────────────
        rules.push(UnificationRule::new(
            "Det-Def", "NP", "NP", vec!["Definiteness"],
            GrammarConstraint::Agreement("Definiteness".into()),
        ));
        rules.push(UnificationRule::new(
            "Existential-Indef", "S", "NP", vec!["Definiteness"],
            GrammarConstraint::Selection("Definiteness".into(), "Definiteness".into()),
        ));
        rules.push(UnificationRule::new(
            "Partitive", "NP", "PP", vec!["Number"],
            GrammarConstraint::Agreement("Number".into()),
        ));
        rules.push(UnificationRule::new(
            "Quantifier-Number", "NP", "NP", vec!["Number"],
            GrammarConstraint::Agreement("Number".into()),
        ));
        rules.push(UnificationRule::new(
            "Mass-Noun-Det", "NP", "NP", vec!["Number"],
            GrammarConstraint::Agreement("Number".into()),
        ));
        rules.push(UnificationRule::new(
            "Demonstrative-Number", "NP", "NP", vec!["Number"],
            GrammarConstraint::Agreement("Number".into()),
        ));
        rules.push(UnificationRule::new(
            "Article-Def", "NP", "NP", vec!["Definiteness"],
            GrammarConstraint::Agreement("Definiteness".into()),
        ));
        rules.push(UnificationRule::new(
            "Some-Indef", "NP", "NP", vec!["Definiteness"],
            GrammarConstraint::Agreement("Definiteness".into()),
        ));
        rules.push(UnificationRule::new(
            "Any-Polarity", "NP", "NP", vec!["Definiteness"],
            GrammarConstraint::Agreement("Definiteness".into()),
        ));
        rules.push(UnificationRule::new(
            "Neg-Polarity-Item", "VP", "NP", vec!["Definiteness"],
            GrammarConstraint::Selection("Definiteness".into(), "Definiteness".into()),
        ));

        // ── 61–70: Transitivity / argument structure ────────────────────
        rules.push(UnificationRule::new(
            "Trans-Requires-Obj", "VP", "NP", vec!["Transitivity"],
            GrammarConstraint::Selection("Transitivity".into(), "Transitivity".into()),
        ));
        rules.push(UnificationRule::new(
            "Intrans-No-Obj", "VP", "*", vec!["Transitivity"],
            GrammarConstraint::Selection("Transitivity".into(), "Transitivity".into()),
        ));
        rules.push(UnificationRule::new(
            "Ditrans-IO", "VP", "NP", vec!["Transitivity"],
            GrammarConstraint::Selection("Transitivity".into(), "Transitivity".into()),
        ));
        rules.push(UnificationRule::new(
            "Copula-Pred", "VP", "AP", vec!["Transitivity"],
            GrammarConstraint::Selection("Transitivity".into(), "Transitivity".into()),
        ));
        rules.push(UnificationRule::new(
            "Unacc-Subj", "VP", "NP", vec!["Transitivity"],
            GrammarConstraint::Selection("Transitivity".into(), "Transitivity".into()),
        ));
        rules.push(UnificationRule::new(
            "Pass-From-Trans", "VP", "VP", vec!["Transitivity", "Voice"],
            GrammarConstraint::Selection("Voice".into(), "Voice".into()),
        ));
        rules.push(UnificationRule::new(
            "Caus-Trans", "VP", "VP", vec!["Transitivity"],
            GrammarConstraint::Selection("Transitivity".into(), "Transitivity".into()),
        ));
        rules.push(UnificationRule::new(
            "Middle-Voice", "VP", "NP", vec!["Voice"],
            GrammarConstraint::Selection("Voice".into(), "Voice".into()),
        ));
        rules.push(UnificationRule::new(
            "Raising-Subj", "NP", "VP", vec!["Case"],
            GrammarConstraint::Government("VP".into(), "Nominative".into()),
        ));
        rules.push(UnificationRule::new(
            "Control-Subj", "NP", "VP", vec!["Person", "Number"],
            GrammarConstraint::Agreement("Person".into()),
        ));

        // ── 71–80: Finiteness / clause-typing ──────────────────────────
        rules.push(UnificationRule::new(
            "Finite-Clause-Tense", "CP", "S", vec!["Finiteness", "Tense"],
            GrammarConstraint::Selection("Finiteness".into(), "Finiteness".into()),
        ));
        rules.push(UnificationRule::new(
            "NonFinite-No-Tense", "VP", "VP", vec!["Finiteness"],
            GrammarConstraint::Selection("Finiteness".into(), "Finiteness".into()),
        ));
        rules.push(UnificationRule::new(
            "Inf-Marker-To", "VP", "VP", vec!["Finiteness"],
            GrammarConstraint::Selection("Finiteness".into(), "Finiteness".into()),
        ));
        rules.push(UnificationRule::new(
            "Gerund-Ing", "VP", "VP", vec!["Finiteness"],
            GrammarConstraint::Selection("Finiteness".into(), "Finiteness".into()),
        ));
        rules.push(UnificationRule::new(
            "Participle-En", "VP", "VP", vec!["Finiteness"],
            GrammarConstraint::Selection("Finiteness".into(), "Finiteness".into()),
        ));
        rules.push(UnificationRule::new(
            "Small-Clause-Pred", "S", "AP", vec!["Finiteness"],
            GrammarConstraint::Selection("Finiteness".into(), "Finiteness".into()),
        ));
        rules.push(UnificationRule::new(
            "Relative-Clause-Finite", "NP", "CP", vec!["Finiteness"],
            GrammarConstraint::Selection("Finiteness".into(), "Finiteness".into()),
        ));
        rules.push(UnificationRule::new(
            "Adverbial-Clause", "S", "S", vec!["Finiteness"],
            GrammarConstraint::Selection("Finiteness".into(), "Finiteness".into()),
        ));
        rules.push(UnificationRule::new(
            "Nominal-Clause", "NP", "CP", vec!["Finiteness"],
            GrammarConstraint::Selection("Finiteness".into(), "Finiteness".into()),
        ));
        rules.push(UnificationRule::new(
            "Cleft-Clause", "S", "CP", vec!["Finiteness"],
            GrammarConstraint::Selection("Finiteness".into(), "Finiteness".into()),
        ));

        rules
    }

    /// Convenience: number of built-in rules.
    pub fn count() -> usize {
        Self::all_rules().len()
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn default_feature_map() -> HashMap<String, Feature> {
    let mut m = HashMap::new();
    m.insert("Number".into(), Feature::Number(NumberValue::Singular));
    m.insert("Person".into(), Feature::Person(PersonValue::Third));
    m.insert("Tense".into(), Feature::Tense(TenseValue::Present));
    m.insert("Aspect".into(), Feature::Aspect(AspectValue::Simple));
    m.insert("Voice".into(), Feature::Voice(VoiceValue::Active));
    m.insert("Mood".into(), Feature::Mood(MoodValue::Indicative));
    m.insert("Case".into(), Feature::Case(CaseValue::Nominative));
    m.insert("Gender".into(), Feature::Gender(GenderValue::Common));
    m.insert(
        "Definiteness".into(),
        Feature::Definiteness(DefinitenessValue::Bare),
    );
    m.insert("Animacy".into(), Feature::Animacy(AnimacyValue::Inanimate));
    m.insert(
        "Transitivity".into(),
        Feature::Transitivity(TransitivityValue::Transitive),
    );
    m.insert(
        "Finiteness".into(),
        Feature::Finiteness(FinitenessValue::Finite),
    );
    m
}

fn parse_feature_string(key: &str, value: &str) -> Option<Feature> {
    match key {
        "Number" => match value {
            "Sing" | "Singular" => Some(Feature::Number(NumberValue::Singular)),
            "Plur" | "Plural" => Some(Feature::Number(NumberValue::Plural)),
            "Uncount" | "Uncountable" => Some(Feature::Number(NumberValue::Uncountable)),
            _ => None,
        },
        "Person" => match value {
            "1" | "First" => Some(Feature::Person(PersonValue::First)),
            "2" | "Second" => Some(Feature::Person(PersonValue::Second)),
            "3" | "Third" => Some(Feature::Person(PersonValue::Third)),
            _ => None,
        },
        "Tense" => match value {
            "Past" => Some(Feature::Tense(TenseValue::Past)),
            "Pres" | "Present" => Some(Feature::Tense(TenseValue::Present)),
            "Fut" | "Future" => Some(Feature::Tense(TenseValue::Future)),
            _ => None,
        },
        "Aspect" => match value {
            "Simple" => Some(Feature::Aspect(AspectValue::Simple)),
            "Prog" | "Progressive" => Some(Feature::Aspect(AspectValue::Progressive)),
            "Perf" | "Perfect" => Some(Feature::Aspect(AspectValue::Perfect)),
            "PerfProg" | "PerfectProgressive" => {
                Some(Feature::Aspect(AspectValue::PerfectProgressive))
            }
            _ => None,
        },
        "Voice" => match value {
            "Act" | "Active" => Some(Feature::Voice(VoiceValue::Active)),
            "Pass" | "Passive" => Some(Feature::Voice(VoiceValue::Passive)),
            _ => None,
        },
        "Mood" => match value {
            "Ind" | "Indicative" => Some(Feature::Mood(MoodValue::Indicative)),
            "Sub" | "Subjunctive" => Some(Feature::Mood(MoodValue::Subjunctive)),
            "Imp" | "Imperative" => Some(Feature::Mood(MoodValue::Imperative)),
            "Int" | "Interrogative" => Some(Feature::Mood(MoodValue::Interrogative)),
            _ => None,
        },
        "Case" => match value {
            "Nom" | "Nominative" => Some(Feature::Case(CaseValue::Nominative)),
            "Acc" | "Accusative" => Some(Feature::Case(CaseValue::Accusative)),
            "Gen" | "Genitive" => Some(Feature::Case(CaseValue::Genitive)),
            "Dat" | "Dative" => Some(Feature::Case(CaseValue::Dative)),
            _ => None,
        },
        "Gender" => match value {
            "Masc" | "Masculine" => Some(Feature::Gender(GenderValue::Masculine)),
            "Fem" | "Feminine" => Some(Feature::Gender(GenderValue::Feminine)),
            "Neut" | "Neuter" => Some(Feature::Gender(GenderValue::Neuter)),
            "Com" | "Common" => Some(Feature::Gender(GenderValue::Common)),
            _ => None,
        },
        "Definiteness" => match value {
            "Def" | "Definite" => Some(Feature::Definiteness(DefinitenessValue::Definite)),
            "Indef" | "Indefinite" => Some(Feature::Definiteness(DefinitenessValue::Indefinite)),
            "Bare" => Some(Feature::Definiteness(DefinitenessValue::Bare)),
            _ => None,
        },
        "Animacy" => match value {
            "Anim" | "Animate" => Some(Feature::Animacy(AnimacyValue::Animate)),
            "Inanim" | "Inanimate" => Some(Feature::Animacy(AnimacyValue::Inanimate)),
            _ => None,
        },
        "Finiteness" => match value {
            "Fin" | "Finite" => Some(Feature::Finiteness(FinitenessValue::Finite)),
            "NonFin" | "NonFinite" => Some(Feature::Finiteness(FinitenessValue::NonFinite)),
            "Inf" | "Infinitive" => Some(Feature::Finiteness(FinitenessValue::Infinitive)),
            "Ger" | "Gerund" => Some(Feature::Finiteness(FinitenessValue::Gerund)),
            "Part" | "Participle" => Some(Feature::Finiteness(FinitenessValue::Participle)),
            _ => None,
        },
        _ => None,
    }
}

/// Return node indices in bottom-up (post-order) traversal.
fn bottom_up_order(tree: &ParseTree) -> Vec<usize> {
    let mut order = Vec::new();
    let mut stack = vec![(tree.root_index, false)];
    while let Some((idx, visited)) = stack.pop() {
        if visited {
            order.push(idx);
            continue;
        }
        if idx >= tree.nodes.len() {
            continue;
        }
        stack.push((idx, true));
        for &child in tree.nodes[idx].children.iter().rev() {
            stack.push((child, false));
        }
    }
    order
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::{np_features, vp_features};
    use shared_types::ParseNode;

    #[test]
    fn test_default_constraints_count() {
        assert!(DefaultConstraints::count() >= 80);
    }

    #[test]
    fn test_engine_unify_compatible() {
        let engine = UnificationEngine::with_defaults();
        let np = np_features(NumberValue::Singular, PersonValue::Third);
        let vp = vp_features(TenseValue::Present, NumberValue::Singular, PersonValue::Third);
        let result = engine.unify(&np, &vp);
        assert!(result.is_success());
    }

    #[test]
    fn test_engine_unify_conflict() {
        let engine = UnificationEngine::with_defaults();
        let np = np_features(NumberValue::Singular, PersonValue::Third);
        let vp = vp_features(TenseValue::Present, NumberValue::Plural, PersonValue::Third);
        let result = engine.unify(&np, &vp);
        assert!(result.is_failure());
    }

    #[test]
    fn test_apply_rule() {
        let engine = UnificationEngine::with_defaults();
        let rule = &engine.rules[0]; // SV-Number
        let np = np_features(NumberValue::Singular, PersonValue::Third);
        let vp = vp_features(TenseValue::Present, NumberValue::Singular, PersonValue::Third);
        let result = engine.apply_rule(rule, &np, &vp);
        assert!(result.is_success());
    }

    #[test]
    fn test_check_agreement_constraint() {
        let engine = UnificationEngine::new();
        let constraint = GrammarConstraint::Agreement("Number".into());
        let mut fb1 = FeatureBundle::new();
        fb1.set("Number", Feature::Number(NumberValue::Singular));
        let mut fb2 = FeatureBundle::new();
        fb2.set("Number", Feature::Number(NumberValue::Plural));
        let v = engine.check_constraint(&constraint, &fb1, &fb2);
        assert!(v.is_some());
    }

    #[test]
    fn test_check_agreement_no_violation() {
        let engine = UnificationEngine::new();
        let constraint = GrammarConstraint::Agreement("Number".into());
        let mut fb1 = FeatureBundle::new();
        fb1.set("Number", Feature::Number(NumberValue::Singular));
        let mut fb2 = FeatureBundle::new();
        fb2.set("Number", Feature::Number(NumberValue::Singular));
        let v = engine.check_constraint(&constraint, &fb1, &fb2);
        assert!(v.is_none());
    }

    #[test]
    fn test_propagate_features() {
        let engine = UnificationEngine::with_defaults();
        let mut nodes = vec![
            ParseNode::new_nonterminal("S", 0, 2),
            ParseNode::new_nonterminal("NP", 0, 1),
            ParseNode::new_nonterminal("VP", 1, 2),
        ];
        nodes[0].children = vec![1, 2];
        nodes[1].parent = Some(0);
        nodes[1].features.insert("Number".into(), "Singular".into());
        nodes[2].parent = Some(0);
        nodes[2].features.insert("Number".into(), "Singular".into());
        let tree = ParseTree::new(nodes, 0);
        let bundles = engine.propagate_features(&tree);
        assert!(bundles.contains_key(&0));
    }

    #[test]
    fn test_occurs_check_always_false_for_atomic() {
        let engine = UnificationEngine::new();
        assert!(!engine.occurs_check("Number", &FeatureBundle::new()));
    }

    #[test]
    fn test_bottom_up_order() {
        let mut nodes = vec![
            ParseNode::new_nonterminal("S", 0, 3),
            ParseNode::new_terminal("NP", "cat", 0),
            ParseNode::new_terminal("VP", "runs", 1),
        ];
        nodes[0].children = vec![1, 2];
        let tree = ParseTree::new(nodes, 0);
        let order = bottom_up_order(&tree);
        assert_eq!(order, vec![1, 2, 0]);
    }

    #[test]
    fn test_parse_feature_string() {
        assert_eq!(
            parse_feature_string("Number", "Sing"),
            Some(Feature::Number(NumberValue::Singular))
        );
        assert_eq!(
            parse_feature_string("Tense", "Past"),
            Some(Feature::Tense(TenseValue::Past))
        );
        assert!(parse_feature_string("Unknown", "xyz").is_none());
    }

    #[test]
    fn test_grammar_constraint_display() {
        let c = GrammarConstraint::Agreement("Number".into());
        assert_eq!(c.to_string(), "Agreement(Number)");
    }
}
