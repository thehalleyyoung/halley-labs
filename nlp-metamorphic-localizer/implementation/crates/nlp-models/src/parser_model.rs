//! Rule-based dependency parser using an arc-eager-like transition system.

use shared_types::{DependencyEdge, DependencyRelation, PosTag, Token};
use serde::{Deserialize, Serialize};

// ── Types ───────────────────────────────────────────────────────────────────

/// The direction a head-finding rule uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HeadDirection {
    LeftToRight,
    RightToLeft,
}

/// A head-finding rule for a given parent POS category.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadRule {
    pub parent_tag: PosTag,
    pub priority: Vec<PosTag>,
    pub direction: HeadDirection,
}

/// Arc-eager transitions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Transition {
    Shift,
    LeftArc(DependencyRelation),
    RightArc(DependencyRelation),
    Reduce,
}

/// Snapshot of the parser's configuration.
#[derive(Debug, Clone)]
pub struct ParserState {
    pub stack: Vec<usize>,
    pub buffer: Vec<usize>,
    pub arcs: Vec<DependencyEdge>,
    pub heads: Vec<Option<usize>>,
}

impl ParserState {
    fn new(n: usize) -> Self {
        Self {
            stack: vec![0], // ROOT sentinel at position 0
            buffer: (1..=n).collect(),
            arcs: Vec::new(),
            heads: vec![None; n + 1],
        }
    }

    fn stack_top(&self) -> Option<usize> {
        self.stack.last().copied()
    }

    fn buffer_front(&self) -> Option<usize> {
        self.buffer.first().copied()
    }

    fn apply(&mut self, t: &Transition) {
        match t {
            Transition::Shift => {
                if let Some(b) = self.buffer.first().copied() {
                    self.buffer.remove(0);
                    self.stack.push(b);
                }
            }
            Transition::LeftArc(rel) => {
                if let (Some(&s), Some(&b)) = (self.stack.last(), self.buffer.first()) {
                    if s != 0 {
                        self.arcs.push(DependencyEdge {
                            head_index: b,
                            dependent_index: s,
                            relation: rel.clone(),
                        });
                        self.heads[s] = Some(b);
                        self.stack.pop();
                    }
                }
            }
            Transition::RightArc(rel) => {
                if let (Some(&s), Some(&b)) = (self.stack.last(), self.buffer.first()) {
                    self.arcs.push(DependencyEdge {
                        head_index: s,
                        dependent_index: b,
                        relation: rel.clone(),
                    });
                    self.heads[b] = Some(s);
                    self.buffer.remove(0);
                    self.stack.push(b);
                }
            }
            Transition::Reduce => {
                if self.stack.len() > 1 {
                    self.stack.pop();
                }
            }
        }
    }

    fn is_terminal(&self) -> bool {
        self.buffer.is_empty()
    }
}

// ── RuleBasedParser ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct RuleBasedParser {
    head_rules: Vec<HeadRule>,
}

impl Default for RuleBasedParser {
    fn default() -> Self {
        Self::new()
    }
}

impl RuleBasedParser {
    pub fn new() -> Self {
        Self {
            head_rules: build_head_rules(),
        }
    }

    /// Parse tokens into dependency edges. Returns edges referencing token
    /// indices (1-based; 0 = ROOT).
    pub fn parse(&self, tokens: &[Token]) -> Vec<DependencyEdge> {
        if tokens.is_empty() {
            return Vec::new();
        }
        let n = tokens.len();
        let tags: Vec<PosTag> = tokens
            .iter()
            .map(|t| t.pos_tag.unwrap_or(PosTag::Noun))
            .collect();
        // Tags indexed 1..=n (index 0 is ROOT)
        let mut all_tags = vec![PosTag::Other]; // ROOT sentinel
        all_tags.extend(tags.clone());

        let mut state = ParserState::new(n);
        let mut safety = 0;
        let max_steps = n * 4 + 10;

        while !state.is_terminal() && safety < max_steps {
            safety += 1;
            let transition = self.select_transition(&state, &all_tags);
            state.apply(&transition);
        }

        // Drain remaining stack → attach to root
        while state.stack.len() > 1 {
            let s = state.stack.pop().unwrap();
            if state.heads[s].is_none() {
                state.arcs.push(DependencyEdge {
                    head_index: 0,
                    dependent_index: s,
                    relation: DependencyRelation::Root,
                });
                state.heads[s] = Some(0);
            }
        }

        // Ensure every token has a head
        for i in 1..=n {
            if state.heads[i].is_none() {
                state.arcs.push(DependencyEdge {
                    head_index: 0,
                    dependent_index: i,
                    relation: DependencyRelation::Root,
                });
            }
        }

        state.arcs
    }

    /// Check whether the dependency tree is projective.
    pub fn is_projective(edges: &[DependencyEdge]) -> bool {
        for a in edges {
            for b in edges {
                if std::ptr::eq(a, b) {
                    continue;
                }
                let (a_min, a_max) = ordered(a.head_index, a.dependent_index);
                let (b_min, b_max) = ordered(b.head_index, b.dependent_index);
                // Check crossing
                if a_min < b_min && b_min < a_max && a_max < b_max {
                    return false;
                }
                if b_min < a_min && a_min < b_max && b_max < a_max {
                    return false;
                }
            }
        }
        true
    }

    /// Reconstruct children lists from edges (0 = root).
    pub fn build_children(edges: &[DependencyEdge], n: usize) -> Vec<Vec<usize>> {
        let mut children = vec![Vec::new(); n + 1];
        for e in edges {
            if e.head_index <= n {
                children[e.head_index].push(e.dependent_index);
            }
        }
        for ch in &mut children {
            ch.sort_unstable();
        }
        children
    }

    // ── transition oracle ───────────────────────────────────────────────────

    fn select_transition(&self, state: &ParserState, tags: &[PosTag]) -> Transition {
        let s = match state.stack_top() {
            Some(s) => s,
            None => return Transition::Shift,
        };
        let b = match state.buffer_front() {
            Some(b) => b,
            None => return Transition::Reduce,
        };

        let s_tag = tags.get(s).copied().unwrap_or(PosTag::Other);
        let b_tag = tags.get(b).copied().unwrap_or(PosTag::Other);

        // ROOT can only take dependents via RightArc
        if s == 0 {
            if b_tag == PosTag::Verb {
                return Transition::RightArc(DependencyRelation::Root);
            }
            return Transition::Shift;
        }

        // Determiners attach to the next noun/adjective (LeftArc from buffer)
        if s_tag == PosTag::Det {
            if matches!(b_tag, PosTag::Noun | PosTag::Adj | PosTag::Num) {
                return Transition::LeftArc(DependencyRelation::Det);
            }
            return Transition::Shift;
        }

        // Adjective before noun → modifier (LeftArc)
        if s_tag == PosTag::Adj && b_tag == PosTag::Noun {
            return Transition::LeftArc(DependencyRelation::Amod);
        }

        // Adverb before verb/adjective → modifier
        if s_tag == PosTag::Adv && matches!(b_tag, PosTag::Verb | PosTag::Adj) {
            return Transition::LeftArc(DependencyRelation::Amod);
        }

        // Noun/Pronoun before Verb → Subject (LeftArc)
        if matches!(s_tag, PosTag::Noun | PosTag::Pron) && b_tag == PosTag::Verb {
            return Transition::LeftArc(DependencyRelation::Nsubj);
        }

        // Verb + Noun/Pronoun → Object (RightArc)
        if s_tag == PosTag::Verb && matches!(b_tag, PosTag::Noun | PosTag::Pron) {
            return Transition::RightArc(DependencyRelation::Dobj);
        }

        // Verb + Determiner → shift (wait for the noun phrase)
        if s_tag == PosTag::Verb && b_tag == PosTag::Det {
            return Transition::Shift;
        }

        // Verb + Adjective → Complement
        if s_tag == PosTag::Verb && b_tag == PosTag::Adj {
            return Transition::RightArc(DependencyRelation::Ccomp);
        }

        // Verb + Adverb → modifier
        if s_tag == PosTag::Verb && b_tag == PosTag::Adv {
            return Transition::RightArc(DependencyRelation::Amod);
        }

        // Verb + Preposition → Preposition (RightArc)
        if s_tag == PosTag::Verb && b_tag == PosTag::Prep {
            return Transition::RightArc(DependencyRelation::Prep);
        }

        // Noun + Preposition → Preposition
        if s_tag == PosTag::Noun && b_tag == PosTag::Prep {
            return Transition::RightArc(DependencyRelation::Prep);
        }

        // Preposition + Noun/Pronoun → Object
        if s_tag == PosTag::Prep && matches!(b_tag, PosTag::Noun | PosTag::Pron) {
            return Transition::RightArc(DependencyRelation::Dobj);
        }

        // Preposition + Determiner → shift
        if s_tag == PosTag::Prep && b_tag == PosTag::Det {
            return Transition::Shift;
        }

        // Auxiliary + Verb → Auxiliary
        if s_tag == PosTag::Verb
            && b_tag == PosTag::Verb
            && state.heads[s].is_none()
        {
            return Transition::LeftArc(DependencyRelation::Aux);
        }

        // Conjunction → Conjunction
        if b_tag == PosTag::Conj {
            return Transition::Shift;
        }
        if s_tag == PosTag::Conj {
            return Transition::LeftArc(DependencyRelation::Conj);
        }

        // Number before noun → modifier
        if s_tag == PosTag::Num && b_tag == PosTag::Noun {
            return Transition::LeftArc(DependencyRelation::Amod);
        }

        // Punctuation → attach left
        if b_tag == PosTag::Punct {
            return Transition::RightArc(DependencyRelation::Punct);
        }
        if s_tag == PosTag::Punct {
            if state.heads[s].is_some() {
                return Transition::Reduce;
            }
            return Transition::LeftArc(DependencyRelation::Punct);
        }

        // If stack item already has a head, reduce
        if state.heads[s].is_some() {
            return Transition::Reduce;
        }

        Transition::Shift
    }
}

// ── Head-finding rules ──────────────────────────────────────────────────────

fn build_head_rules() -> Vec<HeadRule> {
    vec![
        HeadRule {
            parent_tag: PosTag::Verb,
            priority: vec![PosTag::Verb, PosTag::Noun, PosTag::Pron],
            direction: HeadDirection::LeftToRight,
        },
        HeadRule {
            parent_tag: PosTag::Noun,
            priority: vec![PosTag::Noun, PosTag::Adj, PosTag::Det],
            direction: HeadDirection::RightToLeft,
        },
        HeadRule {
            parent_tag: PosTag::Prep,
            priority: vec![PosTag::Noun, PosTag::Pron],
            direction: HeadDirection::RightToLeft,
        },
    ]
}

fn ordered(a: usize, b: usize) -> (usize, usize) {
    if a <= b { (a, b) } else { (b, a) }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tagged(pairs: &[(&str, PosTag)]) -> Vec<Token> {
        pairs
            .iter()
            .enumerate()
            .map(|(i, (w, tag))| {
                let mut t = Token::new(*w, i, 0, w.len());
                t.pos_tag = Some(*tag);
                t
            })
            .collect()
    }

    #[test]
    fn test_simple_sv() {
        let parser = RuleBasedParser::new();
        let tokens = tagged(&[("Dogs", PosTag::Noun), ("run", PosTag::Verb)]);
        let edges = parser.parse(&tokens);
        assert!(!edges.is_empty());
        // "Dogs" should be subject of "run"
        let subj = edges.iter().find(|e| e.dependent_index == 1);
        assert!(subj.is_some());
        assert_eq!(subj.unwrap().relation, DependencyRelation::Nsubj);
    }

    #[test]
    fn test_svo() {
        let parser = RuleBasedParser::new();
        let tokens = tagged(&[
            ("She", PosTag::Pron),
            ("likes", PosTag::Verb),
            ("cats", PosTag::Noun),
        ]);
        let edges = parser.parse(&tokens);
        let obj = edges.iter().find(|e| e.dependent_index == 3);
        assert!(obj.is_some());
        assert_eq!(obj.unwrap().relation, DependencyRelation::Dobj);
    }

    #[test]
    fn test_det_noun() {
        let parser = RuleBasedParser::new();
        let tokens = tagged(&[
            ("The", PosTag::Det),
            ("cat", PosTag::Noun),
            ("sat", PosTag::Verb),
        ]);
        let edges = parser.parse(&tokens);
        let det = edges.iter().find(|e| e.dependent_index == 1 && e.relation == DependencyRelation::Det);
        assert!(det.is_some());
    }

    #[test]
    fn test_adj_noun() {
        let parser = RuleBasedParser::new();
        let tokens = tagged(&[
            ("big", PosTag::Adj),
            ("dog", PosTag::Noun),
        ]);
        let edges = parser.parse(&tokens);
        let m = edges.iter().find(|e| e.dependent_index == 1 && e.relation == DependencyRelation::Amod);
        assert!(m.is_some());
    }

    #[test]
    fn test_prepositional_phrase() {
        let parser = RuleBasedParser::new();
        let tokens = tagged(&[
            ("cat", PosTag::Noun),
            ("on", PosTag::Prep),
            ("mat", PosTag::Noun),
        ]);
        let edges = parser.parse(&tokens);
        let prep_edge = edges.iter().find(|e| e.relation == DependencyRelation::Prep);
        assert!(prep_edge.is_some());
    }

    #[test]
    fn test_projectivity_simple() {
        let edges = vec![
            DependencyEdge { head_index: 0, dependent_index: 2, relation: DependencyRelation::Root },
            DependencyEdge { head_index: 2, dependent_index: 1, relation: DependencyRelation::Nsubj },
            DependencyEdge { head_index: 2, dependent_index: 3, relation: DependencyRelation::Dobj },
        ];
        assert!(RuleBasedParser::is_projective(&edges));
    }

    #[test]
    fn test_projectivity_crossing() {
        let edges = vec![
            DependencyEdge { head_index: 1, dependent_index: 3, relation: DependencyRelation::Other },
            DependencyEdge { head_index: 2, dependent_index: 4, relation: DependencyRelation::Other },
        ];
        // 1→3 and 2→4 cross
        assert!(!RuleBasedParser::is_projective(&edges));
    }

    #[test]
    fn test_empty_input() {
        let parser = RuleBasedParser::new();
        let edges = parser.parse(&[]);
        assert!(edges.is_empty());
    }

    #[test]
    fn test_every_token_has_head() {
        let parser = RuleBasedParser::new();
        let tokens = tagged(&[
            ("The", PosTag::Det),
            ("quick", PosTag::Adj),
            ("brown", PosTag::Adj),
            ("fox", PosTag::Noun),
            ("jumps", PosTag::Verb),
        ]);
        let edges = parser.parse(&tokens);
        for i in 1..=tokens.len() {
            assert!(
                edges.iter().any(|e| e.dependent_index == i),
                "Token {} has no head",
                i
            );
        }
    }

    #[test]
    fn test_build_children() {
        let edges = vec![
            DependencyEdge { head_index: 0, dependent_index: 2, relation: DependencyRelation::Root },
            DependencyEdge { head_index: 2, dependent_index: 1, relation: DependencyRelation::Nsubj },
            DependencyEdge { head_index: 2, dependent_index: 3, relation: DependencyRelation::Dobj },
        ];
        let children = RuleBasedParser::build_children(&edges, 3);
        assert_eq!(children[0], vec![2]);
        assert_eq!(children[2], vec![1, 3]);
    }

    #[test]
    fn test_punctuation_attachment() {
        let parser = RuleBasedParser::new();
        let tokens = tagged(&[
            ("Run", PosTag::Verb),
            (".", PosTag::Punct),
        ]);
        let edges = parser.parse(&tokens);
        let punct = edges.iter().find(|e| e.dependent_index == 2);
        assert!(punct.is_some());
        assert_eq!(punct.unwrap().relation, DependencyRelation::Punct);
    }
}
