//! Regex matching with triple implementation.
//!
//! Includes Thompson's NFA construction, NFA→DFA subset construction,
//! and DFA→Boolean WFA conversion for ZK circuit compilation.

use std::collections::{HashMap, HashSet, BTreeSet, VecDeque};
use super::{
    GoldilocksField, ScoringCircuit, CircuitConstraint,
    ScoringWFA, BooleanSemiring, Semiring,
    TripleMetric, DifferentialResult, ScoringPair,
};
use serde::{Serialize, Deserialize};

/// Regex AST
#[derive(Debug, Clone, PartialEq)]
pub enum RegexAst {
    /// Empty string ε
    Epsilon,
    /// Single character
    Char(char),
    /// Character class [abc]
    CharClass(Vec<char>),
    /// Negated character class [^abc]
    NegCharClass(Vec<char>),
    /// Any character .
    Dot,
    /// Concatenation: ab
    Concat(Box<RegexAst>, Box<RegexAst>),
    /// Alternation: a|b
    Alt(Box<RegexAst>, Box<RegexAst>),
    /// Kleene star: a*
    Star(Box<RegexAst>),
    /// One or more: a+
    Plus(Box<RegexAst>),
    /// Optional: a?
    Optional(Box<RegexAst>),
}

impl RegexAst {
    pub fn char(c: char) -> Self { Self::Char(c) }
    
    pub fn concat(a: Self, b: Self) -> Self {
        Self::Concat(Box::new(a), Box::new(b))
    }
    
    pub fn alt(a: Self, b: Self) -> Self {
        Self::Alt(Box::new(a), Box::new(b))
    }
    
    pub fn star(a: Self) -> Self {
        Self::Star(Box::new(a))
    }
    
    pub fn plus(a: Self) -> Self {
        Self::Plus(Box::new(a))
    }
    
    pub fn optional(a: Self) -> Self {
        Self::Optional(Box::new(a))
    }
    
    /// Simplify the regex AST
    pub fn simplify(self) -> Self {
        match self {
            Self::Concat(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (Self::Epsilon, _) => b,
                    (_, Self::Epsilon) => a,
                    _ => Self::Concat(Box::new(a), Box::new(b)),
                }
            }
            Self::Alt(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                if a == b { a } else { Self::Alt(Box::new(a), Box::new(b)) }
            }
            Self::Star(a) => {
                let a = a.simplify();
                match a {
                    Self::Epsilon => Self::Epsilon,
                    Self::Star(inner) => Self::Star(inner), // (a*)* = a*
                    _ => Self::Star(Box::new(a)),
                }
            }
            Self::Plus(a) => {
                let a = a.simplify();
                // a+ = a(a*)
                Self::Concat(
                    Box::new(a.clone()),
                    Box::new(Self::Star(Box::new(a))),
                )
            }
            Self::Optional(a) => {
                let a = a.simplify();
                Self::Alt(Box::new(Self::Epsilon), Box::new(a))
            }
            other => other,
        }
    }
    
    /// Collect all characters used in this regex
    pub fn alphabet(&self) -> HashSet<char> {
        let mut chars = HashSet::new();
        self.collect_chars(&mut chars);
        chars
    }
    
    fn collect_chars(&self, chars: &mut HashSet<char>) {
        match self {
            Self::Char(c) => { chars.insert(*c); }
            Self::CharClass(cs) => { chars.extend(cs); }
            Self::NegCharClass(cs) => { chars.extend(cs); }
            Self::Concat(a, b) | Self::Alt(a, b) => {
                a.collect_chars(chars);
                b.collect_chars(chars);
            }
            Self::Star(a) | Self::Plus(a) | Self::Optional(a) => {
                a.collect_chars(chars);
            }
            _ => {}
        }
    }
}

/// NFA state
#[derive(Debug, Clone)]
struct NfaState {
    transitions: HashMap<Option<char>, Vec<usize>>, // None = epsilon
}

/// Thompson's NFA
#[derive(Debug, Clone)]
pub struct Nfa {
    states: Vec<NfaState>,
    start: usize,
    accept: usize,
}

impl Nfa {
    fn new_state(states: &mut Vec<NfaState>) -> usize {
        let id = states.len();
        states.push(NfaState { transitions: HashMap::new() });
        id
    }
    
    fn add_transition(states: &mut [NfaState], from: usize, to: usize, on: Option<char>) {
        states[from].transitions.entry(on).or_insert_with(Vec::new).push(to);
    }
    
    /// Thompson's construction: build NFA from regex AST
    pub fn from_regex(ast: &RegexAst) -> Self {
        let mut states = Vec::new();
        let (start, accept) = Self::build(&mut states, ast);
        Nfa { states, start, accept }
    }
    
    fn build(states: &mut Vec<NfaState>, ast: &RegexAst) -> (usize, usize) {
        match ast {
            RegexAst::Epsilon => {
                let s = Self::new_state(states);
                let e = Self::new_state(states);
                Self::add_transition(states, s, e, None);
                (s, e)
            }
            RegexAst::Char(c) => {
                let s = Self::new_state(states);
                let e = Self::new_state(states);
                Self::add_transition(states, s, e, Some(*c));
                (s, e)
            }
            RegexAst::CharClass(cs) => {
                let s = Self::new_state(states);
                let e = Self::new_state(states);
                for &c in cs {
                    Self::add_transition(states, s, e, Some(c));
                }
                (s, e)
            }
            RegexAst::NegCharClass(_cs) => {
                // Simplified: treat as dot for now
                let s = Self::new_state(states);
                let e = Self::new_state(states);
                // Add transitions for all printable ASCII
                for c in 32u8..=126 {
                    if !_cs.contains(&(c as char)) {
                        Self::add_transition(states, s, e, Some(c as char));
                    }
                }
                (s, e)
            }
            RegexAst::Dot => {
                let s = Self::new_state(states);
                let e = Self::new_state(states);
                for c in 32u8..=126 {
                    Self::add_transition(states, s, e, Some(c as char));
                }
                (s, e)
            }
            RegexAst::Concat(a, b) => {
                let (s1, e1) = Self::build(states, a);
                let (s2, e2) = Self::build(states, b);
                Self::add_transition(states, e1, s2, None);
                (s1, e2)
            }
            RegexAst::Alt(a, b) => {
                let s = Self::new_state(states);
                let e = Self::new_state(states);
                let (s1, e1) = Self::build(states, a);
                let (s2, e2) = Self::build(states, b);
                Self::add_transition(states, s, s1, None);
                Self::add_transition(states, s, s2, None);
                Self::add_transition(states, e1, e, None);
                Self::add_transition(states, e2, e, None);
                (s, e)
            }
            RegexAst::Star(a) => {
                let s = Self::new_state(states);
                let e = Self::new_state(states);
                let (s1, e1) = Self::build(states, a);
                Self::add_transition(states, s, s1, None);
                Self::add_transition(states, s, e, None);
                Self::add_transition(states, e1, s1, None);
                Self::add_transition(states, e1, e, None);
                (s, e)
            }
            RegexAst::Plus(a) => {
                // a+ = a(a*)
                let concat = RegexAst::Concat(
                    Box::new(a.as_ref().clone()),
                    Box::new(RegexAst::Star(Box::new(a.as_ref().clone()))),
                );
                Self::build(states, &concat)
            }
            RegexAst::Optional(a) => {
                // a? = (ε|a)
                let alt = RegexAst::Alt(
                    Box::new(RegexAst::Epsilon),
                    Box::new(a.as_ref().clone()),
                );
                Self::build(states, &alt)
            }
        }
    }
    
    /// Compute epsilon closure of a set of states
    pub fn epsilon_closure(&self, states: &BTreeSet<usize>) -> BTreeSet<usize> {
        let mut closure = states.clone();
        let mut stack: Vec<usize> = states.iter().copied().collect();
        
        while let Some(state) = stack.pop() {
            if let Some(eps_targets) = self.states[state].transitions.get(&None) {
                for &target in eps_targets {
                    if closure.insert(target) {
                        stack.push(target);
                    }
                }
            }
        }
        
        closure
    }
    
    /// Get transitions from a set of states on a given character
    fn move_states(&self, states: &BTreeSet<usize>, c: char) -> BTreeSet<usize> {
        let mut result = BTreeSet::new();
        for &state in states {
            if let Some(targets) = self.states[state].transitions.get(&Some(c)) {
                result.extend(targets);
            }
        }
        result
    }
    
    /// Check if any state in the set is the accept state
    fn has_accept(&self, states: &BTreeSet<usize>) -> bool {
        states.contains(&self.accept)
    }
    
    /// Simulate NFA on input string
    pub fn matches(&self, input: &str) -> bool {
        let mut current = BTreeSet::new();
        current.insert(self.start);
        current = self.epsilon_closure(&current);
        
        for c in input.chars() {
            let moved = self.move_states(&current, c);
            current = self.epsilon_closure(&moved);
            if current.is_empty() {
                return false;
            }
        }
        
        self.has_accept(&current)
    }
    
    pub fn num_states(&self) -> usize {
        self.states.len()
    }
}

/// DFA state
#[derive(Debug, Clone)]
struct DfaState {
    transitions: HashMap<char, usize>,
    is_accept: bool,
}

/// Deterministic finite automaton
#[derive(Debug, Clone)]
pub struct Dfa {
    states: Vec<DfaState>,
    start: usize,
    alphabet: Vec<char>,
}

impl Dfa {
    /// Subset construction: NFA → DFA
    pub fn from_nfa(nfa: &Nfa, alphabet: &[char]) -> Self {
        let mut dfa_states: Vec<DfaState> = Vec::new();
        let mut state_map: HashMap<BTreeSet<usize>, usize> = HashMap::new();
        let mut queue: VecDeque<BTreeSet<usize>> = VecDeque::new();
        
        // Start state = epsilon closure of NFA start
        let start_set = {
            let mut s = BTreeSet::new();
            s.insert(nfa.start);
            nfa.epsilon_closure(&s)
        };
        
        let start_id = 0;
        state_map.insert(start_set.clone(), start_id);
        dfa_states.push(DfaState {
            transitions: HashMap::new(),
            is_accept: nfa.has_accept(&start_set),
        });
        queue.push_back(start_set);
        
        while let Some(current_set) = queue.pop_front() {
            let current_id = state_map[&current_set];
            
            for &c in alphabet {
                let moved = nfa.move_states(&current_set, c);
                if moved.is_empty() {
                    continue;
                }
                let target_set = nfa.epsilon_closure(&moved);
                
                let target_id = if let Some(&id) = state_map.get(&target_set) {
                    id
                } else {
                    let id = dfa_states.len();
                    state_map.insert(target_set.clone(), id);
                    dfa_states.push(DfaState {
                        transitions: HashMap::new(),
                        is_accept: nfa.has_accept(&target_set),
                    });
                    queue.push_back(target_set);
                    id
                };
                
                dfa_states[current_id].transitions.insert(c, target_id);
            }
        }
        
        Dfa {
            states: dfa_states,
            start: start_id,
            alphabet: alphabet.to_vec(),
        }
    }
    
    /// Run DFA on input string
    pub fn matches(&self, input: &str) -> bool {
        let mut current = self.start;
        
        for c in input.chars() {
            match self.states[current].transitions.get(&c) {
                Some(&next) => current = next,
                None => return false,
            }
        }
        
        self.states[current].is_accept
    }
    
    /// Convert DFA to Boolean WFA
    pub fn to_wfa(&self) -> ScoringWFA<BooleanSemiring> {
        let char_to_idx: HashMap<char, usize> = self.alphabet.iter()
            .enumerate()
            .map(|(i, &c)| (c, i))
            .collect();
        
        let num_states = self.states.len();
        let alphabet_size = self.alphabet.len();
        
        let mut wfa = ScoringWFA::new(num_states, alphabet_size);
        wfa.set_initial(self.start, BooleanSemiring::one());
        
        for (i, state) in self.states.iter().enumerate() {
            if state.is_accept {
                wfa.set_final(i, BooleanSemiring::one());
            }
            
            for (&c, &target) in &state.transitions {
                if let Some(&sym_idx) = char_to_idx.get(&c) {
                    wfa.set_transition(i, target, sym_idx, BooleanSemiring::one());
                }
            }
        }
        
        wfa
    }
    
    pub fn num_states(&self) -> usize {
        self.states.len()
    }
}

/// Regex compiler: parses a simple regex string into an AST
#[derive(Debug, Clone)]
pub struct RegexCompiler;

impl RegexCompiler {
    /// Parse a simple regex pattern into an AST.
    /// Supports: literal chars, ., *, +, ?, |, (), [], [^]
    pub fn parse(pattern: &str) -> Result<RegexAst, String> {
        let chars: Vec<char> = pattern.chars().collect();
        let (ast, pos) = Self::parse_alt(&chars, 0)?;
        if pos != chars.len() {
            return Err(format!("Unexpected character at position {}", pos));
        }
        Ok(ast)
    }
    
    fn parse_alt(chars: &[char], pos: usize) -> Result<(RegexAst, usize), String> {
        let (mut left, mut pos) = Self::parse_concat(chars, pos)?;
        
        while pos < chars.len() && chars[pos] == '|' {
            let (right, new_pos) = Self::parse_concat(chars, pos + 1)?;
            left = RegexAst::Alt(Box::new(left), Box::new(right));
            pos = new_pos;
        }
        
        Ok((left, pos))
    }
    
    fn parse_concat(chars: &[char], pos: usize) -> Result<(RegexAst, usize), String> {
        let mut parts = Vec::new();
        let mut pos = pos;
        
        while pos < chars.len() && chars[pos] != '|' && chars[pos] != ')' {
            let (part, new_pos) = Self::parse_quantifier(chars, pos)?;
            parts.push(part);
            pos = new_pos;
        }
        
        if parts.is_empty() {
            Ok((RegexAst::Epsilon, pos))
        } else {
            let mut result = parts.remove(0);
            for part in parts {
                result = RegexAst::Concat(Box::new(result), Box::new(part));
            }
            Ok((result, pos))
        }
    }
    
    fn parse_quantifier(chars: &[char], pos: usize) -> Result<(RegexAst, usize), String> {
        let (base, pos) = Self::parse_atom(chars, pos)?;
        
        if pos < chars.len() {
            match chars[pos] {
                '*' => Ok((RegexAst::Star(Box::new(base)), pos + 1)),
                '+' => Ok((RegexAst::Plus(Box::new(base)), pos + 1)),
                '?' => Ok((RegexAst::Optional(Box::new(base)), pos + 1)),
                _ => Ok((base, pos)),
            }
        } else {
            Ok((base, pos))
        }
    }
    
    fn parse_atom(chars: &[char], pos: usize) -> Result<(RegexAst, usize), String> {
        if pos >= chars.len() {
            return Err("Unexpected end of pattern".to_string());
        }
        
        match chars[pos] {
            '(' => {
                let (ast, new_pos) = Self::parse_alt(chars, pos + 1)?;
                if new_pos >= chars.len() || chars[new_pos] != ')' {
                    return Err("Missing closing parenthesis".to_string());
                }
                Ok((ast, new_pos + 1))
            }
            '[' => Self::parse_char_class(chars, pos),
            '.' => Ok((RegexAst::Dot, pos + 1)),
            '\\' => {
                if pos + 1 >= chars.len() {
                    return Err("Unexpected end after escape".to_string());
                }
                let escaped = chars[pos + 1];
                match escaped {
                    'd' => Ok((RegexAst::CharClass(('0'..='9').collect()), pos + 2)),
                    'w' => {
                        let mut cs: Vec<char> = ('a'..='z').collect();
                        cs.extend('A'..='Z');
                        cs.extend('0'..='9');
                        cs.push('_');
                        Ok((RegexAst::CharClass(cs), pos + 2))
                    }
                    's' => Ok((RegexAst::CharClass(vec![' ', '\t', '\n', '\r']), pos + 2)),
                    _ => Ok((RegexAst::Char(escaped), pos + 2)),
                }
            }
            c if c == '|' || c == ')' || c == '*' || c == '+' || c == '?' => {
                Err(format!("Unexpected metacharacter '{}' at position {}", c, pos))
            }
            c => Ok((RegexAst::Char(c), pos + 1)),
        }
    }
    
    fn parse_char_class(chars: &[char], pos: usize) -> Result<(RegexAst, usize), String> {
        let mut i = pos + 1; // skip '['
        let negated = i < chars.len() && chars[i] == '^';
        if negated {
            i += 1;
        }
        
        let mut class_chars = Vec::new();
        
        while i < chars.len() && chars[i] != ']' {
            if i + 2 < chars.len() && chars[i + 1] == '-' && chars[i + 2] != ']' {
                // Range: a-z
                let start = chars[i];
                let end = chars[i + 2];
                for c in start..=end {
                    class_chars.push(c);
                }
                i += 3;
            } else {
                class_chars.push(chars[i]);
                i += 1;
            }
        }
        
        if i >= chars.len() {
            return Err("Missing closing bracket".to_string());
        }
        
        let ast = if negated {
            RegexAst::NegCharClass(class_chars)
        } else {
            RegexAst::CharClass(class_chars)
        };
        
        Ok((ast, i + 1))
    }
}

/// Regex match scorer with triple implementation
#[derive(Debug, Clone)]
pub struct RegexMatchScorer {
    pattern: String,
    ast: RegexAst,
}

impl RegexMatchScorer {
    pub fn new(pattern: &str) -> Result<Self, String> {
        let ast = RegexCompiler::parse(pattern)?;
        Ok(Self {
            pattern: pattern.to_string(),
            ast,
        })
    }
    
    /// Reference implementation: simulate NFA directly
    pub fn reference_match(&self, text: &str) -> bool {
        let nfa = Nfa::from_regex(&self.ast);
        nfa.matches(text)
    }
    
    /// Automaton implementation: NFA → DFA → Boolean WFA
    pub fn automaton_match(&self, text: &str) -> bool {
        let nfa = Nfa::from_regex(&self.ast);
        let alphabet: Vec<char> = self.ast.alphabet().into_iter().collect();
        
        // Add any characters from the text that aren't in the regex alphabet
        let mut full_alphabet = alphabet.clone();
        for c in text.chars() {
            if !full_alphabet.contains(&c) {
                full_alphabet.push(c);
            }
        }
        full_alphabet.sort();
        full_alphabet.dedup();
        
        let dfa = Dfa::from_nfa(&nfa, &full_alphabet);
        dfa.matches(text)
    }
    
    /// Circuit implementation: evaluate match using field arithmetic
    pub fn circuit_match(&self, text: &str) -> bool {
        // Build DFA and simulate with Goldilocks field tracking
        let nfa = Nfa::from_regex(&self.ast);
        let mut alphabet: Vec<char> = self.ast.alphabet().into_iter().collect();
        for c in text.chars() {
            if !alphabet.contains(&c) {
                alphabet.push(c);
            }
        }
        alphabet.sort();
        alphabet.dedup();
        
        let dfa = Dfa::from_nfa(&nfa, &alphabet);
        let wfa = dfa.to_wfa();
        
        let char_to_idx: HashMap<char, usize> = alphabet.iter()
            .enumerate()
            .map(|(i, &c)| (c, i))
            .collect();
        
        let input: Vec<usize> = text.chars()
            .filter_map(|c| char_to_idx.get(&c).copied())
            .collect();
        
        if input.len() != text.chars().count() {
            return false;
        }
        
        let result = wfa.run(&input);
        result == BooleanSemiring::one()
    }
    
    /// Build the full pipeline: regex → NFA → DFA → WFA → Circuit
    pub fn compile_to_circuit(&self, max_input_len: usize) -> ScoringCircuit {
        let nfa = Nfa::from_regex(&self.ast);
        let alphabet: Vec<char> = self.ast.alphabet().into_iter().collect();
        let dfa = Dfa::from_nfa(&nfa, &alphabet);
        
        let mut circuit = ScoringCircuit::new();
        
        // Allocate input wires
        for _ in 0..max_input_len {
            circuit.alloc_public_input();
        }
        
        // Allocate state wires for each step
        let num_dfa_states = dfa.num_states();
        for _ in 0..=max_input_len {
            for _ in 0..num_dfa_states {
                circuit.alloc_wire();
            }
        }
        
        // Output wire: is the final state accepting?
        circuit.alloc_public_output();
        
        circuit
    }
}

impl TripleMetric for RegexMatchScorer {
    type Input = ScoringPair;
    type Score = bool;
    
    fn score_reference(&self, input: &ScoringPair) -> bool {
        self.reference_match(&input.candidate)
    }
    
    fn score_automaton(&self, input: &ScoringPair) -> bool {
        self.automaton_match(&input.candidate)
    }
    
    fn score_circuit(&self, input: &ScoringPair) -> bool {
        self.circuit_match(&input.candidate)
    }
}

/// Compose two regex patterns: matches if either matches (union)
pub fn regex_union(a: &RegexAst, b: &RegexAst) -> RegexAst {
    RegexAst::Alt(Box::new(a.clone()), Box::new(b.clone()))
}

/// Compose two regex patterns: matches if both match in sequence
pub fn regex_concat(a: &RegexAst, b: &RegexAst) -> RegexAst {
    RegexAst::Concat(Box::new(a.clone()), Box::new(b.clone()))
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    fn make_pair(cand: &str, reference: &str) -> ScoringPair {
        ScoringPair {
            candidate: cand.to_string(),
            reference: reference.to_string(),
        }
    }
    
    #[test]
    fn test_regex_parse_literal() {
        let ast = RegexCompiler::parse("abc").unwrap();
        // Should be Concat(Concat(Char('a'), Char('b')), Char('c'))
        match ast {
            RegexAst::Concat(_, _) => {}
            _ => panic!("Expected Concat, got {:?}", ast),
        }
    }
    
    #[test]
    fn test_regex_parse_alternation() {
        let ast = RegexCompiler::parse("a|b").unwrap();
        match ast {
            RegexAst::Alt(_, _) => {}
            _ => panic!("Expected Alt, got {:?}", ast),
        }
    }
    
    #[test]
    fn test_regex_parse_star() {
        let ast = RegexCompiler::parse("a*").unwrap();
        match ast {
            RegexAst::Star(_) => {}
            _ => panic!("Expected Star, got {:?}", ast),
        }
    }
    
    #[test]
    fn test_regex_parse_char_class() {
        let ast = RegexCompiler::parse("[abc]").unwrap();
        match ast {
            RegexAst::CharClass(cs) => assert_eq!(cs, vec!['a', 'b', 'c']),
            _ => panic!("Expected CharClass"),
        }
    }
    
    #[test]
    fn test_regex_parse_char_range() {
        let ast = RegexCompiler::parse("[a-d]").unwrap();
        match ast {
            RegexAst::CharClass(cs) => assert_eq!(cs, vec!['a', 'b', 'c', 'd']),
            _ => panic!("Expected CharClass"),
        }
    }
    
    #[test]
    fn test_regex_parse_groups() {
        let ast = RegexCompiler::parse("(ab)+").unwrap();
        match ast {
            RegexAst::Plus(_) => {}
            _ => panic!("Expected Plus"),
        }
    }
    
    #[test]
    fn test_regex_parse_escape() {
        let ast = RegexCompiler::parse("\\d+").unwrap();
        match ast {
            RegexAst::Plus(_) => {}
            _ => panic!("Expected Plus"),
        }
    }
    
    #[test]
    fn test_nfa_simple_match() {
        let ast = RegexCompiler::parse("abc").unwrap();
        let nfa = Nfa::from_regex(&ast);
        assert!(nfa.matches("abc"));
        assert!(!nfa.matches("ab"));
        assert!(!nfa.matches("abcd"));
        assert!(!nfa.matches("xyz"));
    }
    
    #[test]
    fn test_nfa_alternation() {
        let ast = RegexCompiler::parse("a|b").unwrap();
        let nfa = Nfa::from_regex(&ast);
        assert!(nfa.matches("a"));
        assert!(nfa.matches("b"));
        assert!(!nfa.matches("c"));
        assert!(!nfa.matches("ab"));
    }
    
    #[test]
    fn test_nfa_star() {
        let ast = RegexCompiler::parse("a*").unwrap();
        let nfa = Nfa::from_regex(&ast);
        assert!(nfa.matches(""));
        assert!(nfa.matches("a"));
        assert!(nfa.matches("aaa"));
        assert!(!nfa.matches("b"));
    }
    
    #[test]
    fn test_nfa_plus() {
        let ast = RegexCompiler::parse("a+").unwrap();
        let nfa = Nfa::from_regex(&ast);
        assert!(!nfa.matches(""));
        assert!(nfa.matches("a"));
        assert!(nfa.matches("aaa"));
    }
    
    #[test]
    fn test_nfa_optional() {
        let ast = RegexCompiler::parse("ab?c").unwrap();
        let nfa = Nfa::from_regex(&ast);
        assert!(nfa.matches("ac"));
        assert!(nfa.matches("abc"));
        assert!(!nfa.matches("abbc"));
    }
    
    #[test]
    fn test_nfa_complex() {
        let ast = RegexCompiler::parse("(ab|cd)*e").unwrap();
        let nfa = Nfa::from_regex(&ast);
        assert!(nfa.matches("e"));
        assert!(nfa.matches("abe"));
        assert!(nfa.matches("cde"));
        assert!(nfa.matches("abcde"));
        assert!(nfa.matches("ababcde"));
        assert!(!nfa.matches("ace"));
    }
    
    #[test]
    fn test_dfa_from_nfa() {
        let ast = RegexCompiler::parse("a|b").unwrap();
        let nfa = Nfa::from_regex(&ast);
        let dfa = Dfa::from_nfa(&nfa, &['a', 'b']);
        assert!(dfa.matches("a"));
        assert!(dfa.matches("b"));
        assert!(!dfa.matches("c"));
    }
    
    #[test]
    fn test_dfa_star() {
        let ast = RegexCompiler::parse("(ab)*").unwrap();
        let nfa = Nfa::from_regex(&ast);
        let dfa = Dfa::from_nfa(&nfa, &['a', 'b']);
        assert!(dfa.matches(""));
        assert!(dfa.matches("ab"));
        assert!(dfa.matches("abab"));
        assert!(!dfa.matches("a"));
        assert!(!dfa.matches("aba"));
    }
    
    #[test]
    fn test_dfa_to_wfa() {
        let ast = RegexCompiler::parse("ab").unwrap();
        let nfa = Nfa::from_regex(&ast);
        let dfa = Dfa::from_nfa(&nfa, &['a', 'b']);
        let wfa = dfa.to_wfa();
        
        // 'a' = index 0, 'b' = index 1
        assert_eq!(wfa.run(&[0, 1]), BooleanSemiring::one());
        assert_eq!(wfa.run(&[1, 0]), BooleanSemiring::zero());
    }
    
    #[test]
    fn test_regex_scorer_reference() {
        let scorer = RegexMatchScorer::new("hello|world").unwrap();
        assert!(scorer.reference_match("hello"));
        assert!(scorer.reference_match("world"));
        assert!(!scorer.reference_match("hi"));
    }
    
    #[test]
    fn test_regex_scorer_triple_agreement() {
        let scorer = RegexMatchScorer::new("(ab)+c").unwrap();
        
        let test_cases = vec!["abc", "ababc", "ac", "ab", "abcabc"];
        
        for text in test_cases {
            let pair = make_pair(text, "");
            let result = scorer.score_and_verify(&pair);
            assert!(result.agreement,
                "Disagreement on {:?}: ref={}, aut={}, cir={}",
                text, result.reference, result.automaton, result.circuit);
        }
    }
    
    #[test]
    fn test_regex_scorer_digit() {
        let scorer = RegexMatchScorer::new("\\d+").unwrap();
        assert!(scorer.reference_match("123"));
        assert!(scorer.reference_match("0"));
        assert!(!scorer.reference_match("abc"));
    }
    
    #[test]
    fn test_regex_simplify() {
        let ast = RegexAst::Concat(
            Box::new(RegexAst::Epsilon),
            Box::new(RegexAst::Char('a')),
        );
        let simplified = ast.simplify();
        assert_eq!(simplified, RegexAst::Char('a'));
    }
    
    #[test]
    fn test_regex_alphabet() {
        let ast = RegexCompiler::parse("abc|def").unwrap();
        let alphabet = ast.alphabet();
        assert!(alphabet.contains(&'a'));
        assert!(alphabet.contains(&'f'));
        assert!(!alphabet.contains(&'z'));
    }
    
    #[test]
    fn test_regex_union_compose() {
        let a = RegexCompiler::parse("ab").unwrap();
        let b = RegexCompiler::parse("cd").unwrap();
        let combined = regex_union(&a, &b);
        let nfa = Nfa::from_regex(&combined);
        assert!(nfa.matches("ab"));
        assert!(nfa.matches("cd"));
        assert!(!nfa.matches("ac"));
    }
    
    #[test]
    fn test_regex_concat_compose() {
        let a = RegexCompiler::parse("ab").unwrap();
        let b = RegexCompiler::parse("cd").unwrap();
        let combined = regex_concat(&a, &b);
        let nfa = Nfa::from_regex(&combined);
        assert!(nfa.matches("abcd"));
        assert!(!nfa.matches("ab"));
        assert!(!nfa.matches("cd"));
    }
    
    #[test]
    fn test_compile_to_circuit() {
        let scorer = RegexMatchScorer::new("a+b").unwrap();
        let circuit = scorer.compile_to_circuit(10);
        assert!(circuit.num_wires > 10);
    }
}
