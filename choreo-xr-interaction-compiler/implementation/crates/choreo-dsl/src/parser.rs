//! Recursive-descent parser for the Choreo DSL.
//!
//! Transforms a token stream into an [`ast::Program`] AST.

use crate::ast::Program;
use crate::token::Token;

/// Recursive-descent parser for the Choreo choreography language.
pub struct Parser<'src> {
    source: &'src str,
    tokens: Vec<Token>,
    pos: usize,
}

impl<'src> Parser<'src> {
    /// Create a new parser for the given source text.
    pub fn new(source: &'src str) -> Self {
        Self {
            source,
            tokens: Vec::new(),
            pos: 0,
        }
    }

    /// Parse the source into a [`Program`] AST.
    pub fn parse(&mut self) -> Result<Program, Vec<String>> {
        todo!("implement recursive-descent parsing")
    }
}
