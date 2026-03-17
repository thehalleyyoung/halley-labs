//! Lightweight template engine with variable substitution, conditionals,
//! and iteration.  Used by every code-generation backend to separate
//! target-language boilerplate from automaton-specific logic.

use std::collections::HashMap;
use std::fmt;

use crate::{CodegenError, CodegenResult};

// ---------------------------------------------------------------------------
// Template AST
// ---------------------------------------------------------------------------

/// A parsed segment of a template.
#[derive(Debug, Clone, PartialEq)]
enum TemplateNode {
    /// Literal text – emitted as-is.
    Text(String),
    /// `{{name}}` – replaced by the value of *name* in the context.
    Variable(String),
    /// `{{#if cond}} body {{/if}}` – emitted only when *cond* is truthy.
    Conditional {
        condition: String,
        body: Vec<TemplateNode>,
        else_body: Vec<TemplateNode>,
    },
    /// `{{#each items}} body {{/each}}` – repeated for every element.
    Loop {
        variable: String,
        body: Vec<TemplateNode>,
    },
}

// ---------------------------------------------------------------------------
// Template value (the "model" side of the template)
// ---------------------------------------------------------------------------

/// A value that can be inserted into a template context.
#[derive(Debug, Clone)]
pub enum TemplateValue {
    Str(String),
    Bool(bool),
    Int(i64),
    Float(f64),
    List(Vec<HashMap<String, TemplateValue>>),
}

impl TemplateValue {
    /// Return `true` when the value should be treated as "truthy" by
    /// `{{#if}}` blocks.
    pub fn is_truthy(&self) -> bool {
        match self {
            TemplateValue::Bool(b) => *b,
            TemplateValue::Str(s) => !s.is_empty(),
            TemplateValue::Int(n) => *n != 0,
            TemplateValue::Float(f) => *f != 0.0,
            TemplateValue::List(l) => !l.is_empty(),
        }
    }
}

impl fmt::Display for TemplateValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TemplateValue::Str(s) => write!(f, "{}", s),
            TemplateValue::Bool(b) => write!(f, "{}", b),
            TemplateValue::Int(n) => write!(f, "{}", n),
            TemplateValue::Float(v) => write!(f, "{}", v),
            TemplateValue::List(_) => write!(f, "[list]"),
        }
    }
}

// ---------------------------------------------------------------------------
// TemplateContext
// ---------------------------------------------------------------------------

/// Holds the variables available during template rendering.
#[derive(Debug, Clone, Default)]
pub struct TemplateContext {
    values: HashMap<String, TemplateValue>,
}

impl TemplateContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, key: impl Into<String>, value: TemplateValue) {
        self.values.insert(key.into(), value);
    }

    pub fn set_str(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.values
            .insert(key.into(), TemplateValue::Str(value.into()));
    }

    pub fn set_bool(&mut self, key: impl Into<String>, value: bool) {
        self.values.insert(key.into(), TemplateValue::Bool(value));
    }

    pub fn set_int(&mut self, key: impl Into<String>, value: i64) {
        self.values.insert(key.into(), TemplateValue::Int(value));
    }

    pub fn set_list(
        &mut self,
        key: impl Into<String>,
        items: Vec<HashMap<String, TemplateValue>>,
    ) {
        self.values.insert(key.into(), TemplateValue::List(items));
    }

    pub fn get(&self, key: &str) -> Option<&TemplateValue> {
        self.values.get(key)
    }

    fn merged_with(&self, extra: &HashMap<String, TemplateValue>) -> Self {
        let mut ctx = self.clone();
        for (k, v) in extra {
            ctx.values.insert(k.clone(), v.clone());
        }
        ctx
    }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Parse a template string into an AST of [`TemplateNode`]s.
fn parse_template(input: &str) -> CodegenResult<Vec<TemplateNode>> {
    let mut nodes = Vec::new();
    let mut rest = input;

    while !rest.is_empty() {
        if let Some(pos) = rest.find("{{") {
            if pos > 0 {
                nodes.push(TemplateNode::Text(rest[..pos].to_string()));
            }
            let after_open = &rest[pos + 2..];
            let close = after_open
                .find("}}")
                .ok_or_else(|| CodegenError::Template("unclosed {{".into()))?;
            let tag = after_open[..close].trim();
            rest = &after_open[close + 2..];

            if let Some(stripped) = tag.strip_prefix("#if ") {
                let cond = stripped.trim().to_string();
                let (body, else_body, remaining) = parse_block(rest, "if")?;
                rest = remaining;
                nodes.push(TemplateNode::Conditional {
                    condition: cond,
                    body,
                    else_body,
                });
            } else if let Some(stripped) = tag.strip_prefix("#each ") {
                let var = stripped.trim().to_string();
                let (body, _else_body, remaining) = parse_block(rest, "each")?;
                rest = remaining;
                nodes.push(TemplateNode::Loop {
                    variable: var,
                    body,
                });
            } else if tag.starts_with('/') {
                return Err(CodegenError::Template(format!(
                    "unexpected closing tag: {{{{{}}}}}",
                    tag
                )));
            } else {
                nodes.push(TemplateNode::Variable(tag.to_string()));
            }
        } else {
            nodes.push(TemplateNode::Text(rest.to_string()));
            break;
        }
    }
    Ok(nodes)
}

/// Parse everything until `{{/kind}}`, returning (body, else_body, remaining).
fn parse_block<'a>(
    input: &'a str,
    kind: &str,
) -> CodegenResult<(Vec<TemplateNode>, Vec<TemplateNode>, &'a str)> {
    let close_tag = format!("{{{{/{}}}}}", kind);
    let else_tag = "{{else}}";
    let mut depth: usize = 0;
    let mut pos: usize = 0;
    let mut else_pos: Option<usize> = None;

    let open_prefix = format!("{{{{#{}", kind);

    while pos < input.len() {
        if input[pos..].starts_with(&open_prefix) {
            depth += 1;
            pos += open_prefix.len();
        } else if input[pos..].starts_with(&close_tag) {
            if depth == 0 {
                let before_close = &input[..pos];
                let after = &input[pos + close_tag.len()..];
                return if let Some(ep) = else_pos {
                    let body_src = &input[..ep];
                    let else_src = &input[ep + else_tag.len()..pos];
                    Ok((
                        parse_template(body_src)?,
                        parse_template(else_src)?,
                        after,
                    ))
                } else {
                    Ok((parse_template(before_close)?, Vec::new(), after))
                };
            }
            depth -= 1;
            pos += close_tag.len();
        } else if depth == 0 && input[pos..].starts_with(else_tag) {
            else_pos = Some(pos);
            pos += else_tag.len();
        } else {
            pos += 1;
        }
    }
    Err(CodegenError::Template(format!(
        "unclosed {{{{#{kind}}}}} block"
    )))
}

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------

/// Render a parsed AST against a context.
fn render_nodes(nodes: &[TemplateNode], ctx: &TemplateContext) -> CodegenResult<String> {
    let mut out = String::new();
    for node in nodes {
        match node {
            TemplateNode::Text(t) => out.push_str(t),
            TemplateNode::Variable(name) => {
                if let Some(val) = ctx.get(name) {
                    out.push_str(&val.to_string());
                }
                // Missing variables are silently elided.
            }
            TemplateNode::Conditional {
                condition,
                body,
                else_body,
            } => {
                let truthy = ctx
                    .get(condition)
                    .map(|v| v.is_truthy())
                    .unwrap_or(false);
                if truthy {
                    out.push_str(&render_nodes(body, ctx)?);
                } else {
                    out.push_str(&render_nodes(else_body, ctx)?);
                }
            }
            TemplateNode::Loop { variable, body } => {
                if let Some(TemplateValue::List(items)) = ctx.get(variable) {
                    for (idx, item) in items.iter().enumerate() {
                        let mut child = item.clone();
                        child.insert(
                            "@index".to_string(),
                            TemplateValue::Int(idx as i64),
                        );
                        child.insert(
                            "@first".to_string(),
                            TemplateValue::Bool(idx == 0),
                        );
                        child.insert(
                            "@last".to_string(),
                            TemplateValue::Bool(idx == items.len() - 1),
                        );
                        let inner_ctx = ctx.merged_with(&child);
                        out.push_str(&render_nodes(body, &inner_ctx)?);
                    }
                }
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// A compiled template ready to be rendered against a [`TemplateContext`].
#[derive(Debug, Clone)]
pub struct Template {
    source: String,
    nodes: Vec<TemplateNode>,
}

impl Template {
    /// Parse a template string.  Returns an error when the template is
    /// syntactically invalid (e.g. unclosed blocks).
    pub fn parse(source: &str) -> CodegenResult<Self> {
        let nodes = parse_template(source)?;
        Ok(Self {
            source: source.to_string(),
            nodes,
        })
    }

    /// Render the template with the given context.
    pub fn render(&self, ctx: &TemplateContext) -> CodegenResult<String> {
        render_nodes(&self.nodes, ctx)
    }

    /// Return the original source text.
    pub fn source(&self) -> &str {
        &self.source
    }
}

// ---------------------------------------------------------------------------
// TemplateEngine – registry of named templates
// ---------------------------------------------------------------------------

/// Registry that holds named templates and provides convenience rendering.
#[derive(Debug, Default)]
pub struct TemplateEngine {
    templates: HashMap<String, Template>,
}

impl TemplateEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a template under the given name.
    pub fn register(&mut self, name: impl Into<String>, source: &str) -> CodegenResult<()> {
        let tmpl = Template::parse(source)?;
        self.templates.insert(name.into(), tmpl);
        Ok(())
    }

    /// Render a previously registered template.
    pub fn render(&self, name: &str, ctx: &TemplateContext) -> CodegenResult<String> {
        let tmpl = self
            .templates
            .get(name)
            .ok_or_else(|| CodegenError::Template(format!("template not found: {}", name)))?;
        tmpl.render(ctx)
    }

    /// Check whether a template with the given name exists.
    pub fn has(&self, name: &str) -> bool {
        self.templates.contains_key(name)
    }

    /// Number of registered templates.
    pub fn len(&self) -> usize {
        self.templates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.templates.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Built-in templates for each backend
// ---------------------------------------------------------------------------

/// Built-in Rust backend template.
pub const RUST_STATE_MACHINE_TEMPLATE: &str = r#"/// Auto-generated state machine: {{name}}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum {{name}}State {
{{#each states}}    {{state_name}},
{{/each}}}

impl {{name}}State {
    pub fn is_accepting(&self) -> bool {
        matches!(self, {{#each accepting}}{{#if @first}}{{else}} | {{/if}}Self::{{state_name}}{{/each}})
    }
}

pub struct {{name}}Machine {
    state: {{name}}State,
}

impl {{name}}Machine {
    pub fn new() -> Self {
        Self { state: {{name}}State::{{initial_state}} }
    }

    pub fn current_state(&self) -> {{name}}State {
        self.state
    }

    pub fn step(&mut self, event: &str) -> bool {
        let next = match (self.state, event) {
{{#each transitions}}            ({{name}}State::{{source_name}}, {{event_str}}) => {
{{#if has_guard}}                if {{guard_expr}} { Some({{name}}State::{{target_name}}) } else { None }
{{else}}                Some({{name}}State::{{target_name}})
{{/if}}            }
{{/each}}            _ => None,
        };
        if let Some(s) = next {
            self.state = s;
            true
        } else {
            false
        }
    }

    pub fn is_accepting(&self) -> bool {
        self.state.is_accepting()
    }
}
"#;

/// Built-in C# backend template.
pub const CSHARP_STATE_MACHINE_TEMPLATE: &str = r#"// Auto-generated state machine: {{name}}
using UnityEngine;
using Microsoft.MixedReality.Toolkit;

public enum {{name}}State
{
{{#each states}}    {{state_name}},
{{/each}}}

public class {{name}}StateMachine : MonoBehaviour
{
    public {{name}}State CurrentState { get; private set; }

    void Start()
    {
        CurrentState = {{name}}State.{{initial_state}};
        OnEnterState(CurrentState);
    }

    public bool ProcessEvent(string eventName)
    {
        var next = EvaluateTransition(eventName);
        if (next.HasValue)
        {
            OnExitState(CurrentState);
            CurrentState = next.Value;
            OnEnterState(CurrentState);
            return true;
        }
        return false;
    }

    private {{name}}State? EvaluateTransition(string eventName)
    {
        switch (CurrentState)
        {
{{#each states}}            case {{name}}State.{{state_name}}:
{{#each outgoing}}                if (eventName == {{event_str}}{{#if has_guard}} && {{guard_expr}}{{/if}})
                    return {{name}}State.{{target_name}};
{{/each}}                break;
{{/each}}        }
        return null;
    }

    private void OnEnterState({{name}}State state) { }
    private void OnExitState({{name}}State state) { }
}
"#;

/// Built-in TypeScript backend template.
pub const TYPESCRIPT_STATE_MACHINE_TEMPLATE: &str = r#"// Auto-generated state machine: {{name}}

export enum {{name}}State {
{{#each states}}  {{state_name}},
{{/each}}}

export class {{name}}Machine {
  private state: {{name}}State;

  constructor() {
    this.state = {{name}}State.{{initial_state}};
  }

  get currentState(): {{name}}State {
    return this.state;
  }

  async step(event: string): Promise<boolean> {
    const next = this.evaluate(event);
    if (next !== undefined) {
      this.state = next;
      return true;
    }
    return false;
  }

  private evaluate(event: string): {{name}}State | undefined {
    switch (this.state) {
{{#each states}}      case {{name}}State.{{state_name}}:
{{#each outgoing}}        if (event === {{event_str}}{{#if has_guard}} && {{guard_expr}}{{/if}})
          return {{name}}State.{{target_name}};
{{/each}}        break;
{{/each}}    }
    return undefined;
  }
}
"#;

// ---------------------------------------------------------------------------
// CodeBuffer helper
// ---------------------------------------------------------------------------

/// Indentation-aware string builder for generating source code.
#[derive(Debug, Clone)]
pub struct CodeBuffer {
    buf: String,
    indent: usize,
    indent_str: String,
}

impl CodeBuffer {
    /// Create a new empty buffer with the given indentation string
    /// (typically two or four spaces).
    pub fn new(indent_str: &str) -> Self {
        Self {
            buf: String::new(),
            indent: 0,
            indent_str: indent_str.to_string(),
        }
    }

    /// Append a line at the current indentation level.
    pub fn line(&mut self, text: &str) {
        for _ in 0..self.indent {
            self.buf.push_str(&self.indent_str);
        }
        self.buf.push_str(text);
        self.buf.push('\n');
    }

    /// Append an empty line.
    pub fn blank(&mut self) {
        self.buf.push('\n');
    }

    /// Append raw text without indentation or newline.
    pub fn raw(&mut self, text: &str) {
        self.buf.push_str(text);
    }

    /// Increase indentation by one level.
    pub fn indent(&mut self) {
        self.indent += 1;
    }

    /// Decrease indentation by one level (clamped to zero).
    pub fn dedent(&mut self) {
        self.indent = self.indent.saturating_sub(1);
    }

    /// Convenience: write `{` and indent.
    pub fn open_brace(&mut self, prefix: &str) {
        self.line(&format!("{} {{", prefix));
        self.indent();
    }

    /// Convenience: dedent and write `}`.
    pub fn close_brace(&mut self) {
        self.dedent();
        self.line("}");
    }

    /// Return the assembled source text.
    pub fn finish(self) -> String {
        self.buf
    }

    /// Current length in bytes.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
}

impl fmt::Display for CodeBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.buf)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Template parsing & rendering tests ---------------------------------

    #[test]
    fn test_plain_text() {
        let t = Template::parse("hello world").unwrap();
        let ctx = TemplateContext::new();
        assert_eq!(t.render(&ctx).unwrap(), "hello world");
    }

    #[test]
    fn test_variable_substitution() {
        let t = Template::parse("Hello, {{name}}!").unwrap();
        let mut ctx = TemplateContext::new();
        ctx.set_str("name", "Choreo");
        assert_eq!(t.render(&ctx).unwrap(), "Hello, Choreo!");
    }

    #[test]
    fn test_missing_variable_is_empty() {
        let t = Template::parse("x={{missing}}y").unwrap();
        let ctx = TemplateContext::new();
        assert_eq!(t.render(&ctx).unwrap(), "x=y");
    }

    #[test]
    fn test_conditional_true() {
        let t = Template::parse("{{#if show}}visible{{/if}}").unwrap();
        let mut ctx = TemplateContext::new();
        ctx.set_bool("show", true);
        assert_eq!(t.render(&ctx).unwrap(), "visible");
    }

    #[test]
    fn test_conditional_false() {
        let t = Template::parse("{{#if show}}visible{{/if}}").unwrap();
        let mut ctx = TemplateContext::new();
        ctx.set_bool("show", false);
        assert_eq!(t.render(&ctx).unwrap(), "");
    }

    #[test]
    fn test_conditional_else() {
        let t =
            Template::parse("{{#if show}}yes{{else}}no{{/if}}").unwrap();
        let mut ctx = TemplateContext::new();
        ctx.set_bool("show", false);
        assert_eq!(t.render(&ctx).unwrap(), "no");
    }

    #[test]
    fn test_each_loop() {
        let t = Template::parse("{{#each items}}[{{val}}]{{/each}}").unwrap();
        let mut ctx = TemplateContext::new();
        let items: Vec<HashMap<String, TemplateValue>> = vec![
            [("val".into(), TemplateValue::Str("a".into()))].into(),
            [("val".into(), TemplateValue::Str("b".into()))].into(),
        ];
        ctx.set_list("items", items);
        assert_eq!(t.render(&ctx).unwrap(), "[a][b]");
    }

    #[test]
    fn test_each_loop_index_metadata() {
        let t =
            Template::parse("{{#each items}}{{@index}}:{{val}} {{/each}}").unwrap();
        let mut ctx = TemplateContext::new();
        let items: Vec<HashMap<String, TemplateValue>> = vec![
            [("val".into(), TemplateValue::Str("x".into()))].into(),
            [("val".into(), TemplateValue::Str("y".into()))].into(),
        ];
        ctx.set_list("items", items);
        assert_eq!(t.render(&ctx).unwrap(), "0:x 1:y ");
    }

    #[test]
    fn test_nested_conditional_in_loop() {
        let src = "{{#each items}}{{#if active}}*{{name}}*{{/if}}{{/each}}";
        let t = Template::parse(src).unwrap();
        let mut ctx = TemplateContext::new();
        let items = vec![
            [
                ("active".into(), TemplateValue::Bool(true)),
                ("name".into(), TemplateValue::Str("A".into())),
            ]
            .into(),
            [
                ("active".into(), TemplateValue::Bool(false)),
                ("name".into(), TemplateValue::Str("B".into())),
            ]
            .into(),
        ];
        ctx.set_list("items", items);
        assert_eq!(t.render(&ctx).unwrap(), "*A*");
    }

    #[test]
    fn test_unclosed_block_error() {
        let res = Template::parse("{{#if x}}oops");
        assert!(res.is_err());
    }

    #[test]
    fn test_unclosed_braces_error() {
        let res = Template::parse("hello {{name");
        assert!(res.is_err());
    }

    // -- TemplateEngine registry tests --------------------------------------

    #[test]
    fn test_engine_register_and_render() {
        let mut engine = TemplateEngine::new();
        engine
            .register("greeting", "Hello, {{who}}!")
            .unwrap();
        let mut ctx = TemplateContext::new();
        ctx.set_str("who", "World");
        let out = engine.render("greeting", &ctx).unwrap();
        assert_eq!(out, "Hello, World!");
    }

    #[test]
    fn test_engine_missing_template() {
        let engine = TemplateEngine::new();
        let ctx = TemplateContext::new();
        assert!(engine.render("nope", &ctx).is_err());
    }

    #[test]
    fn test_engine_len() {
        let mut engine = TemplateEngine::new();
        assert!(engine.is_empty());
        engine.register("a", "x").unwrap();
        assert_eq!(engine.len(), 1);
    }

    // -- TemplateValue truthiness tests -------------------------------------

    #[test]
    fn test_truthy_values() {
        assert!(TemplateValue::Bool(true).is_truthy());
        assert!(!TemplateValue::Bool(false).is_truthy());
        assert!(TemplateValue::Str("hi".into()).is_truthy());
        assert!(!TemplateValue::Str("".into()).is_truthy());
        assert!(TemplateValue::Int(1).is_truthy());
        assert!(!TemplateValue::Int(0).is_truthy());
        assert!(TemplateValue::Float(0.1).is_truthy());
        assert!(!TemplateValue::Float(0.0).is_truthy());
        let empty_list: Vec<HashMap<String, TemplateValue>> = vec![];
        assert!(!TemplateValue::List(empty_list).is_truthy());
    }

    // -- CodeBuffer tests ---------------------------------------------------

    #[test]
    fn test_code_buffer_basic() {
        let mut cb = CodeBuffer::new("  ");
        cb.line("fn main() {");
        cb.indent();
        cb.line("println!(\"hello\");");
        cb.dedent();
        cb.line("}");
        let out = cb.finish();
        assert!(out.contains("  println!(\"hello\");"));
        assert!(out.starts_with("fn main() {\n"));
    }

    #[test]
    fn test_code_buffer_open_close_brace() {
        let mut cb = CodeBuffer::new("    ");
        cb.open_brace("struct Foo");
        cb.line("x: i32,");
        cb.close_brace();
        let out = cb.finish();
        assert!(out.contains("struct Foo {"));
        assert!(out.contains("    x: i32,"));
        assert!(out.ends_with("}\n"));
    }

    #[test]
    fn test_code_buffer_blank_line() {
        let mut cb = CodeBuffer::new("  ");
        cb.line("a");
        cb.blank();
        cb.line("b");
        let out = cb.finish();
        assert_eq!(out, "a\n\nb\n");
    }

    // -- Built-in template parse smoke tests --------------------------------

    #[test]
    fn test_builtin_rust_template_parses() {
        assert!(Template::parse(RUST_STATE_MACHINE_TEMPLATE).is_ok());
    }

    #[test]
    fn test_builtin_csharp_template_parses() {
        assert!(Template::parse(CSHARP_STATE_MACHINE_TEMPLATE).is_ok());
    }

    #[test]
    fn test_builtin_typescript_template_parses() {
        assert!(Template::parse(TYPESCRIPT_STATE_MACHINE_TEMPLATE).is_ok());
    }
}
