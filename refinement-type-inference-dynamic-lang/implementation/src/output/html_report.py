from __future__ import annotations

import abc
import collections
import copy
import enum
import hashlib
import html
import io
import json
import os
import re
import textwrap
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _escape(value: Any) -> str:
    return html.escape(str(value))


def _slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", str(text).lower())
    return re.sub(r"[\s_]+", "-", text).strip("-")


def _indent(text: str, level: int = 1, width: int = 2) -> str:
    prefix = " " * (level * width)
    return textwrap.indent(text, prefix)


def _stable_id(*parts: str) -> str:
    raw = "|".join(parts)
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


# ===================================================================
# 1. ReportTemplate – a mini Jinja2-like template engine
# ===================================================================

class _TemplateToken(enum.Enum):
    TEXT = "TEXT"
    VAR = "VAR"
    BLOCK_START = "BLOCK_START"
    BLOCK_END = "BLOCK_END"
    COMMENT = "COMMENT"


@dataclass
class _ParsedNode:
    pass


@dataclass
class _TextNode(_ParsedNode):
    text: str = ""


@dataclass
class _VarNode(_ParsedNode):
    expression: str = ""
    filters: List[Tuple[str, List[str]]] = field(default_factory=list)


@dataclass
class _IfNode(_ParsedNode):
    condition: str = ""
    true_branch: List[_ParsedNode] = field(default_factory=list)
    elif_branches: List[Tuple[str, List[_ParsedNode]]] = field(default_factory=list)
    false_branch: List[_ParsedNode] = field(default_factory=list)


@dataclass
class _ForNode(_ParsedNode):
    var_name: str = ""
    iterable_expr: str = ""
    body: List[_ParsedNode] = field(default_factory=list)


@dataclass
class _BlockNode(_ParsedNode):
    name: str = ""
    body: List[_ParsedNode] = field(default_factory=list)


@dataclass
class _ExtendsNode(_ParsedNode):
    parent_name: str = ""


@dataclass
class _IncludeNode(_ParsedNode):
    template_name: str = ""


@dataclass
class ReportTemplate:
    source: str = ""
    _nodes: List[_ParsedNode] = field(default_factory=list, repr=False)
    _filters: Dict[str, Callable[..., str]] = field(default_factory=dict, repr=False)
    _registry: Dict[str, str] = field(default_factory=dict, repr=False)
    auto_escape: bool = True

    def __post_init__(self) -> None:
        self._TOKEN_RE: re.Pattern[str] = re.compile(
            r"(\{\{.*?\}\}|\{%.*?%\}|\{#.*?#\})", re.DOTALL
        )
        self._register_builtin_filters()
        if self.source:
            self._nodes = self._parse(self._tokenize(self.source))

    # -- filter registration ------------------------------------------------

    def _register_builtin_filters(self) -> None:
        self._filters["escape"] = lambda v: _escape(v)
        self._filters["e"] = lambda v: _escape(v)
        self._filters["upper"] = lambda v: str(v).upper()
        self._filters["lower"] = lambda v: str(v).lower()
        self._filters["title"] = lambda v: str(v).title()
        self._filters["strip"] = lambda v: str(v).strip()
        self._filters["trim"] = lambda v: str(v).strip()
        self._filters["length"] = lambda v: str(len(v))
        self._filters["int"] = lambda v: str(int(v))
        self._filters["float"] = lambda v: str(float(v))
        self._filters["join"] = lambda v, sep=", ": sep.join(str(i) for i in v)
        self._filters["default"] = lambda v, d="": str(d) if not v else str(v)
        self._filters["truncate"] = lambda v, n="80": str(v)[: int(n)] + ("..." if len(str(v)) > int(n) else "")
        self._filters["capitalize"] = lambda v: str(v).capitalize()
        self._filters["replace"] = lambda v, old="", new="": str(v).replace(old, new)
        self._filters["urlencode"] = lambda v: urllib.parse.quote_plus(str(v))
        self._filters["json"] = lambda v: json.dumps(v)
        self._filters["striptags"] = lambda v: re.sub(r"<[^>]+>", "", str(v))
        self._filters["wordcount"] = lambda v: str(len(str(v).split()))
        self._filters["first"] = lambda v: str(v[0]) if v else ""
        self._filters["last"] = lambda v: str(v[-1]) if v else ""
        self._filters["sort"] = lambda v: sorted(v)
        self._filters["reverse"] = lambda v: list(reversed(v)) if isinstance(v, list) else str(v)[::-1]
        self._filters["batch"] = lambda v, n="3": [v[i : i + int(n)] for i in range(0, len(v), int(n))]
        self._filters["slug"] = lambda v: _slugify(str(v))
        self._filters["safe"] = lambda v: str(v)  # mark safe, skip auto-escape
        self._filters["indent"] = lambda v, n="4": textwrap.indent(str(v), " " * int(n))
        self._filters["center"] = lambda v, n="80": str(v).center(int(n))
        self._filters["count"] = lambda v: str(len(v))
        self._filters["abs"] = lambda v: str(abs(int(v)))
        self._filters["round"] = lambda v, n="0": str(round(float(v), int(n)))
        self._filters["string"] = lambda v: str(v)
        self._filters["list"] = lambda v: list(v)
        self._filters["map"] = lambda v, attr="": [getattr(i, attr, i.get(attr, "")) if isinstance(i, dict) else getattr(i, attr, "") for i in v]
        self._filters["select"] = lambda v, attr="": [i for i in v if i]
        self._filters["reject"] = lambda v, attr="": [i for i in v if not i]
        self._filters["unique"] = lambda v: list(dict.fromkeys(v))
        self._filters["tojson"] = lambda v, indent="2": json.dumps(v, indent=int(indent), default=str)
        self._filters["filesizeformat"] = lambda v: _format_filesize(int(v))
        self._filters["pprint"] = lambda v: json.dumps(v, indent=2, default=str)

    def register_filter(self, name: str, fn: Callable[..., str]) -> None:
        self._filters[name] = fn

    def register_template(self, name: str, source: str) -> None:
        self._registry[name] = source

    # -- tokenizer ----------------------------------------------------------

    def _tokenize(self, source: str) -> List[Tuple[_TemplateToken, str]]:
        tokens: List[Tuple[_TemplateToken, str]] = []
        parts = self._TOKEN_RE.split(source)
        for part in parts:
            if not part:
                continue
            if part.startswith("{{") and part.endswith("}}"):
                tokens.append((_TemplateToken.VAR, part[2:-2].strip()))
            elif part.startswith("{%") and part.endswith("%}"):
                tag_body = part[2:-2].strip()
                tokens.append((_TemplateToken.BLOCK_START, tag_body))
            elif part.startswith("{#") and part.endswith("#}"):
                tokens.append((_TemplateToken.COMMENT, part[2:-2].strip()))
            else:
                tokens.append((_TemplateToken.TEXT, part))
        return tokens

    # -- parser -------------------------------------------------------------

    def _parse(self, tokens: List[Tuple[_TemplateToken, str]]) -> List[_ParsedNode]:
        nodes: List[_ParsedNode] = []
        i = 0
        while i < len(tokens):
            tok_type, tok_val = tokens[i]
            if tok_type == _TemplateToken.TEXT:
                nodes.append(_TextNode(text=tok_val))
                i += 1
            elif tok_type == _TemplateToken.COMMENT:
                i += 1
            elif tok_type == _TemplateToken.VAR:
                expr, filt = self._parse_var_expr(tok_val)
                nodes.append(_VarNode(expression=expr, filters=filt))
                i += 1
            elif tok_type == _TemplateToken.BLOCK_START:
                if tok_val.startswith("if "):
                    node, consumed = self._parse_if(tokens, i)
                    nodes.append(node)
                    i += consumed
                elif tok_val.startswith("for "):
                    node, consumed = self._parse_for(tokens, i)
                    nodes.append(node)
                    i += consumed
                elif tok_val.startswith("block "):
                    node, consumed = self._parse_block(tokens, i)
                    nodes.append(node)
                    i += consumed
                elif tok_val.startswith("extends "):
                    name = tok_val[8:].strip().strip("'\"")
                    nodes.append(_ExtendsNode(parent_name=name))
                    i += 1
                elif tok_val.startswith("include "):
                    name = tok_val[8:].strip().strip("'\"")
                    nodes.append(_IncludeNode(template_name=name))
                    i += 1
                else:
                    i += 1
            else:
                i += 1
        return nodes

    def _parse_var_expr(self, raw: str) -> Tuple[str, List[Tuple[str, List[str]]]]:
        parts = raw.split("|")
        expr = parts[0].strip()
        filters: List[Tuple[str, List[str]]] = []
        for p in parts[1:]:
            p = p.strip()
            m = re.match(r"(\w+)\((.*)\)", p)
            if m:
                fname = m.group(1)
                args = [a.strip().strip("'\"") for a in m.group(2).split(",") if a.strip()]
                filters.append((fname, args))
            else:
                filters.append((p, []))
        return expr, filters

    def _parse_if(
        self, tokens: List[Tuple[_TemplateToken, str]], start: int
    ) -> Tuple[_IfNode, int]:
        cond = tokens[start][1][3:].strip()
        node = _IfNode(condition=cond)
        current_branch: List[_ParsedNode] = node.true_branch
        i = start + 1
        depth = 1
        while i < len(tokens):
            tok_type, tok_val = tokens[i]
            if tok_type == _TemplateToken.BLOCK_START:
                if tok_val.startswith("if "):
                    depth += 1
                elif tok_val == "endif":
                    depth -= 1
                    if depth == 0:
                        return node, i - start + 1
                elif tok_val.startswith("elif ") and depth == 1:
                    elif_cond = tok_val[5:].strip()
                    new_branch: List[_ParsedNode] = []
                    node.elif_branches.append((elif_cond, new_branch))
                    current_branch = new_branch
                    i += 1
                    continue
                elif tok_val == "else" and depth == 1:
                    current_branch = node.false_branch
                    i += 1
                    continue
            sub_nodes = self._parse([tokens[i]])
            current_branch.extend(sub_nodes)
            i += 1
        return node, i - start

    def _parse_for(
        self, tokens: List[Tuple[_TemplateToken, str]], start: int
    ) -> Tuple[_ForNode, int]:
        m = re.match(r"for\s+(\w+)\s+in\s+(.+)", tokens[start][1])
        var_name = m.group(1) if m else "item"
        iterable = m.group(2).strip() if m else ""
        node = _ForNode(var_name=var_name, iterable_expr=iterable)
        i = start + 1
        depth = 1
        while i < len(tokens):
            tok_type, tok_val = tokens[i]
            if tok_type == _TemplateToken.BLOCK_START:
                if tok_val.startswith("for "):
                    depth += 1
                elif tok_val == "endfor":
                    depth -= 1
                    if depth == 0:
                        return node, i - start + 1
            sub_nodes = self._parse([tokens[i]])
            node.body.extend(sub_nodes)
            i += 1
        return node, i - start

    def _parse_block(
        self, tokens: List[Tuple[_TemplateToken, str]], start: int
    ) -> Tuple[_BlockNode, int]:
        name = tokens[start][1][6:].strip()
        node = _BlockNode(name=name)
        i = start + 1
        depth = 1
        while i < len(tokens):
            tok_type, tok_val = tokens[i]
            if tok_type == _TemplateToken.BLOCK_START:
                if tok_val.startswith("block "):
                    depth += 1
                elif tok_val == "endblock":
                    depth -= 1
                    if depth == 0:
                        return node, i - start + 1
            sub_nodes = self._parse([tokens[i]])
            node.body.extend(sub_nodes)
            i += 1
        return node, i - start

    # -- evaluator ----------------------------------------------------------

    def _resolve(self, expr: str, ctx: Dict[str, Any]) -> Any:
        expr = expr.strip()
        if expr in ctx:
            return ctx[expr]
        if "." in expr:
            parts = expr.split(".")
            obj = ctx.get(parts[0])
            for part in parts[1:]:
                if obj is None:
                    return ""
                if isinstance(obj, dict):
                    obj = obj.get(part, "")
                else:
                    obj = getattr(obj, part, "")
            return obj
        if "[" in expr:
            m = re.match(r"(\w+)\[(.+?)\]", expr)
            if m:
                base = ctx.get(m.group(1))
                idx_raw = m.group(2).strip().strip("'\"")
                if base is None:
                    return ""
                if isinstance(base, dict):
                    return base.get(idx_raw, "")
                try:
                    return base[int(idx_raw)]
                except (ValueError, IndexError, TypeError):
                    return ""
        if expr.startswith("'") and expr.endswith("'"):
            return expr[1:-1]
        if expr.startswith('"') and expr.endswith('"'):
            return expr[1:-1]
        try:
            return int(expr)
        except ValueError:
            pass
        try:
            return float(expr)
        except ValueError:
            pass
        if expr == "true" or expr == "True":
            return True
        if expr == "false" or expr == "False":
            return False
        if expr == "none" or expr == "None":
            return None
        return ""

    def _eval_condition(self, cond: str, ctx: Dict[str, Any]) -> bool:
        cond = cond.strip()
        if " and " in cond:
            parts = cond.split(" and ", 1)
            return self._eval_condition(parts[0], ctx) and self._eval_condition(parts[1], ctx)
        if " or " in cond:
            parts = cond.split(" or ", 1)
            return self._eval_condition(parts[0], ctx) or self._eval_condition(parts[1], ctx)
        if cond.startswith("not "):
            return not self._eval_condition(cond[4:], ctx)
        m = re.match(r"(.+?)\s*(==|!=|>=|<=|>|<|in|not in)\s*(.+)", cond)
        if m:
            left = self._resolve(m.group(1).strip(), ctx)
            op = m.group(2).strip()
            right = self._resolve(m.group(3).strip(), ctx)
            if op == "==":
                return left == right
            if op == "!=":
                return left != right
            if op == ">":
                return left > right
            if op == "<":
                return left < right
            if op == ">=":
                return left >= right
            if op == "<=":
                return left <= right
            if op == "in":
                return left in right
            if op == "not in":
                return left not in right
        val = self._resolve(cond, ctx)
        return bool(val)

    def _apply_filters(
        self, value: Any, filters: List[Tuple[str, List[str]]]
    ) -> str:
        result = value
        is_safe = False
        for fname, args in filters:
            if fname == "safe":
                is_safe = True
                continue
            fn = self._filters.get(fname)
            if fn:
                try:
                    result = fn(result, *args)
                except Exception:
                    pass
        if self.auto_escape and not is_safe:
            if isinstance(result, str):
                result = _escape(result)
        return str(result) if result is not None else ""

    def _render_nodes(self, nodes: List[_ParsedNode], ctx: Dict[str, Any]) -> str:
        buf = io.StringIO()
        blocks: Dict[str, List[_ParsedNode]] = {}
        extends_parent: Optional[str] = None
        for node in nodes:
            if isinstance(node, _ExtendsNode):
                extends_parent = node.parent_name
            elif isinstance(node, _BlockNode):
                blocks[node.name] = node.body
            elif isinstance(node, _TextNode):
                buf.write(node.text)
            elif isinstance(node, _VarNode):
                val = self._resolve(node.expression, ctx)
                buf.write(self._apply_filters(val, node.filters))
            elif isinstance(node, _IfNode):
                if self._eval_condition(node.condition, ctx):
                    buf.write(self._render_nodes(node.true_branch, ctx))
                else:
                    matched = False
                    for elif_cond, elif_body in node.elif_branches:
                        if self._eval_condition(elif_cond, ctx):
                            buf.write(self._render_nodes(elif_body, ctx))
                            matched = True
                            break
                    if not matched:
                        buf.write(self._render_nodes(node.false_branch, ctx))
            elif isinstance(node, _ForNode):
                iterable = self._resolve(node.iterable_expr, ctx)
                if iterable:
                    items = list(iterable) if not isinstance(iterable, list) else iterable
                    loop_len = len(items)
                    for idx, item in enumerate(items):
                        inner_ctx = dict(ctx)
                        inner_ctx[node.var_name] = item
                        inner_ctx["loop"] = {
                            "index": idx + 1,
                            "index0": idx,
                            "first": idx == 0,
                            "last": idx == loop_len - 1,
                            "length": loop_len,
                            "revindex": loop_len - idx,
                            "revindex0": loop_len - idx - 1,
                        }
                        buf.write(self._render_nodes(node.body, inner_ctx))
            elif isinstance(node, _IncludeNode):
                tmpl_src = self._registry.get(node.template_name, "")
                if tmpl_src:
                    sub = ReportTemplate(source=tmpl_src, _registry=dict(self._registry), _filters=dict(self._filters), auto_escape=self.auto_escape)
                    buf.write(sub.render(ctx))
            elif isinstance(node, _BlockNode):
                buf.write(self._render_nodes(node.body, ctx))
        if extends_parent:
            parent_src = self._registry.get(extends_parent, "")
            if parent_src:
                parent = ReportTemplate(source=parent_src, _registry=dict(self._registry), _filters=dict(self._filters), auto_escape=self.auto_escape)
                parent_nodes = parent._nodes
                resolved: List[_ParsedNode] = []
                for pn in parent_nodes:
                    if isinstance(pn, _BlockNode) and pn.name in blocks:
                        resolved.extend(blocks[pn.name])
                    else:
                        resolved.append(pn)
                return parent._render_nodes(resolved, ctx)
        return buf.getvalue()

    def render(self, context: Dict[str, Any] | None = None) -> str:
        ctx = dict(context or {})
        return self._render_nodes(self._nodes, ctx)


def _format_filesize(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024.0:
            return f"{size:.1f} {unit}"
        size = int(size / 1024)
    return f"{size:.1f} PB"


# ===================================================================
# 2. CssGenerator
# ===================================================================

@dataclass
class CssGenerator:
    dark_mode: bool = False
    primary_color: str = "#2563eb"
    font_family: str = "'Segoe UI', system-ui, -apple-system, sans-serif"
    mono_font: str = "'Fira Code', 'JetBrains Mono', 'Cascadia Code', monospace"

    def _reset(self) -> str:
        return textwrap.dedent("""\
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        html { font-size: 16px; line-height: 1.6; -webkit-text-size-adjust: 100%; }
        body { min-height: 100vh; }
        img, picture, video, canvas, svg { display: block; max-width: 100%; }
        input, button, textarea, select { font: inherit; }
        p, h1, h2, h3, h4, h5, h6 { overflow-wrap: break-word; }
        a { color: inherit; text-decoration: none; }
        ul, ol { list-style: none; }
        table { border-collapse: collapse; width: 100%; }
        """)

    def _typography(self) -> str:
        return textwrap.dedent(f"""\
        body {{ font-family: {self.font_family}; color: var(--fg); background: var(--bg); }}
        code, pre, .mono {{ font-family: {self.mono_font}; font-size: 0.875rem; }}
        h1 {{ font-size: 1.75rem; font-weight: 700; margin-bottom: 0.5em; }}
        h2 {{ font-size: 1.4rem; font-weight: 600; margin-bottom: 0.4em; }}
        h3 {{ font-size: 1.15rem; font-weight: 600; margin-bottom: 0.3em; }}
        h4 {{ font-size: 1rem; font-weight: 600; margin-bottom: 0.25em; }}
        p {{ margin-bottom: 0.75em; }}
        small {{ font-size: 0.8rem; }}
        .text-muted {{ opacity: 0.7; }}
        """)

    def _light_theme(self) -> str:
        return textwrap.dedent("""\
        :root {
          --bg: #ffffff; --fg: #1e293b; --accent: #2563eb;
          --error: #dc2626; --warning: #d97706; --info: #2563eb; --success: #16a34a;
          --code-bg: #f8fafc; --border: #e2e8f0; --sidebar-bg: #f1f5f9;
          --card-bg: #ffffff; --card-shadow: 0 1px 3px rgba(0,0,0,0.1);
          --header-bg: #1e293b; --header-fg: #f8fafc;
          --badge-error-bg: #fef2f2; --badge-error-fg: #dc2626;
          --badge-warn-bg: #fffbeb; --badge-warn-fg: #d97706;
          --badge-info-bg: #eff6ff; --badge-info-fg: #2563eb;
          --badge-success-bg: #f0fdf4; --badge-success-fg: #16a34a;
          --hover-bg: #f1f5f9; --focus-ring: #93c5fd;
          --table-stripe: #f8fafc; --table-header-bg: #f1f5f9;
          --tooltip-bg: #1e293b; --tooltip-fg: #f8fafc;
          --scrollbar-bg: #e2e8f0; --scrollbar-thumb: #94a3b8;
          --link-color: #2563eb; --link-hover: #1d4ed8;
          --selection-bg: #bfdbfe;
        }
        """)

    def _dark_theme(self) -> str:
        return textwrap.dedent("""\
        [data-theme="dark"] {
          --bg: #0f172a; --fg: #e2e8f0; --accent: #3b82f6;
          --error: #f87171; --warning: #fbbf24; --info: #60a5fa; --success: #4ade80;
          --code-bg: #1e293b; --border: #334155; --sidebar-bg: #1e293b;
          --card-bg: #1e293b; --card-shadow: 0 1px 3px rgba(0,0,0,0.4);
          --header-bg: #0f172a; --header-fg: #e2e8f0;
          --badge-error-bg: #451a1a; --badge-error-fg: #f87171;
          --badge-warn-bg: #451a03; --badge-warn-fg: #fbbf24;
          --badge-info-bg: #172554; --badge-info-fg: #60a5fa;
          --badge-success-bg: #052e16; --badge-success-fg: #4ade80;
          --hover-bg: #334155; --focus-ring: #3b82f6;
          --table-stripe: #1e293b; --table-header-bg: #334155;
          --tooltip-bg: #e2e8f0; --tooltip-fg: #0f172a;
          --scrollbar-bg: #334155; --scrollbar-thumb: #64748b;
          --link-color: #60a5fa; --link-hover: #93c5fd;
          --selection-bg: #1e3a5f;
        }
        """)

    def _syntax_colors(self) -> str:
        return textwrap.dedent("""\
        .tok-keyword { color: #8b5cf6; font-weight: 600; }
        .tok-string { color: #059669; }
        .tok-number { color: #d97706; }
        .tok-comment { color: #94a3b8; font-style: italic; }
        .tok-operator { color: #64748b; }
        .tok-builtin { color: #0891b2; }
        .tok-decorator { color: #a855f7; }
        .tok-classname { color: #0ea5e9; font-weight: 600; }
        .tok-funcname { color: #2563eb; }
        .tok-typehint { color: #0d9488; font-style: italic; }
        .tok-punctuation { color: #64748b; }
        .tok-constant { color: #e11d48; }
        .tok-variable { color: var(--fg); }
        .tok-parameter { color: #ea580c; }
        .tok-property { color: #0ea5e9; }
        .tok-regex { color: #e11d48; }
        [data-theme="dark"] .tok-keyword { color: #a78bfa; }
        [data-theme="dark"] .tok-string { color: #34d399; }
        [data-theme="dark"] .tok-number { color: #fbbf24; }
        [data-theme="dark"] .tok-comment { color: #64748b; }
        [data-theme="dark"] .tok-builtin { color: #22d3ee; }
        [data-theme="dark"] .tok-funcname { color: #60a5fa; }
        [data-theme="dark"] .tok-classname { color: #38bdf8; }
        [data-theme="dark"] .tok-typehint { color: #2dd4bf; }
        [data-theme="dark"] .tok-decorator { color: #c084fc; }
        """)

    def _layout(self) -> str:
        return textwrap.dedent("""\
        .page-wrapper { display: flex; min-height: 100vh; }
        .sidebar {
          width: 260px; background: var(--sidebar-bg); border-right: 1px solid var(--border);
          position: sticky; top: 0; height: 100vh; overflow-y: auto;
          padding: 1rem; flex-shrink: 0; z-index: 10;
        }
        .main-content { flex: 1; padding: 2rem; max-width: calc(100% - 260px); overflow-x: hidden; }
        .header {
          background: var(--header-bg); color: var(--header-fg); padding: 1rem 2rem;
          display: flex; align-items: center; justify-content: space-between;
          border-bottom: 1px solid var(--border); position: sticky; top: 0; z-index: 20;
        }
        .footer {
          padding: 1rem 2rem; border-top: 1px solid var(--border);
          text-align: center; font-size: 0.8rem; opacity: 0.7;
        }
        .content-area { max-width: 1200px; margin: 0 auto; }
        """)

    def _cards(self) -> str:
        return textwrap.dedent("""\
        .card {
          background: var(--card-bg); border: 1px solid var(--border);
          border-radius: 8px; padding: 1.25rem; margin-bottom: 1rem;
          box-shadow: var(--card-shadow); transition: box-shadow 0.2s;
        }
        .card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
        .card-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 0.75rem; }
        .card-subtitle { font-size: 0.85rem; color: var(--fg); opacity: 0.7; margin-bottom: 0.5rem; }
        .card-body { font-size: 0.95rem; }
        .card-footer { margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid var(--border); font-size: 0.85rem; }
        .panel { background: var(--code-bg); border: 1px solid var(--border); border-radius: 6px; padding: 1rem; margin-bottom: 1rem; }
        .panel-header { font-weight: 600; margin-bottom: 0.5rem; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border); }
        """)

    def _tables(self) -> str:
        return textwrap.dedent("""\
        table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
        th, td { padding: 0.6rem 0.75rem; text-align: left; border-bottom: 1px solid var(--border); }
        th { background: var(--table-header-bg); font-weight: 600; position: sticky; top: 0; }
        tr:nth-child(even) td { background: var(--table-stripe); }
        tr:hover td { background: var(--hover-bg); }
        .table-wrap { overflow-x: auto; border-radius: 6px; border: 1px solid var(--border); }
        .table-sortable th { cursor: pointer; user-select: none; }
        .table-sortable th:hover { background: var(--hover-bg); }
        .sort-indicator { margin-left: 4px; opacity: 0.5; }
        .sort-indicator.active { opacity: 1; }
        """)

    def _badges(self) -> str:
        return textwrap.dedent("""\
        .badge {
          display: inline-flex; align-items: center; padding: 0.15rem 0.5rem;
          border-radius: 999px; font-size: 0.75rem; font-weight: 600;
          line-height: 1.4; white-space: nowrap;
        }
        .badge-error { background: var(--badge-error-bg); color: var(--badge-error-fg); }
        .badge-warning { background: var(--badge-warn-bg); color: var(--badge-warn-fg); }
        .badge-info { background: var(--badge-info-bg); color: var(--badge-info-fg); }
        .badge-success { background: var(--badge-success-bg); color: var(--badge-success-fg); }
        .badge-count {
          display: inline-flex; align-items: center; justify-content: center;
          min-width: 1.4rem; height: 1.4rem; border-radius: 50%;
          font-size: 0.7rem; font-weight: 700;
        }
        """)

    def _code_blocks(self) -> str:
        return textwrap.dedent("""\
        .code-block {
          background: var(--code-bg); border: 1px solid var(--border);
          border-radius: 6px; overflow-x: auto; position: relative;
          font-family: inherit; margin-bottom: 1rem;
        }
        .code-block pre { margin: 0; padding: 1rem; padding-left: 3.5rem; overflow-x: auto; tab-size: 4; }
        .code-block code { font-family: inherit; }
        .line-numbers {
          position: absolute; left: 0; top: 0; padding: 1rem 0.5rem;
          text-align: right; color: var(--fg); opacity: 0.35;
          user-select: none; border-right: 1px solid var(--border);
          min-width: 3rem; font-size: 0.8rem;
        }
        .line-numbers span { display: block; line-height: 1.6; }
        .code-header {
          display: flex; align-items: center; justify-content: space-between;
          padding: 0.5rem 1rem; border-bottom: 1px solid var(--border);
          background: var(--table-header-bg); border-radius: 6px 6px 0 0;
          font-size: 0.85rem; font-weight: 500;
        }
        .copy-btn {
          background: none; border: 1px solid var(--border); border-radius: 4px;
          padding: 0.2rem 0.6rem; cursor: pointer; font-size: 0.8rem;
          color: var(--fg); transition: background 0.15s;
        }
        .copy-btn:hover { background: var(--hover-bg); }
        .copy-btn.copied { color: var(--success); border-color: var(--success); }
        """)

    def _tooltips(self) -> str:
        return textwrap.dedent("""\
        .tooltip-container { position: relative; display: inline; }
        .tooltip {
          display: none; position: absolute; z-index: 100;
          background: var(--tooltip-bg); color: var(--tooltip-fg);
          padding: 0.5rem 0.75rem; border-radius: 6px;
          font-size: 0.8rem; line-height: 1.4;
          max-width: 400px; white-space: pre-wrap;
          box-shadow: 0 4px 12px rgba(0,0,0,0.2);
          pointer-events: none;
        }
        .tooltip-container:hover .tooltip { display: block; }
        .tooltip.tooltip-top { bottom: 100%; left: 50%; transform: translateX(-50%); margin-bottom: 6px; }
        .tooltip.tooltip-bottom { top: 100%; left: 50%; transform: translateX(-50%); margin-top: 6px; }
        .tooltip.tooltip-left { right: 100%; top: 50%; transform: translateY(-50%); margin-right: 6px; }
        .tooltip.tooltip-right { left: 100%; top: 50%; transform: translateY(-50%); margin-left: 6px; }
        .tooltip::after {
          content: ''; position: absolute; border: 5px solid transparent;
        }
        .tooltip.tooltip-top::after { top: 100%; left: 50%; transform: translateX(-50%); border-top-color: var(--tooltip-bg); }
        .tooltip.tooltip-bottom::after { bottom: 100%; left: 50%; transform: translateX(-50%); border-bottom-color: var(--tooltip-bg); }
        """)

    def _collapsible(self) -> str:
        return textwrap.dedent("""\
        .collapsible { border: 1px solid var(--border); border-radius: 6px; margin-bottom: 0.75rem; overflow: hidden; }
        .collapsible-header {
          display: flex; align-items: center; padding: 0.75rem 1rem;
          cursor: pointer; user-select: none; background: var(--table-header-bg);
          font-weight: 600; transition: background 0.15s;
        }
        .collapsible-header:hover { background: var(--hover-bg); }
        .collapsible-chevron {
          margin-right: 0.5rem; transition: transform 0.2s; display: inline-block;
          width: 0; height: 0; border-left: 5px solid var(--fg); border-top: 4px solid transparent;
          border-bottom: 4px solid transparent;
        }
        .collapsible.open .collapsible-chevron { transform: rotate(90deg); }
        .collapsible-content { padding: 1rem; display: none; }
        .collapsible.open .collapsible-content { display: block; }
        """)

    def _responsive(self) -> str:
        return textwrap.dedent("""\
        @media (max-width: 1024px) {
          .sidebar { width: 220px; }
          .main-content { max-width: calc(100% - 220px); padding: 1.5rem; }
        }
        @media (max-width: 768px) {
          .page-wrapper { flex-direction: column; }
          .sidebar { width: 100%; height: auto; position: relative; border-right: none; border-bottom: 1px solid var(--border); }
          .main-content { max-width: 100%; padding: 1rem; }
          .header { flex-direction: column; gap: 0.5rem; }
          table { font-size: 0.8rem; }
          th, td { padding: 0.4rem 0.5rem; }
        }
        @media (max-width: 480px) {
          html { font-size: 14px; }
          .card { padding: 0.75rem; }
          .code-block pre { font-size: 0.75rem; }
        }
        """)

    def _print(self) -> str:
        return textwrap.dedent("""\
        @media print {
          .sidebar, .header, .footer, .copy-btn, .theme-toggle,
          .search-widget, .export-buttons, .tab-nav { display: none !important; }
          .main-content { max-width: 100% !important; padding: 0 !important; }
          .page-wrapper { display: block !important; }
          .card { box-shadow: none !important; border: 1px solid #ccc !important; page-break-inside: avoid; }
          .collapsible-content { display: block !important; }
          body { color: #000 !important; background: #fff !important; }
          a { color: #000 !important; text-decoration: underline; }
          .code-block { border: 1px solid #ccc !important; }
        }
        """)

    def _extra(self) -> str:
        return textwrap.dedent("""\
        .tab-nav { display: flex; border-bottom: 2px solid var(--border); margin-bottom: 1rem; gap: 0; }
        .tab-btn {
          padding: 0.5rem 1rem; cursor: pointer; border: none; background: none;
          font-weight: 500; color: var(--fg); opacity: 0.6; transition: opacity 0.15s;
          border-bottom: 2px solid transparent; margin-bottom: -2px;
        }
        .tab-btn:hover { opacity: 1; }
        .tab-btn.active { opacity: 1; border-bottom-color: var(--accent); color: var(--accent); }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .search-input {
          width: 100%; padding: 0.5rem 0.75rem; border: 1px solid var(--border);
          border-radius: 6px; background: var(--bg); color: var(--fg);
          font-size: 0.9rem; outline: none; transition: border-color 0.15s;
        }
        .search-input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px var(--focus-ring); }
        .search-results-count { font-size: 0.8rem; margin-top: 0.25rem; opacity: 0.7; }
        .highlight-match { background: var(--selection-bg); padding: 0 2px; border-radius: 2px; }
        .wavy-error { text-decoration: underline wavy var(--error); text-underline-offset: 3px; }
        .wavy-warning { text-decoration: underline wavy var(--warning); text-underline-offset: 3px; }
        .gutter-marker { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; }
        .gutter-error { background: var(--error); }
        .gutter-warning { background: var(--warning); }
        .gutter-info { background: var(--info); }
        .progress-bar-bg { background: var(--border); border-radius: 4px; overflow: hidden; height: 8px; }
        .progress-bar-fill { background: var(--accent); height: 100%; border-radius: 4px; transition: width 0.3s; }
        .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
        .diff-add { background: #dcfce7; }
        .diff-del { background: #fee2e2; }
        .diff-change { background: #fef9c3; }
        [data-theme="dark"] .diff-add { background: #052e16; }
        [data-theme="dark"] .diff-del { background: #450a0a; }
        [data-theme="dark"] .diff-change { background: #422006; }
        .type-tree { padding-left: 1.2rem; border-left: 1px solid var(--border); }
        .type-node { padding: 0.15rem 0; }
        .type-base { color: var(--accent); font-weight: 500; }
        .type-refinement { color: var(--success); font-style: italic; }
        .type-union { color: var(--warning); }
        .type-intersection { color: var(--info); }
        .sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); border: 0; }
        .theme-toggle {
          background: none; border: 1px solid var(--border); border-radius: 6px;
          padding: 0.3rem 0.7rem; cursor: pointer; color: var(--fg);
          font-size: 0.85rem; transition: background 0.15s;
        }
        .theme-toggle:hover { background: var(--hover-bg); }
        .coverage-cell { text-align: center; font-weight: 600; }
        .coverage-high { color: var(--success); }
        .coverage-mid { color: var(--warning); }
        .coverage-low { color: var(--error); }
        .file-tree-item { padding: 0.25rem 0.5rem; cursor: pointer; border-radius: 4px; display: flex; align-items: center; gap: 0.4rem; }
        .file-tree-item:hover { background: var(--hover-bg); }
        .file-tree-item.active { background: var(--selection-bg); font-weight: 600; }
        .file-tree-children { padding-left: 1rem; }
        ::selection { background: var(--selection-bg); }
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: var(--scrollbar-bg); }
        ::-webkit-scrollbar-thumb { background: var(--scrollbar-thumb); border-radius: 4px; }
        .fade-in { animation: fadeIn 0.3s ease-in; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        .annotation-inline {
          background: var(--badge-info-bg); color: var(--badge-info-fg);
          padding: 0 4px; border-radius: 3px; font-size: 0.8em;
          font-family: inherit; margin-left: 0.3em;
        }
        .bug-line { background: rgba(220, 38, 38, 0.08); }
        .warn-line { background: rgba(217, 119, 6, 0.08); }
        """)

    def generate(self) -> str:
        parts = [
            self._reset(),
            self._light_theme(),
            self._dark_theme(),
            self._typography(),
            self._syntax_colors(),
            self._layout(),
            self._cards(),
            self._tables(),
            self._badges(),
            self._code_blocks(),
            self._tooltips(),
            self._collapsible(),
            self._responsive(),
            self._print(),
            self._extra(),
        ]
        return "\n".join(parts)


# ===================================================================
# 3. JavaScriptGenerator
# ===================================================================

@dataclass
class JavaScriptGenerator:

    def _collapsible_toggle(self) -> str:
        return textwrap.dedent("""\
        function initCollapsibles() {
          document.querySelectorAll('.collapsible-header').forEach(function(header) {
            header.addEventListener('click', function() {
              var section = this.closest('.collapsible');
              section.classList.toggle('open');
            });
          });
        }
        function expandAll() {
          document.querySelectorAll('.collapsible').forEach(function(s) { s.classList.add('open'); });
        }
        function collapseAll() {
          document.querySelectorAll('.collapsible').forEach(function(s) { s.classList.remove('open'); });
        }
        """)

    def _tab_switching(self) -> str:
        return textwrap.dedent("""\
        function initTabs() {
          document.querySelectorAll('.tab-btn').forEach(function(btn) {
            btn.addEventListener('click', function() {
              var group = this.dataset.tabGroup;
              var target = this.dataset.tabTarget;
              document.querySelectorAll('.tab-btn[data-tab-group="' + group + '"]').forEach(function(b) { b.classList.remove('active'); });
              document.querySelectorAll('.tab-content[data-tab-group="' + group + '"]').forEach(function(c) { c.classList.remove('active'); });
              this.classList.add('active');
              var el = document.getElementById(target);
              if (el) el.classList.add('active');
            });
          });
        }
        """)

    def _search_filter(self) -> str:
        return textwrap.dedent("""\
        function fuzzyMatch(needle, haystack) {
          needle = needle.toLowerCase();
          haystack = haystack.toLowerCase();
          if (haystack.indexOf(needle) !== -1) return true;
          var ni = 0;
          for (var hi = 0; hi < haystack.length && ni < needle.length; hi++) {
            if (haystack[hi] === needle[ni]) ni++;
          }
          return ni === needle.length;
        }
        function initSearch() {
          var input = document.getElementById('search-input');
          if (!input) return;
          input.addEventListener('input', function() {
            var query = this.value.trim();
            var items = document.querySelectorAll('[data-searchable]');
            var count = 0;
            items.forEach(function(item) {
              var text = item.dataset.searchable || item.textContent;
              if (!query || fuzzyMatch(query, text)) {
                item.style.display = '';
                count++;
                if (query) highlightText(item, query);
                else removeHighlight(item);
              } else {
                item.style.display = 'none';
              }
            });
            var counter = document.getElementById('search-results-count');
            if (counter) counter.textContent = query ? count + ' results' : '';
          });
        }
        function highlightText(el, query) {
          removeHighlight(el);
          var walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null, false);
          var nodes = [];
          while (walker.nextNode()) nodes.push(walker.currentNode);
          nodes.forEach(function(node) {
            var idx = node.textContent.toLowerCase().indexOf(query.toLowerCase());
            if (idx === -1) return;
            var span = document.createElement('span');
            span.className = 'highlight-match';
            var range = document.createRange();
            range.setStart(node, idx);
            range.setEnd(node, idx + query.length);
            range.surroundContents(span);
          });
        }
        function removeHighlight(el) {
          el.querySelectorAll('.highlight-match').forEach(function(h) {
            var parent = h.parentNode;
            parent.replaceChild(document.createTextNode(h.textContent), h);
            parent.normalize();
          });
        }
        """)

    def _sort_tables(self) -> str:
        return textwrap.dedent("""\
        function initSortableTables() {
          document.querySelectorAll('.table-sortable').forEach(function(table) {
            var headers = table.querySelectorAll('th');
            headers.forEach(function(th, colIdx) {
              th.addEventListener('click', function() {
                sortTable(table, colIdx);
              });
            });
          });
        }
        function sortTable(table, colIdx) {
          var tbody = table.querySelector('tbody') || table;
          var rows = Array.from(tbody.querySelectorAll('tr'));
          var dir = table.dataset.sortDir === 'asc' ? 'desc' : 'asc';
          table.dataset.sortDir = dir;
          table.dataset.sortCol = colIdx;
          rows.sort(function(a, b) {
            var aVal = a.cells[colIdx] ? a.cells[colIdx].textContent.trim() : '';
            var bVal = b.cells[colIdx] ? b.cells[colIdx].textContent.trim() : '';
            var aNum = parseFloat(aVal), bNum = parseFloat(bVal);
            if (!isNaN(aNum) && !isNaN(bNum)) {
              return dir === 'asc' ? aNum - bNum : bNum - aNum;
            }
            return dir === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
          });
          rows.forEach(function(row) { tbody.appendChild(row); });
          table.querySelectorAll('.sort-indicator').forEach(function(si) { si.classList.remove('active'); si.textContent = '⇅'; });
          var indicator = table.querySelectorAll('th')[colIdx].querySelector('.sort-indicator');
          if (indicator) { indicator.classList.add('active'); indicator.textContent = dir === 'asc' ? '▲' : '▼'; }
        }
        """)

    def _theme_toggle(self) -> str:
        return textwrap.dedent("""\
        function initThemeToggle() {
          var btn = document.getElementById('theme-toggle');
          if (!btn) return;
          var saved = localStorage.getItem('report-theme');
          if (saved) document.documentElement.setAttribute('data-theme', saved);
          btn.addEventListener('click', function() {
            var current = document.documentElement.getAttribute('data-theme') || 'light';
            var next = current === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', next);
            localStorage.setItem('report-theme', next);
            btn.textContent = next === 'dark' ? '☀️ Light' : '🌙 Dark';
          });
        }
        """)

    def _copy_code(self) -> str:
        return textwrap.dedent("""\
        function initCopyButtons() {
          document.querySelectorAll('.copy-btn').forEach(function(btn) {
            btn.addEventListener('click', function() {
              var block = this.closest('.code-block');
              var code = block.querySelector('code') || block.querySelector('pre');
              var text = code ? code.textContent : '';
              navigator.clipboard.writeText(text).then(function() {
                btn.textContent = '✓ Copied';
                btn.classList.add('copied');
                setTimeout(function() { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 2000);
              }).catch(function() {
                var ta = document.createElement('textarea');
                ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
                document.body.appendChild(ta); ta.select();
                document.execCommand('copy');
                document.body.removeChild(ta);
                btn.textContent = '✓ Copied'; btn.classList.add('copied');
                setTimeout(function() { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 2000);
              });
            });
          });
        }
        """)

    def _keyboard_nav(self) -> str:
        return textwrap.dedent("""\
        function initKeyboardNav() {
          document.addEventListener('keydown', function(e) {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            if (e.key === '/' || e.key === 'f' && (e.ctrlKey || e.metaKey)) {
              e.preventDefault();
              var input = document.getElementById('search-input');
              if (input) input.focus();
            }
            if (e.key === 'Escape') {
              var input = document.getElementById('search-input');
              if (input && document.activeElement === input) { input.blur(); input.value = ''; input.dispatchEvent(new Event('input')); }
            }
            if (e.key === 'e' && !e.ctrlKey && !e.metaKey) expandAll();
            if (e.key === 'c' && !e.ctrlKey && !e.metaKey) collapseAll();
            if (e.key === 't' && !e.ctrlKey && !e.metaKey) {
              var btn = document.getElementById('theme-toggle');
              if (btn) btn.click();
            }
          });
        }
        """)

    def _local_storage(self) -> str:
        return textwrap.dedent("""\
        function savePreference(key, value) {
          try { localStorage.setItem('report-' + key, JSON.stringify(value)); } catch(e) {}
        }
        function loadPreference(key, defaultVal) {
          try {
            var v = localStorage.getItem('report-' + key);
            return v !== null ? JSON.parse(v) : defaultVal;
          } catch(e) { return defaultVal; }
        }
        function initPreferences() {
          var collapsed = loadPreference('collapsed-sections', []);
          collapsed.forEach(function(id) {
            var el = document.getElementById(id);
            if (el) el.classList.remove('open');
          });
          document.querySelectorAll('.collapsible-header').forEach(function(header) {
            header.addEventListener('click', function() {
              var section = this.closest('.collapsible');
              var sId = section.id;
              if (!sId) return;
              var stored = loadPreference('collapsed-sections', []);
              if (section.classList.contains('open')) {
                stored = stored.filter(function(x) { return x !== sId; });
              } else {
                if (stored.indexOf(sId) === -1) stored.push(sId);
              }
              savePreference('collapsed-sections', stored);
            });
          });
        }
        """)

    def _smooth_scroll(self) -> str:
        return textwrap.dedent("""\
        function initSmoothScroll() {
          document.querySelectorAll('a[href^="#"]').forEach(function(a) {
            a.addEventListener('click', function(e) {
              e.preventDefault();
              var target = document.querySelector(this.getAttribute('href'));
              if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                history.pushState(null, '', this.getAttribute('href'));
              }
            });
          });
        }
        """)

    def _tooltip_positioning(self) -> str:
        return textwrap.dedent("""\
        function initTooltips() {
          document.querySelectorAll('.tooltip-container').forEach(function(container) {
            container.addEventListener('mouseenter', function() {
              var tip = this.querySelector('.tooltip');
              if (!tip) return;
              var rect = this.getBoundingClientRect();
              var tipRect = tip.getBoundingClientRect();
              tip.classList.remove('tooltip-top', 'tooltip-bottom', 'tooltip-left', 'tooltip-right');
              if (rect.top > tipRect.height + 20) {
                tip.classList.add('tooltip-top');
              } else if (window.innerHeight - rect.bottom > tipRect.height + 20) {
                tip.classList.add('tooltip-bottom');
              } else if (rect.left > tipRect.width + 20) {
                tip.classList.add('tooltip-left');
              } else {
                tip.classList.add('tooltip-right');
              }
            });
          });
        }
        """)

    def _sidebar_nav(self) -> str:
        return textwrap.dedent("""\
        function initSidebarNav() {
          document.querySelectorAll('.file-tree-item').forEach(function(item) {
            item.addEventListener('click', function() {
              document.querySelectorAll('.file-tree-item.active').forEach(function(a) { a.classList.remove('active'); });
              this.classList.add('active');
              var target = this.dataset.target;
              if (target) {
                var el = document.getElementById(target);
                if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
              }
            });
          });
          var observer = new IntersectionObserver(function(entries) {
            entries.forEach(function(entry) {
              if (entry.isIntersecting) {
                var id = entry.target.id;
                document.querySelectorAll('.file-tree-item.active').forEach(function(a) { a.classList.remove('active'); });
                var item = document.querySelector('.file-tree-item[data-target="' + id + '"]');
                if (item) item.classList.add('active');
              }
            });
          }, { threshold: 0.3 });
          document.querySelectorAll('[data-nav-section]').forEach(function(section) {
            observer.observe(section);
          });
        }
        """)

    def _init(self) -> str:
        return textwrap.dedent("""\
        document.addEventListener('DOMContentLoaded', function() {
          initCollapsibles();
          initTabs();
          initSearch();
          initSortableTables();
          initThemeToggle();
          initCopyButtons();
          initKeyboardNav();
          initPreferences();
          initSmoothScroll();
          initTooltips();
          initSidebarNav();
        });
        """)

    def generate(self) -> str:
        parts = [
            self._collapsible_toggle(),
            self._tab_switching(),
            self._search_filter(),
            self._sort_tables(),
            self._theme_toggle(),
            self._copy_code(),
            self._keyboard_nav(),
            self._local_storage(),
            self._smooth_scroll(),
            self._tooltip_positioning(),
            self._sidebar_nav(),
            self._init(),
        ]
        return "\n".join(parts)


# ===================================================================
# 4. ThemeManager
# ===================================================================

@dataclass
class Theme:
    name: str = "light"
    bg: str = "#ffffff"
    fg: str = "#1e293b"
    accent: str = "#2563eb"
    error: str = "#dc2626"
    warning: str = "#d97706"
    info: str = "#2563eb"
    success: str = "#16a34a"
    code_bg: str = "#f8fafc"
    border: str = "#e2e8f0"
    sidebar_bg: str = "#f1f5f9"
    card_bg: str = "#ffffff"
    header_bg: str = "#1e293b"
    header_fg: str = "#f8fafc"
    hover_bg: str = "#f1f5f9"
    selection_bg: str = "#bfdbfe"
    tooltip_bg: str = "#1e293b"
    tooltip_fg: str = "#f8fafc"
    link_color: str = "#2563eb"

    def as_css_variables(self) -> Dict[str, str]:
        return {
            "--bg": self.bg, "--fg": self.fg, "--accent": self.accent,
            "--error": self.error, "--warning": self.warning,
            "--info": self.info, "--success": self.success,
            "--code-bg": self.code_bg, "--border": self.border,
            "--sidebar-bg": self.sidebar_bg, "--card-bg": self.card_bg,
            "--header-bg": self.header_bg, "--header-fg": self.header_fg,
            "--hover-bg": self.hover_bg, "--selection-bg": self.selection_bg,
            "--tooltip-bg": self.tooltip_bg, "--tooltip-fg": self.tooltip_fg,
            "--link-color": self.link_color,
        }


LIGHT_THEME = Theme(name="light")

DARK_THEME = Theme(
    name="dark",
    bg="#0f172a", fg="#e2e8f0", accent="#3b82f6",
    error="#f87171", warning="#fbbf24", info="#60a5fa", success="#4ade80",
    code_bg="#1e293b", border="#334155", sidebar_bg="#1e293b",
    card_bg="#1e293b", header_bg="#0f172a", header_fg="#e2e8f0",
    hover_bg="#334155", selection_bg="#1e3a5f",
    tooltip_bg="#e2e8f0", tooltip_fg="#0f172a", link_color="#60a5fa",
)


@dataclass
class ThemeManager:
    themes: Dict[str, Theme] = field(default_factory=lambda: {"light": LIGHT_THEME, "dark": DARK_THEME})
    default_theme: str = "light"

    def register_theme(self, theme: Theme) -> None:
        self.themes[theme.name] = theme

    def get_theme(self, name: str | None = None) -> Theme:
        return self.themes.get(name or self.default_theme, LIGHT_THEME)

    def get_css_variables(self, theme_name: str | None = None) -> str:
        theme = self.get_theme(theme_name)
        variables = theme.as_css_variables()
        lines = [f"  {k}: {v};" for k, v in variables.items()]
        selector = ":root" if theme.name == "light" else f'[data-theme="{theme.name}"]'
        return f"{selector} {{\n" + "\n".join(lines) + "\n}\n"

    def get_all_css(self) -> str:
        parts: List[str] = []
        for name in self.themes:
            parts.append(self.get_css_variables(name))
        return "\n".join(parts)


# ===================================================================
# 5. SyntaxHighlighter
# ===================================================================

class TokenType(enum.Enum):
    KEYWORD = "keyword"
    STRING = "string"
    NUMBER = "number"
    COMMENT = "comment"
    OPERATOR = "operator"
    BUILTIN = "builtin"
    DECORATOR = "decorator"
    CLASS_NAME = "classname"
    FUNCTION_NAME = "funcname"
    TYPE_HINT = "typehint"
    PUNCTUATION = "punctuation"
    WHITESPACE = "whitespace"
    IDENTIFIER = "variable"
    CONSTANT = "constant"
    PARAMETER = "parameter"
    PROPERTY = "property"
    REGEX = "regex"
    PLAIN = "plain"


@dataclass
class Token:
    type: TokenType
    value: str
    line: int = 0
    col: int = 0


_PYTHON_KEYWORDS: Set[str] = {
    "False", "None", "True", "and", "as", "assert", "async", "await",
    "break", "class", "continue", "def", "del", "elif", "else", "except",
    "finally", "for", "from", "global", "if", "import", "in", "is",
    "lambda", "nonlocal", "not", "or", "pass", "raise", "return",
    "try", "while", "with", "yield", "match", "case", "type",
}

_PYTHON_BUILTINS: Set[str] = {
    "abs", "all", "any", "bin", "bool", "bytes", "callable", "chr",
    "classmethod", "compile", "complex", "delattr", "dict", "dir",
    "divmod", "enumerate", "eval", "exec", "filter", "float", "format",
    "frozenset", "getattr", "globals", "hasattr", "hash", "help", "hex",
    "id", "input", "int", "isinstance", "issubclass", "iter", "len",
    "list", "locals", "map", "max", "memoryview", "min", "next",
    "object", "oct", "open", "ord", "pow", "print", "property",
    "range", "repr", "reversed", "round", "set", "setattr", "slice",
    "sorted", "staticmethod", "str", "sum", "super", "tuple", "type",
    "vars", "zip", "__import__", "NotImplemented", "Ellipsis",
    "__name__", "__file__", "__doc__",
    "ArithmeticError", "AssertionError", "AttributeError", "BaseException",
    "BlockingIOError", "BrokenPipeError", "BufferError", "BytesWarning",
    "ChildProcessError", "ConnectionAbortedError", "ConnectionError",
    "ConnectionRefusedError", "ConnectionResetError", "DeprecationWarning",
    "EOFError", "EnvironmentError", "Exception", "FileExistsError",
    "FileNotFoundError", "FloatingPointError", "FutureWarning",
    "GeneratorExit", "IOError", "ImportError", "ImportWarning",
    "IndentationError", "IndexError", "InterruptedError", "IsADirectoryError",
    "KeyError", "KeyboardInterrupt", "LookupError", "MemoryError",
    "ModuleNotFoundError", "NameError", "NotADirectoryError",
    "NotImplementedError", "OSError", "OverflowError", "PendingDeprecationWarning",
    "PermissionError", "ProcessLookupError", "RecursionError",
    "ReferenceError", "ResourceWarning", "RuntimeError", "RuntimeWarning",
    "StopAsyncIteration", "StopIteration", "SyntaxError", "SyntaxWarning",
    "SystemError", "SystemExit", "TabError", "TimeoutError", "TypeError",
    "UnboundLocalError", "UnicodeDecodeError", "UnicodeEncodeError",
    "UnicodeError", "UnicodeTranslateError", "UnicodeWarning",
    "UserWarning", "ValueError", "Warning", "ZeroDivisionError",
}

_TS_KEYWORDS: Set[str] = {
    "abstract", "any", "as", "async", "await", "bigint", "boolean",
    "break", "case", "catch", "class", "const", "continue", "debugger",
    "declare", "default", "delete", "do", "else", "enum", "export",
    "extends", "false", "finally", "for", "from", "function", "get",
    "if", "implements", "import", "in", "infer", "instanceof",
    "interface", "is", "keyof", "let", "module", "namespace", "never",
    "new", "null", "number", "object", "of", "package", "private",
    "protected", "public", "readonly", "require", "return", "set",
    "static", "string", "super", "switch", "symbol", "this", "throw",
    "true", "try", "type", "typeof", "undefined", "unique", "unknown",
    "var", "void", "while", "with", "yield",
}


@dataclass
class SyntaxHighlighter:
    _py_token_re: re.Pattern[str] = field(default=None, repr=False)  # type: ignore[assignment]
    _ts_token_re: re.Pattern[str] = field(default=None, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._py_token_re = re.compile(
            r"""(?P<decorator>@\w[\w.]*)"""
            r"""|(?P<triple_dq>\"\"\"[\s\S]*?\"\"\")"""
            r"""|(?P<triple_sq>'''[\s\S]*?''')"""
            r"""|(?P<fstring>f(?P<fq>['\"]).*?(?P=fq))"""
            r"""|(?P<string>(?:"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'))"""
            r"""|(?P<comment>\#.*)"""
            r"""|(?P<number>\b(?:0[xXoObB][\da-fA-F_]+|\d[\d_]*(?:\.[\d_]+)?(?:[eE][+-]?\d+)?j?)\b)"""
            r"""|(?P<operator>[+\-*/%=<>!&|^~@]+|\.{3})"""
            r"""|(?P<punctuation>[(){}\[\]:;,.])"""
            r"""|(?P<identifier>[A-Za-z_]\w*)"""
            r"""|(?P<whitespace>\s+)""",
            re.MULTILINE,
        )
        self._ts_token_re = re.compile(
            r"""(?P<template>`(?:[^`\\]|\\.|\$\{[^}]*\})*`)"""
            r"""|(?P<block_comment>/\*[\s\S]*?\*/)"""
            r"""|(?P<line_comment>//[^\n]*)"""
            r"""|(?P<regex>/(?![/*])(?:[^/\\]|\\.)+/[gimsuy]*)"""
            r"""|(?P<string>(?:"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'))"""
            r"""|(?P<number>\b(?:0[xXoObB][\da-fA-F_]+|\d[\d_]*(?:\.[\d_]+)?(?:[eE][+-]?\d+)?)\b)"""
            r"""|(?P<operator>[+\-*/%=<>!&|^~?]+|\.{3})"""
            r"""|(?P<punctuation>[(){}\[\]:;,.])"""
            r"""|(?P<identifier>[A-Za-z_$]\w*)"""
            r"""|(?P<whitespace>\s+)""",
            re.MULTILINE,
        )

    def _classify_python_identifier(self, word: str, prev_token: Token | None) -> TokenType:
        if word in _PYTHON_KEYWORDS:
            return TokenType.KEYWORD
        if word in _PYTHON_BUILTINS:
            return TokenType.BUILTIN
        if prev_token and prev_token.value in ("def",):
            return TokenType.FUNCTION_NAME
        if prev_token and prev_token.value in ("class",):
            return TokenType.CLASS_NAME
        if word[0].isupper():
            return TokenType.CLASS_NAME
        return TokenType.IDENTIFIER

    def _classify_ts_identifier(self, word: str, prev_token: Token | None) -> TokenType:
        if word in _TS_KEYWORDS:
            return TokenType.KEYWORD
        if prev_token and prev_token.value in ("function",):
            return TokenType.FUNCTION_NAME
        if prev_token and prev_token.value in ("class", "interface", "enum", "type"):
            return TokenType.CLASS_NAME
        if word[0].isupper():
            return TokenType.CLASS_NAME
        return TokenType.IDENTIFIER

    def tokenize(self, code: str, language: str = "python") -> List[Token]:
        pattern = self._py_token_re if language == "python" else self._ts_token_re
        tokens: List[Token] = []
        line_num = 1
        col = 0
        prev: Token | None = None
        for m in pattern.finditer(code):
            val = m.group()
            start = m.start()
            # compute line/col
            line_num = code[:start].count("\n") + 1
            last_nl = code.rfind("\n", 0, start)
            col = start - last_nl - 1 if last_nl >= 0 else start

            group = m.lastgroup
            if group == "whitespace":
                tokens.append(Token(TokenType.WHITESPACE, val, line_num, col))
                prev = tokens[-1]
                continue
            if group in ("comment", "line_comment", "block_comment"):
                tok_type = TokenType.COMMENT
            elif group in ("string", "triple_dq", "triple_sq", "fstring", "template"):
                tok_type = TokenType.STRING
            elif group == "number":
                tok_type = TokenType.NUMBER
            elif group == "operator":
                tok_type = TokenType.OPERATOR
            elif group == "punctuation":
                tok_type = TokenType.PUNCTUATION
            elif group == "decorator":
                tok_type = TokenType.DECORATOR
            elif group == "regex":
                tok_type = TokenType.REGEX
            elif group == "identifier":
                if language == "python":
                    tok_type = self._classify_python_identifier(val, prev)
                else:
                    tok_type = self._classify_ts_identifier(val, prev)
            else:
                tok_type = TokenType.PLAIN

            tok = Token(tok_type, val, line_num, col)
            tokens.append(tok)
            prev = tok
        return tokens

    def highlight(self, code: str, language: str = "python") -> str:
        tokens = self.tokenize(code, language)
        buf = io.StringIO()
        for tok in tokens:
            escaped = _escape(tok.value)
            if tok.type == TokenType.WHITESPACE:
                buf.write(escaped)
            else:
                cls = f"tok-{tok.type.value}"
                buf.write(f'<span class="{cls}">{escaped}</span>')
        return buf.getvalue()

    def highlight_with_line_numbers(self, code: str, language: str = "python", start_line: int = 1) -> str:
        highlighted = self.highlight(code, language)
        lines = highlighted.split("\n")
        line_nums_html = '<div class="line-numbers">'
        for i in range(len(lines)):
            line_nums_html += f"<span>{start_line + i}</span>"
        line_nums_html += "</div>"
        code_html = '<pre><code>' + "\n".join(lines) + "</code></pre>"
        return f'<div class="code-block">{line_nums_html}{code_html}</div>'

    @staticmethod
    def generate_line_numbers(count: int, start: int = 1) -> str:
        parts = [f"<span>{start + i}</span>" for i in range(count)]
        return '<div class="line-numbers">' + "".join(parts) + "</div>"


# ===================================================================
# 6. SourceAnnotator
# ===================================================================

@dataclass
class SourceAnnotation:
    line: int = 0
    col_start: int = 0
    col_end: int = 0
    annotation_type: str = "info"  # info | error | warning
    message: str = ""
    type_info: str = ""
    tooltip: str = ""


@dataclass
class SourceAnnotator:
    highlighter: SyntaxHighlighter = field(default_factory=SyntaxHighlighter)
    language: str = "python"

    def _wrap_annotation(self, text: str, ann: SourceAnnotation) -> str:
        cls = "wavy-error" if ann.annotation_type == "error" else (
            "wavy-warning" if ann.annotation_type == "warning" else ""
        )
        tooltip_html = ""
        if ann.tooltip or ann.message:
            tip_text = _escape(ann.tooltip or ann.message)
            tooltip_html = f'<span class="tooltip tooltip-top">{tip_text}</span>'
        type_badge = ""
        if ann.type_info:
            type_badge = f'<span class="annotation-inline">{_escape(ann.type_info)}</span>'
        return (
            f'<span class="tooltip-container {cls}">'
            f"{text}{type_badge}{tooltip_html}</span>"
        )

    def annotate(self, source_lines: List[str], annotations: List[SourceAnnotation]) -> str:
        anns_by_line: Dict[int, List[SourceAnnotation]] = collections.defaultdict(list)
        for ann in annotations:
            anns_by_line[ann.line].append(ann)

        bug_lines: Set[int] = set()
        warn_lines: Set[int] = set()
        for ann in annotations:
            if ann.annotation_type == "error":
                bug_lines.add(ann.line)
            elif ann.annotation_type == "warning":
                warn_lines.add(ann.line)

        result_lines: List[str] = []
        for idx, raw_line in enumerate(source_lines):
            line_num = idx + 1
            highlighted = self.highlighter.highlight(raw_line, self.language)

            line_anns = sorted(anns_by_line.get(line_num, []), key=lambda a: a.col_start, reverse=True)
            line_html = highlighted
            for ann in line_anns:
                if 0 <= ann.col_start < len(raw_line):
                    end = min(ann.col_end, len(raw_line)) if ann.col_end > ann.col_start else len(raw_line)
                    segment_raw = raw_line[ann.col_start:end]
                    segment_hl = self.highlighter.highlight(segment_raw, self.language)
                    wrapped = self._wrap_annotation(segment_hl, ann)
                    before_raw = raw_line[:ann.col_start]
                    after_raw = raw_line[end:]
                    before_hl = self.highlighter.highlight(before_raw, self.language)
                    after_hl = self.highlighter.highlight(after_raw, self.language)
                    line_html = before_hl + wrapped + after_hl

            gutter = ""
            line_cls = ""
            if line_num in bug_lines:
                gutter = '<span class="gutter-marker gutter-error"></span>'
                line_cls = " bug-line"
            elif line_num in warn_lines:
                gutter = '<span class="gutter-marker gutter-warning"></span>'
                line_cls = " warn-line"

            result_lines.append(
                f'<div class="source-line{line_cls}" id="L{line_num}">'
                f'<span class="line-num">{line_num}</span>'
                f'{gutter}{line_html}</div>'
            )

        return "\n".join(result_lines)


# ===================================================================
# 7. TooltipGenerator
# ===================================================================

@dataclass
class TooltipGenerator:
    position: str = "top"

    def refinement_type_tooltip(self, type_name: str, predicates: List[str], description: str = "") -> str:
        pred_html = ""
        if predicates:
            items = "".join(f"<li>{_escape(p)}</li>" for p in predicates)
            pred_html = f'<div class="tooltip-predicates"><strong>Predicates:</strong><ul>{items}</ul></div>'
        desc_html = f"<div>{_escape(description)}</div>" if description else ""
        return (
            f'<div class="tooltip-content">'
            f'<div class="tooltip-type"><strong>{_escape(type_name)}</strong></div>'
            f'{desc_html}{pred_html}'
            f'</div>'
        )

    def bug_tooltip(self, category: str, severity: str, message: str, fix_hint: str = "") -> str:
        sev_class = f"badge-{severity}" if severity in ("error", "warning", "info") else "badge-info"
        fix_html = f'<div class="tooltip-fix"><em>Fix: {_escape(fix_hint)}</em></div>' if fix_hint else ""
        return (
            f'<div class="tooltip-content">'
            f'<span class="badge {sev_class}">{_escape(severity)}</span> '
            f'<strong>{_escape(category)}</strong>'
            f'<div>{_escape(message)}</div>'
            f'{fix_html}'
            f'</div>'
        )

    def code_context_tooltip(self, code_snippet: str, language: str = "python") -> str:
        hl = SyntaxHighlighter()
        highlighted = hl.highlight(code_snippet, language)
        return (
            f'<div class="tooltip-content tooltip-code">'
            f'<pre><code>{highlighted}</code></pre>'
            f'</div>'
        )

    def predicate_explanation_tooltip(self, predicate: str, explanation: str) -> str:
        return (
            f'<div class="tooltip-content">'
            f'<div><code>{_escape(predicate)}</code></div>'
            f'<div class="text-muted">{_escape(explanation)}</div>'
            f'</div>'
        )

    def wrap_with_tooltip(self, content: str, tooltip_html: str) -> str:
        pos_class = f"tooltip-{self.position}"
        return (
            f'<span class="tooltip-container">'
            f'{content}'
            f'<span class="tooltip {pos_class}">{tooltip_html}</span>'
            f'</span>'
        )


# ===================================================================
# 8. CollapsibleSections
# ===================================================================

@dataclass
class CollapsibleSection:
    section_id: str = ""
    title: str = ""
    content: str = ""
    initially_open: bool = True
    children: List[CollapsibleSection] = field(default_factory=list)
    badge_text: str = ""
    badge_class: str = ""

    def render(self) -> str:
        open_cls = " open" if self.initially_open else ""
        sid = self.section_id or _slugify(self.title)
        badge_html = ""
        if self.badge_text:
            bcls = self.badge_class or "badge-info"
            badge_html = f' <span class="badge {bcls}">{_escape(self.badge_text)}</span>'

        children_html = ""
        if self.children:
            children_html = "\n".join(child.render() for child in self.children)

        return (
            f'<div class="collapsible{open_cls}" id="{_escape(sid)}">'
            f'<div class="collapsible-header">'
            f'<span class="collapsible-chevron"></span>'
            f'{_escape(self.title)}{badge_html}'
            f'</div>'
            f'<div class="collapsible-content">'
            f'{self.content}'
            f'{children_html}'
            f'</div>'
            f'</div>'
        )


@dataclass
class CollapsibleSections:
    sections: List[CollapsibleSection] = field(default_factory=list)

    def add(self, section: CollapsibleSection) -> None:
        self.sections.append(section)

    def render(self) -> str:
        controls = (
            '<div style="margin-bottom:0.5rem;">'
            '<button onclick="expandAll()" class="copy-btn" style="margin-right:0.25rem;">Expand All</button>'
            '<button onclick="collapseAll()" class="copy-btn">Collapse All</button>'
            '</div>'
        )
        body = "\n".join(s.render() for s in self.sections)
        return controls + body


# ===================================================================
# 9. NavigationSidebar
# ===================================================================

@dataclass
class NavFileEntry:
    path: str = ""
    display_name: str = ""
    functions: List[str] = field(default_factory=list)
    bug_count: int = 0
    children: List[NavFileEntry] = field(default_factory=list)


@dataclass
class NavigationSidebar:
    title: str = "Navigation"
    show_search: bool = True

    def _render_file_entry(self, entry: NavFileEntry) -> str:
        target = _slugify(entry.path)
        badge = ""
        if entry.bug_count > 0:
            badge = f' <span class="badge badge-count badge-error">{entry.bug_count}</span>'
        icon = "📄"
        if entry.children:
            icon = "📁"

        funcs_html = ""
        if entry.functions:
            items = "".join(
                f'<div class="file-tree-item" data-target="{_slugify(entry.path + "-" + fn)}" '
                f'data-searchable="{_escape(fn)}">'
                f'  ƒ {_escape(fn)}</div>'
                for fn in entry.functions
            )
            funcs_html = f'<div class="file-tree-children">{items}</div>'

        children_html = ""
        if entry.children:
            children_html = '<div class="file-tree-children">' + "".join(
                self._render_file_entry(c) for c in entry.children
            ) + "</div>"

        display = entry.display_name or entry.path.split("/")[-1]
        return (
            f'<div class="file-tree-item" data-target="{target}" '
            f'data-searchable="{_escape(entry.path)}">'
            f'{icon} {_escape(display)}{badge}'
            f'</div>'
            f'{funcs_html}{children_html}'
        )

    def render(self, files: List[NavFileEntry]) -> str:
        search_html = ""
        if self.show_search:
            search_html = (
                '<div style="margin-bottom:0.75rem;">'
                '<input type="text" id="sidebar-search" class="search-input" '
                'placeholder="Filter files..." oninput="filterSidebar(this.value)">'
                '</div>'
            )
        entries_html = "".join(self._render_file_entry(f) for f in files)
        filter_script = textwrap.dedent("""\
        <script>
        function filterSidebar(query) {
          document.querySelectorAll('.sidebar .file-tree-item').forEach(function(item) {
            var text = item.dataset.searchable || item.textContent;
            item.style.display = (!query || text.toLowerCase().indexOf(query.toLowerCase()) !== -1) ? '' : 'none';
          });
        }
        </script>
        """)
        return (
            f'<nav class="sidebar" aria-label="File navigation">'
            f'<h3>{_escape(self.title)}</h3>'
            f'{search_html}'
            f'<div class="file-tree">{entries_html}</div>'
            f'{filter_script}'
            f'</nav>'
        )


# ===================================================================
# 10. SearchWidget
# ===================================================================

@dataclass
class SearchWidget:
    placeholder: str = "Search functions, files, bugs..."
    widget_id: str = "search-input"

    def render(self) -> str:
        return (
            f'<div class="search-widget" style="margin-bottom:1rem;">'
            f'<input type="text" id="{_escape(self.widget_id)}" '
            f'class="search-input" placeholder="{_escape(self.placeholder)}" '
            f'aria-label="Search">'
            f'<div id="search-results-count" class="search-results-count"></div>'
            f'</div>'
        )


# ===================================================================
# 11. FileReportSection
# ===================================================================

@dataclass
class FileReportSection:
    file_path: str = ""
    source_code: str = ""
    language: str = "python"
    annotations: List[SourceAnnotation] = field(default_factory=list)
    bugs: List[Dict[str, Any]] = field(default_factory=list)
    functions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    coverage: float = 0.0

    def render(self) -> str:
        section_id = _slugify(self.file_path)
        hl = SyntaxHighlighter()
        annotator = SourceAnnotator(highlighter=hl, language=self.language)

        source_lines = self.source_code.split("\n")
        annotated_html = annotator.annotate(source_lines, self.annotations)

        # Metadata
        meta_items = "".join(
            f"<span><strong>{_escape(str(k))}:</strong> {_escape(str(v))}</span> "
            for k, v in self.metadata.items()
        )
        meta_html = f'<div class="card-subtitle">{meta_items}</div>' if self.metadata else ""

        # Coverage bar
        cov_pct = max(0.0, min(100.0, self.coverage * 100))
        cov_cls = "coverage-high" if cov_pct >= 80 else ("coverage-mid" if cov_pct >= 50 else "coverage-low")
        coverage_html = (
            f'<div style="margin-bottom:0.5rem;">'
            f'<span class="{cov_cls}">Coverage: {cov_pct:.0f}%</span>'
            f'<div class="progress-bar-bg"><div class="progress-bar-fill" style="width:{cov_pct}%"></div></div>'
            f'</div>'
        )

        # Bug table
        bug_rows = ""
        for bug in self.bugs:
            sev = bug.get("severity", "info")
            badge_cls = f"badge-{sev}"
            bug_rows += (
                f"<tr>"
                f'<td><span class="badge {badge_cls}">{_escape(sev)}</span></td>'
                f'<td>{_escape(str(bug.get("line", "")))}</td>'
                f'<td>{_escape(bug.get("category", ""))}</td>'
                f'<td>{_escape(bug.get("message", ""))}</td>'
                f"</tr>"
            )
        bug_table = ""
        if bug_rows:
            bug_table = (
                '<div class="table-wrap"><table class="table-sortable">'
                "<thead><tr>"
                '<th>Severity <span class="sort-indicator">⇅</span></th>'
                '<th>Line <span class="sort-indicator">⇅</span></th>'
                '<th>Category <span class="sort-indicator">⇅</span></th>'
                '<th>Message <span class="sort-indicator">⇅</span></th>'
                "</tr></thead><tbody>"
                f"{bug_rows}</tbody></table></div>"
            )

        # Function summary table
        func_rows = ""
        for fn in self.functions:
            func_rows += (
                f"<tr>"
                f'<td>{_escape(fn.get("name", ""))}</td>'
                f'<td><code>{_escape(fn.get("signature", ""))}</code></td>'
                f'<td>{_escape(fn.get("return_type", ""))}</td>'
                f'<td>{fn.get("bug_count", 0)}</td>'
                f"</tr>"
            )
        func_table = ""
        if func_rows:
            func_table = (
                '<div class="table-wrap"><table class="table-sortable">'
                "<thead><tr>"
                '<th>Function <span class="sort-indicator">⇅</span></th>'
                '<th>Signature <span class="sort-indicator">⇅</span></th>'
                '<th>Return Type <span class="sort-indicator">⇅</span></th>'
                '<th>Bugs <span class="sort-indicator">⇅</span></th>'
                "</tr></thead><tbody>"
                f"{func_rows}</tbody></table></div>"
            )

        # Assemble
        code_section = CollapsibleSection(
            section_id=f"{section_id}-source",
            title="Source Code",
            content=(
                f'<div class="code-block">'
                f'<div class="code-header">'
                f'<span>{_escape(self.file_path)}</span>'
                f'<button class="copy-btn">Copy</button>'
                f'</div>'
                f'<div style="position:relative">{annotated_html}</div>'
                f'</div>'
            ),
        )
        bug_section = CollapsibleSection(
            section_id=f"{section_id}-bugs",
            title=f"Bugs ({len(self.bugs)})",
            content=bug_table or '<p class="text-muted">No bugs found.</p>',
            badge_text=str(len(self.bugs)) if self.bugs else "",
            badge_class="badge-error" if self.bugs else "",
        )
        func_section = CollapsibleSection(
            section_id=f"{section_id}-functions",
            title=f"Functions ({len(self.functions)})",
            content=func_table or '<p class="text-muted">No functions.</p>',
        )

        sections = CollapsibleSections(sections=[code_section, bug_section, func_section])

        return (
            f'<section class="card fade-in" id="{section_id}" data-nav-section>'
            f'<h2 class="card-title">📄 {_escape(self.file_path)}</h2>'
            f'{meta_html}'
            f'{coverage_html}'
            f'{sections.render()}'
            f'</section>'
        )


# ===================================================================
# 12. FunctionReportSection
# ===================================================================

@dataclass
class FunctionReportSection:
    name: str = ""
    file_path: str = ""
    signature: str = ""
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    return_type: str = ""
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    cegar_iterations: List[Dict[str, Any]] = field(default_factory=list)
    abstract_states: List[Dict[str, Any]] = field(default_factory=list)

    def render(self) -> str:
        section_id = _slugify(f"{self.file_path}-{self.name}")
        hl = SyntaxHighlighter()
        sig_html = hl.highlight(self.signature, "python")

        # Params table
        param_rows = ""
        for p in self.parameters:
            param_rows += (
                f"<tr>"
                f'<td><code>{_escape(p.get("name", ""))}</code></td>'
                f'<td>{_escape(p.get("declared_type", "Any"))}</td>'
                f'<td>{_escape(p.get("inferred_type", ""))}</td>'
                f'<td>{_escape(p.get("refinement", ""))}</td>'
                f"</tr>"
            )
        param_table = ""
        if param_rows:
            param_table = (
                '<div class="table-wrap"><table>'
                "<thead><tr><th>Parameter</th><th>Declared</th><th>Inferred</th><th>Refinement</th></tr></thead>"
                f"<tbody>{param_rows}</tbody></table></div>"
            )

        # Pre/post conditions
        pre_html = ""
        if self.preconditions:
            items = "".join(f"<li><code>{_escape(p)}</code></li>" for p in self.preconditions)
            pre_html = f'<div class="panel"><div class="panel-header">Preconditions</div><ul style="padding-left:1.5rem;list-style:disc">{items}</ul></div>'
        post_html = ""
        if self.postconditions:
            items = "".join(f"<li><code>{_escape(p)}</code></li>" for p in self.postconditions)
            post_html = f'<div class="panel"><div class="panel-header">Postconditions</div><ul style="padding-left:1.5rem;list-style:disc">{items}</ul></div>'

        # CEGAR trace
        cegar_html = ""
        if self.cegar_iterations:
            viewer = CegarTraceViewer(iterations=self.cegar_iterations)
            cegar_html = viewer.render()

        # Abstract states
        states_html = ""
        if self.abstract_states:
            viewer = AbstractStateViewer()
            states_html = viewer.render(self.abstract_states)

        return (
            f'<section class="card fade-in" id="{section_id}" data-nav-section>'
            f'<h3 class="card-title">ƒ {_escape(self.name)}</h3>'
            f'<div class="code-block"><pre><code>{sig_html}</code></pre></div>'
            f'<div class="card-subtitle">Return type: <code>{_escape(self.return_type)}</code></div>'
            f'{param_table}'
            f'{pre_html}{post_html}'
            f'{cegar_html}'
            f'{states_html}'
            f'</section>'
        )


# ===================================================================
# 13. BugReportSection
# ===================================================================

@dataclass
class BugReportSection:
    bug_id: str = ""
    category: str = ""
    severity: str = "error"
    message: str = ""
    file_path: str = ""
    line: int = 0
    col: int = 0
    context_lines: List[str] = field(default_factory=list)
    context_start_line: int = 1
    trace: List[Dict[str, Any]] = field(default_factory=list)
    fix_suggestions: List[str] = field(default_factory=list)
    related_bugs: List[str] = field(default_factory=list)
    cwe: str = ""

    def render(self) -> str:
        section_id = f"bug-{self.bug_id or _stable_id(self.file_path, str(self.line), self.message)}"
        sev_cls = f"badge-{self.severity}"
        hl = SyntaxHighlighter()

        # Context code
        context_html = ""
        if self.context_lines:
            code_lines: List[str] = []
            for idx, ln in enumerate(self.context_lines):
                ln_num = self.context_start_line + idx
                highlighted = hl.highlight(ln, "python")
                is_bug_line = ln_num == self.line
                cls = ' class="bug-line"' if is_bug_line else ""
                marker = '<span class="gutter-marker gutter-error"></span>' if is_bug_line else ""
                code_lines.append(
                    f'<div{cls}><span class="line-num">{ln_num}</span>{marker}{highlighted}</div>'
                )
            context_html = (
                f'<div class="code-block"><div class="code-header">'
                f'<span>{_escape(self.file_path)}:{self.line}</span></div>'
                f'<pre style="padding-left:3.5rem;position:relative">{"".join(code_lines)}</pre></div>'
            )

        # Trace
        trace_html = ""
        if self.trace:
            steps = ""
            for step in self.trace:
                steps += (
                    f'<div class="panel" style="margin-bottom:0.5rem;">'
                    f'<strong>{_escape(step.get("file", ""))}:{step.get("line", "")}</strong> '
                    f'<span class="text-muted">{_escape(step.get("description", ""))}</span>'
                    f'</div>'
                )
            trace_html = CollapsibleSection(
                section_id=f"{section_id}-trace",
                title="Bug Path Trace",
                content=steps,
                initially_open=False,
            ).render()

        # Fix suggestions
        fix_html = ""
        if self.fix_suggestions:
            items = "".join(f"<li>{_escape(s)}</li>" for s in self.fix_suggestions)
            fix_html = f'<div class="panel"><div class="panel-header">💡 Suggested Fixes</div><ul style="padding-left:1.5rem;list-style:disc">{items}</ul></div>'

        # Related bugs
        related_html = ""
        if self.related_bugs:
            links = " ".join(f'<a href="#bug-{_escape(b)}" class="badge badge-info">{_escape(b)}</a>' for b in self.related_bugs)
            related_html = f'<div style="margin-top:0.5rem;">Related: {links}</div>'

        # CWE
        cwe_html = ""
        if self.cwe:
            cwe_url = f"https://cwe.mitre.org/data/definitions/{self.cwe.replace('CWE-', '')}.html"
            cwe_html = (
                f'<div style="margin-top:0.5rem;">'
                f'<a href="{cwe_url}" target="_blank" rel="noopener" class="badge badge-warning">'
                f'{_escape(self.cwe)}</a></div>'
            )

        return (
            f'<section class="card fade-in" id="{section_id}" data-searchable="{_escape(self.message)}">'
            f'<h3 class="card-title">'
            f'<span class="badge {sev_cls}">{_escape(self.severity)}</span> '
            f'{_escape(self.category)}</h3>'
            f'<p>{_escape(self.message)}</p>'
            f'<div class="card-subtitle">{_escape(self.file_path)} : line {self.line}</div>'
            f'{context_html}'
            f'{trace_html}'
            f'{fix_html}'
            f'{related_html}'
            f'{cwe_html}'
            f'</section>'
        )


# ===================================================================
# 14. TypeVisualization
# ===================================================================

@dataclass
class TypeNode:
    name: str = ""
    kind: str = "base"  # base | union | intersection | refinement | function | tuple | list | dict | optional
    children: List[TypeNode] = field(default_factory=list)
    predicates: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class TypeVisualization:
    max_depth: int = 10

    def render_type(self, type_info: TypeNode, depth: int = 0) -> str:
        if depth > self.max_depth:
            return '<span class="text-muted">…</span>'

        kind_cls = {
            "base": "type-base", "refinement": "type-refinement",
            "union": "type-union", "intersection": "type-intersection",
        }.get(type_info.kind, "type-base")

        predicates_html = ""
        if type_info.predicates:
            preds = ", ".join(_escape(p) for p in type_info.predicates)
            predicates_html = f' <span class="type-refinement">{{ {preds} }}</span>'

        children_html = ""
        if type_info.children:
            if type_info.kind == "union":
                separator = ' <span class="type-union">|</span> '
                parts = [self.render_type(c, depth + 1) for c in type_info.children]
                children_html = f'<div class="type-tree">{separator.join(parts)}</div>'
            elif type_info.kind == "intersection":
                separator = ' <span class="type-intersection">&amp;</span> '
                parts = [self.render_type(c, depth + 1) for c in type_info.children]
                children_html = f'<div class="type-tree">{separator.join(parts)}</div>'
            else:
                items = "".join(
                    f'<div class="type-node">{self.render_type(c, depth + 1)}</div>'
                    for c in type_info.children
                )
                children_html = f'<div class="type-tree">{items}</div>'

        if type_info.children and depth < 3:
            section = CollapsibleSection(
                section_id=f"type-{_stable_id(type_info.name, str(depth))}",
                title=f"{type_info.name}",
                content=children_html,
                initially_open=depth == 0,
            )
            return (
                f'<span class="{kind_cls}">{_escape(type_info.name)}</span>'
                f'{predicates_html}'
                f'{section.render()}'
            )

        return (
            f'<span class="{kind_cls}">{_escape(type_info.name)}</span>'
            f'{predicates_html}'
            f'{children_html}'
        )


# ===================================================================
# 15. ControlFlowGraphRenderer
# ===================================================================

@dataclass
class CFGBlock:
    block_id: str = ""
    label: str = ""
    statements: List[str] = field(default_factory=list)
    is_entry: bool = False
    is_exit: bool = False


@dataclass
class CFGEdge:
    source: str = ""
    target: str = ""
    label: str = ""
    is_back_edge: bool = False


@dataclass
class CFGData:
    blocks: List[CFGBlock] = field(default_factory=list)
    edges: List[CFGEdge] = field(default_factory=list)


@dataclass
class ControlFlowGraphRenderer:
    node_width: int = 160
    node_height: int = 60
    layer_gap: int = 90
    node_gap: int = 40
    padding: int = 40

    def _topological_layers(self, cfg: CFGData) -> List[List[str]]:
        adjacency: Dict[str, List[str]] = {b.block_id: [] for b in cfg.blocks}
        in_degree: Dict[str, int] = {b.block_id: 0 for b in cfg.blocks}
        for e in cfg.edges:
            if not e.is_back_edge:
                adjacency.setdefault(e.source, []).append(e.target)
                in_degree[e.target] = in_degree.get(e.target, 0) + 1

        layers: List[List[str]] = []
        queue = [bid for bid, deg in in_degree.items() if deg == 0]
        visited: Set[str] = set()
        while queue:
            layers.append(list(queue))
            visited.update(queue)
            next_queue: List[str] = []
            for bid in queue:
                for succ in adjacency.get(bid, []):
                    in_degree[succ] -= 1
                    if in_degree[succ] <= 0 and succ not in visited:
                        next_queue.append(succ)
                        visited.add(succ)
            queue = next_queue

        # add any remaining (e.g., in cycles)
        remaining = [b.block_id for b in cfg.blocks if b.block_id not in visited]
        if remaining:
            layers.append(remaining)
        return layers

    def render_svg(self, cfg: CFGData) -> str:
        layers = self._topological_layers(cfg)
        block_map: Dict[str, CFGBlock] = {b.block_id: b for b in cfg.blocks}

        positions: Dict[str, Tuple[int, int]] = {}
        max_width = 0
        for layer_idx, layer in enumerate(layers):
            y = self.padding + layer_idx * (self.node_height + self.layer_gap)
            total_w = len(layer) * self.node_width + (len(layer) - 1) * self.node_gap
            start_x = self.padding
            for node_idx, bid in enumerate(layer):
                x = start_x + node_idx * (self.node_width + self.node_gap)
                positions[bid] = (x, y)
                max_width = max(max_width, x + self.node_width)

        svg_w = max_width + self.padding
        svg_h = self.padding * 2 + len(layers) * (self.node_height + self.layer_gap)

        parts: List[str] = []
        parts.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w} {svg_h}" '
            f'width="{svg_w}" height="{svg_h}" style="font-family:sans-serif;font-size:12px;">'
        )
        # Arrowhead marker
        parts.append(
            '<defs>'
            '<marker id="arrow" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">'
            '<polygon points="0 0, 10 3.5, 0 7" fill="var(--fg, #333)"/>'
            '</marker>'
            '<marker id="arrow-back" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">'
            '<polygon points="0 0, 10 3.5, 0 7" fill="var(--warning, #d97706)"/>'
            '</marker>'
            '</defs>'
        )

        # Edges
        for edge in cfg.edges:
            if edge.source not in positions or edge.target not in positions:
                continue
            sx, sy = positions[edge.source]
            tx, ty = positions[edge.target]
            sx_center = sx + self.node_width // 2
            sy_bottom = sy + self.node_height
            tx_center = tx + self.node_width // 2
            ty_top = ty

            stroke_color = "var(--warning, #d97706)" if edge.is_back_edge else "var(--fg, #333)"
            marker = "url(#arrow-back)" if edge.is_back_edge else "url(#arrow)"
            dash = ' stroke-dasharray="5,3"' if edge.is_back_edge else ""

            if ty_top < sy_bottom:
                # back-edge: curve around
                mid_x = max(sx_center, tx_center) + self.node_width // 2 + 30
                parts.append(
                    f'<path d="M{sx_center},{sy_bottom} C{mid_x},{sy_bottom + 40} {mid_x},{ty_top - 40} {tx_center},{ty_top}" '
                    f'stroke="{stroke_color}" fill="none" stroke-width="1.5" marker-end="{marker}"{dash}/>'
                )
            else:
                mid_y = (sy_bottom + ty_top) // 2
                parts.append(
                    f'<path d="M{sx_center},{sy_bottom} C{sx_center},{mid_y} {tx_center},{mid_y} {tx_center},{ty_top}" '
                    f'stroke="{stroke_color}" fill="none" stroke-width="1.5" marker-end="{marker}"{dash}/>'
                )

            if edge.label:
                lx = (sx_center + tx_center) // 2
                ly = (sy_bottom + ty_top) // 2 - 5
                parts.append(
                    f'<text x="{lx}" y="{ly}" text-anchor="middle" '
                    f'fill="{stroke_color}" font-size="10">{_escape(edge.label)}</text>'
                )

        # Nodes
        for block in cfg.blocks:
            if block.block_id not in positions:
                continue
            x, y = positions[block.block_id]
            fill = "var(--code-bg, #f8fafc)"
            stroke = "var(--border, #e2e8f0)"
            rx = 6
            if block.is_entry:
                fill = "var(--badge-success-bg, #f0fdf4)"
                stroke = "var(--success, #16a34a)"
                rx = 20
            elif block.is_exit:
                fill = "var(--badge-error-bg, #fef2f2)"
                stroke = "var(--error, #dc2626)"
                rx = 20

            parts.append(
                f'<rect x="{x}" y="{y}" width="{self.node_width}" height="{self.node_height}" '
                f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
            )
            label = block.label or block.block_id
            text_y = y + 20
            parts.append(
                f'<text x="{x + self.node_width // 2}" y="{text_y}" text-anchor="middle" '
                f'font-weight="bold" fill="var(--fg, #333)">{_escape(label)}</text>'
            )
            if block.statements:
                for si, stmt in enumerate(block.statements[:2]):
                    stmt_y = text_y + 14 + si * 14
                    truncated = stmt if len(stmt) <= 20 else stmt[:17] + "..."
                    parts.append(
                        f'<text x="{x + self.node_width // 2}" y="{stmt_y}" text-anchor="middle" '
                        f'font-size="10" fill="var(--fg, #666)">{_escape(truncated)}</text>'
                    )

        parts.append("</svg>")
        return "\n".join(parts)


# ===================================================================
# 16. AbstractStateViewer
# ===================================================================

@dataclass
class AbstractStateViewer:
    highlight_changes: bool = True

    def render(self, states: List[Dict[str, Any]]) -> str:
        if not states:
            return '<p class="text-muted">No abstract states available.</p>'

        all_vars: List[str] = []
        seen: Set[str] = set()
        for state in states:
            for v in state.get("variables", {}):
                if v not in seen:
                    all_vars.append(v)
                    seen.add(v)

        header = "<th>Program Point</th>" + "".join(f"<th>{_escape(v)}</th>" for v in all_vars)
        rows: List[str] = []
        prev_values: Dict[str, str] = {}
        for state in states:
            pp = state.get("program_point", "")
            variables = state.get("variables", {})
            cells: List[str] = [f"<td><strong>{_escape(str(pp))}</strong></td>"]
            for v in all_vars:
                val = str(variables.get(v, "⊥"))
                changed = self.highlight_changes and prev_values.get(v) is not None and prev_values.get(v) != val
                cls = ' class="diff-change"' if changed else ""
                cells.append(f"<td{cls}><code>{_escape(val)}</code></td>")
                prev_values[v] = val
            rows.append("<tr>" + "".join(cells) + "</tr>")

        section = CollapsibleSection(
            section_id=f"abstract-states-{_stable_id(*[str(s.get('program_point', '')) for s in states[:3]])}",
            title="Abstract States",
            content=(
                f'<div class="table-wrap"><table>'
                f"<thead><tr>{header}</tr></thead>"
                f'<tbody>{"".join(rows)}</tbody>'
                f"</table></div>"
            ),
            initially_open=False,
        )
        return section.render()


# ===================================================================
# 17. CegarTraceViewer
# ===================================================================

@dataclass
class CegarTraceViewer:
    iterations: List[Dict[str, Any]] = field(default_factory=list)

    def render(self) -> str:
        if not self.iterations:
            return '<p class="text-muted">No CEGAR iterations.</p>'

        rows: List[str] = []
        for it in self.iterations:
            num = it.get("iteration", 0)
            predicates = it.get("predicates_added", [])
            result = it.get("result", "unknown")
            counterexample = it.get("counterexample", "")

            result_badge_cls = "badge-success" if result == "safe" else (
                "badge-error" if result == "unsafe" else "badge-info"
            )
            preds_html = ", ".join(f"<code>{_escape(p)}</code>" for p in predicates) if predicates else "—"
            ce_html = ""
            if counterexample:
                ce_html = (
                    f'<div class="panel" style="margin-top:0.25rem;font-size:0.85rem">'
                    f'<strong>Counterexample:</strong> <code>{_escape(str(counterexample))}</code></div>'
                )

            rows.append(
                f"<tr>"
                f"<td>{num}</td>"
                f"<td>{preds_html}</td>"
                f'<td><span class="badge {result_badge_cls}">{_escape(result)}</span></td>'
                f"<td>{ce_html}</td>"
                f"</tr>"
            )

        converged = self.iterations[-1].get("result", "") == "safe" if self.iterations else False
        indicator = (
            '<span class="badge badge-success">✓ Converged</span>'
            if converged else
            '<span class="badge badge-warning">⋯ Did not converge</span>'
        )

        section = CollapsibleSection(
            section_id=f"cegar-trace-{_stable_id(str(len(self.iterations)))}",
            title=f"CEGAR Iterations ({len(self.iterations)})",
            content=(
                f'<div style="margin-bottom:0.5rem">{indicator}</div>'
                f'<div class="table-wrap"><table>'
                f"<thead><tr><th>#</th><th>Predicates Added</th><th>Result</th><th>Details</th></tr></thead>"
                f'<tbody>{"".join(rows)}</tbody>'
                f"</table></div>"
            ),
            initially_open=False,
        )
        return section.render()


# ===================================================================
# 18. DependencyGraphViewer
# ===================================================================

@dataclass
class DepNode:
    name: str = ""
    scc_id: int = -1


@dataclass
class DepEdge:
    source: str = ""
    target: str = ""


@dataclass
class DependencyGraphViewer:
    node_radius: int = 30
    layer_gap: int = 120
    node_gap: int = 80
    padding: int = 50

    def _compute_layers(self, nodes: List[DepNode], edges: List[DepEdge]) -> List[List[str]]:
        names = {n.name for n in nodes}
        adj: Dict[str, List[str]] = {n.name: [] for n in nodes}
        in_deg: Dict[str, int] = {n.name: 0 for n in nodes}
        for e in edges:
            if e.source in names and e.target in names:
                adj[e.source].append(e.target)
                in_deg[e.target] = in_deg.get(e.target, 0) + 1

        layers: List[List[str]] = []
        visited: Set[str] = set()
        queue = [n for n, d in in_deg.items() if d == 0]
        while queue:
            layers.append(queue)
            visited.update(queue)
            nxt: List[str] = []
            for n in queue:
                for s in adj.get(n, []):
                    in_deg[s] -= 1
                    if in_deg[s] <= 0 and s not in visited:
                        nxt.append(s)
                        visited.add(s)
            queue = nxt
        rem = [n.name for n in nodes if n.name not in visited]
        if rem:
            layers.append(rem)
        return layers

    def render_svg(self, nodes: List[DepNode], edges: List[DepEdge]) -> str:
        layers = self._compute_layers(nodes, edges)
        scc_map: Dict[str, int] = {n.name: n.scc_id for n in nodes}

        scc_colors = [
            "#3b82f6", "#ef4444", "#22c55e", "#f59e0b",
            "#8b5cf6", "#ec4899", "#14b8a6", "#f97316",
        ]

        positions: Dict[str, Tuple[int, int]] = {}
        max_x = 0
        for li, layer in enumerate(layers):
            y = self.padding + li * self.layer_gap
            for ni, name in enumerate(layer):
                x = self.padding + ni * self.node_gap + self.node_radius
                positions[name] = (x, y)
                max_x = max(max_x, x + self.node_radius)

        svg_w = max_x + self.padding
        svg_h = self.padding * 2 + len(layers) * self.layer_gap

        parts: List[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w} {svg_h}" '
            f'width="{svg_w}" height="{svg_h}" style="font-family:sans-serif;font-size:11px;">',
            '<defs><marker id="dep-arrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">'
            '<polygon points="0 0, 8 3, 0 6" fill="var(--fg, #333)"/></marker></defs>',
        ]

        for e in edges:
            if e.source in positions and e.target in positions:
                sx, sy = positions[e.source]
                tx, ty = positions[e.target]
                parts.append(
                    f'<line x1="{sx}" y1="{sy + self.node_radius}" '
                    f'x2="{tx}" y2="{ty - self.node_radius}" '
                    f'stroke="var(--fg, #999)" stroke-width="1" marker-end="url(#dep-arrow)"/>'
                )

        for node in nodes:
            if node.name not in positions:
                continue
            x, y = positions[node.name]
            color = scc_colors[node.scc_id % len(scc_colors)] if node.scc_id >= 0 else "var(--code-bg, #f8fafc)"
            stroke = scc_colors[node.scc_id % len(scc_colors)] if node.scc_id >= 0 else "var(--border, #ccc)"
            parts.append(
                f'<circle cx="{x}" cy="{y}" r="{self.node_radius}" '
                f'fill="{color}" fill-opacity="0.15" stroke="{stroke}" stroke-width="2"/>'
            )
            parts.append(
                f'<text x="{x}" y="{y + 4}" text-anchor="middle" fill="var(--fg, #333)" '
                f'font-size="10">{_escape(node.name)}</text>'
            )

        parts.append("</svg>")
        return "\n".join(parts)


# ===================================================================
# 19. CoverageHeatmap
# ===================================================================

@dataclass
class CoverageEntry:
    name: str = ""
    status: str = "analyzed"  # analyzed | skipped | timeout
    confidence: float = 1.0


@dataclass
class CoverageHeatmap:

    def _color_for_entry(self, entry: CoverageEntry) -> str:
        if entry.status == "skipped":
            return "var(--border, #e2e8f0)"
        if entry.status == "timeout":
            return "var(--warning, #d97706)"
        r = int(220 - 180 * entry.confidence)
        g = int(60 + 180 * entry.confidence)
        return f"rgb({r}, {g}, 80)"

    def render(self, coverage: List[CoverageEntry]) -> str:
        if not coverage:
            return '<p class="text-muted">No coverage data.</p>'

        cells: List[str] = []
        analyzed = sum(1 for c in coverage if c.status == "analyzed")
        total = len(coverage)
        for entry in coverage:
            color = self._color_for_entry(entry)
            tooltip_text = f"{entry.name}: {entry.status} (conf: {entry.confidence:.0%})"
            cells.append(
                f'<span class="tooltip-container" style="display:inline-block;width:24px;height:24px;'
                f'background:{color};border-radius:3px;margin:2px;cursor:default;" '
                f'title="{_escape(tooltip_text)}">'
                f'<span class="tooltip tooltip-top">{_escape(tooltip_text)}</span></span>'
            )

        pct = (analyzed / total * 100) if total > 0 else 0
        return (
            f'<div class="card">'
            f'<div class="card-title">Analysis Coverage</div>'
            f'<p>{analyzed}/{total} functions analyzed ({pct:.0f}%)</p>'
            f'<div style="display:flex;flex-wrap:wrap;margin-top:0.5rem;">{"".join(cells)}</div>'
            f'<div style="margin-top:0.5rem;font-size:0.8rem;" class="text-muted">'
            f'Green = high confidence, Yellow = timeout, Gray = skipped</div>'
            f'</div>'
        )


# ===================================================================
# 20. StatisticsSection
# ===================================================================

@dataclass
class StatisticsSection:

    def _bar_chart_svg(self, data: Dict[str, int], title: str = "", width: int = 400, height: int = 200) -> str:
        if not data:
            return ""
        max_val = max(data.values()) if data.values() else 1
        bar_w = max(20, (width - 60) // len(data) - 10)
        colors = ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#8b5cf6", "#ec4899", "#14b8a6", "#f97316"]

        parts: List[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}" style="font-family:sans-serif;">'
        ]
        if title:
            parts.append(f'<text x="{width // 2}" y="16" text-anchor="middle" font-weight="bold" font-size="13" fill="var(--fg, #333)">{_escape(title)}</text>')

        chart_top = 30
        chart_bottom = height - 30
        chart_height = chart_bottom - chart_top

        for idx, (label, value) in enumerate(data.items()):
            x = 50 + idx * (bar_w + 10)
            bar_h = (value / max_val) * chart_height if max_val > 0 else 0
            y = chart_bottom - bar_h
            color = colors[idx % len(colors)]
            parts.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bar_h}" fill="{color}" rx="3"/>')
            parts.append(f'<text x="{x + bar_w // 2}" y="{y - 4}" text-anchor="middle" font-size="10" fill="var(--fg, #333)">{value}</text>')
            parts.append(f'<text x="{x + bar_w // 2}" y="{chart_bottom + 14}" text-anchor="middle" font-size="9" fill="var(--fg, #666)">{_escape(label[:10])}</text>')

        # Axis
        parts.append(f'<line x1="45" y1="{chart_top}" x2="45" y2="{chart_bottom}" stroke="var(--border, #ccc)" stroke-width="1"/>')
        parts.append(f'<line x1="45" y1="{chart_bottom}" x2="{width - 10}" y2="{chart_bottom}" stroke="var(--border, #ccc)" stroke-width="1"/>')

        parts.append("</svg>")
        return "\n".join(parts)

    def render(self, stats: Dict[str, Any]) -> str:
        total_funcs = stats.get("total_functions", 0)
        total_time = stats.get("total_time_seconds", 0)
        total_bugs = stats.get("total_bugs", 0)
        total_predicates = stats.get("total_predicates", 0)
        avg_cegar = stats.get("avg_cegar_iterations", 0)
        bugs_by_category: Dict[str, int] = stats.get("bugs_by_category", {})
        cegar_per_func: Dict[str, int] = stats.get("cegar_per_function", {})

        summary_cards = (
            '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:0.75rem;margin-bottom:1rem;">'
            f'<div class="panel" style="text-align:center"><div style="font-size:1.5rem;font-weight:700">{total_funcs}</div><div class="text-muted">Functions Analyzed</div></div>'
            f'<div class="panel" style="text-align:center"><div style="font-size:1.5rem;font-weight:700">{total_bugs}</div><div class="text-muted">Bugs Found</div></div>'
            f'<div class="panel" style="text-align:center"><div style="font-size:1.5rem;font-weight:700">{total_predicates}</div><div class="text-muted">Predicates</div></div>'
            f'<div class="panel" style="text-align:center"><div style="font-size:1.5rem;font-weight:700">{total_time:.1f}s</div><div class="text-muted">Total Time</div></div>'
            f'<div class="panel" style="text-align:center"><div style="font-size:1.5rem;font-weight:700">{avg_cegar:.1f}</div><div class="text-muted">Avg CEGAR Iters</div></div>'
            '</div>'
        )

        bug_chart = ""
        if bugs_by_category:
            bug_chart = (
                f'<div class="panel">'
                f'{self._bar_chart_svg(bugs_by_category, "Bugs by Category", 500, 220)}'
                f'</div>'
            )

        cegar_chart = ""
        if cegar_per_func:
            top_funcs = dict(sorted(cegar_per_func.items(), key=lambda x: x[1], reverse=True)[:8])
            cegar_chart = (
                f'<div class="panel">'
                f'{self._bar_chart_svg(top_funcs, "CEGAR Iterations per Function", 500, 220)}'
                f'</div>'
            )

        return (
            f'<section class="card" id="statistics" data-nav-section>'
            f'<h2 class="card-title">📊 Analysis Statistics</h2>'
            f'{summary_cards}'
            f'<div class="two-col">{bug_chart}{cegar_chart}</div>'
            f'</section>'
        )


# ===================================================================
# 21. DiffViewer
# ===================================================================

@dataclass
class DiffViewer:

    def _diff_items(
        self,
        old_items: Dict[str, Any],
        new_items: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Tuple[Any, Any]]]:
        added = {k: v for k, v in new_items.items() if k not in old_items}
        removed = {k: v for k, v in old_items.items() if k not in new_items}
        changed = {
            k: (old_items[k], new_items[k])
            for k in old_items
            if k in new_items and old_items[k] != new_items[k]
        }
        return added, removed, changed

    def render(self, old_results: Dict[str, Any], new_results: Dict[str, Any]) -> str:
        old_types: Dict[str, Any] = old_results.get("types", {})
        new_types: Dict[str, Any] = new_results.get("types", {})
        old_bugs_raw: List[Dict[str, Any]] = old_results.get("bugs", [])
        new_bugs_raw: List[Dict[str, Any]] = new_results.get("bugs", [])

        old_bugs = {b.get("id", str(i)): b for i, b in enumerate(old_bugs_raw)}
        new_bugs = {b.get("id", str(i)): b for i, b in enumerate(new_bugs_raw)}

        type_added, type_removed, type_changed = self._diff_items(old_types, new_types)
        bug_added, bug_removed, bug_changed = self._diff_items(old_bugs, new_bugs)

        rows: List[str] = []
        for k, v in type_added.items():
            rows.append(f'<tr class="diff-add"><td>+</td><td>{_escape(k)}</td><td></td><td>{_escape(str(v))}</td></tr>')
        for k, v in type_removed.items():
            rows.append(f'<tr class="diff-del"><td>-</td><td>{_escape(k)}</td><td>{_escape(str(v))}</td><td></td></tr>')
        for k, (old_v, new_v) in type_changed.items():
            rows.append(f'<tr class="diff-change"><td>~</td><td>{_escape(k)}</td><td>{_escape(str(old_v))}</td><td>{_escape(str(new_v))}</td></tr>')

        type_table = (
            '<h4>Type Changes</h4>'
            '<div class="table-wrap"><table>'
            '<thead><tr><th></th><th>Name</th><th>Old</th><th>New</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>'
        ) if rows else '<p class="text-muted">No type changes.</p>'

        bug_rows: List[str] = []
        for k, v in bug_added.items():
            bug_rows.append(
                f'<tr class="diff-add"><td><span class="badge badge-error">New Bug</span></td>'
                f'<td>{_escape(v.get("message", k))}</td></tr>'
            )
        for k, v in bug_removed.items():
            bug_rows.append(
                f'<tr class="diff-del"><td><span class="badge badge-success">Fixed</span></td>'
                f'<td>{_escape(v.get("message", k))}</td></tr>'
            )

        bug_table = (
            '<h4>Bug Changes</h4>'
            '<div class="table-wrap"><table>'
            '<thead><tr><th>Status</th><th>Description</th></tr></thead>'
            f'<tbody>{"".join(bug_rows)}</tbody></table></div>'
        ) if bug_rows else '<p class="text-muted">No bug changes.</p>'

        summary_added = len(type_added) + len(bug_added)
        summary_removed = len(type_removed) + len(bug_removed)
        summary_changed = len(type_changed) + len(bug_changed)

        return (
            f'<section class="card" id="diff-viewer">'
            f'<h2 class="card-title">🔄 Diff: Old vs New Analysis</h2>'
            f'<div style="margin-bottom:0.5rem;">'
            f'<span class="badge badge-success">+{summary_added} added</span> '
            f'<span class="badge badge-error">-{summary_removed} removed</span> '
            f'<span class="badge badge-warning">~{summary_changed} changed</span>'
            f'</div>'
            f'{type_table}'
            f'{bug_table}'
            f'</section>'
        )


# ===================================================================
# 22. ProgressIndicator
# ===================================================================

@dataclass
class ProgressPhase:
    name: str = ""
    status: str = "pending"  # pending | running | done | error
    progress: float = 0.0


@dataclass
class ProgressIndicator:
    width: int = 500
    height: int = 60

    def _phase_icon(self, status: str) -> str:
        icons = {"pending": "○", "running": "◉", "done": "✓", "error": "✗"}
        return icons.get(status, "○")

    def _phase_color(self, status: str) -> str:
        colors = {
            "pending": "var(--border, #ccc)",
            "running": "var(--accent, #2563eb)",
            "done": "var(--success, #16a34a)",
            "error": "var(--error, #dc2626)",
        }
        return colors.get(status, "var(--border, #ccc)")

    def render(self, progress: List[ProgressPhase]) -> str:
        if not progress:
            return ""

        total_done = sum(1 for p in progress if p.status == "done")
        overall = total_done / len(progress) if progress else 0

        # SVG progress bar
        bar_y = 15
        bar_h = 8
        bar_x = 10
        bar_w = self.width - 20

        parts: List[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}" '
            f'width="{self.width}" height="{self.height}" style="font-family:sans-serif;">',
            f'<rect x="{bar_x}" y="{bar_y}" width="{bar_w}" height="{bar_h}" rx="4" fill="var(--border, #e2e8f0)"/>',
            f'<rect x="{bar_x}" y="{bar_y}" width="{bar_w * overall}" height="{bar_h}" rx="4" fill="var(--accent, #2563eb)"/>',
        ]

        phase_w = bar_w / len(progress)
        for idx, phase in enumerate(progress):
            cx = bar_x + phase_w * idx + phase_w / 2
            color = self._phase_color(phase.status)
            icon = self._phase_icon(phase.status)
            parts.append(
                f'<circle cx="{cx}" cy="{bar_y + bar_h / 2}" r="6" fill="white" stroke="{color}" stroke-width="2"/>'
            )
            parts.append(
                f'<text x="{cx}" y="{bar_y + bar_h / 2 + 4}" text-anchor="middle" font-size="8" fill="{color}">{icon}</text>'
            )
            parts.append(
                f'<text x="{cx}" y="{bar_y + bar_h + 20}" text-anchor="middle" font-size="9" fill="var(--fg, #333)">{_escape(phase.name)}</text>'
            )

        parts.append("</svg>")
        svg = "\n".join(parts)

        return (
            f'<div class="panel">'
            f'<div class="panel-header">Analysis Progress ({overall:.0%})</div>'
            f'{svg}'
            f'</div>'
        )


# ===================================================================
# 23. ComparisonView
# ===================================================================

@dataclass
class ComparisonView:

    def render(self, old: Dict[str, Any], new: Dict[str, Any]) -> str:
        old_html = self._render_column(old, "Previous")
        new_html = self._render_column(new, "Current")
        return (
            f'<section class="card" id="comparison-view">'
            f'<h2 class="card-title">⚖ Comparison View</h2>'
            f'<div class="two-col">{old_html}{new_html}</div>'
            f'</section>'
        )

    def _render_column(self, data: Dict[str, Any], label: str) -> str:
        functions = data.get("functions", [])
        bugs = data.get("bugs", [])
        types_count = len(data.get("types", {}))

        func_list = "".join(f"<li>{_escape(f)}</li>" for f in functions[:20])
        bug_list = "".join(
            f'<li><span class="badge badge-{b.get("severity", "info")}">'
            f'{_escape(b.get("severity", ""))}</span> {_escape(b.get("message", ""))}</li>'
            for b in bugs[:20]
        )

        return (
            f'<div class="panel">'
            f'<div class="panel-header">{_escape(label)}</div>'
            f'<p><strong>Functions:</strong> {len(functions)}</p>'
            f'<p><strong>Types inferred:</strong> {types_count}</p>'
            f'<p><strong>Bugs:</strong> {len(bugs)}</p>'
            f'<ul style="padding-left:1.5rem;list-style:disc">{func_list}</ul>'
            f'{("<h4>Bugs</h4><ul style=padding-left:1.5rem;list-style:disc>" + bug_list + "</ul>") if bug_list else ""}'
            f'</div>'
        )


# ===================================================================
# 24. ExportOptions
# ===================================================================

@dataclass
class ExportOptions:

    def _print_css(self) -> str:
        return textwrap.dedent("""\
        @media print {
          body { font-size: 10pt; color: #000; background: #fff; }
          .sidebar, .header, .footer, .copy-btn, .theme-toggle,
          .search-widget, .export-buttons, .tab-nav, .collapsible-chevron { display: none !important; }
          .main-content { max-width: 100% !important; padding: 0 !important; }
          .page-wrapper { display: block !important; }
          .collapsible-content { display: block !important; }
          .card { box-shadow: none !important; border: 1px solid #ccc !important; break-inside: avoid; }
          a[href]::after { content: " (" attr(href) ")"; font-size: 0.8em; color: #666; }
          .code-block { border: 1px solid #ccc !important; font-size: 8pt; }
          pre { white-space: pre-wrap !important; word-wrap: break-word !important; }
        }
        """)

    def render_export_buttons(self) -> str:
        return (
            '<div class="export-buttons" style="display:flex;gap:0.5rem;margin-bottom:1rem;">'
            '<button class="copy-btn" onclick="window.print()">🖨 Print / PDF</button>'
            '<button class="copy-btn" onclick="exportJSON()">📋 Export JSON</button>'
            '</div>'
            '<script>'
            'function exportJSON() {'
            '  var data = document.getElementById("report-data");'
            '  if (data) {'
            '    var blob = new Blob([data.textContent], {type: "application/json"});'
            '    var url = URL.createObjectURL(blob);'
            '    var a = document.createElement("a"); a.href = url; a.download = "report.json";'
            '    document.body.appendChild(a); a.click(); document.body.removeChild(a);'
            '    URL.revokeObjectURL(url);'
            '  }'
            '}'
            '</script>'
        )

    def get_print_css(self) -> str:
        return self._print_css()


# ===================================================================
# 25. HtmlReportGenerator – main orchestrator
# ===================================================================

@dataclass
class HtmlReportGenerator:
    title: str = "Refinement Type Inference Report"
    theme: str = "light"
    show_sidebar: bool = True
    show_statistics: bool = True
    show_diff: bool = False
    old_results: Dict[str, Any] = field(default_factory=dict)

    _css_gen: CssGenerator = field(default_factory=CssGenerator, repr=False)
    _js_gen: JavaScriptGenerator = field(default_factory=JavaScriptGenerator, repr=False)
    _theme_mgr: ThemeManager = field(default_factory=ThemeManager, repr=False)
    _export: ExportOptions = field(default_factory=ExportOptions, repr=False)

    def _build_head(self) -> str:
        css = self._css_gen.generate()
        return (
            '<head>'
            f'<meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{_escape(self.title)}</title>'
            f'<style>{css}</style>'
            '</head>'
        )

    def _build_header(self) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return (
            '<header class="header">'
            f'<h1>{_escape(self.title)}</h1>'
            '<div style="display:flex;align-items:center;gap:0.75rem;">'
            f'<span class="text-muted">{now}</span>'
            '<button id="theme-toggle" class="theme-toggle">🌙 Dark</button>'
            '</div>'
            '</header>'
        )

    def _build_sidebar(self, analysis_results: Dict[str, Any]) -> str:
        files_data = analysis_results.get("files", [])
        nav_entries: List[NavFileEntry] = []
        for fd in files_data:
            fpath = fd.get("path", "unknown")
            funcs = [f.get("name", "") for f in fd.get("functions", [])]
            bug_count = len(fd.get("bugs", []))
            nav_entries.append(NavFileEntry(
                path=fpath, display_name=fpath.split("/")[-1],
                functions=funcs, bug_count=bug_count,
            ))
        sidebar = NavigationSidebar()
        return sidebar.render(nav_entries)

    def _build_file_sections(self, analysis_results: Dict[str, Any]) -> str:
        files_data = analysis_results.get("files", [])
        parts: List[str] = []
        for fd in files_data:
            annotations: List[SourceAnnotation] = []
            for ann in fd.get("annotations", []):
                annotations.append(SourceAnnotation(
                    line=ann.get("line", 0),
                    col_start=ann.get("col_start", 0),
                    col_end=ann.get("col_end", 0),
                    annotation_type=ann.get("type", "info"),
                    message=ann.get("message", ""),
                    type_info=ann.get("type_info", ""),
                    tooltip=ann.get("tooltip", ""),
                ))
            section = FileReportSection(
                file_path=fd.get("path", ""),
                source_code=fd.get("source", ""),
                language=fd.get("language", "python"),
                annotations=annotations,
                bugs=fd.get("bugs", []),
                functions=fd.get("functions", []),
                metadata=fd.get("metadata", {}),
                coverage=fd.get("coverage", 0.0),
            )
            parts.append(section.render())

            for fn in fd.get("functions", []):
                fn_section = FunctionReportSection(
                    name=fn.get("name", ""),
                    file_path=fd.get("path", ""),
                    signature=fn.get("signature", ""),
                    parameters=fn.get("parameters", []),
                    return_type=fn.get("return_type", ""),
                    preconditions=fn.get("preconditions", []),
                    postconditions=fn.get("postconditions", []),
                    cegar_iterations=fn.get("cegar_iterations", []),
                    abstract_states=fn.get("abstract_states", []),
                )
                parts.append(fn_section.render())

        return "\n".join(parts)

    def _build_bug_sections(self, analysis_results: Dict[str, Any]) -> str:
        all_bugs: List[Dict[str, Any]] = analysis_results.get("bugs", [])
        for fd in analysis_results.get("files", []):
            for bug in fd.get("bugs", []):
                enriched = dict(bug)
                enriched.setdefault("file_path", fd.get("path", ""))
                all_bugs.append(enriched)

        if not all_bugs:
            return '<section class="card"><h2 class="card-title">🐛 Bugs</h2><p class="text-muted">No bugs found. 🎉</p></section>'

        parts: List[str] = [f'<h2 id="bugs-section" data-nav-section>🐛 Bugs ({len(all_bugs)})</h2>']
        for bug in all_bugs:
            section = BugReportSection(
                bug_id=bug.get("id", ""),
                category=bug.get("category", ""),
                severity=bug.get("severity", "error"),
                message=bug.get("message", ""),
                file_path=bug.get("file_path", ""),
                line=bug.get("line", 0),
                col=bug.get("col", 0),
                context_lines=bug.get("context_lines", []),
                context_start_line=bug.get("context_start_line", 1),
                trace=bug.get("trace", []),
                fix_suggestions=bug.get("fix_suggestions", []),
                related_bugs=bug.get("related_bugs", []),
                cwe=bug.get("cwe", ""),
            )
            parts.append(section.render())
        return "\n".join(parts)

    def _build_statistics(self, analysis_results: Dict[str, Any]) -> str:
        stats_data = analysis_results.get("statistics", {})
        if not stats_data:
            files = analysis_results.get("files", [])
            total_funcs = sum(len(f.get("functions", [])) for f in files)
            total_bugs = sum(len(f.get("bugs", [])) for f in files) + len(analysis_results.get("bugs", []))
            stats_data = {
                "total_functions": total_funcs,
                "total_bugs": total_bugs,
                "total_time_seconds": analysis_results.get("total_time", 0),
                "total_predicates": analysis_results.get("total_predicates", 0),
                "avg_cegar_iterations": analysis_results.get("avg_cegar", 0),
                "bugs_by_category": analysis_results.get("bugs_by_category", {}),
                "cegar_per_function": analysis_results.get("cegar_per_function", {}),
            }
        section = StatisticsSection()
        return section.render(stats_data)

    def _build_coverage(self, analysis_results: Dict[str, Any]) -> str:
        cov_data = analysis_results.get("coverage", [])
        entries: List[CoverageEntry] = []
        for c in cov_data:
            entries.append(CoverageEntry(
                name=c.get("name", ""),
                status=c.get("status", "analyzed"),
                confidence=c.get("confidence", 1.0),
            ))
        if not entries:
            files = analysis_results.get("files", [])
            for fd in files:
                for fn in fd.get("functions", []):
                    entries.append(CoverageEntry(
                        name=fn.get("name", "unknown"),
                        status="analyzed",
                        confidence=fd.get("coverage", 0.8),
                    ))
        heatmap = CoverageHeatmap()
        return heatmap.render(entries)

    def _build_progress(self, analysis_results: Dict[str, Any]) -> str:
        phases_data = analysis_results.get("progress", [])
        phases: List[ProgressPhase] = []
        if phases_data:
            for p in phases_data:
                phases.append(ProgressPhase(
                    name=p.get("name", ""),
                    status=p.get("status", "done"),
                    progress=p.get("progress", 0.0),
                ))
        else:
            phases = [
                ProgressPhase(name="Parse", status="done", progress=1.0),
                ProgressPhase(name="CFG Build", status="done", progress=1.0),
                ProgressPhase(name="Type Inference", status="done", progress=1.0),
                ProgressPhase(name="CEGAR", status="done", progress=1.0),
                ProgressPhase(name="Bug Detection", status="done", progress=1.0),
            ]
        indicator = ProgressIndicator()
        return indicator.render(phases)

    def _build_diff(self, analysis_results: Dict[str, Any]) -> str:
        if not self.show_diff or not self.old_results:
            return ""
        diff_viewer = DiffViewer()
        return diff_viewer.render(self.old_results, analysis_results)

    def _build_cfg_section(self, analysis_results: Dict[str, Any]) -> str:
        cfgs = analysis_results.get("cfgs", [])
        if not cfgs:
            return ""
        parts: List[str] = ['<h2 id="cfg-section" data-nav-section>🔀 Control Flow Graphs</h2>']
        renderer = ControlFlowGraphRenderer()
        for cfg_data in cfgs:
            blocks = [
                CFGBlock(
                    block_id=b.get("id", ""),
                    label=b.get("label", ""),
                    statements=b.get("statements", []),
                    is_entry=b.get("is_entry", False),
                    is_exit=b.get("is_exit", False),
                )
                for b in cfg_data.get("blocks", [])
            ]
            edges = [
                CFGEdge(
                    source=e.get("source", ""),
                    target=e.get("target", ""),
                    label=e.get("label", ""),
                    is_back_edge=e.get("is_back_edge", False),
                )
                for e in cfg_data.get("edges", [])
            ]
            cfg = CFGData(blocks=blocks, edges=edges)
            svg = renderer.render_svg(cfg)
            section = CollapsibleSection(
                section_id=f"cfg-{_slugify(cfg_data.get('name', 'unknown'))}",
                title=f"CFG: {cfg_data.get('name', 'Unknown')}",
                content=f'<div style="overflow-x:auto;">{svg}</div>',
                initially_open=False,
            )
            parts.append(section.render())
        return "\n".join(parts)

    def _build_dep_graph(self, analysis_results: Dict[str, Any]) -> str:
        deps = analysis_results.get("dependencies", {})
        if not deps:
            return ""
        nodes_data = deps.get("nodes", [])
        edges_data = deps.get("edges", [])
        nodes = [DepNode(name=n.get("name", ""), scc_id=n.get("scc_id", -1)) for n in nodes_data]
        edges = [DepEdge(source=e.get("source", ""), target=e.get("target", "")) for e in edges_data]
        viewer = DependencyGraphViewer()
        svg = viewer.render_svg(nodes, edges)
        section = CollapsibleSection(
            section_id="dep-graph",
            title="Function Dependency Graph",
            content=f'<div style="overflow-x:auto;">{svg}</div>',
            initially_open=False,
        )
        return f'<div id="dep-graph-section" data-nav-section>{section.render()}</div>'

    def _build_types_section(self, analysis_results: Dict[str, Any]) -> str:
        types_data = analysis_results.get("types", {})
        if not types_data:
            return ""
        viz = TypeVisualization()
        parts: List[str] = ['<h2 id="types-section" data-nav-section>📐 Inferred Types</h2>']
        for name, type_info in types_data.items():
            node = self._build_type_node(type_info)
            rendered = viz.render_type(node)
            parts.append(
                f'<div class="card" data-searchable="{_escape(name)}">'
                f'<h4>{_escape(name)}</h4>{rendered}</div>'
            )
        return "\n".join(parts)

    def _build_type_node(self, info: Any) -> TypeNode:
        if isinstance(info, str):
            return TypeNode(name=info, kind="base")
        if isinstance(info, dict):
            children = [self._build_type_node(c) for c in info.get("children", [])]
            return TypeNode(
                name=info.get("name", ""),
                kind=info.get("kind", "base"),
                children=children,
                predicates=info.get("predicates", []),
                description=info.get("description", ""),
            )
        return TypeNode(name=str(info), kind="base")

    def _build_footer(self) -> str:
        return (
            '<footer class="footer">'
            'Generated by <strong>Refinement Type Inference Engine</strong> '
            f'on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
            '</footer>'
        )

    def _build_report_data(self, analysis_results: Dict[str, Any]) -> str:
        try:
            data_json = json.dumps(analysis_results, indent=2, default=str)
        except (TypeError, ValueError):
            data_json = "{}"
        return f'<script id="report-data" type="application/json">{data_json}</script>'

    def generate_report(self, analysis_results: Dict[str, Any]) -> str:
        head = self._build_head()
        header = self._build_header()
        search_widget = SearchWidget().render()
        export_btns = self._export.render_export_buttons()
        progress = self._build_progress(analysis_results)
        statistics = self._build_statistics(analysis_results) if self.show_statistics else ""
        coverage = self._build_coverage(analysis_results)
        file_sections = self._build_file_sections(analysis_results)
        bug_sections = self._build_bug_sections(analysis_results)
        types_section = self._build_types_section(analysis_results)
        cfg_section = self._build_cfg_section(analysis_results)
        dep_graph = self._build_dep_graph(analysis_results)
        diff_section = self._build_diff(analysis_results)

        sidebar_html = ""
        if self.show_sidebar:
            sidebar_html = self._build_sidebar(analysis_results)

        footer = self._build_footer()
        js = self._js_gen.generate()
        report_data = self._build_report_data(analysis_results)
        print_css = self._export.get_print_css()

        main_content = (
            f'{search_widget}'
            f'{export_btns}'
            f'{progress}'
            f'{statistics}'
            f'{coverage}'
            f'{diff_section}'
            f'{types_section}'
            f'{file_sections}'
            f'{bug_sections}'
            f'{cfg_section}'
            f'{dep_graph}'
        )

        body = (
            f'<body>'
            f'{header}'
            f'<div class="page-wrapper">'
            f'{sidebar_html}'
            f'<main class="main-content"><div class="content-area">{main_content}</div></main>'
            f'</div>'
            f'{footer}'
            f'{report_data}'
            f'<style>{print_css}</style>'
            f'<script>{js}</script>'
            f'</body>'
        )

        return f'<!DOCTYPE html>\n<html lang="en">\n{head}\n{body}\n</html>'

    def write_to_file(self, path: str | Path, analysis_results: Dict[str, Any]) -> None:
        report_html = self.generate_report(analysis_results)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(report_html, encoding="utf-8")

    def generate_summary_page(self, module_reports: Dict[str, Dict[str, Any]]) -> str:
        total_files = sum(len(r.get("files", [])) for r in module_reports.values())
        total_bugs = sum(
            sum(len(f.get("bugs", [])) for f in r.get("files", [])) + len(r.get("bugs", []))
            for r in module_reports.values()
        )
        total_funcs = sum(
            sum(len(f.get("functions", [])) for f in r.get("files", []))
            for r in module_reports.values()
        )

        rows: List[str] = []
        for module_name, results in module_reports.items():
            m_files = len(results.get("files", []))
            m_funcs = sum(len(f.get("functions", [])) for f in results.get("files", []))
            m_bugs = sum(len(f.get("bugs", [])) for f in results.get("files", [])) + len(results.get("bugs", []))
            sev_cls = "badge-error" if m_bugs > 0 else "badge-success"
            rows.append(
                f"<tr>"
                f'<td><a href="#{_slugify(module_name)}">{_escape(module_name)}</a></td>'
                f"<td>{m_files}</td>"
                f"<td>{m_funcs}</td>"
                f'<td><span class="badge {sev_cls}">{m_bugs}</span></td>'
                f"</tr>"
            )

        head = self._build_head()
        header = self._build_header()
        footer = self._build_footer()
        js = self._js_gen.generate()

        summary_content = (
            '<div class="content-area">'
            '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:0.75rem;margin-bottom:1.5rem;">'
            f'<div class="panel" style="text-align:center"><div style="font-size:2rem;font-weight:700">{len(module_reports)}</div><div class="text-muted">Modules</div></div>'
            f'<div class="panel" style="text-align:center"><div style="font-size:2rem;font-weight:700">{total_files}</div><div class="text-muted">Files</div></div>'
            f'<div class="panel" style="text-align:center"><div style="font-size:2rem;font-weight:700">{total_funcs}</div><div class="text-muted">Functions</div></div>'
            f'<div class="panel" style="text-align:center"><div style="font-size:2rem;font-weight:700">{total_bugs}</div><div class="text-muted">Bugs</div></div>'
            '</div>'
            '<div class="table-wrap"><table class="table-sortable">'
            '<thead><tr>'
            '<th>Module <span class="sort-indicator">⇅</span></th>'
            '<th>Files <span class="sort-indicator">⇅</span></th>'
            '<th>Functions <span class="sort-indicator">⇅</span></th>'
            '<th>Bugs <span class="sort-indicator">⇅</span></th>'
            '</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>'
            '</div>'
        )

        body = (
            f'<body>{header}'
            f'<main class="main-content">{summary_content}</main>'
            f'{footer}'
            f'<script>{js}</script>'
            f'</body>'
        )
        return f'<!DOCTYPE html>\n<html lang="en">\n{head}\n{body}\n</html>'
