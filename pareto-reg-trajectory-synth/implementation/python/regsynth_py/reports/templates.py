"""
Report templates for HTML, text, and JSON formats.

Provides reusable templates and formatting utilities for all report types
in the RegSynth compliance analysis pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
import html as html_module
import json


@dataclass
class Template:
    """A named report template."""

    name: str
    format_type: str  # "html", "text", "json"
    template: str


class Templates:
    """Singleton-like template registry for all report formats."""

    _instance = None

    HTML_REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
{css}
</style>
</head>
<body>
<header class="report-header">
  <div class="header-inner">
    <h1>{title}</h1>
    <p class="report-date">Generated: {date}</p>
  </div>
</header>
<main class="report-content">
{content}
</main>
<footer class="report-footer">
  <p>RegSynth Compliance Report &mdash; Generated {date}</p>
</footer>
</body>
</html>"""

    TEXT_REPORT_TEMPLATE = """{title}
{'=' * 72}

Generated: {date}

{content}

{'=' * 72}
End of Report
"""

    JSON_REPORT_SCHEMA = {
        "type": "object",
        "required": ["report_type", "generated_at", "data"],
        "properties": {
            "report_type": {"type": "string"},
            "generated_at": {"type": "string"},
            "version": {"type": "string"},
            "data": {"type": "object"},
        },
    }

    DEFAULT_CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen,
               Ubuntu, Cantarell, 'Helvetica Neue', sans-serif;
  line-height: 1.6; color: #1a1a2e; background: #f8f9fa; font-size: 15px;
}
.report-header {
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
  color: #fff; padding: 2rem 0; margin-bottom: 2rem;
}
.header-inner { max-width: 960px; margin: 0 auto; padding: 0 1.5rem; }
.report-header h1 { font-size: 1.75rem; font-weight: 700; }
.report-date { opacity: 0.85; margin-top: 0.25rem; font-size: 0.9rem; }
.report-content { max-width: 960px; margin: 0 auto; padding: 0 1.5rem 3rem; }
.report-footer {
  text-align: center; padding: 1.5rem; color: #666;
  border-top: 1px solid #dee2e6; font-size: 0.85rem; margin-top: 2rem;
}
h2 { font-size: 1.35rem; color: #1a1a2e; margin: 2rem 0 1rem;
     padding-bottom: 0.4rem; border-bottom: 2px solid #0f3460; }
h3 { font-size: 1.1rem; color: #16213e; margin: 1.5rem 0 0.75rem; }
table { width: 100%; border-collapse: collapse; margin: 1rem 0 1.5rem; }
th, td { padding: 0.6rem 0.75rem; text-align: left; border-bottom: 1px solid #dee2e6; }
th { background: #e9ecef; font-weight: 600; font-size: 0.85rem;
     text-transform: uppercase; letter-spacing: 0.03em; color: #495057; }
tr:hover td { background: #f1f3f5; }
.badge {
  display: inline-block; padding: 0.2rem 0.6rem; border-radius: 3px;
  font-size: 0.78rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.04em;
}
.badge-high, .badge-critical { background: #ffe0e0; color: #c0392b; }
.badge-medium { background: #fff3cd; color: #856404; }
.badge-low { background: #d4edda; color: #155724; }
.badge-compliant, .badge-valid, .badge-pass { background: #d4edda; color: #155724; }
.badge-non-compliant, .badge-expired, .badge-fail { background: #ffe0e0; color: #c0392b; }
.badge-partial, .badge-warning { background: #fff3cd; color: #856404; }
.badge-revoked { background: #e2e3e5; color: #383d41; }
.metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
               gap: 1rem; margin: 1.5rem 0; }
.metric-card {
  background: #fff; border: 1px solid #dee2e6; border-radius: 6px;
  padding: 1.25rem; text-align: center;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #0f3460; }
.metric-label { font-size: 0.85rem; color: #666; margin-top: 0.25rem; }
.section { background: #fff; border: 1px solid #dee2e6; border-radius: 6px;
           padding: 1.5rem; margin: 1.5rem 0; }
.bar-chart { margin: 1rem 0; }
.bar-row { display: flex; align-items: center; margin: 0.4rem 0; }
.bar-label { width: 120px; font-size: 0.85rem; color: #495057; }
.bar-track { flex: 1; background: #e9ecef; border-radius: 3px; height: 22px;
             overflow: hidden; }
.bar-fill { height: 100%; border-radius: 3px; display: flex; align-items: center;
            padding-left: 0.5rem; font-size: 0.75rem; color: #fff; font-weight: 600; }
.bar-fill-high { background: #c0392b; }
.bar-fill-medium { background: #f39c12; }
.bar-fill-low { background: #27ae60; }
.bar-fill-default { background: #0f3460; }
ul.rec-list { list-style: none; padding: 0; }
ul.rec-list li { padding: 0.75rem; margin: 0.5rem 0; background: #fff;
                 border-left: 4px solid #0f3460; border-radius: 0 4px 4px 0; }
ul.rec-list li.priority-high { border-left-color: #c0392b; }
ul.rec-list li.priority-medium { border-left-color: #f39c12; }
ul.rec-list li.priority-low { border-left-color: #27ae60; }
.phase-card { background: #fff; border: 1px solid #dee2e6; border-radius: 6px;
              padding: 1.25rem; margin: 1rem 0; }
.phase-card h3 { margin-top: 0; }
.check-pass::before { content: '\\2713 '; color: #27ae60; font-weight: 700; }
.check-fail::before { content: '\\2717 '; color: #c0392b; font-weight: 700; }
.cert-header { text-align: center; padding: 2rem; background: #fff;
               border: 2px solid #dee2e6; border-radius: 8px; margin: 1rem 0; }
.cert-header .cert-status { font-size: 1.5rem; margin-top: 0.5rem; }
"""

    _REPORT_TYPE_HTML = {
        "compliance": "<h2>Compliance Analysis Report</h2>\n{content}",
        "conflict": "<h2>Conflict Analysis Report</h2>\n{content}",
        "roadmap": "<h2>Compliance Roadmap</h2>\n{content}",
        "certificate": "<h2>Certificate Verification Report</h2>\n{content}",
    }

    _REPORT_TYPE_TEXT = {
        "compliance": "COMPLIANCE ANALYSIS REPORT\n{'=' * 40}\n\n{content}",
        "conflict": "CONFLICT ANALYSIS REPORT\n{'=' * 40}\n\n{content}",
        "roadmap": "COMPLIANCE ROADMAP\n{'=' * 40}\n\n{content}",
        "certificate": "CERTIFICATE VERIFICATION REPORT\n{'=' * 40}\n\n{content}",
    }

    _REPORT_TYPE_JSON_SCHEMAS = {
        "compliance": {
            "type": "object",
            "required": ["score", "frameworks", "obligations", "gaps"],
            "properties": {
                "score": {"type": "number"},
                "frameworks": {"type": "array"},
                "obligations": {"type": "array"},
                "gaps": {"type": "array"},
                "recommendations": {"type": "array"},
            },
        },
        "conflict": {
            "type": "object",
            "required": ["conflicts", "summary"],
            "properties": {
                "conflicts": {"type": "array"},
                "summary": {"type": "object"},
                "resolutions": {"type": "array"},
            },
        },
        "roadmap": {
            "type": "object",
            "required": ["phases", "timeline"],
            "properties": {
                "phases": {"type": "array"},
                "timeline": {"type": "object"},
                "budget": {"type": "object"},
                "milestones": {"type": "array"},
            },
        },
        "certificate": {
            "type": "object",
            "required": ["certificate_id", "status", "checks"],
            "properties": {
                "certificate_id": {"type": "string"},
                "status": {"type": "string"},
                "checks": {"type": "array"},
                "trust_score": {"type": "number"},
            },
        },
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ------------------------------------------------------------------ #
    # Template accessors
    # ------------------------------------------------------------------ #

    def get_html_template(self, report_type: str) -> str:
        """Return the HTML inner-content template for a given report type."""
        return self._REPORT_TYPE_HTML.get(
            report_type,
            "<h2>Report</h2>\n{content}",
        )

    def get_text_template(self, report_type: str) -> str:
        """Return the plain-text template for a given report type."""
        return self._REPORT_TYPE_TEXT.get(
            report_type,
            "REPORT\n{'=' * 40}\n\n{content}",
        )

    def get_json_schema(self, report_type: str) -> dict:
        """Return the expected JSON schema dict for a given report type."""
        return self._REPORT_TYPE_JSON_SCHEMAS.get(
            report_type,
            self.JSON_REPORT_SCHEMA,
        )

    # ------------------------------------------------------------------ #
    # Rendering helpers
    # ------------------------------------------------------------------ #

    def render_html(self, template: str, **kwargs) -> str:
        """Render an HTML template using keyword substitution.

        Falls back gracefully when a placeholder key is missing – the
        placeholder is left as-is so that callers can do multi-pass
        rendering.
        """
        css = kwargs.pop("css", self.DEFAULT_CSS)
        date = kwargs.pop("date", datetime.now().strftime("%Y-%m-%d %H:%M"))
        result = template
        for key, value in {**kwargs, "css": css, "date": date}.items():
            result = result.replace("{" + key + "}", str(value))
        return result

    def render_text(self, template: str, **kwargs) -> str:
        """Render a plain-text template using keyword substitution."""
        date = kwargs.pop("date", datetime.now().strftime("%Y-%m-%d %H:%M"))
        result = template
        for key, value in {**kwargs, "date": date}.items():
            result = result.replace("{" + key + "}", str(value))
        return result

    def render_json(self, data: dict, schema: dict) -> str:
        """Serialise *data* to JSON after lightweight schema validation.

        Validation checks that all ``required`` keys are present and that
        top-level value types match (``string``, ``number``, ``array``,
        ``object``).  Raises ``ValueError`` on failure.
        """
        _TYPE_MAP = {
            "string": str,
            "number": (int, float),
            "array": list,
            "object": dict,
        }
        required = schema.get("required", [])
        props = schema.get("properties", {})
        for key in required:
            if key not in data:
                raise ValueError(
                    f"Schema validation failed: missing required key '{key}'"
                )
        for key, spec in props.items():
            if key in data:
                expected = _TYPE_MAP.get(spec.get("type"))
                if expected and not isinstance(data[key], expected):
                    raise ValueError(
                        f"Schema validation failed: '{key}' must be "
                        f"{spec['type']}, got {type(data[key]).__name__}"
                    )
        return json.dumps(data, indent=2, default=str)

    # ------------------------------------------------------------------ #
    # Formatting utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def format_date(date_str: str) -> str:
        """Parse an ISO-ish date string and return a human-readable form."""
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%B %d, %Y")
            except ValueError:
                continue
        return date_str

    @staticmethod
    def format_currency(amount: float, currency: str = "EUR") -> str:
        """Format *amount* with thousands separators and currency symbol."""
        symbols = {"EUR": "€", "USD": "$", "GBP": "£", "CHF": "CHF "}
        sym = symbols.get(currency, currency + " ")
        formatted = f"{amount:,.2f}"
        return f"{sym}{formatted}"

    @staticmethod
    def format_percentage(value: float) -> str:
        """Format *value* (0-100) as a percentage string."""
        return f"{value:.1f}%"

    @staticmethod
    def format_risk_badge(level: str) -> str:
        """Return an HTML ``<span>`` badge coloured by risk level."""
        level_lower = level.lower().strip()
        css_class = {
            "critical": "badge-critical",
            "high": "badge-high",
            "medium": "badge-medium",
            "low": "badge-low",
        }.get(level_lower, "badge-medium")
        safe = html_module.escape(level)
        return f'<span class="badge {css_class}">{safe}</span>'

    @staticmethod
    def format_status_badge(status: str) -> str:
        """Return an HTML ``<span>`` badge coloured by status."""
        status_lower = status.lower().strip().replace(" ", "-")
        css_class = {
            "compliant": "badge-compliant",
            "non-compliant": "badge-non-compliant",
            "partial": "badge-partial",
            "valid": "badge-valid",
            "expired": "badge-expired",
            "revoked": "badge-revoked",
            "pass": "badge-pass",
            "fail": "badge-fail",
            "warning": "badge-warning",
        }.get(status_lower, "badge-partial")
        safe = html_module.escape(status)
        return f'<span class="badge {css_class}">{safe}</span>'

    @staticmethod
    def escape_html(text: str) -> str:
        """Escape HTML special characters."""
        return html_module.escape(str(text))

    @staticmethod
    def truncate(text: str, max_length: int = 200) -> str:
        """Truncate *text* to *max_length*, appending ``...`` if needed."""
        if len(text) <= max_length:
            return text
        return text[: max_length - 3].rstrip() + "..."
