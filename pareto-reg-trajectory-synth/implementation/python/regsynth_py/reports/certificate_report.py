"""
Certificate verification report generator.

Produces HTML, plain-text, and JSON reports summarising the validity and
verification status of a compliance certificate.
"""

import json
import os
from datetime import datetime

from .templates import Templates


class CertificateReportGenerator:
    """Generate certificate verification reports in multiple formats."""

    def __init__(self):
        self._tpl = Templates()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def generate(self, certificate_data: dict, format_type: str = "html") -> str:
        dispatch = {
            "html": self.generate_html,
            "text": self.generate_text,
            "json": self.generate_json,
        }
        handler = dispatch.get(format_type)
        if handler is None:
            raise ValueError(f"Unsupported format: {format_type}")
        return handler(certificate_data)

    # ------------------------------------------------------------------ #
    # HTML report
    # ------------------------------------------------------------------ #

    def generate_html(self, data: dict) -> str:
        sections = [
            self._certificate_header(data),
            self._details_table(data),
            self._verification_checks_table(data.get("checks", [])),
            self._framework_coverage(data.get("frameworks", [])),
            self._validity_section(data),
            self._chain_of_trust(data),
            self._signature_info(data.get("signature", {})),
        ]
        content = "\n".join(sections)
        return self._tpl.render_html(
            self._tpl.HTML_REPORT_TEMPLATE,
            title="Certificate Verification Report",
            content=content,
        )

    # ------------------------------------------------------------------ #
    # Text report
    # ------------------------------------------------------------------ #

    def generate_text(self, data: dict) -> str:
        checks = data.get("checks", [])
        frameworks = data.get("frameworks", [])
        trust_score = self._compute_trust_score(checks)
        valid = self._is_valid(data)
        days_left = self._days_until_expiry(data.get("expiry_date", ""))

        status = data.get("verification_status", "UNKNOWN").upper()
        lines = [
            "CERTIFICATE VERIFICATION REPORT",
            "=" * 72,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            f"Status: {status}",
            f"Certificate ID: {data.get('certificate_id', '—')}",
            f"Subject:        {data.get('subject', '—')}",
            f"Issuer:         {data.get('issuer', '—')}",
            f"Issued:         {data.get('issued_date', '—')}",
            f"Expires:        {data.get('expiry_date', '—')}",
            f"Days remaining: {days_left if days_left >= 0 else 'EXPIRED'}",
            f"Trust score:    {trust_score:.1f}/100",
            "",
            "VERIFICATION CHECKS",
            "-" * 40,
        ]
        for chk in checks:
            name = chk.get("name", "?")
            result = "PASS" if chk.get("passed", False) else "FAIL"
            lines.append(f"  [{result}] {name}")
        lines.append("")

        lines.append("FRAMEWORK COVERAGE")
        lines.append("-" * 40)
        for fw in frameworks:
            name = fw.get("name", str(fw)) if isinstance(fw, dict) else str(fw)
            lines.append(f"  - {name}")
        lines.append("")

        sig = data.get("signature", {})
        if sig:
            lines.append("SIGNATURE INFORMATION")
            lines.append("-" * 40)
            lines.append(f"  Algorithm: {sig.get('algorithm', '—')}")
            lines.append(f"  Key ID:    {sig.get('key_id', '—')}")
            valid_sig = "Valid" if sig.get("valid", False) else "Invalid"
            lines.append(f"  Status:    {valid_sig}")
            lines.append("")

        lines.append("=" * 72)
        lines.append("End of Report")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # JSON report
    # ------------------------------------------------------------------ #

    def generate_json(self, data: dict) -> str:
        checks = data.get("checks", [])
        trust_score = self._compute_trust_score(checks)
        valid = self._is_valid(data)

        report_data = {
            "certificate_id": data.get("certificate_id", ""),
            "status": data.get("verification_status", "unknown"),
            "checks": checks,
            "trust_score": round(trust_score, 2),
            "valid": valid,
            "days_until_expiry": self._days_until_expiry(
                data.get("expiry_date", "")
            ),
            "subject": data.get("subject", ""),
            "issuer": data.get("issuer", ""),
            "issued_date": data.get("issued_date", ""),
            "expiry_date": data.get("expiry_date", ""),
            "frameworks": [
                f.get("name", str(f)) if isinstance(f, dict) else str(f)
                for f in data.get("frameworks", [])
            ],
        }
        schema = self._tpl.get_json_schema("certificate")
        return self._tpl.render_json(report_data, schema)

    # ------------------------------------------------------------------ #
    # Section builders (HTML)
    # ------------------------------------------------------------------ #

    def _certificate_header(self, data: dict) -> str:
        status = data.get("verification_status", "unknown").upper()
        badge = self._tpl.format_status_badge(status.lower())
        cert_id = self._tpl.escape_html(data.get("certificate_id", "—"))
        subject = self._tpl.escape_html(data.get("subject", "—"))
        return (
            '<div class="cert-header">'
            f"<h2>Certificate {cert_id}</h2>"
            f"<p>{subject}</p>"
            f'<div class="cert-status">{badge}</div>'
            "</div>"
        )

    def _details_table(self, data: dict) -> str:
        fields = [
            ("Certificate ID", data.get("certificate_id", "—")),
            ("Subject", data.get("subject", "—")),
            ("Issuer", data.get("issuer", "—")),
            ("Issued Date", data.get("issued_date", "—")),
            ("Expiry Date", data.get("expiry_date", "—")),
            ("Verification Status", data.get("verification_status", "—")),
        ]
        rows = "".join(
            f"<tr><td><strong>{self._tpl.escape_html(label)}</strong></td>"
            f"<td>{self._tpl.escape_html(str(value))}</td></tr>"
            for label, value in fields
        )
        return (
            "<h2>Certificate Details</h2>"
            f"<table><tbody>{rows}</tbody></table>"
        )

    def _verification_checks_table(self, checks: list) -> str:
        if not checks:
            return '<div class="section"><h2>Verification Checks</h2><p>No checks recorded.</p></div>'
        rows = []
        for chk in checks:
            name = self._tpl.escape_html(chk.get("name", "?"))
            passed = chk.get("passed", False)
            css_cls = "check-pass" if passed else "check-fail"
            detail = self._tpl.escape_html(chk.get("detail", "—"))
            rows.append(
                f'<tr><td class="{css_cls}">{name}</td>'
                f"<td>{self._tpl.format_status_badge('pass' if passed else 'fail')}</td>"
                f"<td>{detail}</td></tr>"
            )
        return (
            "<h2>Verification Checks</h2>"
            "<table><thead><tr><th>Check</th><th>Result</th>"
            f"<th>Detail</th></tr></thead><tbody>{''.join(rows)}"
            "</tbody></table>"
        )

    def _framework_coverage(self, frameworks: list) -> str:
        if not frameworks:
            return ""
        items = []
        for fw in frameworks:
            name = fw.get("name", str(fw)) if isinstance(fw, dict) else str(fw)
            items.append(f"<li>{self._tpl.escape_html(name)}</li>")
        return (
            '<div class="section"><h2>Framework Coverage</h2>'
            f"<ul>{''.join(items)}</ul></div>"
        )

    def _validity_section(self, data: dict) -> str:
        days = self._days_until_expiry(data.get("expiry_date", ""))
        valid = self._is_valid(data)
        issued = self._tpl.escape_html(data.get("issued_date", "—"))
        expiry = self._tpl.escape_html(data.get("expiry_date", "—"))

        if days < 0:
            status_text = f'<span style="color:#c0392b"><strong>EXPIRED</strong> ({abs(days)} days ago)</span>'
        elif days <= 30:
            status_text = f'<span style="color:#f39c12"><strong>EXPIRING SOON</strong> ({days} days remaining)</span>'
        else:
            status_text = f'<span style="color:#27ae60"><strong>VALID</strong> ({days} days remaining)</span>'

        return (
            '<div class="section"><h2>Validity Period</h2>'
            f"<p>Issued: <strong>{issued}</strong></p>"
            f"<p>Expires: <strong>{expiry}</strong></p>"
            f"<p>Status: {status_text}</p></div>"
        )

    def _chain_of_trust(self, data: dict) -> str:
        """Render a simple chain-of-trust visualisation."""
        subject = self._tpl.escape_html(data.get("subject", "Subject"))
        issuer = self._tpl.escape_html(data.get("issuer", "Issuer"))
        cert_id = self._tpl.escape_html(data.get("certificate_id", "—"))
        return (
            '<div class="section"><h2>Chain of Trust</h2>'
            '<div style="text-align:center;padding:1rem;">'
            f'<div style="display:inline-block;border:2px solid #0f3460;'
            f'border-radius:6px;padding:0.75rem 1.5rem;margin:0.5rem">'
            f"<strong>{issuer}</strong><br><small>Issuer</small></div>"
            '<div style="font-size:1.5rem;">&#x2193;</div>'
            f'<div style="display:inline-block;border:2px solid #27ae60;'
            f'border-radius:6px;padding:0.75rem 1.5rem;margin:0.5rem">'
            f"<strong>{subject}</strong><br><small>Certificate {cert_id}</small></div>"
            "</div></div>"
        )

    def _signature_info(self, signature: dict) -> str:
        if not signature:
            return ""
        algo = self._tpl.escape_html(signature.get("algorithm", "—"))
        key_id = self._tpl.escape_html(signature.get("key_id", "—"))
        sig_valid = signature.get("valid", False)
        badge = self._tpl.format_status_badge("valid" if sig_valid else "fail")

        return (
            '<div class="section"><h2>Digital Signature</h2>'
            "<table>"
            f"<tr><td><strong>Algorithm</strong></td><td>{algo}</td></tr>"
            f"<tr><td><strong>Key ID</strong></td><td>{key_id}</td></tr>"
            f"<tr><td><strong>Signature Valid</strong></td><td>{badge}</td></tr>"
            "</table></div>"
        )

    # ------------------------------------------------------------------ #
    # Analytics
    # ------------------------------------------------------------------ #

    def _compute_trust_score(self, checks: list) -> float:
        """Return 0-100 score based on the ratio of passed checks."""
        if not checks:
            return 0.0
        passed = sum(1 for c in checks if c.get("passed", False))
        return (passed / len(checks)) * 100.0

    def _is_valid(self, data: dict) -> bool:
        """Determine if the certificate is currently valid."""
        status = data.get("verification_status", "").lower()
        if status in ("expired", "revoked", "invalid"):
            return False
        days = self._days_until_expiry(data.get("expiry_date", ""))
        if days < 0:
            return False
        checks = data.get("checks", [])
        if checks and not all(c.get("passed", False) for c in checks):
            return False
        return True

    def _days_until_expiry(self, expiry_date: str) -> int:
        """Return the number of days until *expiry_date* (negative if past)."""
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(expiry_date, fmt)
                delta = dt - datetime.now()
                return delta.days
            except ValueError:
                continue
        return -1

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, content: str, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write(content)
