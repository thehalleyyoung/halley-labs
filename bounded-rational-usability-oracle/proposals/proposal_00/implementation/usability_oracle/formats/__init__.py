"""
usability_oracle.formats — Multi-format accessibility tree support.

Provides parsers and converters for ARIA, axe-core, Chrome DevTools,
Android, iOS, and Windows accessibility tree formats.
"""

from __future__ import annotations

from usability_oracle.formats.aria import ARIAParser
from usability_oracle.formats.axe_core import AxeCoreParser
from usability_oracle.formats.chrome_devtools import ChromeDevToolsParser
from usability_oracle.formats.android import AndroidParser
from usability_oracle.formats.ios import IOSParser
from usability_oracle.formats.windows import WindowsUIAParser
from usability_oracle.formats.converters import FormatConverter

__all__ = [
    "ARIAParser",
    "AxeCoreParser",
    "ChromeDevToolsParser",
    "AndroidParser",
    "IOSParser",
    "WindowsUIAParser",
    "FormatConverter",
]
