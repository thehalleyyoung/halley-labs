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
from usability_oracle.formats.playwright import PlaywrightParser
from usability_oracle.formats.selenium_webdriver import SeleniumParser
from usability_oracle.formats.cypress import CypressParser
from usability_oracle.formats.puppeteer import PuppeteerParser
from usability_oracle.formats.react_testing_library import ReactTestingLibraryParser
from usability_oracle.formats.storybook import StorybookParser
from usability_oracle.formats.pa11y import Pa11yParser
from usability_oracle.formats.testing_library_queries import TestingLibraryQueriesParser
from usability_oracle.formats.converters import FormatConverter
from usability_oracle.formats.registry import FormatRegistry, FormatInfo
from usability_oracle.formats.converters2 import FormatConverter2
from usability_oracle.formats.schema_validator import (
    SchemaValidator,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
)

__all__ = [
    "ARIAParser",
    "AxeCoreParser",
    "ChromeDevToolsParser",
    "AndroidParser",
    "IOSParser",
    "WindowsUIAParser",
    "FormatConverter",
    "FormatRegistry",
    "FormatInfo",
    "PlaywrightParser",
    "SeleniumParser",
    "CypressParser",
    "PuppeteerParser",
    "ReactTestingLibraryParser",
    "StorybookParser",
    "Pa11yParser",
    "TestingLibraryQueriesParser",
    "FormatConverter2",
    "SchemaValidator",
    "ValidationResult",
    "ValidationIssue",
    "ValidationSeverity",
]
