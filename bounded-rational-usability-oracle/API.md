# API Reference — Bounded-Rational Usability Oracle

> Complete programmatic API for `usability_oracle` v0.1.0.
> For CLI usage, see the [README](README.md). For theoretical background, see the [tool paper](tool_paper.pdf).

---

## Table of Contents

- [Quick Start](#quick-start)
- [Pipeline API](#pipeline-api)
  - [PipelineRunner](#pipelinerunner)
  - [PipelineResult](#pipelineresult)
  - [StageResult](#stageresult)
- [Accessibility Trees](#accessibility-trees)
  - [AccessibilityTree](#accessibilitytree)
  - [AccessibilityNode](#accessibilitynode)
  - [BoundingBox](#boundingbox)
  - [AccessibilityState](#accessibilitystate)
- [Parsers & Format Support](#parsers--format-support)
  - [FormatRegistry](#formatregistry)
  - [HTMLAccessibilityParser](#htmlaccessibilityparser)
  - [JSONAccessibilityParser](#jsonaccessibilityparser)
  - [PlaywrightParser](#playwrightparser)
  - [PuppeteerParser](#puppeteerparser)
  - [SeleniumParser](#seleniumparser)
  - [CypressParser](#cypressparser)
  - [ReactTestingLibraryParser](#reacttestinglibraryparser)
  - [TestingLibraryQueriesParser](#testinglibraryqueriesparser)
  - [StorybookParser](#storybookparser)
  - [AxeCoreParser](#axecoreparser)
  - [Pa11yParser](#pa11yparser)
  - [ChromeDevToolsParser](#chromedevtoolsparser)
  - [AndroidParser](#androidparser)
  - [IOSParser](#iosparser)
  - [WindowsUIAParser](#windowsuiaparser)
- [Cognitive Models](#cognitive-models)
  - [FittsLaw](#fittslaw)
  - [HickHymanLaw](#hickhymanlaw)
  - [WorkingMemoryModel](#workingmemorymodel)
  - [VisualSearchModel](#visualsearchmodel)
- [Cost Algebra](#cost-algebra)
  - [CostElement](#costelement)
  - [SequentialComposer](#sequentialcomposer)
  - [ParallelComposer](#parallelcomposer)
  - [ContextModulator](#contextmodulator)
  - [TaskGraphComposer](#taskgraphcomposer)
  - [SoundnessVerifier](#soundnessverifier)
- [Comparison & Regression Testing](#comparison--regression-testing)
  - [PairedComparator](#pairedcomparator)
  - [RegressionTester](#regressiontester)
  - [ErrorBoundComputer](#errorboundcomputer)
  - [ComparisonResult](#comparisonresult)
- [Bisimulation](#bisimulation)
  - [CognitiveDistanceComputer](#cognitivedistancecomputer)
  - [PartitionRefinement](#partitionrefinement)
  - [QuotientMDPBuilder](#quotientmdpbuilder)
- [Bottleneck Classification](#bottleneck-classification)
  - [BottleneckClassifier](#bottleneckclassifier)
  - [BottleneckResult](#bottleneckresult)
  - [BottleneckReport](#bottleneckreport)
- [Repair Synthesis](#repair-synthesis)
  - [RepairSynthesizer](#repairsynthesizer)
  - [RepairCandidate](#repaircandidate)
- [Benchmarking](#benchmarking)
  - [BenchmarkSuite](#benchmarksuite)
  - [BenchmarkMetrics](#benchmarkmetrics)
  - [SyntheticUIGenerator](#syntheticuigenerator)
  - [MutationGenerator](#mutationgenerator)
  - [DatasetManager](#datasetmanager)
- [Enumerations](#enumerations)

---

## Quick Start

```python
from usability_oracle.pipeline.runner import PipelineRunner
from usability_oracle.accessibility import HTMLAccessibilityParser

parser = HTMLAccessibilityParser()
tree_before = parser.parse(open("before.html").read())
tree_after  = parser.parse(open("after.html").read())

runner = PipelineRunner()
result = runner.run(source_a=tree_before, source_b=tree_after)

print(result.success)        # True
print(result.final_result)   # ComparisonResult with verdict, cost delta, severity
```

---

## Pipeline API

### PipelineRunner

**Import:** `from usability_oracle.pipeline.runner import PipelineRunner`

The top-level orchestrator. Runs the 10-stage pipeline (parse → align → cost → MDP → bisimulate → policy → compare → bottleneck → repair → output) with caching, retry logic, and timing instrumentation.

```python
class PipelineRunner:
    def __init__(
        self,
        config: FullPipelineConfig | None = None,
        registry: StageRegistry | None = None,
        cache: ResultCache | None = None,
    ) -> None: ...

    def run(
        self,
        config: FullPipelineConfig | None = None,
        source_a: Any = None,
        source_b: Any = None,
        task_spec: Any = None,
    ) -> PipelineResult: ...
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `FullPipelineConfig \| None` | Pipeline configuration. Uses defaults if `None`. |
| `source_a` | `Any` | Before-version UI (AccessibilityTree, HTML string, or file path) |
| `source_b` | `Any` | After-version UI |
| `task_spec` | `Any` | Task specification (YAML path or TaskSpec object) |

**Example:**
```python
runner = PipelineRunner()

# From AccessibilityTree objects
result = runner.run(source_a=tree_a, source_b=tree_b)

# From HTML strings
result = runner.run(source_a="<html>...</html>", source_b="<html>...</html>")
```

---

### PipelineResult

**Import:** `from usability_oracle.pipeline.runner import PipelineResult`

```python
@dataclass
class PipelineResult:
    stages: dict[str, StageResult]     # Results per stage
    final_result: Any                   # ComparisonResult or AnalysisResult
    timing: dict[str, float]            # Per-stage timing in seconds
    cache_hits: int                     # Number of cache hits
    success: bool                       # Whether pipeline completed
    errors: list[str]                   # Error messages (if any)
```

---

### StageResult

```python
@dataclass
class StageResult:
    stage: PipelineStage     # Enum: PARSE, ALIGN, COST, MDP, etc.
    output: Any              # Stage-specific output
    timing: float            # Time in seconds
    errors: list[str]        # Stage-specific errors
    cached: bool             # Whether result was cached
```

---

## Accessibility Trees

### AccessibilityTree

**Import:** `from usability_oracle.accessibility import AccessibilityTree`

The core data structure representing a UI's accessibility tree.

```python
@dataclass
class AccessibilityTree:
    root: AccessibilityNode
    metadata: dict[str, Any] = field(default_factory=dict)
    node_index: dict[str, AccessibilityNode] = field(default_factory=dict)

    def build_index(self) -> None: ...
    def get_node(self, node_id: str) -> AccessibilityNode | None: ...
    def get_interactive_nodes(self) -> list[AccessibilityNode]: ...
    def get_focusable_nodes(self) -> list[AccessibilityNode]: ...
    def get_nodes_by_role(self, role: str) -> list[AccessibilityNode]: ...
    def lca(self, node_a: str, node_b: str) -> AccessibilityNode | None: ...
    def subtree(self, node_id: str) -> AccessibilityTree: ...
    def path_between(self, a: str, b: str) -> list[str]: ...
    def iter_bfs(self) -> Iterator[AccessibilityNode]: ...
    def iter_dfs(self) -> Iterator[AccessibilityNode]: ...
    def validate(self) -> list[str]: ...
    def to_dict(self) -> dict: ...
    def to_json(self) -> str: ...

    @classmethod
    def from_dict(cls, data: dict) -> AccessibilityTree: ...
    @classmethod
    def from_json(cls, json_str: str) -> AccessibilityTree: ...
```

---

### AccessibilityNode

**Import:** `from usability_oracle.accessibility import AccessibilityNode`

```python
@dataclass
class AccessibilityNode:
    id: str                                          # Unique node identifier
    role: str                                        # ARIA role ("button", "link", ...)
    name: str                                        # Accessible name
    description: str = ""                            # Accessible description
    bounding_box: BoundingBox | None = None          # Screen coordinates
    properties: dict[str, Any] = field(...)          # Custom properties
    state: AccessibilityState = field(...)           # Interaction state
    children: list[AccessibilityNode] = field(...)   # Child nodes
    parent_id: str | None = None                     # Parent node ID
    depth: int = 0                                   # Tree depth (0 = root)
    index_in_parent: int = 0                         # Sibling index

    def is_interactive(self) -> bool: ...
    def is_visible(self) -> bool: ...
    def is_focusable(self) -> bool: ...
    def semantic_hash(self) -> str: ...
    def subtree_size(self) -> int: ...
    def iter_preorder(self) -> Iterator[AccessibilityNode]: ...
    def iter_postorder(self) -> Iterator[AccessibilityNode]: ...
    def to_dict(self) -> dict: ...

    @classmethod
    def from_dict(cls, data: dict) -> AccessibilityNode: ...
```

---

### BoundingBox

**Import:** `from usability_oracle.accessibility import BoundingBox`

```python
@dataclass
class BoundingBox:
    x: float          # Left edge
    y: float          # Top edge
    width: float      # Width in pixels
    height: float     # Height in pixels

    @property
    def right(self) -> float: ...
    @property
    def bottom(self) -> float: ...
    @property
    def center_x(self) -> float: ...
    @property
    def center_y(self) -> float: ...
    @property
    def area(self) -> float: ...

    def contains_point(self, x: float, y: float) -> bool: ...
    def contains(self, other: BoundingBox) -> bool: ...
    def overlaps(self, other: BoundingBox) -> bool: ...
    def intersection(self, other: BoundingBox) -> BoundingBox | None: ...
    def union(self, other: BoundingBox) -> BoundingBox: ...
    def distance_to(self, other: BoundingBox) -> float: ...
    def to_dict(self) -> dict: ...

    @classmethod
    def from_dict(cls, data: dict) -> BoundingBox: ...
```

---

### AccessibilityState

**Import:** `from usability_oracle.accessibility import AccessibilityState`

```python
@dataclass
class AccessibilityState:
    focused: bool = False
    selected: bool = False
    expanded: bool = False
    checked: bool | None = None      # Tri-state
    disabled: bool = False
    hidden: bool = False
    required: bool = False
    readonly: bool = False
    pressed: bool | None = None      # For toggle buttons
    value: str | None = None         # Current value

    @classmethod
    def from_dict(cls, data: dict) -> AccessibilityState: ...
```

---

## Parsers & Format Support

All parsers produce `AccessibilityTree` objects. The `FormatRegistry` auto-detects format from content.

### FormatRegistry

**Import:** `from usability_oracle.formats import FormatRegistry`

Singleton registry for format detection and parser lookup.

```python
class FormatRegistry:
    def detect(self, content: str) -> FormatInfo | None: ...
    def get_parser(self, format_id: str) -> Any: ...
    def list_formats(self) -> list[FormatInfo]: ...

    def register(
        self,
        format_id: str,
        parser_class: str,
        extensions: Sequence[str] = (),
        mime_types: Sequence[str] = (),
        name: str = "",
        detector: Callable[[str], bool] | None = None,
    ) -> None: ...
```

**Supported formats:**

| Format ID | Parser Class | Input Source |
|-----------|-------------|-------------|
| `html-aria` | `ARIAParser` | Any HTML with ARIA attributes |
| `chrome-devtools` | `ChromeDevToolsParser` | Chrome DevTools Protocol |
| `axe-core` | `AxeCoreParser` | axe-core JSON output |
| `playwright` | `PlaywrightParser` | Playwright `page.accessibility.snapshot()` |
| `puppeteer` | `PuppeteerParser` | Puppeteer `page.accessibility.snapshot()` |
| `selenium` | `SeleniumParser` | Selenium WebDriver a11y extraction |
| `cypress` | `CypressParser` | cypress-axe / cypress-audit results |
| `react-testing-library` | `ReactTestingLibraryParser` | `prettyDOM()` / `logRoles()` output |
| `testing-library-queries` | `TestingLibraryQueriesParser` | `screen.getByRole()` serialized |
| `storybook` | `StorybookParser` | `@storybook/addon-a11y` JSON |
| `pa11y` | `Pa11yParser` | pa11y JSON output |
| `android-json` | `AndroidParser` | Android `uiautomator dump` |
| `ios-a11y` | `IOSParser` | iOS XCTest accessibility snapshot |
| `windows-uia` | `WindowsUIAParser` | Windows UI Automation tree |

---

### HTMLAccessibilityParser

**Import:** `from usability_oracle.accessibility import HTMLAccessibilityParser`

Parses raw HTML into an AccessibilityTree, respecting ARIA roles, labels, and landmarks.

```python
class HTMLAccessibilityParser:
    def parse(self, html: str) -> AccessibilityTree: ...
```

**Example:**
```python
parser = HTMLAccessibilityParser()
tree = parser.parse("""
<html>
  <body>
    <nav aria-label="Main">
      <a href="/home">Home</a>
      <a href="/about">About</a>
    </nav>
    <main>
      <form aria-label="Login">
        <label for="email">Email</label>
        <input id="email" type="email" required>
        <button type="submit">Sign In</button>
      </form>
    </main>
  </body>
</html>
""")
print(len(tree.get_interactive_nodes()))  # 3
```

---

### JSONAccessibilityParser

**Import:** `from usability_oracle.accessibility import JSONAccessibilityParser`

Auto-detects JSON format (Chrome DevTools, axe-core, or generic) and dispatches to the appropriate parser.

```python
class JSONAccessibilityParser:
    def parse(self, json_str: str) -> AccessibilityTree: ...
    def parse_file(self, path: Path) -> AccessibilityTree: ...
    def parse_dict(self, data: Any) -> AccessibilityTree: ...
```

---

### PlaywrightParser

**Import:** `from usability_oracle.formats import PlaywrightParser`

Parses [Playwright](https://playwright.dev/) accessibility snapshots.

```python
class PlaywrightParser:
    def parse(self, data: str | dict) -> AccessibilityTree: ...
```

**How to obtain input:**
```javascript
// In Playwright (JavaScript/TypeScript)
const snapshot = await page.accessibility.snapshot();
fs.writeFileSync('snapshot.json', JSON.stringify(snapshot));
```
```python
# In Playwright (Python)
snapshot = await page.accessibility.snapshot()
import json; open("snapshot.json", "w").write(json.dumps(snapshot))
```

**Example:**
```python
import json
from usability_oracle.formats import PlaywrightParser

snapshot = json.load(open("snapshot.json"))
tree = PlaywrightParser().parse(snapshot)
```

---

### PuppeteerParser

**Import:** `from usability_oracle.formats import PuppeteerParser`

Parses [Puppeteer](https://pptr.dev/) accessibility snapshots.

```python
class PuppeteerParser:
    def parse(self, data: str | dict) -> AccessibilityTree: ...
```

**How to obtain input:**
```javascript
const snapshot = await page.accessibility.snapshot();
fs.writeFileSync('snapshot.json', JSON.stringify(snapshot));
```

---

### SeleniumParser

**Import:** `from usability_oracle.formats import SeleniumParser`

Parses Selenium WebDriver accessibility extractions.

```python
class SeleniumParser:
    def parse(self, data: str | dict) -> AccessibilityTree: ...
```

**How to obtain input:**
```python
from selenium import webdriver

driver = webdriver.Chrome()
driver.get("https://example.com")
a11y_tree = driver.execute_script("""
    function extractA11y(el) {
        return {
            tag: el.tagName.toLowerCase(),
            role: el.getAttribute('role') || el.tagName.toLowerCase(),
            'aria-label': el.getAttribute('aria-label') || el.innerText?.slice(0, 80),
            rect: el.getBoundingClientRect(),
            attributes: Object.fromEntries(
                Array.from(el.attributes).map(a => [a.name, a.value])
            ),
            children: Array.from(el.children).map(extractA11y)
        };
    }
    return extractA11y(document.body);
""")
```

---

### CypressParser

**Import:** `from usability_oracle.formats import CypressParser`

Parses [cypress-axe](https://github.com/component-driven/cypress-axe) and [cypress-audit](https://github.com/mfrachet/cypress-audit) results.

```python
class CypressParser:
    def parse(self, data: str | dict) -> AccessibilityTree: ...
```

**How to obtain input:**
```javascript
// cypress/e2e/a11y.cy.js
cy.injectAxe();
cy.checkA11y(null, null, (violations) => {
  cy.writeFile('cypress-results.json', { violations, passes: [] });
});
```

---

### ReactTestingLibraryParser

**Import:** `from usability_oracle.formats import ReactTestingLibraryParser`

Parses [React Testing Library](https://testing-library.com/docs/react-testing-library/intro/) output: `prettyDOM()` HTML strings or `logRoles()` role-based output.

```python
class ReactTestingLibraryParser:
    def parse(self, data: str | dict) -> AccessibilityTree: ...
```

**How to obtain input:**
```javascript
import { render, screen } from '@testing-library/react';
import { logRoles } from '@testing-library/dom';

const { container } = render(<MyComponent />);
// Option 1: HTML string
const html = prettyDOM(container);
// Option 2: logRoles output (captured)
logRoles(container);
```

---

### TestingLibraryQueriesParser

**Import:** `from usability_oracle.formats import TestingLibraryQueriesParser`

Parses serialized output from Testing Library's role-based queries.

```python
class TestingLibraryQueriesParser:
    def parse(self, data: str | dict | list) -> AccessibilityTree: ...
```

**Input format:**
```json
[
  {"role": "button", "name": "Submit", "disabled": false, "pressed": false},
  {"role": "textbox", "name": "Email", "value": "", "required": true},
  {"role": "heading", "name": "Login", "level": 1}
]
```

---

### StorybookParser

**Import:** `from usability_oracle.formats import StorybookParser`

Parses [@storybook/addon-a11y](https://storybook.js.org/addons/@storybook/addon-a11y) JSON results.

```python
class StorybookParser:
    def parse(self, data: str | dict) -> AccessibilityTree: ...
```

**Input format:**
```json
{
  "storyId": "button--primary",
  "kind": "Button",
  "name": "Primary",
  "axeResults": {
    "violations": [...],
    "passes": [...],
    "incomplete": [...]
  }
}
```

---

### AxeCoreParser

**Import:** `from usability_oracle.formats import AxeCoreParser`

Parses [axe-core](https://github.com/dequelabs/axe-core) JSON output (v3.x and v4.x).

```python
class AxeCoreParser:
    def parse(self, data: str | dict) -> AxeResult: ...
    def to_bottleneck_annotations(self, axe_result: AxeResult) -> list[dict]: ...
    def severity_score(self, axe_result: AxeResult) -> float: ...
    def summary(self, axe_result: AxeResult) -> dict: ...
```

---

### Pa11yParser

**Import:** `from usability_oracle.formats import Pa11yParser`

Parses [pa11y](https://pa11y.org/) JSON output.

```python
class Pa11yParser:
    def parse(self, data: str | list) -> AccessibilityTree: ...
```

**How to obtain input:**
```bash
pa11y https://example.com --reporter json > results.json
```

---

### ChromeDevToolsParser

**Import:** `from usability_oracle.formats import ChromeDevToolsParser`

Parses Chrome DevTools Protocol `Accessibility.getFullAXTree` output.

```python
class ChromeDevToolsParser:
    def parse(self, data: str | dict | list) -> AccessibilityTree: ...
    def to_cdp_format(self, tree: AccessibilityTree) -> list[dict]: ...
```

---

### AndroidParser

**Import:** `from usability_oracle.formats import AndroidParser`

Parses Android `uiautomator dump` XML or JSON output.

```python
class AndroidParser:
    def parse(self, data: str | dict) -> AccessibilityTree: ...
```

---

### IOSParser

**Import:** `from usability_oracle.formats import IOSParser`

Parses iOS XCTest accessibility snapshot JSON.

```python
class IOSParser:
    def parse(self, data: str | dict) -> AccessibilityTree: ...
```

---

### WindowsUIAParser

**Import:** `from usability_oracle.formats import WindowsUIAParser`

Parses Windows UI Automation tree JSON exports.

```python
class WindowsUIAParser:
    def parse(self, data: str | dict) -> AccessibilityTree: ...
```

---

## Cognitive Models

### FittsLaw

**Import:** `from usability_oracle.cognitive.fitts import FittsLaw`

Models motor targeting time using the Shannon formulation.

```python
class FittsLaw:
    DEFAULT_A: float = 0.050   # 50ms intercept
    DEFAULT_B: float = 0.150   # 150ms per bit

    @staticmethod
    def predict(distance: float, width: float,
                a: float = DEFAULT_A, b: float = DEFAULT_B) -> float: ...

    @staticmethod
    def index_of_difficulty(distance: float, width: float) -> float: ...

    @staticmethod
    def throughput(distance: float, width: float,
                   movement_time: float) -> float: ...

    @staticmethod
    def effective_width(distance: float, movement_time: float,
                        a: float = DEFAULT_A, b: float = DEFAULT_B) -> float: ...
```

**Example:**
```python
from usability_oracle.cognitive.fitts import FittsLaw

# Time to click a 50px button 300px away
mt = FittsLaw.predict(distance=300, width=50)
print(f"{mt:.3f}s")  # ~0.44s

# Index of difficulty
id = FittsLaw.index_of_difficulty(300, 50)
print(f"{id:.2f} bits")  # ~2.81 bits
```

---

### HickHymanLaw

**Import:** `from usability_oracle.cognitive.hick import HickHymanLaw`

Models choice reaction time as a function of the number of alternatives.

```python
class HickHymanLaw:
    DEFAULT_A: float = 0.200   # 200ms base reaction time
    DEFAULT_B: float = 0.155   # 155ms per bit of information

    @staticmethod
    def predict(n_alternatives: int,
                a: float = DEFAULT_A, b: float = DEFAULT_B) -> float: ...

    @staticmethod
    def predict_unequal_probabilities(
        probabilities: Sequence[float],
        a: float = DEFAULT_A, b: float = DEFAULT_B,
    ) -> float: ...

    @staticmethod
    def entropy(probabilities: Sequence[float] | np.ndarray) -> float: ...

    @staticmethod
    def information_gain(prior: float, posterior: float) -> float: ...

    @staticmethod
    def practice_effect(n_trials: int, learning_rate: float = 0.1) -> float: ...
```

**Example:**
```python
from usability_oracle.cognitive.hick import HickHymanLaw

# Choice time for a menu with 8 items
rt = HickHymanLaw.predict(8)
print(f"{rt:.3f}s")  # ~0.665s

# With unequal probabilities (frequent vs. rare items)
rt = HickHymanLaw.predict_unequal_probabilities([0.4, 0.3, 0.2, 0.1])
print(f"{rt:.3f}s")  # Lower: skewed distribution reduces entropy
```

---

### WorkingMemoryModel

**Import:** `from usability_oracle.cognitive.working_memory import WorkingMemoryModel`

Models working memory capacity, decay, and interference.

```python
class WorkingMemoryModel:
    @staticmethod
    def predict_recall_probability(
        n_items: int,
        retention_interval: float = 0.0,
        capacity: int = 4,
    ) -> float: ...

    @staticmethod
    def interference_factor(
        n_similar_items: int,
        similarity: float = 0.5,
    ) -> float: ...

    @staticmethod
    def load_cost(
        n_items: int,
        item_complexity: float = 1.0,
    ) -> float: ...
```

---

### VisualSearchModel

**Import:** `from usability_oracle.cognitive.visual_search import VisualSearchModel`

Models visual search time under serial, parallel, and guided paradigms.

```python
class VisualSearchModel:
    @staticmethod
    def predict_serial(n_items: int, time_per_item: float = 0.050) -> float: ...

    @staticmethod
    def predict_parallel(n_items: int, pop_out_factor: float = 0.8) -> float: ...

    @staticmethod
    def predict_guided(
        n_items: int,
        n_distractors: int,
        guidance_strength: float = 0.5,
    ) -> float: ...

    @staticmethod
    def saliency_from_structure(node: AccessibilityNode) -> float: ...
```

---

## Cost Algebra

### CostElement

**Import:** `from usability_oracle.algebra import CostElement`

```python
@dataclass
class CostElement:
    mu: float           # Expected cost (seconds)
    sigma_sq: float     # Variance
    kappa: float        # Capacity utilization [0, 1]
    lambda_: float      # Interference susceptibility [0, 1]
```

---

### SequentialComposer

**Import:** `from usability_oracle.algebra import SequentialComposer`

Implements the `⊕` operator: sequential composition with load coupling.

```python
class SequentialComposer:
    def compose(self, a: CostElement, b: CostElement,
                coupling: float = 0.1) -> CostElement: ...
    def compose_chain(self, elements: Sequence[CostElement]) -> CostElement: ...
    def sensitivity(self, a: CostElement, b: CostElement) -> dict: ...
```

---

### ParallelComposer

**Import:** `from usability_oracle.algebra import ParallelComposer`

Implements the `⊗` operator: parallel composition with interference.

```python
class ParallelComposer:
    def compose(self, a: CostElement, b: CostElement,
                interference: float = 0.2) -> CostElement: ...
    def interference_factor(self, a: CostElement, b: CostElement) -> float: ...
```

---

### ContextModulator

**Import:** `from usability_oracle.algebra import ContextModulator`

Implements the `Δ` operator: context-dependent cost modulation.

```python
class ContextModulator:
    def modulate(
        self,
        element: CostElement,
        context: dict[str, float],  # fatigue, load, practice, stress
    ) -> CostElement: ...
```

---

### TaskGraphComposer

**Import:** `from usability_oracle.algebra import TaskGraphComposer`

Composes costs over a DAG of task steps.

```python
class TaskGraphComposer:
    def compose(self, graph: nx.DiGraph) -> CostElement: ...
    def critical_path(self, graph: nx.DiGraph) -> list[str]: ...
```

---

### SoundnessVerifier

**Import:** `from usability_oracle.algebra import SoundnessVerifier`

Verifies algebraic axioms (monotonicity, triangle inequality, commutativity).

```python
class SoundnessVerifier:
    def verify_monotonicity(self, a: CostElement, b: CostElement) -> bool: ...
    def verify_triangle_inequality(self, a, b, c: CostElement) -> bool: ...
    def verify_all(self, elements: list[CostElement]) -> dict[str, bool]: ...
```

---

## Comparison & Regression Testing

### PairedComparator

**Import:** `from usability_oracle.comparison.paired import PairedComparator`

```python
class PairedComparator:
    def __init__(
        self,
        beta: float = 1.0,
        n_trajectories: int = 500,
        significance_level: float = 0.05,
        min_effect_size: float = 0.2,
    ) -> None: ...

    def compare(
        self,
        mdp_a: MDP, mdp_b: MDP,
        alignment: AlignmentResult,
        task: TaskSpec,
        config: dict | None = None,
    ) -> ComparisonResult: ...
```

---

### RegressionTester

**Import:** `from usability_oracle.comparison.regression import RegressionTester`

```python
class RegressionTester:
    def test(
        self,
        cost_samples_a: np.ndarray,
        cost_samples_b: np.ndarray,
        alpha: float = 0.05,
    ) -> HypothesisResult: ...

    def test_multiple(
        self,
        tests: list[tuple[np.ndarray, np.ndarray]],
        alpha: float = 0.05,
    ) -> list[HypothesisResult]: ...
```

---

### ErrorBoundComputer

**Import:** `from usability_oracle.comparison.error_bounds import ErrorBoundComputer`

```python
class ErrorBoundComputer:
    def compute_abstraction_error(self, epsilon: float, horizon: int,
                                   beta: float) -> float: ...
    def compute_sampling_error(self, n_samples: int, variance: float,
                                alpha: float) -> float: ...
    def compute_required_samples(self, desired_error: float,
                                  variance: float) -> int: ...
    def full_analysis(self, ...) -> dict: ...
```

---

### ComparisonResult

```python
@dataclass
class ComparisonResult:
    verdict: RegressionVerdict        # REGRESSION | IMPROVEMENT | NEUTRAL | INCONCLUSIVE
    cost_delta: CostElement           # Cost difference (B - A)
    confidence_interval: tuple[float, float]
    severity: Severity                # LOW | MEDIUM | HIGH | CRITICAL
    p_value: float
    effect_size: float
    bottlenecks: list[BottleneckResult]
    timing: dict[str, float]
```

---

## Bisimulation

### CognitiveDistanceComputer

**Import:** `from usability_oracle.bisimulation import CognitiveDistanceComputer`

```python
class CognitiveDistanceComputer:
    def compute_distance(self, s1: Any, s2: Any, beta: float) -> float: ...
    def compute_distance_matrix(self, states: list, beta: float) -> np.ndarray: ...
```

---

### PartitionRefinement

**Import:** `from usability_oracle.bisimulation import PartitionRefinement`

```python
class PartitionRefinement:
    def refine(self, mdp: MDP, beta: float,
               epsilon: float = 0.01) -> dict[str, int]: ...
```

---

### QuotientMDPBuilder

**Import:** `from usability_oracle.bisimulation import QuotientMDPBuilder`

```python
class QuotientMDPBuilder:
    def build(self, mdp: MDP, partition: dict[str, int]) -> MDP: ...
    def verify_quotient(self, original: MDP, quotient: MDP) -> bool: ...
```

---

## Bottleneck Classification

### BottleneckClassifier

**Import:** `from usability_oracle.bottleneck import BottleneckClassifier`

```python
class BottleneckClassifier:
    def __init__(
        self,
        beta: float = 1.0,
        min_confidence: float = 0.7,
        max_bottlenecks: int = 5,
    ) -> None: ...

    def classify(
        self,
        mdp: MDP,
        policy: Policy,
        trajectory_stats: dict,
        cost_breakdown: dict,
    ) -> list[BottleneckResult]: ...
```

---

### BottleneckResult

```python
@dataclass
class BottleneckResult:
    type: BottleneckType    # PERCEPTUAL_OVERLOAD | CHOICE_PARALYSIS | ...
    severity_score: float   # 0.0–1.0
    impact_score: float     # Estimated cost contribution
    description: str        # Human-readable explanation
    affected_nodes: list[str]
    repair_hint: str        # Suggested fix
```

---

### BottleneckReport

```python
@dataclass
class BottleneckReport:
    bottlenecks: list[BottleneckResult]
    summary: dict[str, Any]

    def by_type(self) -> dict[BottleneckType, list[BottleneckResult]]: ...
    def type_distribution(self) -> dict[str, float]: ...
```

---

## Repair Synthesis

### RepairSynthesizer

**Import:** `from usability_oracle.smt_repair import RepairSynthesizer`

Z3-backed repair synthesis that proposes minimal UI mutations to restore the cost envelope.

```python
class RepairSynthesizer:
    def synthesize(
        self,
        mdp: MDP,
        bottlenecks: list[BottleneckResult],
        constraints: dict | None = None,
        timeout: int = 30,
    ) -> RepairResult: ...
```

---

### RepairCandidate

```python
@dataclass
class RepairCandidate:
    mutations: list[UIMutation]
    expected_cost_reduction: float
    feasibility_score: float
    description: str
```

---

## Benchmarking

### BenchmarkSuite

**Import:** `from usability_oracle.benchmarks.suite import BenchmarkSuite`

```python
class BenchmarkSuite:
    def __init__(
        self,
        pipeline_fn: Callable | None = None,
        positive_class: RegressionVerdict = RegressionVerdict.REGRESSION,
        verbose: bool = False,
    ) -> None: ...

    def run(
        self,
        config: dict | None = None,
        cases: list[BenchmarkCase] | None = None,
    ) -> BenchmarkReport: ...

    def run_with_warmup(
        self,
        cases: list[BenchmarkCase],
        n_warmup: int = 3,
        n_repeats: int = 5,
    ) -> dict[str, Any]: ...

    def profile_memory(
        self,
        cases: list[BenchmarkCase],
    ) -> list[dict[str, Any]]: ...
```

---

### BenchmarkMetrics

**Import:** `from usability_oracle.benchmarks.metrics import BenchmarkMetrics`

```python
class BenchmarkMetrics:
    @staticmethod
    def accuracy(results: list[BenchmarkResult]) -> float: ...
    @staticmethod
    def precision(results, positive_class="regression") -> float: ...
    @staticmethod
    def recall(results, positive_class="regression") -> float: ...
    @staticmethod
    def f1_score(results, positive_class="regression") -> float: ...
    @staticmethod
    def matthews_correlation(results, positive_class=None) -> float: ...
    @staticmethod
    def cohens_kappa(results) -> float: ...
    @staticmethod
    def rank_correlation(model_rankings, human_rankings) -> float: ...
    @staticmethod
    def full_summary(results, positive_class=None) -> dict: ...
    @staticmethod
    def per_category_accuracy(results) -> dict[str, float]: ...
    @staticmethod
    def sensitivity_by_severity(results, n_bins=5) -> dict: ...
    @staticmethod
    def scalability_fit(sizes, times) -> dict: ...
```

---

### SyntheticUIGenerator

**Import:** `from usability_oracle.benchmarks.generators import SyntheticUIGenerator`

```python
class SyntheticUIGenerator:
    def __init__(self, seed: int | None = None) -> None: ...

    def generate_form(self, n_fields: int = 6, complexity: str = "medium") -> AccessibilityTree: ...
    def generate_navigation(self, n_items: int = 8, depth: int = 2) -> AccessibilityTree: ...
    def generate_dashboard(self, n_widgets: int = 6) -> AccessibilityTree: ...
    def generate_search_results(self, n_results: int = 10) -> AccessibilityTree: ...
    def generate_settings_page(self, n_settings: int = 10) -> AccessibilityTree: ...
    def generate_data_table(self, n_rows: int = 10, n_cols: int = 5) -> AccessibilityTree: ...
    def generate_wizard(self, n_steps: int = 4) -> AccessibilityTree: ...
    def generate_modal_dialog(self, n_buttons: int = 3) -> AccessibilityTree: ...
```

---

### MutationGenerator

**Import:** `from usability_oracle.benchmarks.mutations import MutationGenerator`

```python
class MutationGenerator:
    def __init__(self, seed: int | None = None) -> None: ...

    def apply_perceptual_overload(self, tree, severity=0.5) -> AccessibilityTree: ...
    def apply_choice_paralysis(self, tree, severity=0.5) -> AccessibilityTree: ...
    def apply_motor_difficulty(self, tree, severity=0.5) -> AccessibilityTree: ...
    def apply_memory_decay(self, tree, severity=0.5) -> AccessibilityTree: ...
    def apply_interference(self, tree, severity=0.5) -> AccessibilityTree: ...
    def apply_random_mutation(self, tree, seed=None) -> tuple[AccessibilityTree, str]: ...
    def apply_label_removal(self, tree, severity=0.5) -> AccessibilityTree: ...
    def apply_tab_order_scramble(self, tree, severity=0.5) -> AccessibilityTree: ...
    def apply_focus_trap(self, tree, severity=0.5) -> AccessibilityTree: ...
    def apply_contrast_reduction(self, tree, severity=0.5) -> AccessibilityTree: ...
    def apply_landmark_removal(self, tree, severity=0.5) -> AccessibilityTree: ...
```

---

### DatasetManager

**Import:** `from usability_oracle.benchmarks.datasets import DatasetManager`

```python
class DatasetManager:
    def __init__(self, data_dir: Path | None = None, seed: int = 42) -> None: ...

    def load(self, name: str) -> list[BenchmarkCase]: ...
    def list_available(self) -> list[str]: ...
    def generate_synthetic(self, n_cases: int = 20, config: dict | None = None) -> list[BenchmarkCase]: ...
    def save(self, cases: list[BenchmarkCase], path: Path) -> None: ...
```

**Built-in datasets:** `"smoke"`, `"perceptual"`, `"choice"`, `"motor"`, `"memory"`, `"interference"`

---

## Enumerations

**Import:** `from usability_oracle.core.enums import ...`

```python
class RegressionVerdict(Enum):
    REGRESSION = "regression"
    IMPROVEMENT = "improvement"
    NEUTRAL = "neutral"
    INCONCLUSIVE = "inconclusive"

class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class BottleneckType(Enum):
    PERCEPTUAL_OVERLOAD = "perceptual_overload"
    CHOICE_PARALYSIS = "choice_paralysis"
    MOTOR_DIFFICULTY = "motor_difficulty"
    MEMORY_DECAY = "memory_decay"
    CROSS_CHANNEL_INTERFERENCE = "cross_channel_interference"

class PipelineStage(Enum):
    PARSE = "parse"
    ALIGN = "align"
    COST = "cost"
    MDP = "mdp"
    BISIMULATION = "bisimulation"
    POLICY = "policy"
    COMPARISON = "comparison"
    BOTTLENECK = "bottleneck"
    REPAIR = "repair"
    OUTPUT = "output"
```

---

## Error Handling

All public API methods raise typed exceptions:

```python
from usability_oracle.core.exceptions import (
    OracleError,              # Base exception
    ParseError,               # Input parsing failed
    ValidationError,          # Invalid configuration or input
    PipelineError,            # Pipeline execution failed
    BisimulationError,        # State abstraction failed
    ComparisonError,          # Comparison failed
    RepairSynthesisError,     # SMT repair failed (e.g., timeout)
)
```

---

## Configuration

The pipeline is configured via YAML or Python dicts:

```yaml
# oracle-config.yaml
pipeline:
  beta_range: [0.1, 20.0]
  n_trajectories: 500
  significance_level: 0.05

bisimulation:
  enabled: true
  epsilon: 0.005

algebra:
  mode: "compositional"   # or "additive"
  coupling: 0.1
  interference: 0.2

bottleneck:
  enabled: true
  min_confidence: 0.7
  max_bottlenecks: 5

repair:
  enabled: false
  timeout_sec: 30
  max_repairs: 3

output:
  format: "json"   # json | sarif | html | console
```

---

*For CLI usage, see the [README](README.md). For theoretical background, see the [tool paper](tool_paper.pdf).*
