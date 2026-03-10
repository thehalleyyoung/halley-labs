# Task Specification DSL Reference

## Bounded-Rational Usability Oracle — Task DSL

This document describes the YAML-based task specification language used to define user
tasks for the usability oracle. Task specifications tell the system *what* a user is
trying to accomplish in a UI, enabling construction of task-state MDPs and cognitive
cost analysis.

---

## Table of Contents

- [Overview](#overview)
- [YAML Schema](#yaml-schema)
- [Task Steps](#task-steps)
- [Task Flows](#task-flows)
- [Task Specifications](#task-specifications)
- [Action Types](#action-types)
- [Target Selectors](#target-selectors)
- [Preconditions and Postconditions](#preconditions-and-postconditions)
- [Dependencies](#dependencies)
- [Success Criteria](#success-criteria)
- [Pre-Built Templates](#pre-built-templates)
- [Task Inference](#task-inference)
- [Validation](#validation)
- [Task Graphs](#task-graphs)
- [Implementation Reference](#implementation-reference)
- [Complete Examples](#complete-examples)

---

## Overview

A task specification describes a user's goal and the sequence of interactions required
to achieve it. The oracle uses this to:

1. **Build a task-state MDP** where states encode which steps are completed and the
   current UI focus position
2. **Identify goal states** where the task is complete
3. **Compute cognitive costs** for each transition (step) in the task
4. **Detect regressions** when a UI change increases the cost of completing the task

### Hierarchy

```
TaskSpec
├── spec_id: str
├── name: str
├── description: str
├── flows: list[TaskFlow]
│   └── TaskFlow
│       ├── flow_id: str
│       ├── name: str
│       ├── steps: list[TaskStep]
│       │   └── TaskStep
│       │       ├── step_id: str
│       │       ├── action_type: str
│       │       ├── target_role: str
│       │       ├── target_name: str
│       │       └── ...
│       ├── success_criteria: list
│       └── max_time: float
└── initial_state: dict
```

---

## YAML Schema

### Minimal Example

```yaml
spec_id: login
name: User Login
flows:
  - flow_id: login
    name: Login
    steps:
      - step_id: click-login
        action_type: CLICK
        target_role: BUTTON
        target_name: "Log In"
```

### Full Example

```yaml
spec_id: search-and-purchase
name: Product Search and Purchase
description: >
  User searches for a product, filters results, selects an item,
  and adds it to cart.

initial_state:
  page: "home"
  cart_items: 0

flows:
  - flow_id: search
    name: Search for Product
    description: Enter search query and submit
    steps:
      - step_id: find-search
        action_type: READ
        target_role: TEXTBOX
        target_name: "Search"
        description: "Locate the search field"
        optional: false

      - step_id: enter-query
        action_type: TYPE
        target_role: TEXTBOX
        target_name: "Search"
        input_value: "wireless headphones"
        depends_on: [find-search]

      - step_id: submit-search
        action_type: CLICK
        target_role: BUTTON
        target_name: "Search"
        depends_on: [enter-query]
        postconditions:
          - results_visible: true

    success_criteria:
      - type: state_reached
        condition: "results_visible"
    max_time: 15.0

  - flow_id: filter-and-select
    name: Filter and Select Product
    steps:
      - step_id: apply-filter
        action_type: SELECT
        target_role: COMBOBOX
        target_name: "Price Range"
        input_value: "$50-$100"
        preconditions:
          - results_visible: true

      - step_id: select-product
        action_type: CLICK
        target_role: LINK
        target_selector: ".product-card:first-child"
        depends_on: [apply-filter]

      - step_id: add-to-cart
        action_type: CLICK
        target_role: BUTTON
        target_name: "Add to Cart"
        depends_on: [select-product]
        timeout: 5.0

    success_criteria:
      - type: element_state
        target: "cart-badge"
        state: { text: "1" }
    max_time: 30.0
```

---

## Task Steps

A `TaskStep` is a single user interaction.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `step_id` | str | ✓ | Unique identifier within the flow |
| `action_type` | str | ✓ | Type of user action (see [Action Types](#action-types)) |
| `target_role` | str | | ARIA role of the target element |
| `target_name` | str | | Accessible name of the target element |
| `target_selector` | str | | CSS-like selector for the target |
| `input_value` | str | | Value to input (for TYPE and SELECT actions) |
| `preconditions` | list | | Conditions that must hold before this step |
| `postconditions` | list | | Conditions that hold after this step |
| `optional` | bool | | If true, step can be skipped (default: false) |
| `description` | str | | Human-readable description |
| `timeout` | float | | Maximum expected time for this step (seconds) |
| `depends_on` | list[str] | | Step IDs that must complete before this step |
| `metadata` | dict | | Arbitrary key-value metadata |

### Implementation

```python
from usability_oracle.taskspec.models import TaskStep

step = TaskStep(
    step_id="enter-email",
    action_type="TYPE",
    target_role="TEXTBOX",
    target_name="Email Address",
    input_value="user@example.com",
    depends_on=["find-email-field"],
    description="Enter the user's email address",
)

# Serialization
d = step.to_dict()
step2 = TaskStep.from_dict(d)
```

---

## Task Flows

A `TaskFlow` is an ordered sequence of steps representing one way to accomplish a sub-goal.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `flow_id` | str | ✓ | Unique identifier |
| `name` | str | ✓ | Human-readable name |
| `steps` | list[TaskStep] | ✓ | Ordered steps |
| `success_criteria` | list | | Conditions for flow completion |
| `max_time` | float | | Maximum expected duration (seconds) |
| `description` | str | | Human-readable description |
| `metadata` | dict | | Arbitrary metadata |

### Useful Methods

```python
from usability_oracle.taskspec.models import TaskFlow

flow.step_ids()          # List of step IDs
flow.get_step("enter-email")  # Lookup by ID
flow.required_steps()    # Non-optional steps
flow.input_steps()       # Steps with input_value (TYPE, SELECT)
flow.action_type_counts() # {'CLICK': 3, 'TYPE': 2, 'SELECT': 1}
```

---

## Task Specifications

A `TaskSpec` groups related flows into a complete task specification.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `spec_id` | str | ✓ | Unique identifier |
| `name` | str | ✓ | Human-readable name |
| `description` | str | | Detailed description |
| `flows` | list[TaskFlow] | ✓ | Task flows |
| `initial_state` | dict | | Starting state assumptions |
| `metadata` | dict | | Arbitrary metadata |

---

## Action Types

Supported action types correspond to `Action` constants in `mdp/models.py`:

| Action Type | Description | Requires `input_value` | Typical Cost |
|-------------|-------------|----------------------|-------------|
| `CLICK` | Click/tap an element | No | 0.3s (Fitts') |
| `TYPE` | Enter text into a field | Yes | 0.5s + typing time |
| `TAB` | Tab to next focusable element | No | 0.2s |
| `SCROLL` | Scroll the viewport | No | 0.4s |
| `NAVIGATE` | Navigate to a new page/view | No | 0.6s |
| `READ` | Read/scan content | No | 0.3s + visual search |
| `SELECT` | Select from a dropdown/list | Yes | Hick–Hyman based |
| `BACK` | Navigate back | No | 0.3s |

---

## Target Selectors

Steps target UI elements using three selectors (checked in order):

### 1. Role + Name (Preferred)

```yaml
target_role: BUTTON
target_name: "Submit Order"
```

Matches any `AccessibilityNode` with the given role and accessible name.

### 2. Role Only

```yaml
target_role: TEXTBOX
```

Matches the first matching role (use `target_name` for disambiguation).

### 3. CSS-Like Selector

```yaml
target_selector: "#checkout-button"
target_selector: ".product-card:first-child a"
```

Matches by ID or class-based selector (less portable across UI versions).

### Supported Roles

All `AccessibilityRole` enum values:

```
BUTTON, LINK, TEXTBOX, CHECKBOX, COMBOBOX, RADIO, SLIDER, SWITCH,
MENU, MENUITEM, TAB, TABPANEL, DIALOG, ALERT, HEADING, LIST,
LISTITEM, TABLE, ROW, CELL, IMAGE, NAVIGATION, MAIN, SEARCH,
FORM, BANNER, COMPLEMENTARY, CONTENTINFO, REGION, SEPARATOR,
TOOLBAR, TREE, TREEITEM, GRID, GRIDCELL, PROGRESSBAR, SPINBUTTON,
STATUS, TIMER, TOOLTIP, GENERIC
```

---

## Preconditions and Postconditions

### Preconditions

Conditions that must be true before a step can execute:

```yaml
preconditions:
  - results_visible: true
  - page: "checkout"
  - cart_items: { gte: 1 }
```

### Postconditions

Conditions guaranteed after a step completes:

```yaml
postconditions:
  - form_submitted: true
  - page: "confirmation"
```

### Condition Operators

| Operator | Syntax | Example |
|----------|--------|---------|
| Equals | `key: value` | `page: "home"` |
| Greater than or equal | `key: { gte: n }` | `cart_items: { gte: 1 }` |
| Less than or equal | `key: { lte: n }` | `errors: { lte: 0 }` |
| Contains | `key: { contains: str }` | `text: { contains: "Success" }` |
| Not | `key: { not: value }` | `state: { not: "disabled" }` |

---

## Dependencies

Steps can declare dependencies on other steps using `depends_on`:

```yaml
steps:
  - step_id: enter-username
    action_type: TYPE
    target_role: TEXTBOX
    target_name: "Username"

  - step_id: enter-password
    action_type: TYPE
    target_role: TEXTBOX
    target_name: "Password"
    depends_on: [enter-username]

  - step_id: click-submit
    action_type: CLICK
    target_role: BUTTON
    target_name: "Sign In"
    depends_on: [enter-username, enter-password]
```

This forms a DAG:

```
enter-username ──→ enter-password ──→ click-submit
       └─────────────────────────────→
```

Steps without dependencies can execute in any order (or conceptually in parallel).

---

## Success Criteria

### State Reached

```yaml
success_criteria:
  - type: state_reached
    condition: "logged_in"
```

### Element State

```yaml
success_criteria:
  - type: element_state
    target: "success-message"
    state:
      visible: true
      text: "Welcome back!"
```

### URL Match

```yaml
success_criteria:
  - type: url_match
    pattern: "/dashboard*"
```

---

## Pre-Built Templates

The `TaskTemplates` class provides pre-built specifications for common interaction
patterns:

### Login Form

```python
from usability_oracle.taskspec.templates import TaskTemplates

spec = TaskTemplates.login_form(
    username_label="Email",
    password_label="Password",
    submit_label="Log In",
)
```

### Search and Select

```python
spec = TaskTemplates.search_and_select(
    search_label="Search products",
    result_type="product-card",
    has_filters=True,
)
```

### Form Fill

```python
spec = TaskTemplates.form_fill(
    fields=[
        ("First Name", "TEXTBOX", "John"),
        ("Last Name", "TEXTBOX", "Doe"),
        ("Country", "COMBOBOX", "United States"),
        ("Terms", "CHECKBOX", "true"),
    ],
    submit_label="Submit",
)
```

### Multi-Step Wizard

```python
spec = TaskTemplates.multi_step_wizard(
    step_count=4,
    step_names=["Personal Info", "Address", "Payment", "Review"],
)
```

### Navigation

```python
spec = TaskTemplates.navigation(
    path=["Home", "Products", "Electronics", "Headphones"],
)
```

### Available Templates

| Template | Method | Description |
|----------|--------|-------------|
| Login | `login_form()` | Username + password + submit |
| Search | `search_and_select()` | Search field + results + selection |
| Form | `form_fill()` | Multiple fields + submit |
| Wizard | `multi_step_wizard()` | Multi-step form with navigation |
| Navigation | `navigation()` | Menu navigation to target page |
| Shopping cart | `shopping_cart()` | Add to cart + checkout |
| Settings | `settings()` | Toggle settings on a preferences page |

---

## Task Inference

For common UI patterns, tasks can be **automatically inferred** from the accessibility
tree structure:

```python
from usability_oracle.taskspec.inference import infer_tasks

# Automatically detect forms, navigation, search patterns
specs = infer_tasks(accessibility_tree)
for spec in specs:
    print(f"Inferred: {spec.name} ({len(spec.flows)} flows)")
```

The inference engine detects:

- **Form submissions** (groups of TEXTBOX + BUTTON)
- **Navigation paths** (NAVIGATION → LINK hierarchies)
- **Search patterns** (SEARCH role + TEXTBOX + results region)
- **Tab interfaces** (TAB + TABPANEL groups)
- **Modal interactions** (DIALOG with BUTTON children)

---

## Validation

### Task Spec Validation

```python
from usability_oracle.taskspec.validator import TaskSpecValidator

validator = TaskSpecValidator()
result = validator.validate(task_spec)

if not result.valid:
    for issue in result.issues:
        print(f"[{issue.severity}] {issue.message}")
```

### Validation Checks

| Check | Description |
|-------|-------------|
| Unique step IDs | No duplicate step_id within a flow |
| Valid action types | All action_types are recognized |
| Dependency graph acyclic | No circular dependencies |
| Dependencies exist | All depends_on reference valid step_ids |
| Roles valid | All target_roles are valid ARIA roles |
| Input values present | TYPE and SELECT steps have input_value |

### UI Compatibility Validation

```bash
usability-oracle validate --task-spec-file task.yaml --ui-source ui.html
```

Checks that task targets (roles + names) exist in the actual UI.

---

## Task Graphs

The `TaskGraph` class constructs a dependency DAG from task steps:

```python
from usability_oracle.taskspec.models import TaskGraph

graph = TaskGraph(task_flow)

# Properties
graph.roots()          # Steps with no dependencies
graph.leaves()         # Steps with no dependents
graph.topological_order()  # Valid execution ordering
graph.critical_path()  # Longest dependency chain
graph.parallelizable() # Steps that can execute concurrently
```

The MDP builder uses the task graph to:

1. Determine valid action orderings at each state
2. Track task progress (which steps are completed)
3. Identify goal states (all required steps completed)

---

## Implementation Reference

| Class | Module | Key Methods |
|-------|--------|-------------|
| `TaskStep` | `taskspec/models.py` | `to_dict()`, `from_dict()` |
| `TaskFlow` | `taskspec/models.py` | `step_ids()`, `get_step()`, `required_steps()` |
| `TaskSpec` | `taskspec/models.py` | Container for flows |
| `TaskGraph` | `taskspec/models.py` | `topological_order()`, `critical_path()` |
| `TaskDSLParser` | `taskspec/dsl.py` | `parse()`, `parse_file()`, `to_yaml()` |
| `TaskTemplates` | `taskspec/templates.py` | `login_form()`, `search_and_select()`, etc. |
| `TaskSpecValidator` | `taskspec/validator.py` | `validate()` |

---

## Complete Examples

### Example: E-Commerce Checkout

```yaml
spec_id: checkout
name: E-Commerce Checkout
description: Complete a purchase from cart to confirmation

flows:
  - flow_id: review-cart
    name: Review Cart
    steps:
      - step_id: verify-items
        action_type: READ
        target_role: TABLE
        target_name: "Cart Items"

      - step_id: proceed
        action_type: CLICK
        target_role: BUTTON
        target_name: "Proceed to Checkout"
        depends_on: [verify-items]

    success_criteria:
      - type: state_reached
        condition: "checkout_page"
    max_time: 20.0

  - flow_id: enter-shipping
    name: Enter Shipping Info
    steps:
      - step_id: address-line-1
        action_type: TYPE
        target_role: TEXTBOX
        target_name: "Address Line 1"
        input_value: "123 Main St"

      - step_id: city
        action_type: TYPE
        target_role: TEXTBOX
        target_name: "City"
        input_value: "Springfield"

      - step_id: state
        action_type: SELECT
        target_role: COMBOBOX
        target_name: "State"
        input_value: "IL"

      - step_id: zip
        action_type: TYPE
        target_role: TEXTBOX
        target_name: "ZIP Code"
        input_value: "62704"

      - step_id: continue
        action_type: CLICK
        target_role: BUTTON
        target_name: "Continue to Payment"
        depends_on: [address-line-1, city, state, zip]

    success_criteria:
      - type: state_reached
        condition: "payment_page"
    max_time: 60.0

  - flow_id: payment
    name: Enter Payment
    steps:
      - step_id: card-number
        action_type: TYPE
        target_role: TEXTBOX
        target_name: "Card Number"
        input_value: "4111111111111111"

      - step_id: expiry
        action_type: TYPE
        target_role: TEXTBOX
        target_name: "Expiration"
        input_value: "12/25"

      - step_id: cvv
        action_type: TYPE
        target_role: TEXTBOX
        target_name: "CVV"
        input_value: "123"

      - step_id: place-order
        action_type: CLICK
        target_role: BUTTON
        target_name: "Place Order"
        depends_on: [card-number, expiry, cvv]

    success_criteria:
      - type: element_state
        target: "confirmation-message"
        state: { visible: true }
    max_time: 45.0
```
