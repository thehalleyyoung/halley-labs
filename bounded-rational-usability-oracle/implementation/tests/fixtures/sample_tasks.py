"""Sample task specifications for testing."""

from __future__ import annotations

from usability_oracle.taskspec.models import TaskStep, TaskFlow, TaskSpec


def make_login_task() -> TaskSpec:
    """A simple login form-filling task."""
    steps = [
        TaskStep(step_id="s1", action_type="click", target_role="textfield",
                 target_name="Username", description="Focus username"),
        TaskStep(step_id="s2", action_type="type", target_role="textfield",
                 target_name="Username", input_value="admin",
                 description="Type username", depends_on=["s1"]),
        TaskStep(step_id="s3", action_type="click", target_role="textfield",
                 target_name="Password", description="Focus password",
                 depends_on=["s2"]),
        TaskStep(step_id="s4", action_type="type", target_role="textfield",
                 target_name="Password", input_value="secret",
                 description="Type password", depends_on=["s3"]),
        TaskStep(step_id="s5", action_type="click", target_role="button",
                 target_name="Submit", description="Submit form",
                 depends_on=["s4"]),
    ]
    flow = TaskFlow(flow_id="login", name="Login Flow", steps=steps,
                    success_criteria=["logged_in"])
    return TaskSpec(spec_id="login", name="Login", flows=[flow])


def make_search_task() -> TaskSpec:
    """A search-and-select task."""
    steps = [
        TaskStep(step_id="s1", action_type="click", target_role="searchbox",
                 target_name="Search", description="Focus search box"),
        TaskStep(step_id="s2", action_type="type", target_role="searchbox",
                 target_name="Search", input_value="widget",
                 description="Type query", depends_on=["s1"]),
        TaskStep(step_id="s3", action_type="click", target_role="button",
                 target_name="Search", description="Submit search",
                 depends_on=["s2"]),
        TaskStep(step_id="s4", action_type="click", target_role="link",
                 target_name="Widget Pro", description="Select result",
                 depends_on=["s3"]),
    ]
    flow = TaskFlow(flow_id="search", name="Search Flow", steps=steps)
    return TaskSpec(spec_id="search", name="Search Task", flows=[flow])


def make_navigation_task() -> TaskSpec:
    """A multi-step navigation task."""
    steps = [
        TaskStep(step_id="n1", action_type="click", target_role="link",
                 target_name="Products", description="Nav to products"),
        TaskStep(step_id="n2", action_type="scroll", target_role="main",
                 target_name="Product List", description="Scroll down",
                 depends_on=["n1"]),
        TaskStep(step_id="n3", action_type="click", target_role="button",
                 target_name="Add to Cart", description="Add item",
                 depends_on=["n2"]),
    ]
    flow = TaskFlow(flow_id="nav", name="Navigation", steps=steps)
    return TaskSpec(spec_id="nav", name="Navigation Task", flows=[flow])


def make_dialog_task() -> TaskSpec:
    """A dialog confirmation task."""
    steps = [
        TaskStep(step_id="d1", action_type="verify", target_role="heading",
                 target_name="Confirm Deletion", description="Read heading"),
        TaskStep(step_id="d2", action_type="click", target_role="button",
                 target_name="OK", description="Confirm",
                 depends_on=["d1"]),
    ]
    flow = TaskFlow(flow_id="dialog", name="Dialog", steps=steps)
    return TaskSpec(spec_id="dialog", name="Dialog Task", flows=[flow])
