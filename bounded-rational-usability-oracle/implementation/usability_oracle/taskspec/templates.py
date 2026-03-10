"""
usability_oracle.taskspec.templates — Pre-built task specification templates.

Provides factory methods for commonly occurring UI interaction patterns.
Templates generate fully-formed :class:`TaskSpec` objects that can be used
directly or customised after generation.

Supported patterns
------------------
* Login forms
* Search and select workflows
* Generic form filling
* Page navigation
* Multi-step wizards
* Shopping cart interactions
* Settings modification
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from usability_oracle.taskspec.models import TaskFlow, TaskSpec, TaskStep


def _gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


class TaskTemplates:
    """Library of pre-built :class:`TaskSpec` templates.

    All methods are class methods and return fully-formed :class:`TaskSpec`
    objects with populated steps, flows, and success criteria.

    Usage::

        spec = TaskTemplates.login_form()
        spec = TaskTemplates.form_fill(["First Name", "Last Name", "Email"])
        spec = TaskTemplates.multi_step_wizard(4)
    """

    # -- login form ----------------------------------------------------------

    @classmethod
    def login_form(
        cls,
        *,
        username_label: str = "Username",
        password_label: str = "Password",
        submit_label: str = "Sign In",
        has_remember_me: bool = False,
        has_forgot_password: bool = False,
    ) -> TaskSpec:
        """Generate a login form task specification.

        Parameters
        ----------
        username_label, password_label, submit_label : str
            Labels for the form widgets.
        has_remember_me : bool
            Include a "Remember me" checkbox step.
        has_forgot_password : bool
            Include an alternative "forgot password" flow.
        """
        steps: List[TaskStep] = [
            TaskStep(
                step_id=_gen_id("step"),
                action_type="click",
                target_role="textfield",
                target_name=username_label,
                description=f"Click the {username_label} field",
            ),
            TaskStep(
                step_id=_gen_id("step"),
                action_type="type",
                target_role="textfield",
                target_name=username_label,
                input_value="user@example.com",
                description=f"Enter username/email",
            ),
            TaskStep(
                step_id=_gen_id("step"),
                action_type="click",
                target_role="textfield",
                target_name=password_label,
                description=f"Click the {password_label} field",
            ),
            TaskStep(
                step_id=_gen_id("step"),
                action_type="type",
                target_role="textfield",
                target_name=password_label,
                input_value="password123",
                description=f"Enter password",
            ),
        ]

        if has_remember_me:
            steps.append(TaskStep(
                step_id=_gen_id("step"),
                action_type="click",
                target_role="checkbox",
                target_name="Remember me",
                optional=True,
                description="Toggle 'Remember me' checkbox",
            ))

        steps.append(TaskStep(
            step_id=_gen_id("step"),
            action_type="click",
            target_role="button",
            target_name=submit_label,
            description=f"Click {submit_label}",
            postconditions=["authenticated == true", "page == /dashboard"],
        ))

        main_flow = TaskFlow(
            flow_id=_gen_id("flow"),
            name="standard_login",
            steps=steps,
            success_criteria=["authenticated == true", "page == /dashboard"],
            max_time=30.0,
            description="Standard username/password login",
        )

        flows = [main_flow]

        if has_forgot_password:
            forgot_flow = TaskFlow(
                flow_id=_gen_id("flow"),
                name="forgot_password",
                steps=[
                    TaskStep(
                        step_id=_gen_id("step"),
                        action_type="click",
                        target_role="link",
                        target_name="Forgot password?",
                        description="Click forgot password link",
                    ),
                    TaskStep(
                        step_id=_gen_id("step"),
                        action_type="click",
                        target_role="textfield",
                        target_name="Email",
                        description="Click email field",
                    ),
                    TaskStep(
                        step_id=_gen_id("step"),
                        action_type="type",
                        target_role="textfield",
                        target_name="Email",
                        input_value="user@example.com",
                        description="Enter email for password reset",
                    ),
                    TaskStep(
                        step_id=_gen_id("step"),
                        action_type="click",
                        target_role="button",
                        target_name="Send Reset Link",
                        description="Submit password reset request",
                        postconditions=["reset_email_sent"],
                    ),
                ],
                success_criteria=["reset_email_sent"],
                max_time=20.0,
                description="Forgotten password recovery flow",
            )
            flows.append(forgot_flow)

        return TaskSpec(
            spec_id=_gen_id("spec"),
            name="login_form",
            description="User authentication via login form",
            flows=flows,
            initial_state={"page": "/login", "authenticated": False},
            metadata={"template": "login_form"},
        )

    # -- search and select ---------------------------------------------------

    @classmethod
    def search_and_select(
        cls,
        *,
        search_label: str = "Search",
        result_type: str = "link",
        has_filters: bool = False,
    ) -> TaskSpec:
        """Generate a search → select result task specification."""
        steps: List[TaskStep] = [
            TaskStep(
                step_id=_gen_id("step"),
                action_type="click",
                target_role="searchbox",
                target_name=search_label,
                description=f"Click {search_label} box",
            ),
            TaskStep(
                step_id=_gen_id("step"),
                action_type="type",
                target_role="searchbox",
                target_name=search_label,
                input_value="<search_query>",
                description="Enter search query",
                postconditions=["search_submitted"],
            ),
        ]

        if has_filters:
            steps.append(TaskStep(
                step_id=_gen_id("step"),
                action_type="click",
                target_role="combobox",
                target_name="Filter",
                optional=True,
                description="Open filter dropdown",
            ))
            steps.append(TaskStep(
                step_id=_gen_id("step"),
                action_type="select",
                target_role="option",
                target_name="<filter_value>",
                input_value="<filter_value>",
                optional=True,
                description="Select a filter option",
            ))

        steps.extend([
            TaskStep(
                step_id=_gen_id("step"),
                action_type="wait",
                target_role="region",
                target_name="Results",
                description="Wait for search results",
                timeout=5.0,
                preconditions=["search_submitted"],
                postconditions=["results_loaded"],
            ),
            TaskStep(
                step_id=_gen_id("step"),
                action_type="click",
                target_role=result_type,
                target_name="<target_result>",
                description="Select desired search result",
                preconditions=["results_loaded"],
                postconditions=["result_selected"],
            ),
        ])

        flow = TaskFlow(
            flow_id=_gen_id("flow"),
            name="search_and_select",
            steps=steps,
            success_criteria=["result_selected"],
            max_time=20.0,
        )
        return TaskSpec(
            spec_id=_gen_id("spec"),
            name="search_and_select",
            description="Search for an item and select from results",
            flows=[flow],
            metadata={"template": "search_and_select"},
        )

    # -- generic form fill ---------------------------------------------------

    @classmethod
    def form_fill(
        cls,
        fields: List[str],
        *,
        form_name: str = "form",
        submit_label: str = "Submit",
    ) -> TaskSpec:
        """Generate a form-fill task for the given field labels.

        Parameters
        ----------
        fields : list[str]
            Field labels in order of appearance.
        form_name : str
            Name for the form (used in spec naming).
        submit_label : str
            Text on the submit button.
        """
        steps: List[TaskStep] = []
        for i, field_name in enumerate(fields):
            steps.append(TaskStep(
                step_id=_gen_id("step"),
                action_type="click",
                target_role="textfield",
                target_name=field_name,
                description=f"Focus {field_name} field",
            ))
            steps.append(TaskStep(
                step_id=_gen_id("step"),
                action_type="type",
                target_role="textfield",
                target_name=field_name,
                input_value=f"<{field_name.lower().replace(' ', '_')}_value>",
                description=f"Enter {field_name}",
            ))

        steps.append(TaskStep(
            step_id=_gen_id("step"),
            action_type="click",
            target_role="button",
            target_name=submit_label,
            description=f"Click {submit_label}",
            postconditions=[f"{form_name}_submitted"],
        ))

        flow = TaskFlow(
            flow_id=_gen_id("flow"),
            name=f"fill_{form_name}",
            steps=steps,
            success_criteria=[f"{form_name}_submitted"],
        )
        return TaskSpec(
            spec_id=_gen_id("spec"),
            name=f"form_fill_{form_name}",
            description=f"Fill and submit the {form_name} form ({len(fields)} fields)",
            flows=[flow],
            metadata={"template": "form_fill", "field_count": len(fields)},
        )

    # -- navigation ----------------------------------------------------------

    @classmethod
    def navigation(
        cls,
        target: str,
        *,
        via: str = "link",
        from_page: str = "/",
    ) -> TaskSpec:
        """Generate a single-step navigation task.

        Parameters
        ----------
        target : str
            The name / label of the link / button to click.
        via : str
            The role of the navigation element (``link``, ``tab``, etc.).
        from_page : str
            The starting page.
        """
        steps = [
            TaskStep(
                step_id=_gen_id("step"),
                action_type="click",
                target_role=via,
                target_name=target,
                description=f"Navigate to {target}",
                postconditions=[f"page == /{target.lower().replace(' ', '-')}"],
            ),
        ]
        flow = TaskFlow(
            flow_id=_gen_id("flow"),
            name=f"navigate_to_{target.lower().replace(' ', '_')}",
            steps=steps,
            success_criteria=[f"page == /{target.lower().replace(' ', '-')}"],
        )
        return TaskSpec(
            spec_id=_gen_id("spec"),
            name=f"navigation_{target}",
            description=f"Navigate from {from_page} to {target}",
            flows=[flow],
            initial_state={"page": from_page},
            metadata={"template": "navigation"},
        )

    # -- multi-step wizard ---------------------------------------------------

    @classmethod
    def multi_step_wizard(
        cls,
        n_steps: int,
        *,
        wizard_name: str = "wizard",
        fields_per_step: int = 3,
        has_back_button: bool = True,
    ) -> TaskSpec:
        """Generate a multi-step wizard task specification.

        Parameters
        ----------
        n_steps : int
            Number of wizard pages/steps.
        wizard_name : str
            Name for the wizard.
        fields_per_step : int
            Number of fields per wizard page.
        has_back_button : bool
            Whether "Back" buttons are available.
        """
        if n_steps < 1:
            raise ValueError("Wizard must have at least 1 step.")

        all_steps: List[TaskStep] = []
        for page_idx in range(n_steps):
            page_num = page_idx + 1
            # Fields for this wizard page
            for field_idx in range(fields_per_step):
                field_name = f"Step {page_num} Field {field_idx + 1}"
                all_steps.append(TaskStep(
                    step_id=_gen_id("step"),
                    action_type="click",
                    target_role="textfield",
                    target_name=field_name,
                    description=f"Focus {field_name}",
                ))
                all_steps.append(TaskStep(
                    step_id=_gen_id("step"),
                    action_type="type",
                    target_role="textfield",
                    target_name=field_name,
                    input_value=f"<{field_name.lower().replace(' ', '_')}_value>",
                    description=f"Fill {field_name}",
                ))

            # Navigation button
            if page_idx < n_steps - 1:
                all_steps.append(TaskStep(
                    step_id=_gen_id("step"),
                    action_type="click",
                    target_role="button",
                    target_name="Next",
                    description=f"Advance to step {page_num + 1}",
                    postconditions=[f"wizard_page == {page_num + 1}"],
                ))
            else:
                all_steps.append(TaskStep(
                    step_id=_gen_id("step"),
                    action_type="click",
                    target_role="button",
                    target_name="Finish",
                    description="Complete wizard",
                    postconditions=[f"{wizard_name}_completed"],
                ))

        flow = TaskFlow(
            flow_id=_gen_id("flow"),
            name=f"{wizard_name}_forward",
            steps=all_steps,
            success_criteria=[f"{wizard_name}_completed"],
            max_time=float(n_steps * 30),
        )
        return TaskSpec(
            spec_id=_gen_id("spec"),
            name=f"wizard_{wizard_name}",
            description=f"{n_steps}-step wizard: {wizard_name}",
            flows=[flow],
            initial_state={"wizard_page": 1},
            metadata={
                "template": "multi_step_wizard",
                "n_steps": n_steps,
                "fields_per_step": fields_per_step,
            },
        )

    # -- shopping cart -------------------------------------------------------

    @classmethod
    def shopping_cart(
        cls,
        *,
        n_items: int = 1,
        has_quantity: bool = True,
        has_promo_code: bool = False,
    ) -> TaskSpec:
        """Generate a shopping cart checkout task specification."""
        steps: List[TaskStep] = []

        for i in range(n_items):
            item_name = f"Product {i + 1}" if n_items > 1 else "Product"
            steps.append(TaskStep(
                step_id=_gen_id("step"),
                action_type="click",
                target_role="button",
                target_name=f"Add {item_name} to Cart",
                description=f"Add {item_name} to cart",
                postconditions=[f"cart_contains_{item_name.lower().replace(' ', '_')}"],
            ))
            if has_quantity:
                steps.append(TaskStep(
                    step_id=_gen_id("step"),
                    action_type="select",
                    target_role="spinbutton",
                    target_name=f"Quantity for {item_name}",
                    input_value="1",
                    optional=True,
                    description=f"Set quantity for {item_name}",
                ))

        # Navigate to cart
        steps.append(TaskStep(
            step_id=_gen_id("step"),
            action_type="click",
            target_role="link",
            target_name="View Cart",
            description="Open shopping cart",
            postconditions=["page == /cart"],
        ))

        if has_promo_code:
            steps.extend([
                TaskStep(
                    step_id=_gen_id("step"),
                    action_type="click",
                    target_role="textfield",
                    target_name="Promo Code",
                    optional=True,
                    description="Click promo code field",
                ),
                TaskStep(
                    step_id=_gen_id("step"),
                    action_type="type",
                    target_role="textfield",
                    target_name="Promo Code",
                    input_value="<promo_code>",
                    optional=True,
                    description="Enter promo code",
                ),
                TaskStep(
                    step_id=_gen_id("step"),
                    action_type="click",
                    target_role="button",
                    target_name="Apply",
                    optional=True,
                    description="Apply promo code",
                ),
            ])

        # Checkout
        steps.append(TaskStep(
            step_id=_gen_id("step"),
            action_type="click",
            target_role="button",
            target_name="Proceed to Checkout",
            description="Proceed to checkout",
            postconditions=["checkout_started"],
        ))

        flow = TaskFlow(
            flow_id=_gen_id("flow"),
            name="add_and_checkout",
            steps=steps,
            success_criteria=["checkout_started"],
            max_time=60.0,
        )
        return TaskSpec(
            spec_id=_gen_id("spec"),
            name="shopping_cart",
            description=f"Add {n_items} item(s) to cart and proceed to checkout",
            flows=[flow],
            initial_state={"page": "/products", "cart_empty": True},
            metadata={"template": "shopping_cart", "n_items": n_items},
        )

    # -- settings change -----------------------------------------------------

    @classmethod
    def settings_change(
        cls,
        setting: str,
        value: str,
        *,
        settings_page: str = "/settings",
        setting_type: str = "textfield",
        requires_save: bool = True,
    ) -> TaskSpec:
        """Generate a settings modification task.

        Parameters
        ----------
        setting : str
            Name / label of the setting to change.
        value : str
            New value to set.
        settings_page : str
            URL / path to the settings page.
        setting_type : str
            The role of the setting widget (textfield, checkbox, combobox, …).
        requires_save : bool
            Whether an explicit Save/Apply button must be clicked.
        """
        steps: List[TaskStep] = [
            TaskStep(
                step_id=_gen_id("step"),
                action_type="navigate",
                target_role="link",
                target_name="Settings",
                description=f"Navigate to settings page",
                postconditions=[f"page == {settings_page}"],
            ),
        ]

        if setting_type in ("checkbox", "switch"):
            steps.append(TaskStep(
                step_id=_gen_id("step"),
                action_type="click",
                target_role=setting_type,
                target_name=setting,
                description=f"Toggle {setting}",
                postconditions=[f"{setting.lower().replace(' ', '_')} == {value}"],
            ))
        elif setting_type in ("combobox", "listbox"):
            steps.append(TaskStep(
                step_id=_gen_id("step"),
                action_type="click",
                target_role=setting_type,
                target_name=setting,
                description=f"Open {setting} dropdown",
            ))
            steps.append(TaskStep(
                step_id=_gen_id("step"),
                action_type="select",
                target_role="option",
                target_name=value,
                input_value=value,
                description=f"Select {value}",
            ))
        else:
            steps.append(TaskStep(
                step_id=_gen_id("step"),
                action_type="click",
                target_role=setting_type,
                target_name=setting,
                description=f"Click {setting} field",
            ))
            steps.append(TaskStep(
                step_id=_gen_id("step"),
                action_type="type",
                target_role=setting_type,
                target_name=setting,
                input_value=value,
                description=f"Enter new value for {setting}",
            ))

        if requires_save:
            steps.append(TaskStep(
                step_id=_gen_id("step"),
                action_type="click",
                target_role="button",
                target_name="Save",
                description="Save settings",
                postconditions=["settings_saved"],
            ))

        flow = TaskFlow(
            flow_id=_gen_id("flow"),
            name=f"change_{setting.lower().replace(' ', '_')}",
            steps=steps,
            success_criteria=["settings_saved"] if requires_save else [
                f"{setting.lower().replace(' ', '_')} == {value}"
            ],
        )
        return TaskSpec(
            spec_id=_gen_id("spec"),
            name=f"settings_{setting.lower().replace(' ', '_')}",
            description=f"Change {setting} to {value}",
            flows=[flow],
            initial_state={"page": "/"},
            metadata={"template": "settings_change", "setting": setting, "value": value},
        )
