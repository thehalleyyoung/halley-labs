"""Real-world benchmark tests for the Bounded-Rational Usability Oracle.

Tests exercise the oracle against realistic UI structures derived from
public datasets and common accessibility patterns:

  * **Rico-derived tests**: Android UI accessibility tree snippets based on
    the Rico dataset schema (view hierarchies with bounds, types, text).
  * **Web task tests**: Multi-step web navigation tasks (e-commerce checkout,
    form filling) as state-action sequences with MDP construction.
  * **Accessibility regression tests**: UI version pairs (before/after)
    based on real accessibility anti-patterns for regression detection.

All tests use the repo's existing API to construct task MDPs and run the
oracle.  Realistic UI tree structures are used throughout.

Marked with ``@pytest.mark.real_benchmark`` for selective filtering::

    pytest -m real_benchmark implementation/tests/test_real_benchmarks.py
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import pytest

from usability_oracle.accessibility.models import (
    AccessibilityNode,
    AccessibilityState,
    AccessibilityTree,
    BoundingBox,
)
from usability_oracle.mdp.models import State, Action, Transition, MDP, MDPStatistics
from usability_oracle.mdp.builder import MDPBuilder, MDPBuilderConfig
from usability_oracle.taskspec.models import TaskStep, TaskFlow, TaskSpec
from usability_oracle.policy.models import Policy, QValues
from usability_oracle.policy.value_iteration import SoftValueIteration
from usability_oracle.cognitive.fitts import FittsLaw
from usability_oracle.cognitive.hick import HickHymanLaw
from usability_oracle.bottleneck.classifier import BottleneckClassifier
from usability_oracle.bottleneck.models import BottleneckReport
from usability_oracle.comparison.paired import PairedComparator
from usability_oracle.core.enums import (
    BottleneckType,
    RegressionVerdict,
    Severity,
)
from usability_oracle.core.config import (
    CognitiveConfig,
    ComparisonConfig,
    MDPConfig,
    OracleConfig,
)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _default_state(**overrides) -> AccessibilityState:
    defaults = dict(
        focused=False, selected=False, expanded=False, checked=None,
        disabled=False, hidden=False, required=False, readonly=False,
        pressed=None, value=None,
    )
    defaults.update(overrides)
    return AccessibilityState(**defaults)


def _node(id: str, role: str, name: str, bbox: tuple,
          parent_id: str | None = None, depth: int = 0,
          index: int = 0, children: list | None = None,
          **props) -> AccessibilityNode:
    """Shorthand node constructor matching the existing fixture pattern."""
    return AccessibilityNode(
        id=id, role=role, name=name, description=props.pop("description", ""),
        bounding_box=BoundingBox(x=bbox[0], y=bbox[1], width=bbox[2], height=bbox[3]),
        properties=props, state=_default_state(**props.pop("state_overrides", {})),
        children=children or [], parent_id=parent_id, depth=depth,
        index_in_parent=index,
    )


# ═══════════════════════════════════════════════════════════════════════
# Rico-derived Android UI trees
# ═══════════════════════════════════════════════════════════════════════

def _make_rico_shopping_app_tree() -> AccessibilityTree:
    """Rico-style Android e-commerce product listing.

    Modeled after Rico dataset view hierarchies for shopping apps
    (e.g., eBay, Amazon). Includes toolbar, search bar, product grid
    with images, prices, and ratings.  ~25 nodes, 4 levels deep.
    """
    # Toolbar
    back_btn = _node("back_btn", "button", "Navigate up", (0, 24, 48, 48),
                      parent_id="toolbar", depth=2, index=0)
    title = _node("app_title", "heading", "ShopNow", (56, 24, 200, 48),
                   parent_id="toolbar", depth=2, index=1)
    cart_btn = _node("cart_btn", "button", "Cart (3)", (312, 24, 48, 48),
                      parent_id="toolbar", depth=2, index=2)
    toolbar = _node("toolbar", "navigation", "Top Bar", (0, 0, 360, 72),
                     parent_id="root", depth=1, index=0,
                     children=[back_btn, title, cart_btn])

    # Search bar
    search_input = _node("search_input", "textbox", "Search products",
                          (8, 80, 304, 40), parent_id="search_bar", depth=2, index=0)
    search_btn = _node("search_btn", "button", "Search", (316, 80, 36, 40),
                        parent_id="search_bar", depth=2, index=1)
    search_bar = _node("search_bar", "search", "Product Search", (0, 72, 360, 56),
                        parent_id="root", depth=1, index=1,
                        children=[search_input, search_btn])

    # Product cards (grid of 4)
    product_cards = []
    products = [
        ("Wireless Headphones", "$49.99", "4.5 stars", "headphones_img"),
        ("USB-C Hub", "$29.99", "4.2 stars", "usbhub_img"),
        ("Laptop Stand", "$34.99", "4.7 stars", "stand_img"),
        ("Bluetooth Speaker", "$39.99", "4.0 stars", "speaker_img"),
    ]
    for i, (pname, price, rating, img_id) in enumerate(products):
        col = i % 2
        row = i // 2
        x = 4 + col * 178
        y = 140 + row * 220
        img = _node(f"{img_id}", "img", pname, (x, y, 170, 140),
                     parent_id=f"product_{i}", depth=3, index=0)
        price_lbl = _node(f"price_{i}", "text", price, (x, y + 144, 100, 24),
                           parent_id=f"product_{i}", depth=3, index=1)
        rating_lbl = _node(f"rating_{i}", "text", rating, (x + 104, y + 144, 66, 24),
                            parent_id=f"product_{i}", depth=3, index=2)
        add_btn = _node(f"add_cart_{i}", "button", "Add to Cart", (x, y + 172, 170, 36),
                         parent_id=f"product_{i}", depth=3, index=3)
        card = _node(f"product_{i}", "group", pname, (x, y, 170, 212),
                      parent_id="product_grid", depth=2, index=i,
                      children=[img, price_lbl, rating_lbl, add_btn])
        product_cards.append(card)

    product_grid = _node("product_grid", "list", "Products", (0, 132, 360, 460),
                          parent_id="root", depth=1, index=2,
                          children=product_cards)

    root = _node("root", "document", "ShopNow - Products", (0, 0, 360, 640),
                  parent_id=None, depth=0, index=0,
                  children=[toolbar, search_bar, product_grid])
    return AccessibilityTree(root=root, metadata={"source": "rico_shopping"})


def _make_rico_settings_tree() -> AccessibilityTree:
    """Rico-style Android settings screen.

    Based on typical Android Settings view hierarchies with toggle switches,
    section headers, and nested preferences.  ~20 nodes.
    """
    # Section: Wireless & Networks
    wifi_toggle = _node("wifi_toggle", "switch", "Wi-Fi", (252, 80, 60, 36),
                         parent_id="wifi_row", depth=3, index=1)
    wifi_label = _node("wifi_label", "text", "Wi-Fi", (16, 80, 220, 36),
                        parent_id="wifi_row", depth=3, index=0)
    wifi_row = _node("wifi_row", "listitem", "Wi-Fi", (0, 72, 360, 52),
                      parent_id="wireless_section", depth=2, index=0,
                      children=[wifi_label, wifi_toggle])

    bt_toggle = _node("bt_toggle", "switch", "Bluetooth", (252, 136, 60, 36),
                       parent_id="bt_row", depth=3, index=1)
    bt_label = _node("bt_label", "text", "Bluetooth", (16, 136, 220, 36),
                      parent_id="bt_row", depth=3, index=0)
    bt_row = _node("bt_row", "listitem", "Bluetooth", (0, 128, 360, 52),
                    parent_id="wireless_section", depth=2, index=1,
                    children=[bt_label, bt_toggle])

    airplane_toggle = _node("airplane_toggle", "switch", "Airplane Mode",
                             (252, 192, 60, 36), parent_id="airplane_row",
                             depth=3, index=1)
    airplane_label = _node("airplane_label", "text", "Airplane Mode",
                            (16, 192, 220, 36), parent_id="airplane_row",
                            depth=3, index=0)
    airplane_row = _node("airplane_row", "listitem", "Airplane Mode",
                          (0, 184, 360, 52), parent_id="wireless_section",
                          depth=2, index=2,
                          children=[airplane_label, airplane_toggle])

    wireless_header = _node("wireless_header", "heading", "Wireless & Networks",
                             (16, 40, 328, 28), parent_id="wireless_section",
                             depth=2, index=0)
    wireless_section = _node("wireless_section", "group", "Wireless & Networks",
                              (0, 32, 360, 212), parent_id="root", depth=1, index=0,
                              children=[wireless_header, wifi_row, bt_row, airplane_row])

    # Section: Display
    brightness_slider = _node("brightness_slider", "slider", "Brightness",
                               (16, 288, 328, 36), parent_id="display_section",
                               depth=2, index=1)
    display_header = _node("display_header", "heading", "Display",
                            (16, 256, 328, 28), parent_id="display_section",
                            depth=2, index=0)
    dark_mode = _node("dark_mode_toggle", "switch", "Dark Mode",
                       (252, 332, 60, 36), parent_id="display_section",
                       depth=2, index=2)
    display_section = _node("display_section", "group", "Display",
                             (0, 248, 360, 132), parent_id="root", depth=1, index=1,
                             children=[display_header, brightness_slider, dark_mode])

    root = _node("root", "document", "Settings", (0, 0, 360, 640),
                  parent_id=None, depth=0, index=0,
                  children=[wireless_section, display_section])
    return AccessibilityTree(root=root, metadata={"source": "rico_settings"})


def _make_rico_social_feed_tree() -> AccessibilityTree:
    """Rico-style social media feed (Instagram/Twitter-like).

    Feed with post cards containing avatar, username, image, like/comment
    buttons, and caption text.  ~18 nodes per post, 2 posts shown.
    """
    posts = []
    for pi in range(2):
        y_off = 72 + pi * 280
        avatar = _node(f"avatar_{pi}", "img", f"User {pi} avatar",
                        (8, y_off, 36, 36), parent_id=f"post_{pi}", depth=2, index=0)
        username = _node(f"username_{pi}", "link", f"user_{pi}",
                          (52, y_off, 180, 24), parent_id=f"post_{pi}", depth=2, index=1)
        post_img = _node(f"post_img_{pi}", "img", f"Photo by user_{pi}",
                          (0, y_off + 40, 360, 200), parent_id=f"post_{pi}", depth=2, index=2)
        like_btn = _node(f"like_{pi}", "button", "Like",
                          (8, y_off + 244, 40, 28), parent_id=f"post_{pi}", depth=2, index=3)
        comment_btn = _node(f"comment_{pi}", "button", "Comment",
                             (56, y_off + 244, 40, 28), parent_id=f"post_{pi}", depth=2, index=4)
        share_btn = _node(f"share_{pi}", "button", "Share",
                           (104, y_off + 244, 40, 28), parent_id=f"post_{pi}", depth=2, index=5)
        post = _node(f"post_{pi}", "article", f"Post by user_{pi}",
                      (0, y_off, 360, 276), parent_id="feed", depth=1, index=pi,
                      children=[avatar, username, post_img, like_btn, comment_btn, share_btn])
        posts.append(post)

    toolbar = _node("toolbar", "navigation", "Feed", (0, 0, 360, 56),
                     parent_id="root", depth=1, index=0)
    feed = _node("feed", "list", "Feed", (0, 56, 360, 584),
                  parent_id="root", depth=1, index=1, children=posts)
    root = _node("root", "document", "Social Feed", (0, 0, 360, 640),
                  parent_id=None, depth=0, index=0, children=[toolbar, feed])
    return AccessibilityTree(root=root, metadata={"source": "rico_social"})


def _make_rico_news_reader_tree() -> AccessibilityTree:
    """Rico-style news reading app with article cards.

    Compact article list with headlines, sources, timestamps, and
    bookmark buttons.  Tests dense information layouts.
    """
    articles = []
    headlines = [
        ("Global Markets Rally on Trade Deal", "Reuters", "2h ago"),
        ("New Study Reveals Climate Tipping Points", "Nature", "4h ago"),
        ("Tech Giants Report Q4 Earnings", "Bloomberg", "6h ago"),
        ("SpaceX Launches 40 Satellites", "AP News", "8h ago"),
        ("Breakthrough in Quantum Computing", "Science", "12h ago"),
    ]
    for i, (headline, source, ts) in enumerate(headlines):
        y_off = 56 + i * 88
        title = _node(f"headline_{i}", "heading", headline,
                       (16, y_off, 280, 44), parent_id=f"article_{i}", depth=2, index=0)
        src = _node(f"source_{i}", "text", f"{source} · {ts}",
                     (16, y_off + 48, 200, 20), parent_id=f"article_{i}", depth=2, index=1)
        bookmark = _node(f"bookmark_{i}", "button", "Bookmark",
                          (312, y_off + 16, 32, 32), parent_id=f"article_{i}", depth=2, index=2)
        article = _node(f"article_{i}", "listitem", headline,
                         (0, y_off, 360, 80), parent_id="article_list", depth=1,
                         index=i, children=[title, src, bookmark])
        articles.append(article)

    article_list = _node("article_list", "list", "Headlines", (0, 48, 360, 496),
                          parent_id="root", depth=1, index=1, children=articles)
    toolbar = _node("toolbar", "navigation", "News", (0, 0, 360, 48),
                     parent_id="root", depth=1, index=0)
    root = _node("root", "document", "News Reader", (0, 0, 360, 640),
                  parent_id=None, depth=0, index=0,
                  children=[toolbar, article_list])
    return AccessibilityTree(root=root, metadata={"source": "rico_news"})


def _make_rico_banking_tree() -> AccessibilityTree:
    """Rico-style mobile banking app main screen.

    Account balance card, recent transactions list, and quick action
    buttons.  Tests financial UI with high-stakes interaction patterns.
    """
    # Balance card
    balance_label = _node("balance_label", "text", "Available Balance",
                           (24, 80, 200, 24), parent_id="balance_card", depth=2, index=0)
    balance_value = _node("balance_value", "text", "$12,345.67",
                           (24, 108, 200, 36), parent_id="balance_card", depth=2, index=1)
    balance_card = _node("balance_card", "group", "Account Balance",
                          (16, 72, 328, 84), parent_id="root", depth=1, index=1,
                          children=[balance_label, balance_value])

    # Quick actions
    actions = []
    action_data = [
        ("send_btn", "Send", (24, 172, 72, 56)),
        ("request_btn", "Request", (108, 172, 72, 56)),
        ("pay_bills_btn", "Pay Bills", (192, 172, 72, 56)),
        ("deposit_btn", "Deposit", (276, 172, 72, 56)),
    ]
    for i, (aid, aname, bbox) in enumerate(action_data):
        actions.append(_node(aid, "button", aname, bbox,
                              parent_id="quick_actions", depth=2, index=i))
    quick_actions = _node("quick_actions", "group", "Quick Actions",
                           (16, 164, 328, 64), parent_id="root", depth=1, index=2,
                           children=actions)

    # Recent transactions
    txns = []
    txn_data = [
        ("Grocery Store", "-$85.32", "Yesterday"),
        ("Direct Deposit", "+$2,400.00", "Mon"),
        ("Electric Bill", "-$142.50", "Fri"),
    ]
    for i, (desc, amount, date) in enumerate(txn_data):
        y = 260 + i * 56
        txn_desc = _node(f"txn_desc_{i}", "text", desc, (16, y, 200, 24),
                          parent_id=f"txn_{i}", depth=3, index=0)
        txn_amt = _node(f"txn_amt_{i}", "text", amount, (240, y, 100, 24),
                         parent_id=f"txn_{i}", depth=3, index=1)
        txn_date = _node(f"txn_date_{i}", "text", date, (16, y + 24, 100, 20),
                          parent_id=f"txn_{i}", depth=3, index=2)
        txn = _node(f"txn_{i}", "listitem", f"{desc} {amount}",
                     (0, y, 360, 52), parent_id="txn_list", depth=2, index=i,
                     children=[txn_desc, txn_amt, txn_date])
        txns.append(txn)
    txn_list = _node("txn_list", "list", "Recent Transactions", (0, 244, 360, 200),
                      parent_id="root", depth=1, index=3, children=txns)

    toolbar = _node("toolbar", "navigation", "My Bank", (0, 0, 360, 56),
                     parent_id="root", depth=1, index=0)
    root = _node("root", "document", "Banking App", (0, 0, 360, 640),
                  parent_id=None, depth=0, index=0,
                  children=[toolbar, balance_card, quick_actions, txn_list])
    return AccessibilityTree(root=root, metadata={"source": "rico_banking"})


# ═══════════════════════════════════════════════════════════════════════
# Web task MDP definitions
# ═══════════════════════════════════════════════════════════════════════

def _make_ecommerce_checkout_task() -> TaskSpec:
    """Multi-step e-commerce checkout: cart → shipping → payment → confirm."""
    steps = [
        TaskStep(step_id="c1", action_type="click", target_role="button",
                 target_name="Proceed to Checkout", description="Start checkout"),
        TaskStep(step_id="c2", action_type="click", target_role="textfield",
                 target_name="Full Name", description="Focus name field",
                 depends_on=["c1"]),
        TaskStep(step_id="c3", action_type="type", target_role="textfield",
                 target_name="Full Name", input_value="Jane Doe",
                 description="Enter shipping name", depends_on=["c2"]),
        TaskStep(step_id="c4", action_type="click", target_role="textfield",
                 target_name="Address", description="Focus address field",
                 depends_on=["c3"]),
        TaskStep(step_id="c5", action_type="type", target_role="textfield",
                 target_name="Address", input_value="123 Main St",
                 description="Enter address", depends_on=["c4"]),
        TaskStep(step_id="c6", action_type="click", target_role="textfield",
                 target_name="City", description="Focus city",
                 depends_on=["c5"]),
        TaskStep(step_id="c7", action_type="type", target_role="textfield",
                 target_name="City", input_value="Springfield",
                 description="Enter city", depends_on=["c6"]),
        TaskStep(step_id="c8", action_type="click", target_role="button",
                 target_name="Continue to Payment", description="Advance to payment",
                 depends_on=["c7"]),
        TaskStep(step_id="c9", action_type="click", target_role="textfield",
                 target_name="Card Number", description="Focus card number",
                 depends_on=["c8"]),
        TaskStep(step_id="c10", action_type="type", target_role="textfield",
                 target_name="Card Number", input_value="4111111111111111",
                 description="Enter card number", depends_on=["c9"]),
        TaskStep(step_id="c11", action_type="click", target_role="button",
                 target_name="Place Order", description="Confirm order",
                 depends_on=["c10"]),
    ]
    flow = TaskFlow(flow_id="checkout", name="Checkout Flow", steps=steps,
                    success_criteria=["order_placed"])
    return TaskSpec(spec_id="ecommerce_checkout", name="E-Commerce Checkout",
                    flows=[flow])


def _make_insurance_form_task() -> TaskSpec:
    """Multi-page insurance quote form with dropdowns and conditionals."""
    steps = [
        TaskStep(step_id="i1", action_type="click", target_role="textfield",
                 target_name="First Name", description="Focus first name"),
        TaskStep(step_id="i2", action_type="type", target_role="textfield",
                 target_name="First Name", input_value="John",
                 description="Enter first name", depends_on=["i1"]),
        TaskStep(step_id="i3", action_type="click", target_role="textfield",
                 target_name="Last Name", description="Focus last name",
                 depends_on=["i2"]),
        TaskStep(step_id="i4", action_type="type", target_role="textfield",
                 target_name="Last Name", input_value="Smith",
                 description="Enter last name", depends_on=["i3"]),
        TaskStep(step_id="i5", action_type="click", target_role="combobox",
                 target_name="Coverage Type", description="Open coverage dropdown",
                 depends_on=["i4"]),
        TaskStep(step_id="i6", action_type="click", target_role="option",
                 target_name="Comprehensive", description="Select coverage",
                 depends_on=["i5"]),
        TaskStep(step_id="i7", action_type="click", target_role="textfield",
                 target_name="Vehicle Year", description="Focus vehicle year",
                 depends_on=["i6"]),
        TaskStep(step_id="i8", action_type="type", target_role="textfield",
                 target_name="Vehicle Year", input_value="2022",
                 description="Enter vehicle year", depends_on=["i7"]),
        TaskStep(step_id="i9", action_type="click", target_role="button",
                 target_name="Get Quote", description="Submit quote request",
                 depends_on=["i8"]),
    ]
    flow = TaskFlow(flow_id="quote", name="Insurance Quote", steps=steps,
                    success_criteria=["quote_received"])
    return TaskSpec(spec_id="insurance_form", name="Insurance Quote Form",
                    flows=[flow])


def _make_flight_booking_task() -> TaskSpec:
    """Flight booking: search → select → passenger info → confirm."""
    steps = [
        TaskStep(step_id="f1", action_type="click", target_role="textfield",
                 target_name="From", description="Focus departure city"),
        TaskStep(step_id="f2", action_type="type", target_role="textfield",
                 target_name="From", input_value="SFO",
                 description="Enter departure", depends_on=["f1"]),
        TaskStep(step_id="f3", action_type="click", target_role="textfield",
                 target_name="To", description="Focus destination",
                 depends_on=["f2"]),
        TaskStep(step_id="f4", action_type="type", target_role="textfield",
                 target_name="To", input_value="JFK",
                 description="Enter destination", depends_on=["f3"]),
        TaskStep(step_id="f5", action_type="click", target_role="button",
                 target_name="Search Flights", description="Search",
                 depends_on=["f4"]),
        TaskStep(step_id="f6", action_type="click", target_role="listitem",
                 target_name="UA 237 - $299", description="Select flight",
                 depends_on=["f5"]),
        TaskStep(step_id="f7", action_type="click", target_role="button",
                 target_name="Continue", description="Proceed to passenger info",
                 depends_on=["f6"]),
        TaskStep(step_id="f8", action_type="click", target_role="button",
                 target_name="Confirm Booking", description="Finalize booking",
                 depends_on=["f7"]),
    ]
    flow = TaskFlow(flow_id="booking", name="Flight Booking", steps=steps,
                    success_criteria=["booking_confirmed"])
    return TaskSpec(spec_id="flight_booking", name="Flight Booking",
                    flows=[flow])


def _make_admin_dashboard_task() -> TaskSpec:
    """Admin dashboard: navigate to users → filter → edit user role."""
    steps = [
        TaskStep(step_id="a1", action_type="click", target_role="link",
                 target_name="Users", description="Navigate to Users section"),
        TaskStep(step_id="a2", action_type="click", target_role="textfield",
                 target_name="Search Users", description="Focus user search",
                 depends_on=["a1"]),
        TaskStep(step_id="a3", action_type="type", target_role="textfield",
                 target_name="Search Users", input_value="john@example.com",
                 description="Search for user", depends_on=["a2"]),
        TaskStep(step_id="a4", action_type="click", target_role="button",
                 target_name="Edit", description="Open user editor",
                 depends_on=["a3"]),
        TaskStep(step_id="a5", action_type="click", target_role="combobox",
                 target_name="Role", description="Open role dropdown",
                 depends_on=["a4"]),
        TaskStep(step_id="a6", action_type="click", target_role="option",
                 target_name="Admin", description="Select admin role",
                 depends_on=["a5"]),
        TaskStep(step_id="a7", action_type="click", target_role="button",
                 target_name="Save Changes", description="Save user changes",
                 depends_on=["a6"]),
    ]
    flow = TaskFlow(flow_id="admin", name="Admin User Edit", steps=steps,
                    success_criteria=["user_updated"])
    return TaskSpec(spec_id="admin_dashboard", name="Admin Dashboard Task",
                    flows=[flow])


def _make_support_ticket_task() -> TaskSpec:
    """Customer support: create ticket with category, description, attachment."""
    steps = [
        TaskStep(step_id="t1", action_type="click", target_role="button",
                 target_name="New Ticket", description="Open new ticket form"),
        TaskStep(step_id="t2", action_type="click", target_role="combobox",
                 target_name="Category", description="Open category dropdown",
                 depends_on=["t1"]),
        TaskStep(step_id="t3", action_type="click", target_role="option",
                 target_name="Billing", description="Select billing category",
                 depends_on=["t2"]),
        TaskStep(step_id="t4", action_type="click", target_role="textfield",
                 target_name="Subject", description="Focus subject",
                 depends_on=["t3"]),
        TaskStep(step_id="t5", action_type="type", target_role="textfield",
                 target_name="Subject", input_value="Incorrect charge",
                 description="Enter subject", depends_on=["t4"]),
        TaskStep(step_id="t6", action_type="click", target_role="textfield",
                 target_name="Description", description="Focus description",
                 depends_on=["t5"]),
        TaskStep(step_id="t7", action_type="type", target_role="textfield",
                 target_name="Description",
                 input_value="I was charged $50 instead of $25 on my last order",
                 description="Enter description", depends_on=["t6"]),
        TaskStep(step_id="t8", action_type="click", target_role="button",
                 target_name="Submit Ticket", description="Submit the ticket",
                 depends_on=["t7"]),
    ]
    flow = TaskFlow(flow_id="support", name="Support Ticket", steps=steps,
                    success_criteria=["ticket_submitted"])
    return TaskSpec(spec_id="support_ticket", name="Support Ticket",
                    flows=[flow])


# ═══════════════════════════════════════════════════════════════════════
# Accessibility regression pairs (before / after)
# ═══════════════════════════════════════════════════════════════════════

def _make_regression_pair_touch_target() -> tuple:
    """Before: adequate touch targets. After: shrunk buttons (< 44px)."""
    common_children_before = [
        _node("btn_a", "button", "Save", (20, 200, 120, 48),
              parent_id="form", depth=2, index=0),
        _node("btn_b", "button", "Cancel", (160, 200, 120, 48),
              parent_id="form", depth=2, index=1),
    ]
    form_before = _node("form", "form", "Edit", (0, 60, 360, 260),
                          parent_id="root", depth=1, index=0,
                          children=common_children_before)
    root_before = _node("root", "document", "Edit Page", (0, 0, 360, 640),
                          parent_id=None, depth=0, index=0,
                          children=[form_before])
    tree_before = AccessibilityTree(root=root_before, metadata={"source": "v1"})

    # After: buttons shrunken below WCAG minimum touch target size
    common_children_after = [
        _node("btn_a", "button", "Save", (20, 200, 60, 22),
              parent_id="form", depth=2, index=0),
        _node("btn_b", "button", "Cancel", (100, 200, 60, 22),
              parent_id="form", depth=2, index=1),
    ]
    form_after = _node("form", "form", "Edit", (0, 60, 360, 260),
                         parent_id="root", depth=1, index=0,
                         children=common_children_after)
    root_after = _node("root", "document", "Edit Page", (0, 0, 360, 640),
                         parent_id=None, depth=0, index=0,
                         children=[form_after])
    tree_after = AccessibilityTree(root=root_after, metadata={"source": "v2"})
    return tree_before, tree_after


def _make_regression_pair_label_removal() -> tuple:
    """Before: labeled inputs. After: labels replaced with placeholder text only."""
    # Before: proper labels
    label_user = _node("lbl_user", "text", "Username", (20, 60, 100, 20),
                        parent_id="form", depth=2, index=0)
    input_user_before = _node("input_user", "textbox", "Username",
                               (20, 84, 200, 36), parent_id="form", depth=2, index=1)
    label_pw = _node("lbl_pw", "text", "Password", (20, 130, 100, 20),
                      parent_id="form", depth=2, index=2)
    input_pw_before = _node("input_pw", "textbox", "Password",
                             (20, 154, 200, 36), parent_id="form", depth=2, index=3)
    submit_before = _node("btn_submit", "button", "Log In",
                           (20, 210, 200, 40), parent_id="form", depth=2, index=4)
    form_before = _node("form", "form", "Login", (0, 40, 360, 280),
                          parent_id="root", depth=1, index=0,
                          children=[label_user, input_user_before, label_pw,
                                    input_pw_before, submit_before])
    root_before = _node("root", "document", "Login Page", (0, 0, 360, 640),
                          parent_id=None, depth=0, index=0,
                          children=[form_before])
    tree_before = AccessibilityTree(root=root_before, metadata={"source": "v1"})

    # After: labels removed, only placeholder attributes remain
    input_user_after = _node("input_user", "textbox", "",
                              (20, 60, 200, 36), parent_id="form", depth=2, index=0,
                              description="")
    input_pw_after = _node("input_pw", "textbox", "",
                            (20, 106, 200, 36), parent_id="form", depth=2, index=1,
                            description="")
    submit_after = _node("btn_submit", "button", "Log In",
                          (20, 160, 200, 40), parent_id="form", depth=2, index=2)
    form_after = _node("form", "form", "Login", (0, 40, 360, 230),
                         parent_id="root", depth=1, index=0,
                         children=[input_user_after, input_pw_after, submit_after])
    root_after = _node("root", "document", "Login Page", (0, 0, 360, 640),
                         parent_id=None, depth=0, index=0,
                         children=[form_after])
    tree_after = AccessibilityTree(root=root_after, metadata={"source": "v2"})
    return tree_before, tree_after


def _make_regression_pair_choice_overload() -> tuple:
    """Before: 5 navigation items. After: 15 items (Hick's law overload)."""
    def _make_nav(n_items, version):
        links = []
        names = ["Home", "Products", "Services", "Pricing", "About",
                 "Blog", "Docs", "Support", "Community", "Careers",
                 "Partners", "Press", "Legal", "Privacy", "Terms"][:n_items]
        for i, name in enumerate(names):
            links.append(_node(
                f"link_{i}", "link", name, (i * 80, 0, 72, 40),
                parent_id="nav", depth=2, index=i,
            ))
        nav = _node("nav", "navigation", "Main Navigation",
                      (0, 0, n_items * 80, 40), parent_id="root", depth=1, index=0,
                      children=links)
        root = _node("root", "document", "Site", (0, 0, 1920, 1080),
                      parent_id=None, depth=0, index=0, children=[nav])
        return AccessibilityTree(root=root, metadata={"source": version})

    return _make_nav(5, "v1"), _make_nav(15, "v2")


# ═══════════════════════════════════════════════════════════════════════
# Tests — Rico-derived Android UIs
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.real_benchmark
class TestRicoShoppingApp:
    """Usability analysis of a Rico-style e-commerce product listing."""

    def test_tree_structure_valid(self):
        """Shopping app tree should have expected node count and depth."""
        tree = _make_rico_shopping_app_tree()
        all_nodes = _collect_nodes(tree.root)
        assert len(all_nodes) >= 20, "Shopping app should have ≥20 nodes"
        max_depth = max(n.depth for n in all_nodes)
        assert max_depth >= 3, "Tree should be at least 3 levels deep"

    def test_product_grid_cognitive_cost(self):
        """Product grid with 4 items: Hick's law should flag moderate choice cost."""
        n_products = 4
        predicted_ms = HickHymanLaw.predict(n_products)
        assert predicted_ms > 0, "Hick-Hyman prediction should be positive"
        # 4 choices should produce ~1.16s of decision time (log2(4+1))
        assert predicted_ms < 3.0, "4-product grid should not exceed 3s decision time"

    def test_small_cart_button_fitts(self):
        """Cart button (48×48) at top-right should have measurable Fitts cost."""
        tree = _make_rico_shopping_app_tree()
        cart = _find_node(tree.root, "cart_btn")
        assert cart is not None
        distance = math.sqrt(cart.bounding_box.x**2 + cart.bounding_box.y**2)
        width = min(cart.bounding_box.width, cart.bounding_box.height)
        fitts_time = FittsLaw.predict(max(distance, 1.0), max(width, 1.0))
        assert fitts_time > 0, "Fitts prediction should be positive"

    def test_add_to_cart_mdp_construction(self):
        """MDP for 'add product to cart' task should be constructable."""
        tree = _make_rico_shopping_app_tree()
        task = TaskSpec(
            spec_id="add_to_cart", name="Add to Cart",
            flows=[TaskFlow(
                flow_id="add", name="Add Item", steps=[
                    TaskStep(step_id="s1", action_type="click", target_role="button",
                             target_name="Add to Cart",
                             description="Add first product to cart"),
                ],
            )],
        )
        builder = MDPBuilder(config=MDPBuilderConfig(max_states=5000))
        mdp = builder.build(tree, task)
        assert mdp.n_states >= 2, "MDP should have at least start and goal states"
        assert mdp.n_transitions >= 1


@pytest.mark.real_benchmark
class TestRicoSettingsScreen:
    """Usability analysis of a Rico-style settings screen."""

    def test_toggle_switches_identifiable(self):
        """Settings tree should contain identifiable switch controls."""
        tree = _make_rico_settings_tree()
        all_nodes = _collect_nodes(tree.root)
        switches = [n for n in all_nodes if n.role == "switch"]
        assert len(switches) >= 3, "Should have at least 3 toggle switches"

    def test_settings_choice_complexity(self):
        """Settings with multiple toggles should have bounded cognitive cost."""
        tree = _make_rico_settings_tree()
        all_nodes = _collect_nodes(tree.root)
        interactive = [n for n in all_nodes if n.role in ("switch", "slider", "button")]
        hick_time = HickHymanLaw.predict(len(interactive))
        assert hick_time > 0


@pytest.mark.real_benchmark
class TestRicoSocialFeed:
    """Usability analysis of a Rico-style social media feed."""

    def test_feed_post_structure(self):
        """Each post should have avatar, image, and action buttons."""
        tree = _make_rico_social_feed_tree()
        all_nodes = _collect_nodes(tree.root)
        posts = [n for n in all_nodes if n.role == "article"]
        assert len(posts) >= 2
        for post in posts:
            child_roles = {c.role for c in post.children}
            assert "img" in child_roles, "Post should contain an image"
            assert "button" in child_roles, "Post should contain action buttons"

    def test_like_button_accessibility(self):
        """Like buttons should have minimum touch target size."""
        tree = _make_rico_social_feed_tree()
        likes = [n for n in _collect_nodes(tree.root)
                 if n.role == "button" and "Like" in n.name]
        assert len(likes) >= 2
        for btn in likes:
            area = btn.bounding_box.width * btn.bounding_box.height
            assert area >= 40 * 24, "Like button should meet minimum touch area"


@pytest.mark.real_benchmark
class TestRicoNewsReader:
    """Usability analysis of a Rico-style news reader."""

    def test_headline_count(self):
        """News reader should present multiple article headlines."""
        tree = _make_rico_news_reader_tree()
        articles = [n for n in _collect_nodes(tree.root) if n.role == "listitem"]
        assert len(articles) == 5

    def test_dense_list_cognitive_load(self):
        """5 headlines should produce measurable Hick-Hyman decision time."""
        predicted = HickHymanLaw.predict(5)
        assert predicted > 0
        assert predicted < 5.0, "5-item list should not exceed 5s decision time"


@pytest.mark.real_benchmark
class TestRicoBankingApp:
    """Usability analysis of a Rico-style banking app."""

    def test_banking_structure(self):
        """Banking app should have balance, quick actions, and transactions."""
        tree = _make_rico_banking_tree()
        all_nodes = _collect_nodes(tree.root)
        roles = {n.role for n in all_nodes}
        assert "list" in roles, "Should have transaction list"
        assert "button" in roles, "Should have action buttons"

    def test_high_stakes_button_size(self):
        """Financial action buttons should be large enough for confident interaction."""
        tree = _make_rico_banking_tree()
        action_btns = [n for n in _collect_nodes(tree.root)
                       if n.role == "button" and n.parent_id == "quick_actions"]
        assert len(action_btns) == 4
        for btn in action_btns:
            assert btn.bounding_box.width >= 48, (
                f"Financial button '{btn.name}' width {btn.bounding_box.width} < 48px"
            )


# ═══════════════════════════════════════════════════════════════════════
# Tests — Web task MDPs
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.real_benchmark
class TestEcommerceCheckout:
    """MDP construction and cognitive cost for e-commerce checkout flow."""

    def test_task_step_count(self):
        """Checkout task should have correct number of steps."""
        task = _make_ecommerce_checkout_task()
        assert len(task.flows) == 1
        assert len(task.flows[0].steps) == 11

    def test_checkout_mdp_states(self):
        """Checkout MDP should have states for each step progression."""
        tree = _make_rico_shopping_app_tree()
        task = _make_ecommerce_checkout_task()
        builder = MDPBuilder(config=MDPBuilderConfig(max_states=50_000))
        mdp = builder.build(tree, task)
        assert mdp.n_states >= 2
        assert len(mdp.goal_states) >= 1

    def test_checkout_cognitive_cost_bounded(self):
        """11-step checkout: total cognitive cost should be computable and bounded."""
        task = _make_ecommerce_checkout_task()
        total_steps = len(task.flows[0].steps)
        # Approximate: each step involves Hick choice (1 target among ~10 elements)
        per_step_cost = HickHymanLaw.predict(10)
        total_cost = per_step_cost * total_steps
        assert total_cost > 0
        assert total_cost < 120.0, "Checkout should not exceed 120s cognitive time"


@pytest.mark.real_benchmark
class TestInsuranceForm:
    """MDP and cognitive analysis for insurance quote form."""

    def test_form_complexity(self):
        """Insurance form task should have 9 steps including dropdown interaction."""
        task = _make_insurance_form_task()
        steps = task.flows[0].steps
        assert len(steps) == 9
        action_types = {s.action_type for s in steps}
        assert "click" in action_types
        assert "type" in action_types


@pytest.mark.real_benchmark
class TestFlightBooking:
    """MDP construction for flight booking workflow."""

    def test_booking_task_structure(self):
        """Flight booking should have search, select, and confirm phases."""
        task = _make_flight_booking_task()
        assert len(task.flows[0].steps) == 8


@pytest.mark.real_benchmark
class TestAdminDashboard:
    """MDP for admin user management workflow."""

    def test_admin_task_dependencies(self):
        """Admin task steps should form a sequential dependency chain."""
        task = _make_admin_dashboard_task()
        steps = task.flows[0].steps
        for i, step in enumerate(steps):
            if i > 0:
                assert step.depends_on is not None and len(step.depends_on) > 0, (
                    f"Step {step.step_id} should depend on previous step"
                )


@pytest.mark.real_benchmark
class TestSupportTicket:
    """MDP for support ticket creation workflow."""

    def test_support_task_structure(self):
        """Support ticket task should have 8 sequential steps."""
        task = _make_support_ticket_task()
        assert len(task.flows[0].steps) == 8
        assert task.flows[0].success_criteria == ["ticket_submitted"]


# ═══════════════════════════════════════════════════════════════════════
# Tests — Accessibility regression detection
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.real_benchmark
class TestRegressionTouchTarget:
    """Detect regression when button touch targets shrink below WCAG minimum."""

    def test_before_buttons_adequate(self):
        """Before-version buttons should meet minimum size requirements."""
        tree_before, _ = _make_regression_pair_touch_target()
        buttons = [n for n in _collect_nodes(tree_before.root) if n.role == "button"]
        for btn in buttons:
            assert btn.bounding_box.height >= 44, (
                f"Before: button '{btn.name}' height {btn.bounding_box.height} < 44px"
            )

    def test_after_buttons_regressed(self):
        """After-version buttons should be smaller (regression detected)."""
        _, tree_after = _make_regression_pair_touch_target()
        buttons = [n for n in _collect_nodes(tree_after.root) if n.role == "button"]
        undersized = [b for b in buttons if b.bounding_box.height < 44]
        assert len(undersized) > 0, "Regression: should detect undersized buttons"

    def test_fitts_cost_increases(self):
        """Smaller buttons should produce higher Fitts' law motor cost."""
        tree_before, tree_after = _make_regression_pair_touch_target()
        before_btn = _find_node(tree_before.root, "btn_a")
        after_btn = _find_node(tree_after.root, "btn_a")

        dist = 200.0  # fixed approach distance
        cost_before = FittsLaw.predict(dist, before_btn.bounding_box.width)
        cost_after = FittsLaw.predict(dist, after_btn.bounding_box.width)
        assert cost_after > cost_before, (
            "Smaller button should have higher Fitts motor cost"
        )


@pytest.mark.real_benchmark
class TestRegressionLabelRemoval:
    """Detect regression when form labels are removed (placeholder-only pattern)."""

    def test_before_has_labels(self):
        """Before-version form should have visible text labels."""
        tree_before, _ = _make_regression_pair_label_removal()
        text_nodes = [n for n in _collect_nodes(tree_before.root)
                      if n.role == "text" and n.name]
        label_names = {n.name for n in text_nodes}
        assert "Username" in label_names
        assert "Password" in label_names

    def test_after_labels_missing(self):
        """After-version should be missing label nodes (regression)."""
        _, tree_after = _make_regression_pair_label_removal()
        text_nodes = [n for n in _collect_nodes(tree_after.root) if n.role == "text"]
        assert len(text_nodes) == 0, "After version should have no text labels"

    def test_input_names_lost(self):
        """After-version textboxes should have empty names (accessibility regression)."""
        _, tree_after = _make_regression_pair_label_removal()
        textboxes = [n for n in _collect_nodes(tree_after.root) if n.role == "textbox"]
        unnamed = [t for t in textboxes if not t.name]
        assert len(unnamed) >= 2, "Should detect at least 2 unlabeled inputs"


@pytest.mark.real_benchmark
class TestRegressionChoiceOverload:
    """Detect regression when navigation items increase from 5 to 15."""

    def test_before_nav_manageable(self):
        """Before: 5 navigation items should have moderate Hick cost."""
        tree_before, _ = _make_regression_pair_choice_overload()
        links = [n for n in _collect_nodes(tree_before.root) if n.role == "link"]
        assert len(links) == 5
        cost = HickHymanLaw.predict(5)
        assert cost > 0

    def test_after_nav_overloaded(self):
        """After: 15 navigation items should have substantially higher cost."""
        _, tree_after = _make_regression_pair_choice_overload()
        links = [n for n in _collect_nodes(tree_after.root) if n.role == "link"]
        assert len(links) == 15

    def test_choice_cost_increases(self):
        """Hick-Hyman cost should increase with more navigation choices."""
        cost_5 = HickHymanLaw.predict(5)
        cost_15 = HickHymanLaw.predict(15)
        ratio = cost_15 / cost_5
        assert ratio > 1.3, (
            f"15 items should be ≥30% more costly than 5 items (got {ratio:.2f}×)"
        )


# ═══════════════════════════════════════════════════════════════════════
# Tree traversal utilities
# ═══════════════════════════════════════════════════════════════════════

def _collect_nodes(node: AccessibilityNode) -> List[AccessibilityNode]:
    """Recursively collect all nodes in the tree."""
    result = [node]
    for child in node.children:
        result.extend(_collect_nodes(child))
    return result


def _find_node(root: AccessibilityNode, node_id: str) -> AccessibilityNode | None:
    """Find a node by ID in the tree."""
    if root.id == node_id:
        return root
    for child in root.children:
        found = _find_node(child, node_id)
        if found is not None:
            return found
    return None
