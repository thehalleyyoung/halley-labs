"""Tests for usability_oracle.accessibility.roles — ARIA role taxonomy.

Covers RoleTaxonomy classification (is_interactive, is_landmark, is_widget,
is_structure), hierarchy queries (parent_role, ancestors, depth, is_descendant_of,
children, descendants), semantic similarity, taxonomy distance, containment
checks, and utility methods.
"""

from __future__ import annotations

import pytest

from usability_oracle.accessibility.roles import RoleTaxonomy


@pytest.fixture
def taxonomy() -> RoleTaxonomy:
    """Return a fresh RoleTaxonomy instance."""
    return RoleTaxonomy()


# ── Classification queries ────────────────────────────────────────────────────


class TestIsInteractive:
    """Tests for RoleTaxonomy.is_interactive()."""

    def test_button_is_interactive(self, taxonomy):
        """button should be classified as interactive."""
        assert taxonomy.is_interactive("button") is True

    def test_textbox_is_interactive(self, taxonomy):
        """textbox should be classified as interactive."""
        assert taxonomy.is_interactive("textbox") is True

    def test_checkbox_is_interactive(self, taxonomy):
        """checkbox should be classified as interactive."""
        assert taxonomy.is_interactive("checkbox") is True

    def test_link_is_interactive(self, taxonomy):
        """link should be classified as interactive."""
        assert taxonomy.is_interactive("link") is True

    def test_document_not_interactive(self, taxonomy):
        """document should not be interactive."""
        assert taxonomy.is_interactive("document") is False

    def test_heading_not_interactive(self, taxonomy):
        """heading should not be interactive."""
        assert taxonomy.is_interactive("heading") is False

    def test_unknown_not_interactive(self, taxonomy):
        """An unknown role should not be interactive."""
        assert taxonomy.is_interactive("foobar") is False


class TestIsLandmark:
    """Tests for RoleTaxonomy.is_landmark()."""

    def test_navigation_is_landmark(self, taxonomy):
        """navigation should be a landmark."""
        assert taxonomy.is_landmark("navigation") is True

    def test_main_is_landmark(self, taxonomy):
        """main should be a landmark."""
        assert taxonomy.is_landmark("main") is True

    def test_banner_is_landmark(self, taxonomy):
        """banner should be a landmark."""
        assert taxonomy.is_landmark("banner") is True

    def test_form_is_landmark(self, taxonomy):
        """form should be a landmark."""
        assert taxonomy.is_landmark("form") is True

    def test_search_is_landmark(self, taxonomy):
        """search should be a landmark."""
        assert taxonomy.is_landmark("search") is True

    def test_button_not_landmark(self, taxonomy):
        """button should not be a landmark."""
        assert taxonomy.is_landmark("button") is False


class TestIsWidget:
    """Tests for RoleTaxonomy.is_widget()."""

    def test_button_is_widget(self, taxonomy):
        """button should be classified as a widget."""
        assert taxonomy.is_widget("button") is True

    def test_dialog_is_widget(self, taxonomy):
        """dialog should be classified as a widget."""
        assert taxonomy.is_widget("dialog") is True

    def test_slider_is_widget(self, taxonomy):
        """slider should be a widget."""
        assert taxonomy.is_widget("slider") is True

    def test_document_not_widget(self, taxonomy):
        """document should not be a widget."""
        assert taxonomy.is_widget("document") is False


class TestIsStructure:
    """Tests for RoleTaxonomy.is_structure()."""

    def test_heading_is_structure(self, taxonomy):
        """heading should be a structure role."""
        assert taxonomy.is_structure("heading") is True

    def test_table_is_structure(self, taxonomy):
        """table should be a structure role."""
        assert taxonomy.is_structure("table") is True

    def test_list_is_structure(self, taxonomy):
        """list should be a structure role."""
        assert taxonomy.is_structure("list") is True

    def test_separator_is_structure(self, taxonomy):
        """separator should be a structure role."""
        assert taxonomy.is_structure("separator") is True

    def test_button_not_structure(self, taxonomy):
        """button should not be a structure role."""
        assert taxonomy.is_structure("button") is False


# ── is_valid_role ─────────────────────────────────────────────────────────────


class TestIsValidRole:
    """Tests for RoleTaxonomy.is_valid_role()."""

    def test_button_is_valid(self, taxonomy):
        """button should be a valid role."""
        assert taxonomy.is_valid_role("button") is True

    def test_roletype_is_valid(self, taxonomy):
        """roletype (the root of the hierarchy) should be valid."""
        assert taxonomy.is_valid_role("roletype") is True

    def test_random_string_not_valid(self, taxonomy):
        """An arbitrary string should not be valid."""
        assert taxonomy.is_valid_role("banana") is False

    def test_all_interactive_roles_are_valid(self, taxonomy):
        """Every interactive role that appears in the hierarchy should be valid."""
        all_roles = taxonomy.all_roles()
        for role in taxonomy.INTERACTIVE_ROLES:
            if role in all_roles:
                assert taxonomy.is_valid_role(role), f"{role} should be valid"

    def test_all_landmark_roles_are_valid(self, taxonomy):
        """Every landmark role should also be a valid role."""
        for role in taxonomy.LANDMARK_ROLES:
            assert taxonomy.is_valid_role(role), f"{role} should be valid"


# ── Hierarchy queries ─────────────────────────────────────────────────────────


class TestHierarchyQueries:
    """Tests for parent_role, ancestors, depth, is_descendant_of, children, descendants."""

    def test_parent_role_of_button(self, taxonomy):
        """button's parent role should be 'command'."""
        assert taxonomy.parent_role("button") == "command"

    def test_parent_role_of_roletype(self, taxonomy):
        """roletype should have no parent."""
        assert taxonomy.parent_role("roletype") is None

    def test_parent_role_of_unknown(self, taxonomy):
        """Unknown role should have no parent."""
        assert taxonomy.parent_role("banana") is None

    def test_ancestors_of_button(self, taxonomy):
        """Ancestors of button should include command, widget, roletype."""
        ancs = taxonomy.ancestors("button")
        assert "command" in ancs
        assert "widget" in ancs
        assert "roletype" in ancs

    def test_ancestors_of_roletype(self, taxonomy):
        """roletype should have no ancestors."""
        assert taxonomy.ancestors("roletype") == []

    def test_depth_of_roletype(self, taxonomy):
        """roletype should have depth 0."""
        assert taxonomy.depth("roletype") == 0

    def test_depth_of_button(self, taxonomy):
        """button should have depth >= 2 (roletype -> widget -> command -> button)."""
        assert taxonomy.depth("button") >= 2

    def test_depth_increases_down_hierarchy(self, taxonomy):
        """Deeper roles should have higher depth values."""
        assert taxonomy.depth("button") > taxonomy.depth("widget")
        assert taxonomy.depth("widget") > taxonomy.depth("roletype")

    def test_is_descendant_of_self(self, taxonomy):
        """A role should be a descendant of itself."""
        assert taxonomy.is_descendant_of("button", "button") is True

    def test_is_descendant_of_ancestor(self, taxonomy):
        """button should be a descendant of widget."""
        assert taxonomy.is_descendant_of("button", "widget") is True

    def test_not_descendant_of_sibling(self, taxonomy):
        """button should not be a descendant of link."""
        assert taxonomy.is_descendant_of("button", "link") is False

    def test_children_of_command(self, taxonomy):
        """children of 'command' should include button, link, menuitem."""
        kids = taxonomy.children("command")
        assert "button" in kids
        assert "link" in kids
        assert "menuitem" in kids

    def test_children_of_leaf(self, taxonomy):
        """A leaf role with no hierarchy children should return empty list."""
        # 'dialog' children in the hierarchy
        kids = taxonomy.children("alertdialog")
        assert kids == []  # alertdialog has no children defined

    def test_descendants_of_widget(self, taxonomy):
        """descendants of 'widget' should be a large set including button, textbox, etc."""
        desc = taxonomy.descendants("widget")
        assert "button" in desc
        assert "textbox" in desc
        assert "checkbox" in desc
        assert isinstance(desc, set)

    def test_descendants_of_leaf(self, taxonomy):
        """A leaf role should have no descendants."""
        desc = taxonomy.descendants("alertdialog")
        assert desc == set()


# ── Semantic similarity ───────────────────────────────────────────────────────


class TestSemanticSimilarity:
    """Tests for semantic_similarity() and taxonomy_distance()."""

    def test_same_role_similarity_one(self, taxonomy):
        """Same role should have similarity 1.0."""
        assert taxonomy.semantic_similarity("button", "button") == 1.0

    def test_similarity_in_range(self, taxonomy):
        """Similarity of distinct valid roles should be in (0, 1)."""
        sim = taxonomy.semantic_similarity("button", "link")
        assert 0.0 < sim < 1.0

    def test_similarity_unknown_role_zero(self, taxonomy):
        """Unknown roles should get similarity 0.0."""
        assert taxonomy.semantic_similarity("button", "foobar") == 0.0

    def test_siblings_more_similar_than_distant(self, taxonomy):
        """Sibling roles should be more similar than distant roles."""
        sim_siblings = taxonomy.semantic_similarity("button", "link")
        sim_distant = taxonomy.semantic_similarity("button", "table")
        assert sim_siblings > sim_distant

    def test_taxonomy_distance_same_role(self, taxonomy):
        """Distance from a role to itself should be 0."""
        assert taxonomy.taxonomy_distance("button", "button") == 0

    def test_taxonomy_distance_non_negative(self, taxonomy):
        """Taxonomy distance should always be non-negative."""
        d = taxonomy.taxonomy_distance("button", "textbox")
        assert d >= 0
        assert isinstance(d, int)

    def test_taxonomy_distance_symmetric(self, taxonomy):
        """Taxonomy distance should be symmetric."""
        d1 = taxonomy.taxonomy_distance("button", "link")
        d2 = taxonomy.taxonomy_distance("link", "button")
        assert d1 == d2

    def test_parent_child_distance_one(self, taxonomy):
        """Distance between adjacent parent and child should be 1."""
        d = taxonomy.taxonomy_distance("command", "button")
        assert d == 1


# ── Containment queries ──────────────────────────────────────────────────────


class TestContainment:
    """Tests for can_contain() and expected_children()."""

    def test_list_can_contain_listitem(self, taxonomy):
        """list should be allowed to contain listitem."""
        assert taxonomy.can_contain("list", "listitem") is True

    def test_list_cannot_contain_button(self, taxonomy):
        """list should not be allowed to contain button."""
        assert taxonomy.can_contain("list", "button") is False

    def test_unrestricted_parent_can_contain_anything(self, taxonomy):
        """A role with no expected children restriction should allow any child."""
        # 'document' has no entry in EXPECTED_CHILDREN
        assert taxonomy.can_contain("document", "button") is True
        assert taxonomy.can_contain("document", "table") is True

    def test_tablist_can_contain_tab(self, taxonomy):
        """tablist should allow tab children."""
        assert taxonomy.can_contain("tablist", "tab") is True

    def test_menu_can_contain_menuitem(self, taxonomy):
        """menu should allow menuitem children."""
        assert taxonomy.can_contain("menu", "menuitem") is True

    def test_expected_children_returns_set(self, taxonomy):
        """expected_children should return a set."""
        ec = taxonomy.expected_children("list")
        assert isinstance(ec, set)
        assert "listitem" in ec

    def test_expected_children_empty_for_unknown(self, taxonomy):
        """expected_children for a role with no restrictions should be empty set."""
        ec = taxonomy.expected_children("button")
        assert ec == set()


# ── Utility methods ───────────────────────────────────────────────────────────


class TestUtilities:
    """Tests for all_roles() and role_category()."""

    def test_all_roles_returns_frozenset(self, taxonomy):
        """all_roles() should return a frozenset."""
        roles = taxonomy.all_roles()
        assert isinstance(roles, frozenset)

    def test_all_roles_contains_known(self, taxonomy):
        """all_roles() should contain standard roles."""
        roles = taxonomy.all_roles()
        assert "button" in roles
        assert "roletype" in roles
        assert "document" in roles

    def test_all_roles_has_reasonable_size(self, taxonomy):
        """The taxonomy should contain a substantial number of roles."""
        assert len(taxonomy.all_roles()) > 30

    def test_role_category_widget(self, taxonomy):
        """button should be categorised as 'widget'."""
        assert taxonomy.role_category("button") == "widget"

    def test_role_category_landmark(self, taxonomy):
        """navigation should be categorised as 'landmark'."""
        assert taxonomy.role_category("navigation") == "landmark"

    def test_role_category_structure(self, taxonomy):
        """heading should be categorised as 'structure'."""
        assert taxonomy.role_category("heading") == "structure"

    def test_role_category_other(self, taxonomy):
        """A role not in widget/landmark/structure should be 'other'."""
        # roletype isn't in any of those sets
        assert taxonomy.role_category("roletype") == "other"

    def test_repr(self, taxonomy):
        """__repr__ should mention the role count."""
        r = repr(taxonomy)
        assert "RoleTaxonomy" in r
        assert "roles=" in r
