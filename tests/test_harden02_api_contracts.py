# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""H02-R06: API contracts validation tests.

Tests for evaluate_schema_compatibility() and schema_diff().
"""
import pytest

from humeris.domain.api_contracts import (
    ApiCompatibilityResult,
    evaluate_schema_compatibility,
    schema_diff,
)


class TestSchemaCompatibility:
    """Test evaluate_schema_compatibility() decision matrix."""

    def test_no_changes_compatible(self):
        """No field changes → compatible=True, no breaking changes."""
        fields = {"name", "age", "email"}
        result = evaluate_schema_compatibility(
            previous_fields=fields,
            current_fields=fields,
            migration_notes="",
            version_bumped=False,
        )
        assert result.compatible is True
        assert result.breaking_changes == ()
        assert result.requires_version_bump is False

    def test_addition_only_compatible(self):
        """Adding fields (no removal) → compatible=True."""
        result = evaluate_schema_compatibility(
            previous_fields={"name", "age"},
            current_fields={"name", "age", "email"},
            migration_notes="",
            version_bumped=False,
        )
        assert result.compatible is True
        assert result.breaking_changes == ()
        assert result.requires_version_bump is False

    def test_removal_without_version_bump_incompatible(self):
        """Removing field without version bump → compatible=False."""
        result = evaluate_schema_compatibility(
            previous_fields={"name", "age", "email"},
            current_fields={"name", "age"},
            migration_notes="Removed email field",
            version_bumped=False,
        )
        assert result.compatible is False
        assert "removed:email" in result.breaking_changes
        assert result.requires_version_bump is True

    def test_removal_without_migration_notes_incompatible(self):
        """Removing field without migration notes → compatible=False."""
        result = evaluate_schema_compatibility(
            previous_fields={"name", "age", "email"},
            current_fields={"name", "age"},
            migration_notes="",
            version_bumped=True,
        )
        assert result.compatible is False
        assert result.migration_notes_present is False

    def test_removal_with_version_bump_and_notes_compatible(self):
        """Removing field with both version bump and notes → compatible=True."""
        result = evaluate_schema_compatibility(
            previous_fields={"name", "age", "email"},
            current_fields={"name", "age"},
            migration_notes="Removed email field — use contact_email instead",
            version_bumped=True,
        )
        assert result.compatible is True
        assert "removed:email" in result.breaking_changes
        assert result.requires_version_bump is True
        assert result.migration_notes_present is True

    def test_multiple_removals(self):
        """Multiple field removals all tracked as breaking."""
        result = evaluate_schema_compatibility(
            previous_fields={"a", "b", "c", "d"},
            current_fields={"a"},
            migration_notes="Major schema rework",
            version_bumped=True,
        )
        assert result.compatible is True
        assert len(result.breaking_changes) == 3
        assert "removed:b" in result.breaking_changes
        assert "removed:c" in result.breaking_changes
        assert "removed:d" in result.breaking_changes

    def test_whitespace_only_notes_not_present(self):
        """Whitespace-only migration notes count as not present."""
        result = evaluate_schema_compatibility(
            previous_fields={"name", "email"},
            current_fields={"name"},
            migration_notes="   \n  \t  ",
            version_bumped=True,
        )
        assert result.compatible is False
        assert result.migration_notes_present is False

    def test_empty_previous_fields(self):
        """Empty previous → any current fields are additions only."""
        result = evaluate_schema_compatibility(
            previous_fields=set(),
            current_fields={"name", "age"},
            migration_notes="",
            version_bumped=False,
        )
        assert result.compatible is True

    def test_empty_current_fields(self):
        """Empty current → all previous fields removed (breaking)."""
        result = evaluate_schema_compatibility(
            previous_fields={"name", "age"},
            current_fields=set(),
            migration_notes="Complete schema removal",
            version_bumped=True,
        )
        assert result.compatible is True
        assert len(result.breaking_changes) == 2


class TestSchemaDiff:
    """Test schema_diff() output correctness."""

    def test_diff_no_changes(self):
        """No changes produces empty added/removed, full unchanged."""
        fields = {"a", "b", "c"}
        diff = schema_diff(fields, fields)
        assert diff["added"] == []
        assert diff["removed"] == []
        assert sorted(diff["unchanged"]) == ["a", "b", "c"]

    def test_diff_additions(self):
        diff = schema_diff({"a"}, {"a", "b", "c"})
        assert sorted(diff["added"]) == ["b", "c"]
        assert diff["removed"] == []
        assert diff["unchanged"] == ["a"]

    def test_diff_removals(self):
        diff = schema_diff({"a", "b", "c"}, {"a"})
        assert diff["added"] == []
        assert sorted(diff["removed"]) == ["b", "c"]
        assert diff["unchanged"] == ["a"]

    def test_diff_mixed(self):
        diff = schema_diff({"a", "b"}, {"b", "c"})
        assert diff["added"] == ["c"]
        assert diff["removed"] == ["a"]
        assert diff["unchanged"] == ["b"]

    def test_diff_empty_to_populated(self):
        diff = schema_diff(set(), {"x", "y"})
        assert sorted(diff["added"]) == ["x", "y"]
        assert diff["removed"] == []
        assert diff["unchanged"] == []
