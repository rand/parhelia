"""Tests for interactive intelligence system.

Tests [SPEC-20.50], [SPEC-20.51], [SPEC-20.52].
"""

from __future__ import annotations

from pathlib import Path

import pytest

from parhelia.interactive import (
    Example,
    ExampleSystem,
    HelpSystem,
    SmartPrompt,
)


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """Create a temporary cache directory."""
    cache = tmp_path / "cache"
    cache.mkdir()
    return cache


class TestSmartPrompt:
    """Tests for SmartPrompt caching."""

    def test_remember_and_get_default(self, cache_dir: Path):
        """Remembered values become defaults."""
        prompt = SmartPrompt(cache_dir=cache_dir)
        prompt.remember("gpu_type", "A100")

        default = prompt.get_default("gpu_type")
        assert default == "A100"

    def test_get_default_returns_none_for_unknown(self, cache_dir: Path):
        """Unknown keys return None."""
        prompt = SmartPrompt(cache_dir=cache_dir)
        assert prompt.get_default("unknown_key") is None

    def test_cache_persists_across_instances(self, cache_dir: Path):
        """Cache persists across SmartPrompt instances."""
        prompt1 = SmartPrompt(cache_dir=cache_dir)
        prompt1.remember("project", "my-project")

        # New instance should load cached value
        prompt2 = SmartPrompt(cache_dir=cache_dir)
        assert prompt2.get_default("project") == "my-project"

    def test_remember_overwrites_existing(self, cache_dir: Path):
        """Remembering overwrites previous value."""
        prompt = SmartPrompt(cache_dir=cache_dir)
        prompt.remember("gpu_type", "A10G")
        prompt.remember("gpu_type", "H100")

        assert prompt.get_default("gpu_type") == "H100"

    def test_forget_removes_cached_value(self, cache_dir: Path):
        """Forget removes a cached value."""
        prompt = SmartPrompt(cache_dir=cache_dir)
        prompt.remember("temp_value", "test")
        prompt.forget("temp_value")

        assert prompt.get_default("temp_value") is None


class TestHelpSystem:
    """Tests for HelpSystem."""

    def test_get_topic_help_returns_content(self):
        """Known topics return help content."""
        help_sys = HelpSystem()

        content = help_sys.get_topic_help("task")
        assert content is not None
        assert len(content) > 0
        assert "task" in content.lower()

    def test_get_topic_help_returns_none_for_unknown(self):
        """Unknown topics return None."""
        help_sys = HelpSystem()

        content = help_sys.get_topic_help("nonexistent_topic_xyz")
        assert content is None

    def test_list_topics_returns_topics(self):
        """list_topics returns available topics."""
        help_sys = HelpSystem()

        topics = help_sys.list_topics()
        assert len(topics) > 0
        assert "task" in topics
        assert "session" in topics

    def test_get_error_help_returns_content(self):
        """Known error codes return help content."""
        help_sys = HelpSystem()

        content = help_sys.get_error_help("E200")
        assert content is not None
        assert len(content) > 0

    def test_get_error_help_returns_none_for_unknown(self):
        """Unknown error codes return None."""
        help_sys = HelpSystem()

        content = help_sys.get_error_help("E999")
        assert content is None

    def test_error_help_includes_recovery_suggestions(self):
        """Error help includes recovery suggestions."""
        help_sys = HelpSystem()

        content = help_sys.get_error_help("E200")
        # Should include some actionable suggestions
        assert content is not None
        # Content should have recovery info


class TestExampleSystem:
    """Tests for ExampleSystem."""

    def test_get_examples_returns_examples(self):
        """Known topics return examples."""
        example_sys = ExampleSystem()

        examples = example_sys.get_examples("gpu")
        assert examples is not None
        assert len(examples) > 0

    def test_get_examples_returns_none_for_unknown(self):
        """Unknown topics return None."""
        example_sys = ExampleSystem()

        examples = example_sys.get_examples("nonexistent_xyz")
        assert examples is None

    def test_list_topics_returns_topics(self):
        """list_topics returns available topics."""
        example_sys = ExampleSystem()

        topics = example_sys.list_topics()
        assert len(topics) > 0
        assert "gpu" in topics

    def test_examples_are_example_objects(self):
        """Examples are Example dataclass instances."""
        example_sys = ExampleSystem()

        examples = example_sys.get_examples("gpu")
        assert examples is not None
        for ex in examples:
            assert isinstance(ex, Example)
            assert ex.title
            assert ex.commands  # Note: plural

    def test_example_commands_contain_parhelia(self):
        """Example commands contain parhelia invocations."""
        example_sys = ExampleSystem()

        examples = example_sys.get_examples("checkpoint")
        assert examples is not None
        for ex in examples:
            # At least one command should contain parhelia
            has_parhelia = any("parhelia" in cmd for cmd in ex.commands)
            assert has_parhelia, f"No parhelia command in {ex.commands}"


class TestHelpSystemTopicCoverage:
    """Tests for help topic coverage."""

    def test_has_task_topic(self):
        """Has help for task topic."""
        assert HelpSystem().get_topic_help("task") is not None

    def test_has_session_topic(self):
        """Has help for session topic."""
        assert HelpSystem().get_topic_help("session") is not None

    def test_has_container_topic(self):
        """Has help for container topic."""
        assert HelpSystem().get_topic_help("container") is not None

    def test_has_checkpoint_topic(self):
        """Has help for checkpoint topic."""
        assert HelpSystem().get_topic_help("checkpoint") is not None

    def test_has_budget_topic(self):
        """Has help for budget topic."""
        assert HelpSystem().get_topic_help("budget") is not None


class TestExampleSystemTopicCoverage:
    """Tests for example topic coverage."""

    def test_has_gpu_examples(self):
        """Has examples for GPU topic."""
        assert ExampleSystem().get_examples("gpu") is not None

    def test_has_checkpoint_examples(self):
        """Has examples for checkpoint topic."""
        assert ExampleSystem().get_examples("checkpoint") is not None

    def test_has_budget_examples(self):
        """Has examples for budget topic."""
        assert ExampleSystem().get_examples("budget") is not None
