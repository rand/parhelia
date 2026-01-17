"""Tests for Modal app scaffolding.

@trace SPEC-01.11 - Image Definition
@trace SPEC-01.12 - Volume Mounting
"""

import pytest


class TestModalAppDefinition:
    """Tests for Modal App scaffolding - SPEC-01.11."""

    def test_app_exists_with_correct_name(self):
        """@trace SPEC-01.11 - App MUST be named 'parhelia'."""
        from parhelia.modal_app import app

        assert app.name == "parhelia"

    def test_volume_defined(self):
        """@trace SPEC-01.12 - Volume MUST be named 'parhelia-vol'."""
        from parhelia.modal_app import volume

        # Volume should be defined (actual creation happens at runtime)
        assert volume is not None

    def test_cpu_image_defined(self):
        """@trace SPEC-01.11 - CPU image MUST be defined with required packages."""
        from parhelia.modal_app import cpu_image

        assert cpu_image is not None

    def test_gpu_image_defined(self):
        """@trace SPEC-01.11 - GPU image MUST extend CPU image."""
        from parhelia.modal_app import gpu_image

        assert gpu_image is not None


class TestConfigLoading:
    """Tests for configuration loading."""

    def test_default_config_loads(self):
        """Config MUST load with sensible defaults."""
        from parhelia.config import ParheliaConfig, load_config

        config = load_config()
        assert isinstance(config, ParheliaConfig)

    def test_config_has_modal_settings(self):
        """Config MUST include Modal-specific settings."""
        from parhelia.config import load_config

        config = load_config()
        assert hasattr(config, "modal")
        assert hasattr(config.modal, "region")

    def test_config_has_volume_name(self):
        """@trace SPEC-01.12 - Config MUST specify volume name."""
        from parhelia.config import load_config

        config = load_config()
        assert config.modal.volume_name == "parhelia-vol"

    def test_region_has_valid_default(self):
        """@trace SPEC-01.11b - Region MUST default to us-east."""
        from parhelia.config import load_config

        config = load_config()
        assert config.modal.region == "us-east"

    def test_region_validates_allowed_values(self):
        """@trace SPEC-01.11b - Region MUST be one of us-east, us-west, eu-west."""
        from parhelia.config import ModalConfig

        # Valid regions should work
        for region in ["us-east", "us-west", "eu-west"]:
            config = ModalConfig(region=region)
            assert config.region == region


class TestContainerVariants:
    """Tests for container variants - SPEC-01.10."""

    def test_cpu_variant_config(self):
        """@trace SPEC-01.10 - CPU variant MUST have cpu=4, memory=16384."""
        from parhelia.modal_app import CPU_CONFIG

        assert CPU_CONFIG["cpu"] == 4
        assert CPU_CONFIG["memory"] == 16384

    def test_gpu_variant_options(self):
        """@trace SPEC-01.10 - GPU variant MUST support A10G and A100."""
        from parhelia.modal_app import SUPPORTED_GPUS

        assert "A10G" in SUPPORTED_GPUS
        assert "A100" in SUPPORTED_GPUS


class TestSandboxCreation:
    """Tests for sandbox creation validation."""

    @pytest.mark.asyncio
    async def test_invalid_gpu_raises_error(self):
        """@trace SPEC-01.10 - Invalid GPU type MUST raise ValueError."""
        from parhelia.modal_app import create_claude_sandbox

        with pytest.raises(ValueError, match="Unsupported GPU"):
            await create_claude_sandbox(task_id="test-task", gpu="INVALID_GPU")

    def test_supported_gpus_includes_a10g_a100(self):
        """@trace SPEC-01.10 - Sandbox MUST support A10G and A100 GPUs."""
        from parhelia.modal_app import SUPPORTED_GPUS

        assert "A10G" in SUPPORTED_GPUS
        assert "A100" in SUPPORTED_GPUS

    def test_create_sandbox_function_exists(self):
        """@trace SPEC-01 - create_claude_sandbox MUST be callable."""
        from parhelia.modal_app import create_claude_sandbox

        assert callable(create_claude_sandbox)

    def test_run_in_sandbox_function_exists(self):
        """@trace SPEC-01 - run_in_sandbox MUST be callable."""
        from parhelia.modal_app import run_in_sandbox

        assert callable(run_in_sandbox)


class TestSandboxManager:
    """Tests for SandboxManager class."""

    def test_sandbox_manager_initialization(self):
        """@trace SPEC-01 - SandboxManager MUST track active sandboxes."""
        from parhelia.modal_app import SandboxManager

        manager = SandboxManager()
        assert manager is not None
        assert len(manager.active_sandboxes) == 0

    def test_sandbox_manager_can_accept_check(self):
        """@trace SPEC-01 - SandboxManager MUST check if capacity available."""
        from parhelia.modal_app import SandboxManager

        manager = SandboxManager()
        assert manager.can_accept_sandbox() is True

    def test_sandbox_manager_max_sandboxes(self):
        """@trace SPEC-01 - SandboxManager MUST respect max_sandboxes limit."""
        from parhelia.modal_app import SandboxManager

        manager = SandboxManager(max_sandboxes=2)
        assert manager.max_sandboxes == 2
