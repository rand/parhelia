"""Tests for the container reconciler.

Tests cover:
- Orphan detection (containers in Modal but not in DB)
- Stale container marking (containers in DB but not in Modal)
- State drift correction (mismatched state between DB and Modal)
- Event emission on changes
- Background loop operation
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest

from parhelia.reconciler import (
    ContainerReconciler,
    MockModalClient,
    ModalSandboxInfo,
    ModalSandboxStatus,
    ReconcileResult,
    ReconcilerConfig,
)
from parhelia.state import (
    Container,
    ContainerState,
    EventType,
    HealthStatus,
    StateStore,
)


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database path."""
    return tmp_path / "test_state.db"


@pytest.fixture
def state_store(temp_db):
    """Create a StateStore with temporary database."""
    return StateStore(temp_db)


@pytest.fixture
def mock_modal():
    """Create a MockModalClient."""
    return MockModalClient()


@pytest.fixture
def reconciler(state_store, mock_modal):
    """Create a ContainerReconciler with mock dependencies."""
    config = ReconcilerConfig(
        poll_interval_seconds=1,  # Fast for tests
        stale_threshold_seconds=60,
        auto_terminate_orphans=False,
    )
    return ContainerReconciler(
        state_store=state_store,
        modal_client=mock_modal,
        config=config,
    )


class TestReconcileResult:
    """Tests for ReconcileResult dataclass."""

    def test_empty_result_str(self):
        """Empty result reports no changes."""
        result = ReconcileResult()
        assert "no changes" in str(result)

    def test_result_with_orphans(self):
        """Result with orphans shows count."""
        result = ReconcileResult(orphans_detected=3)
        assert "3 orphans" in str(result)

    def test_result_with_multiple_changes(self):
        """Result with multiple change types shows all."""
        result = ReconcileResult(
            orphans_detected=2,
            stale_marked=1,
            drift_corrected=3,
            duration_seconds=1.5,
        )
        result_str = str(result)
        assert "2 orphans" in result_str
        assert "1 stale" in result_str
        assert "3 drift" in result_str
        assert "1.5" in result_str

    def test_result_with_errors(self):
        """Result with errors shows error count."""
        result = ReconcileResult(errors=["error1", "error2"])
        assert "2 errors" in str(result)


class TestMockModalClient:
    """Tests for MockModalClient."""

    @pytest.mark.asyncio
    async def test_list_sandboxes_empty(self, mock_modal):
        """List returns empty when no sandboxes."""
        result = await mock_modal.list_sandboxes()
        assert result == []

    @pytest.mark.asyncio
    async def test_list_sandboxes_with_data(self, mock_modal):
        """List returns added sandboxes."""
        sandbox = ModalSandboxInfo(
            sandbox_id="sb-123",
            status=ModalSandboxStatus.RUNNING,
            app_name="parhelia",
            parhelia_managed=True,
        )
        mock_modal.add_sandbox(sandbox)

        result = await mock_modal.list_sandboxes()
        assert len(result) == 1
        assert result[0].sandbox_id == "sb-123"

    @pytest.mark.asyncio
    async def test_list_sandboxes_filters_by_app(self, mock_modal):
        """List filters by app name when provided."""
        mock_modal.add_sandbox(
            ModalSandboxInfo(
                sandbox_id="sb-1",
                status=ModalSandboxStatus.RUNNING,
                app_name="parhelia",
            )
        )
        mock_modal.add_sandbox(
            ModalSandboxInfo(
                sandbox_id="sb-2",
                status=ModalSandboxStatus.RUNNING,
                app_name="other-app",
            )
        )

        result = await mock_modal.list_sandboxes(app_name="parhelia")
        assert len(result) == 1
        assert result[0].sandbox_id == "sb-1"

    @pytest.mark.asyncio
    async def test_get_sandbox(self, mock_modal):
        """Get returns specific sandbox."""
        sandbox = ModalSandboxInfo(
            sandbox_id="sb-123",
            status=ModalSandboxStatus.RUNNING,
        )
        mock_modal.add_sandbox(sandbox)

        result = await mock_modal.get_sandbox("sb-123")
        assert result is not None
        assert result.sandbox_id == "sb-123"

    @pytest.mark.asyncio
    async def test_get_sandbox_not_found(self, mock_modal):
        """Get returns None for unknown sandbox."""
        result = await mock_modal.get_sandbox("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_terminate_sandbox(self, mock_modal):
        """Terminate marks sandbox as terminated."""
        mock_modal.add_sandbox(
            ModalSandboxInfo(
                sandbox_id="sb-123",
                status=ModalSandboxStatus.RUNNING,
            )
        )

        success = await mock_modal.terminate_sandbox("sb-123")
        assert success is True
        assert "sb-123" in mock_modal.terminated_ids

        # Verify status changed
        sandbox = await mock_modal.get_sandbox("sb-123")
        assert sandbox.status == ModalSandboxStatus.TERMINATED

    @pytest.mark.asyncio
    async def test_terminate_sandbox_not_found(self, mock_modal):
        """Terminate returns False for unknown sandbox."""
        success = await mock_modal.terminate_sandbox("nonexistent")
        assert success is False

    @pytest.mark.asyncio
    async def test_call_counting(self, mock_modal):
        """Mock counts API calls."""
        mock_modal.add_sandbox(
            ModalSandboxInfo(sandbox_id="sb-1", status=ModalSandboxStatus.RUNNING)
        )

        await mock_modal.list_sandboxes()
        await mock_modal.list_sandboxes()
        await mock_modal.get_sandbox("sb-1")

        assert mock_modal.list_calls == 2
        assert mock_modal.get_calls == 1


class TestOrphanDetection:
    """Tests for orphan container detection."""

    @pytest.mark.asyncio
    async def test_detects_orphan_container(self, state_store, mock_modal, reconciler):
        """Reconciler detects containers in Modal but not in DB."""
        # Add sandbox to Modal but not to DB
        mock_modal.add_sandbox(
            ModalSandboxInfo(
                sandbox_id="sb-orphan",
                status=ModalSandboxStatus.RUNNING,
                app_name="parhelia",
                parhelia_managed=True,
                task_id="task-123",
            )
        )

        result = await reconciler.reconcile()

        assert result.orphans_detected == 1
        assert len(result.orphan_ids) == 1

        # Verify orphan record was created
        orphans = state_store.get_orphaned_containers()
        assert len(orphans) == 1
        assert orphans[0].modal_sandbox_id == "sb-orphan"
        assert orphans[0].state == ContainerState.ORPHANED
        assert orphans[0].task_id == "task-123"

    @pytest.mark.asyncio
    async def test_orphan_creates_event(self, state_store, mock_modal, reconciler):
        """Orphan detection emits ORPHAN_DETECTED event."""
        mock_modal.add_sandbox(
            ModalSandboxInfo(
                sandbox_id="sb-orphan",
                status=ModalSandboxStatus.RUNNING,
                app_name="parhelia",
                parhelia_managed=True,
            )
        )

        await reconciler.reconcile()

        events = state_store.get_events(event_type=EventType.ORPHAN_DETECTED)
        assert len(events) == 1
        assert "sb-orphan" in events[0].message

    @pytest.mark.asyncio
    async def test_ignores_non_parhelia_sandboxes(
        self, state_store, mock_modal, reconciler
    ):
        """Reconciler ignores sandboxes not managed by Parhelia."""
        mock_modal.add_sandbox(
            ModalSandboxInfo(
                sandbox_id="sb-other",
                status=ModalSandboxStatus.RUNNING,
                parhelia_managed=False,  # Not Parhelia's
            )
        )

        result = await reconciler.reconcile()

        assert result.orphans_detected == 0
        assert len(state_store.get_orphaned_containers()) == 0

    @pytest.mark.asyncio
    async def test_auto_terminate_orphan(self, state_store, mock_modal):
        """Auto-terminates orphans when configured."""
        config = ReconcilerConfig(auto_terminate_orphans=True)
        reconciler = ContainerReconciler(state_store, mock_modal, config)

        mock_modal.add_sandbox(
            ModalSandboxInfo(
                sandbox_id="sb-orphan",
                status=ModalSandboxStatus.RUNNING,
                app_name="parhelia",
                parhelia_managed=True,
            )
        )

        await reconciler.reconcile()

        # Verify termination was called
        assert "sb-orphan" in mock_modal.terminated_ids

        # Verify container state updated
        containers = state_store.get_containers_by_state(ContainerState.TERMINATED)
        assert len(containers) == 1


class TestStaleContainerMarking:
    """Tests for stale container detection."""

    @pytest.mark.asyncio
    async def test_marks_stale_container(self, state_store, mock_modal, reconciler):
        """Reconciler marks containers in DB but not in Modal as terminated."""
        # Create container in DB
        container = Container.create(
            modal_sandbox_id="sb-stale",
            task_id="task-123",
        )
        container.state = ContainerState.RUNNING
        state_store.create_container(container)

        # Don't add to Modal - it's "gone"

        result = await reconciler.reconcile()

        assert result.stale_marked == 1
        assert container.id in result.stale_ids

        # Verify container state updated
        updated = state_store.get_container(container.id)
        assert updated.state == ContainerState.TERMINATED
        assert updated.terminated_at is not None

    @pytest.mark.asyncio
    async def test_stale_creates_terminated_event(
        self, state_store, mock_modal, reconciler
    ):
        """Stale container marking emits CONTAINER_TERMINATED event."""
        container = Container.create(modal_sandbox_id="sb-stale")
        container.state = ContainerState.RUNNING
        state_store.create_container(container)

        await reconciler.reconcile()

        events = state_store.get_events(event_type=EventType.CONTAINER_TERMINATED)
        assert len(events) >= 1
        # Find the event for our container
        container_events = [e for e in events if e.container_id == container.id]
        assert len(container_events) == 1

    @pytest.mark.asyncio
    async def test_only_checks_active_containers(
        self, state_store, mock_modal, reconciler
    ):
        """Only RUNNING/CREATED containers are checked for staleness."""
        # Create terminated container
        terminated = Container.create(modal_sandbox_id="sb-already-terminated")
        terminated.state = ContainerState.TERMINATED
        state_store.containers.save(terminated)

        result = await reconciler.reconcile()

        # Should not mark already-terminated as stale again
        assert result.stale_marked == 0


class TestStateDriftCorrection:
    """Tests for state drift detection and correction."""

    @pytest.mark.asyncio
    async def test_corrects_state_drift(self, state_store, mock_modal, reconciler):
        """Reconciler corrects when DB state doesn't match Modal."""
        # Create container in DB as CREATED
        container = Container.create(modal_sandbox_id="sb-drift")
        container.state = ContainerState.CREATED
        state_store.create_container(container)

        # Modal reports it as RUNNING
        mock_modal.add_sandbox(
            ModalSandboxInfo(
                sandbox_id="sb-drift",
                status=ModalSandboxStatus.RUNNING,
                app_name="parhelia",
                parhelia_managed=True,
            )
        )

        result = await reconciler.reconcile()

        assert result.drift_corrected == 1
        assert container.id in result.drift_ids

        # Verify state was corrected
        updated = state_store.get_container(container.id)
        assert updated.state == ContainerState.RUNNING

    @pytest.mark.asyncio
    async def test_drift_creates_event(self, state_store, mock_modal, reconciler):
        """State drift correction emits STATE_DRIFT_CORRECTED event."""
        container = Container.create(modal_sandbox_id="sb-drift")
        container.state = ContainerState.CREATED
        state_store.create_container(container)

        mock_modal.add_sandbox(
            ModalSandboxInfo(
                sandbox_id="sb-drift",
                status=ModalSandboxStatus.RUNNING,
                app_name="parhelia",
                parhelia_managed=True,
            )
        )

        await reconciler.reconcile()

        events = state_store.get_events(event_type=EventType.STATE_DRIFT_CORRECTED)
        assert len(events) == 1
        # old_value/new_value are stored in details via log_event's **kwargs
        assert events[0].details["old_value"] == "created"
        assert events[0].details["new_value"] == "running"

    @pytest.mark.asyncio
    async def test_no_drift_when_states_match(self, state_store, mock_modal, reconciler):
        """No drift correction when states already match."""
        container = Container.create(modal_sandbox_id="sb-ok")
        container.state = ContainerState.RUNNING
        state_store.create_container(container)

        mock_modal.add_sandbox(
            ModalSandboxInfo(
                sandbox_id="sb-ok",
                status=ModalSandboxStatus.RUNNING,
                app_name="parhelia",
                parhelia_managed=True,
            )
        )

        result = await reconciler.reconcile()

        assert result.drift_corrected == 0

    @pytest.mark.asyncio
    async def test_updates_cost_from_modal(self, state_store, mock_modal, reconciler):
        """Reconciler updates cost info from Modal."""
        container = Container.create(modal_sandbox_id="sb-cost")
        container.state = ContainerState.RUNNING
        container.cost_accrued_usd = 0.0
        state_store.create_container(container)

        mock_modal.add_sandbox(
            ModalSandboxInfo(
                sandbox_id="sb-cost",
                status=ModalSandboxStatus.RUNNING,
                app_name="parhelia",
                parhelia_managed=True,
                cost_usd=1.50,
            )
        )

        await reconciler.reconcile()

        updated = state_store.get_container(container.id)
        assert updated.cost_accrued_usd == 1.50


class TestEventEmission:
    """Tests for event emission during reconciliation."""

    @pytest.mark.asyncio
    async def test_reconcile_failed_event_on_error(self, state_store, mock_modal):
        """Emits RECONCILE_FAILED event when reconciliation fails."""

        class FailingClient(MockModalClient):
            async def list_sandboxes(self, app_name=None):
                raise RuntimeError("API error")

        reconciler = ContainerReconciler(state_store, FailingClient())

        result = await reconciler.reconcile()

        assert len(result.errors) == 1
        assert "API error" in result.errors[0]

        events = state_store.get_events(event_type=EventType.RECONCILE_FAILED)
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_multiple_events_in_single_reconcile(
        self, state_store, mock_modal, reconciler
    ):
        """Multiple changes in one reconcile cycle emit multiple events."""
        # Setup: one stale container, one orphan
        stale = Container.create(modal_sandbox_id="sb-stale")
        stale.state = ContainerState.RUNNING
        state_store.create_container(stale)

        mock_modal.add_sandbox(
            ModalSandboxInfo(
                sandbox_id="sb-orphan",
                status=ModalSandboxStatus.RUNNING,
                app_name="parhelia",
                parhelia_managed=True,
            )
        )

        result = await reconciler.reconcile()

        assert result.stale_marked == 1
        assert result.orphans_detected == 1

        # Should have events for both
        all_events = state_store.get_events(limit=100)
        event_types = {e.event_type for e in all_events}
        assert EventType.CONTAINER_TERMINATED in event_types
        assert EventType.ORPHAN_DETECTED in event_types


class TestBackgroundLoop:
    """Tests for background reconciliation loop."""

    @pytest.mark.asyncio
    async def test_run_and_stop(self, state_store, mock_modal):
        """Background loop runs and can be stopped."""
        config = ReconcilerConfig(poll_interval_seconds=1)
        reconciler = ContainerReconciler(state_store, mock_modal, config)

        # Start in background
        task = asyncio.create_task(reconciler.run_background(interval_seconds=1))

        # Let it run a couple cycles
        await asyncio.sleep(0.5)
        assert reconciler.is_running

        # Stop it
        reconciler.stop()
        await asyncio.sleep(0.2)

        assert not reconciler.is_running
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_reconcile_called_in_loop(self, state_store, mock_modal):
        """Background loop calls reconcile at intervals."""
        config = ReconcilerConfig(poll_interval_seconds=1)
        reconciler = ContainerReconciler(state_store, mock_modal, config)

        # Track reconcile calls via list_sandboxes calls
        task = asyncio.create_task(reconciler.run_background(interval_seconds=0.1))

        await asyncio.sleep(0.35)  # Should allow ~3 cycles

        reconciler.stop()
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            task.cancel()

        # Should have made multiple list calls
        assert mock_modal.list_calls >= 2


class TestComplexScenarios:
    """Integration tests for complex reconciliation scenarios."""

    @pytest.mark.asyncio
    async def test_mixed_changes(self, state_store, mock_modal, reconciler):
        """Handles multiple types of changes in single cycle."""
        # Container 1: Running and healthy (no change needed)
        c1 = Container.create(modal_sandbox_id="sb-1")
        c1.state = ContainerState.RUNNING
        state_store.create_container(c1)
        mock_modal.add_sandbox(
            ModalSandboxInfo(
                sandbox_id="sb-1",
                status=ModalSandboxStatus.RUNNING,
                app_name="parhelia",
                parhelia_managed=True,
            )
        )

        # Container 2: In DB but gone from Modal (stale)
        c2 = Container.create(modal_sandbox_id="sb-2")
        c2.state = ContainerState.RUNNING
        state_store.create_container(c2)
        # Not in Modal

        # Container 3: In Modal but not DB (orphan)
        mock_modal.add_sandbox(
            ModalSandboxInfo(
                sandbox_id="sb-3",
                status=ModalSandboxStatus.RUNNING,
                app_name="parhelia",
                parhelia_managed=True,
            )
        )

        # Container 4: State drift (DB=CREATED, Modal=RUNNING)
        c4 = Container.create(modal_sandbox_id="sb-4")
        c4.state = ContainerState.CREATED
        state_store.create_container(c4)
        mock_modal.add_sandbox(
            ModalSandboxInfo(
                sandbox_id="sb-4",
                status=ModalSandboxStatus.RUNNING,
                app_name="parhelia",
                parhelia_managed=True,
            )
        )

        result = await reconciler.reconcile()

        assert result.stale_marked == 1
        assert result.orphans_detected == 1
        assert result.drift_corrected == 1
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_idempotent_reconciliation(
        self, state_store, mock_modal, reconciler
    ):
        """Running reconcile twice with no changes is idempotent."""
        # Setup a consistent state
        container = Container.create(modal_sandbox_id="sb-stable")
        container.state = ContainerState.RUNNING
        state_store.create_container(container)
        mock_modal.add_sandbox(
            ModalSandboxInfo(
                sandbox_id="sb-stable",
                status=ModalSandboxStatus.RUNNING,
                app_name="parhelia",
                parhelia_managed=True,
            )
        )

        # First reconcile
        result1 = await reconciler.reconcile()
        assert result1.orphans_detected == 0
        assert result1.stale_marked == 0
        assert result1.drift_corrected == 0

        # Second reconcile - should have same result
        result2 = await reconciler.reconcile()
        assert result2.orphans_detected == 0
        assert result2.stale_marked == 0
        assert result2.drift_corrected == 0

    @pytest.mark.asyncio
    async def test_orphan_not_re_detected(self, state_store, mock_modal, reconciler):
        """Once an orphan is recorded, it's not detected again."""
        mock_modal.add_sandbox(
            ModalSandboxInfo(
                sandbox_id="sb-orphan",
                status=ModalSandboxStatus.RUNNING,
                app_name="parhelia",
                parhelia_managed=True,
            )
        )

        # First reconcile - detects orphan
        result1 = await reconciler.reconcile()
        assert result1.orphans_detected == 1

        # Second reconcile - orphan is now in DB (as ORPHANED)
        # The reconciler checks by modal_sandbox_id, so it won't re-detect
        result2 = await reconciler.reconcile()
        assert result2.orphans_detected == 0  # Not re-detected

        # Verify we still only have one orphan record
        orphans = state_store.get_orphaned_containers()
        assert len(orphans) == 1


class TestReconcilerConfig:
    """Tests for reconciler configuration."""

    @pytest.mark.asyncio
    async def test_default_config(self, state_store, mock_modal):
        """Reconciler works with default config."""
        reconciler = ContainerReconciler(state_store, mock_modal)

        assert reconciler.config.poll_interval_seconds == 60
        assert reconciler.config.auto_terminate_orphans is False

    @pytest.mark.asyncio
    async def test_custom_config(self, state_store, mock_modal):
        """Reconciler accepts custom config."""
        config = ReconcilerConfig(
            poll_interval_seconds=30,
            stale_threshold_seconds=120,
            auto_terminate_orphans=True,
        )
        reconciler = ContainerReconciler(state_store, mock_modal, config)

        assert reconciler.config.poll_interval_seconds == 30
        assert reconciler.config.auto_terminate_orphans is True
