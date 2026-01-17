"""Result aggregation for parallel task execution.

Implements:
- [SPEC-05.13] Result Aggregation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from parhelia.orchestrator import Task


@dataclass
class ExecutionMetrics:
    """Metrics from task execution.

    Implements [SPEC-05.13].
    """

    duration_seconds: float = 0.0
    cost_usd: float = 0.0
    tokens_used: int = 0


@dataclass
class Artifact:
    """A file artifact from task execution.

    Implements [SPEC-05.13].
    """

    path: str
    content: str
    action: Literal["created", "modified", "deleted"]


@dataclass
class WorkResult:
    """Result from a single work unit.

    Implements [SPEC-05.13].
    """

    work_unit_id: str
    worker_id: str
    status: Literal["success", "partial", "failed"]
    output: str = ""
    error: str | None = None
    artifacts: list[Artifact] = field(default_factory=list)
    metrics: ExecutionMetrics = field(default_factory=ExecutionMetrics)


@dataclass
class AggregatedResult:
    """Aggregated result from parallel work units.

    Implements [SPEC-05.13].
    """

    task_id: str
    status: Literal["success", "partial", "failed"]
    summary: str = ""
    results: list[WorkResult] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    total_cost: float = 0.0
    total_duration: float = 0.0


class ResultAggregator:
    """Aggregate results from parallel work units.

    Implements [SPEC-05.13].

    The aggregator:
    - Combines results from multiple workers
    - Handles partial failures
    - Merges file artifacts
    - Detects and resolves conflicts
    """

    async def aggregate(
        self,
        task: Task,
        results: list[WorkResult],
    ) -> AggregatedResult:
        """Aggregate results from all work units.

        Args:
            task: The original task.
            results: Results from individual work units.

        Returns:
            Aggregated result with combined artifacts and metrics.
        """
        # Check for failures
        failed = [r for r in results if r.status == "failed"]
        succeeded = [r for r in results if r.status == "success"]

        # Determine overall status
        if len(failed) == len(results):
            status = "failed"
            summary = self._summarize_failures(failed)
        elif failed:
            status = "partial"
            summary = f"{len(succeeded)}/{len(results)} succeeded"
        else:
            status = "success"
            summary = f"All {len(results)} work units completed"

        # Merge artifacts
        merged_artifacts = await self._merge_artifacts(results)

        # Calculate totals
        total_cost = sum(r.metrics.cost_usd for r in results)
        # For parallel execution, duration is max (concurrent), not sum
        total_duration = max(
            (r.metrics.duration_seconds for r in results),
            default=0.0,
        )

        return AggregatedResult(
            task_id=task.id,
            status=status,
            summary=summary,
            results=results,
            artifacts=merged_artifacts,
            total_cost=total_cost,
            total_duration=total_duration,
        )

    def _summarize_failures(self, failed: list[WorkResult]) -> str:
        """Create summary of failed work units.

        Args:
            failed: List of failed work results.

        Returns:
            Summary string.
        """
        errors = [r.error or "Unknown error" for r in failed]
        return f"Failed: {'; '.join(errors[:3])}"

    async def _merge_artifacts(
        self,
        results: list[WorkResult],
    ) -> list[Artifact]:
        """Merge artifacts from all results.

        Args:
            results: Work results containing artifacts.

        Returns:
            Merged list of artifacts, with conflicts resolved.
        """
        artifacts_by_path: dict[str, list[Artifact]] = {}

        for result in results:
            for artifact in result.artifacts:
                if artifact.path not in artifacts_by_path:
                    artifacts_by_path[artifact.path] = []
                artifacts_by_path[artifact.path].append(artifact)

        merged: list[Artifact] = []
        for path, versions in artifacts_by_path.items():
            if len(versions) == 1:
                merged.append(versions[0])
            else:
                # Conflict - use last version for now
                # Future: implement proper conflict resolution
                resolved = await self._resolve_conflict(path, versions)
                merged.append(resolved)

        return merged

    async def _resolve_conflict(
        self,
        path: str,
        versions: list[Artifact],
    ) -> Artifact:
        """Resolve conflict between multiple versions of an artifact.

        Args:
            path: The conflicting file path.
            versions: Multiple versions of the artifact.

        Returns:
            Resolved artifact.
        """
        # Simple strategy: take the last version
        # Future: implement three-way merge or user prompt
        return versions[-1]
