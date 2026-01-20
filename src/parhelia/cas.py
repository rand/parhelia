"""Content-Addressable Storage (REAPI-inspired).

Implements:
- [SPEC-08.10] Digest Format
- [SPEC-08.11] Directory Proto
- [SPEC-08.12] CAS API
- [SPEC-08.13] Merkle Operations
- [SPEC-08.14] Incremental Checkpoints
- [SPEC-08.15] Action Cache
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import aiofiles
import aiofiles.os


class CASError(Exception):
    """Base exception for CAS errors."""


class CorruptBlobError(CASError):
    """Raised when blob content doesn't match expected digest."""


class BlobNotFoundError(CASError):
    """Raised when blob is not found in CAS."""


# =============================================================================
# [SPEC-08.10] Digest Format
# =============================================================================


@dataclass(frozen=True)
class Digest:
    """Content identifier (REAPI-compatible).

    Implements [SPEC-08.10].
    """

    hash: str  # Lowercase hex SHA-256
    size_bytes: int  # Size for verification

    @classmethod
    def from_content(cls, content: bytes) -> Digest:
        """Create digest from content bytes."""
        return cls(
            hash=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
        )

    def __str__(self) -> str:
        """Return string representation: sha256:prefix:size."""
        return f"sha256:{self.hash[:12]}:{self.size_bytes}"


# =============================================================================
# [SPEC-08.11] Directory Proto
# =============================================================================


@dataclass
class FileNode:
    """File entry in directory.

    Implements [SPEC-08.11].
    """

    name: str  # Filename (not path)
    digest: Digest  # Content hash
    is_executable: bool


@dataclass
class DirectoryNode:
    """Subdirectory entry.

    Implements [SPEC-08.11].
    """

    name: str  # Directory name
    digest: Digest  # Hash of Directory proto


@dataclass
class SymlinkNode:
    """Symlink entry.

    Implements [SPEC-08.11].
    """

    name: str  # Link name
    target: str  # Link target path


@dataclass
class Directory:
    """Merkle tree node (REAPI-compatible).

    Implements [SPEC-08.11].
    """

    files: list[FileNode] = field(default_factory=list)  # Sorted by name
    directories: list[DirectoryNode] = field(default_factory=list)  # Sorted by name
    symlinks: list[SymlinkNode] = field(default_factory=list)  # Sorted by name

    def serialize(self) -> bytes:
        """Serialize to canonical JSON form."""
        data = {
            "files": [
                {
                    "name": f.name,
                    "digest": {"hash": f.digest.hash, "size_bytes": f.digest.size_bytes},
                    "is_executable": f.is_executable,
                }
                for f in sorted(self.files, key=lambda x: x.name)
            ],
            "directories": [
                {
                    "name": d.name,
                    "digest": {"hash": d.digest.hash, "size_bytes": d.digest.size_bytes},
                }
                for d in sorted(self.directories, key=lambda x: x.name)
            ],
            "symlinks": [
                {"name": s.name, "target": s.target}
                for s in sorted(self.symlinks, key=lambda x: x.name)
            ],
        }
        return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @classmethod
    def deserialize(cls, data: bytes) -> Directory:
        """Deserialize from JSON bytes."""
        obj = json.loads(data.decode("utf-8"))
        return cls(
            files=[
                FileNode(
                    name=f["name"],
                    digest=Digest(hash=f["digest"]["hash"], size_bytes=f["digest"]["size_bytes"]),
                    is_executable=f["is_executable"],
                )
                for f in obj.get("files", [])
            ],
            directories=[
                DirectoryNode(
                    name=d["name"],
                    digest=Digest(hash=d["digest"]["hash"], size_bytes=d["digest"]["size_bytes"]),
                )
                for d in obj.get("directories", [])
            ],
            symlinks=[
                SymlinkNode(name=s["name"], target=s["target"]) for s in obj.get("symlinks", [])
            ],
        )

    def to_digest(self) -> Digest:
        """Compute digest of this directory."""
        content = self.serialize()
        return Digest.from_content(content)


# =============================================================================
# [SPEC-08.12] Content-Addressable Storage API
# =============================================================================


class ContentAddressableStorage:
    """REAPI-inspired CAS implementation.

    Implements [SPEC-08.12].
    """

    def __init__(self, root_path: str = "/vol/parhelia/cas"):
        self.root_path = Path(root_path)
        self.blobs_path = self.root_path / "blobs" / "sha256"

    async def contains(self, digest: Digest) -> bool:
        """Check if blob exists in CAS."""
        path = self._blob_path(digest)
        return await aiofiles.os.path.exists(str(path))

    async def find_missing(self, digests: list[Digest]) -> list[Digest]:
        """Return digests not present in CAS (REAPI FindMissingBlobs)."""
        missing = []
        for digest in digests:
            if not await self.contains(digest):
                missing.append(digest)
        return missing

    async def read_blob(self, digest: Digest) -> bytes:
        """Read blob content by digest."""
        path = self._blob_path(digest)

        if not await aiofiles.os.path.exists(str(path)):
            raise BlobNotFoundError(f"Blob not found: {digest}")

        async with aiofiles.open(str(path), "rb") as f:
            content = await f.read()

        # Verify integrity
        actual_digest = Digest.from_content(content)
        if actual_digest != digest:
            raise CorruptBlobError(f"Expected {digest}, got {actual_digest}")

        return content

    async def write_blob(self, content: bytes) -> Digest:
        """Write blob to CAS, return digest."""
        digest = Digest.from_content(content)

        # Skip if already exists (deduplication)
        if await self.contains(digest):
            return digest

        path = self._blob_path(digest)
        await aiofiles.os.makedirs(str(path.parent), exist_ok=True)

        # Atomic write using temp file
        temp_path = path.with_suffix(f".tmp.{os.getpid()}")
        async with aiofiles.open(str(temp_path), "wb") as f:
            await f.write(content)

        # Atomic rename
        await aiofiles.os.rename(str(temp_path), str(path))

        return digest

    def _blob_path(self, digest: Digest) -> Path:
        """Get storage path for digest."""
        # Shard by first 2 chars for filesystem efficiency
        return self.blobs_path / digest.hash[:2] / digest.hash

    def _blob_path_by_hash(self, hash: str) -> Path:
        """Get storage path from hash string only."""
        return self.blobs_path / hash[:2] / hash

    async def read_blob_by_hash(self, hash: str) -> tuple[bytes, Digest]:
        """Read blob content by hash only (for GC purposes).

        Returns:
            Tuple of (content, actual_digest) where actual_digest has correct size.
        """
        path = self._blob_path_by_hash(hash)

        if not await aiofiles.os.path.exists(str(path)):
            raise BlobNotFoundError(f"Blob not found: {hash}")

        async with aiofiles.open(str(path), "rb") as f:
            content = await f.read()

        actual_digest = Digest.from_content(content)
        return content, actual_digest


# =============================================================================
# [SPEC-08.13] Merkle Tree Operations
# =============================================================================


@dataclass
class TreeDiff:
    """Result of comparing two Merkle trees."""

    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)


class MerkleTreeBuilder:
    """Build Merkle trees from directories.

    Implements [SPEC-08.13].
    """

    EXCLUDE_PATTERNS = {
        "node_modules",
        "__pycache__",
        ".git",
        "target",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        "dist",
        "build",
        ".next",
        ".nuxt",
    }

    def __init__(self, cas: ContentAddressableStorage):
        self.cas = cas

    async def build_tree(self, path: str) -> Digest:
        """Build Merkle tree from directory, return root digest."""
        return await self._build_directory(path)

    async def _build_directory(self, path: str) -> Digest:
        """Recursively build directory tree."""
        files: list[FileNode] = []
        directories: list[DirectoryNode] = []
        symlinks: list[SymlinkNode] = []

        dir_path = Path(path)

        for entry in sorted(dir_path.iterdir(), key=lambda e: e.name):
            # Skip excluded patterns
            if self._should_exclude(entry.name):
                continue

            if entry.is_symlink():
                symlinks.append(
                    SymlinkNode(
                        name=entry.name,
                        target=str(os.readlink(entry)),
                    )
                )
            elif entry.is_file():
                async with aiofiles.open(str(entry), "rb") as f:
                    content = await f.read()
                digest = await self.cas.write_blob(content)
                files.append(
                    FileNode(
                        name=entry.name,
                        digest=digest,
                        is_executable=os.access(entry, os.X_OK),
                    )
                )
            elif entry.is_dir():
                digest = await self._build_directory(str(entry))
                directories.append(
                    DirectoryNode(
                        name=entry.name,
                        digest=digest,
                    )
                )

        directory = Directory(files=files, directories=directories, symlinks=symlinks)
        dir_content = directory.serialize()
        return await self.cas.write_blob(dir_content)

    def _should_exclude(self, name: str) -> bool:
        """Check if name should be excluded."""
        return name in self.EXCLUDE_PATTERNS


class MerkleTreeDiff:
    """Compare Merkle trees efficiently.

    Implements [SPEC-08.13].
    """

    def __init__(self, cas: ContentAddressableStorage):
        self.cas = cas

    async def diff(self, old_root: Digest, new_root: Digest) -> TreeDiff:
        """Find differences between two trees."""
        added: list[str] = []
        modified: list[str] = []
        deleted: list[str] = []

        await self._diff_recursive(old_root, new_root, "", added, modified, deleted)

        return TreeDiff(added=added, modified=modified, deleted=deleted)

    async def _diff_recursive(
        self,
        old_digest: Digest | None,
        new_digest: Digest | None,
        path: str,
        added: list[str],
        modified: list[str],
        deleted: list[str],
    ) -> None:
        """Recursive tree comparison."""
        if old_digest == new_digest:
            # Subtrees identical - skip entirely
            return

        if old_digest is None and new_digest is not None:
            # New subtree - all files are added
            await self._collect_all(new_digest, path, added)
            return

        if old_digest is not None and new_digest is None:
            # Deleted subtree - all files are deleted
            await self._collect_all(old_digest, path, deleted)
            return

        # Both exist but different - compare contents
        assert old_digest is not None and new_digest is not None

        old_dir = await self._load_directory(old_digest)
        new_dir = await self._load_directory(new_digest)

        # Compare files
        old_files = {f.name: f for f in old_dir.files}
        new_files = {f.name: f for f in new_dir.files}

        for name in set(old_files.keys()) | set(new_files.keys()):
            file_path = f"{path}/{name}" if path else name
            old_file = old_files.get(name)
            new_file = new_files.get(name)

            if old_file is None:
                added.append(file_path)
            elif new_file is None:
                deleted.append(file_path)
            elif old_file.digest != new_file.digest:
                modified.append(file_path)

        # Recurse into subdirectories
        old_dirs = {d.name: d for d in old_dir.directories}
        new_dirs = {d.name: d for d in new_dir.directories}

        for name in set(old_dirs.keys()) | set(new_dirs.keys()):
            dir_path = f"{path}/{name}" if path else name
            old_subdir = old_dirs.get(name)
            new_subdir = new_dirs.get(name)

            await self._diff_recursive(
                old_subdir.digest if old_subdir else None,
                new_subdir.digest if new_subdir else None,
                dir_path,
                added,
                modified,
                deleted,
            )

    async def _load_directory(self, digest: Digest) -> Directory:
        """Load directory from CAS."""
        content = await self.cas.read_blob(digest)
        return Directory.deserialize(content)

    async def _collect_all(self, digest: Digest, path: str, collection: list[str]) -> None:
        """Collect all files in a tree."""
        directory = await self._load_directory(digest)

        for f in directory.files:
            file_path = f"{path}/{f.name}" if path else f.name
            collection.append(file_path)

        for d in directory.directories:
            dir_path = f"{path}/{d.name}" if path else d.name
            await self._collect_all(d.digest, dir_path, collection)


# =============================================================================
# [SPEC-08.14] Incremental Workspace Checkpoint
# =============================================================================


@dataclass
class Checkpoint:
    """Workspace checkpoint using CAS.

    Implements [SPEC-08.14].
    """

    workspace_root: Digest
    changed_files: list[str] = field(default_factory=list)


class IncrementalCheckpointManager:
    """Checkpoint manager using CAS for efficiency.

    Implements [SPEC-08.14].
    """

    def __init__(self, cas: ContentAddressableStorage):
        self.cas = cas
        self.tree_builder = MerkleTreeBuilder(cas)
        self.tree_diff = MerkleTreeDiff(cas)

    async def create_checkpoint(
        self,
        working_directory: str,
        previous_checkpoint: Checkpoint | None = None,
    ) -> Checkpoint:
        """Create checkpoint with incremental storage."""
        # Build Merkle tree of current workspace
        new_root = await self.tree_builder.build_tree(working_directory)

        # Compute what changed (for metadata)
        changed_files: list[str] = []
        if previous_checkpoint:
            diff = await self.tree_diff.diff(
                previous_checkpoint.workspace_root,
                new_root,
            )
            changed_files = diff.added + diff.modified + diff.deleted

        return Checkpoint(
            workspace_root=new_root,
            changed_files=changed_files,
        )

    async def restore_checkpoint(
        self,
        checkpoint: Checkpoint,
        target_dir: str,
        from_checkpoint: Checkpoint | None = None,
    ) -> None:
        """Restore workspace from checkpoint, incrementally if possible."""
        target_path = Path(target_dir)

        if from_checkpoint and from_checkpoint.workspace_root:
            # Incremental restore - only sync changes
            diff = await self.tree_diff.diff(
                from_checkpoint.workspace_root,
                checkpoint.workspace_root,
            )

            # Delete removed files
            for path in diff.deleted:
                full_path = target_path / path
                if full_path.exists():
                    full_path.unlink()

            # Restore added/modified files
            for path in diff.added + diff.modified:
                await self._restore_file(checkpoint.workspace_root, path, target_dir)

        else:
            # Full restore
            await self._restore_tree(checkpoint.workspace_root, target_dir)

    async def _restore_tree(self, root_digest: Digest, target_dir: str) -> None:
        """Restore entire tree from CAS."""
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)

        directory = Directory.deserialize(await self.cas.read_blob(root_digest))

        # Restore files
        for f in directory.files:
            file_path = target_path / f.name
            content = await self.cas.read_blob(f.digest)
            async with aiofiles.open(str(file_path), "wb") as out:
                await out.write(content)
            if f.is_executable:
                os.chmod(file_path, 0o755)

        # Restore symlinks
        for s in directory.symlinks:
            link_path = target_path / s.name
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            link_path.symlink_to(s.target)

        # Restore subdirectories
        for d in directory.directories:
            subdir_path = target_path / d.name
            await self._restore_tree(d.digest, str(subdir_path))

    async def _restore_file(self, root_digest: Digest, file_path: str, target_dir: str) -> None:
        """Restore single file from tree."""
        target_path = Path(target_dir)
        parts = file_path.split("/")

        # Navigate to the file's directory
        current_digest = root_digest
        for part in parts[:-1]:
            directory = Directory.deserialize(await self.cas.read_blob(current_digest))
            for d in directory.directories:
                if d.name == part:
                    current_digest = d.digest
                    break

        # Find and restore the file
        directory = Directory.deserialize(await self.cas.read_blob(current_digest))
        file_name = parts[-1]
        for f in directory.files:
            if f.name == file_name:
                full_path = target_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                content = await self.cas.read_blob(f.digest)
                async with aiofiles.open(str(full_path), "wb") as out:
                    await out.write(content)
                if f.is_executable:
                    os.chmod(full_path, 0o755)
                break


# =============================================================================
# [SPEC-08.15] Action Cache
# =============================================================================


@dataclass
class Action:
    """Cacheable action definition (REAPI-inspired).

    Implements [SPEC-08.15].
    """

    command: list[str]  # Command to execute
    working_directory: str  # Relative to workspace root
    input_root: Digest  # Merkle tree of inputs
    environment: dict[str, str]  # Sorted env vars
    timeout_seconds: int

    def to_digest(self) -> Digest:
        """Compute action cache key."""
        canonical = json.dumps(
            {
                "command": self.command,
                "working_directory": self.working_directory,
                "input_root": str(self.input_root),
                "environment": sorted(self.environment.items()),
            },
            sort_keys=True,
        )
        return Digest.from_content(canonical.encode())


@dataclass
class ActionResult:
    """Cached result of action execution.

    Implements [SPEC-08.15].
    """

    exit_code: int
    stdout_digest: Digest | None
    stderr_digest: Digest | None
    output_files: list[tuple[str, Digest]] = field(default_factory=list)  # (path, digest) pairs
    execution_metadata: dict = field(default_factory=dict)


class ActionCache:
    """Cache for deterministic task results.

    Implements [SPEC-08.15].
    """

    CACHEABLE_COMMANDS = {
        # Build commands
        "npm run build",
        "yarn build",
        "pnpm build",
        "cargo build",
        "go build",
        "make",
        # Test commands
        "npm test",
        "pytest",
        "cargo test",
        "go test",
        # Lint commands
        "eslint",
        "ruff",
        "clippy",
        "golangci-lint",
    }

    def __init__(self, cas: ContentAddressableStorage):
        self.cas = cas
        self.actions_path = cas.root_path / "actions" / "sha256"

    async def lookup(self, action: Action) -> ActionResult | None:
        """Look up cached result for action."""
        action_digest = action.to_digest()
        cache_path = self.actions_path / action_digest.hash[:2] / action_digest.hash

        if not await aiofiles.os.path.exists(str(cache_path)):
            return None

        async with aiofiles.open(str(cache_path), "r") as f:
            data = json.loads(await f.read())

        # Reconstruct ActionResult
        stdout_digest = None
        if data.get("stdout_digest"):
            stdout_digest = Digest(
                hash=data["stdout_digest"]["hash"],
                size_bytes=data["stdout_digest"]["size_bytes"],
            )

        stderr_digest = None
        if data.get("stderr_digest"):
            stderr_digest = Digest(
                hash=data["stderr_digest"]["hash"],
                size_bytes=data["stderr_digest"]["size_bytes"],
            )

        output_files = [
            (path, Digest(hash=d["hash"], size_bytes=d["size_bytes"]))
            for path, d in data.get("output_files", [])
        ]

        return ActionResult(
            exit_code=data["exit_code"],
            stdout_digest=stdout_digest,
            stderr_digest=stderr_digest,
            output_files=output_files,
            execution_metadata=data.get("execution_metadata", {}),
        )

    async def store(self, action: Action, result: ActionResult) -> None:
        """Store action result in cache."""
        action_digest = action.to_digest()
        cache_path = self.actions_path / action_digest.hash[:2] / action_digest.hash

        await aiofiles.os.makedirs(str(cache_path.parent), exist_ok=True)

        # Serialize result
        data = {
            "exit_code": result.exit_code,
            "stdout_digest": (
                {"hash": result.stdout_digest.hash, "size_bytes": result.stdout_digest.size_bytes}
                if result.stdout_digest
                else None
            ),
            "stderr_digest": (
                {"hash": result.stderr_digest.hash, "size_bytes": result.stderr_digest.size_bytes}
                if result.stderr_digest
                else None
            ),
            "output_files": [
                (path, {"hash": d.hash, "size_bytes": d.size_bytes})
                for path, d in result.output_files
            ],
            "execution_metadata": result.execution_metadata,
        }

        async with aiofiles.open(str(cache_path), "w") as f:
            await f.write(json.dumps(data))

    def is_cacheable(self, command: list[str]) -> bool:
        """Check if command output can be cached."""
        cmd_str = " ".join(command)
        return any(cacheable in cmd_str for cacheable in self.CACHEABLE_COMMANDS)


# =============================================================================
# CAS Garbage Collection
# =============================================================================


@dataclass
class GCConfig:
    """Configuration for garbage collection."""

    max_size_bytes: int = 10 * 1024 * 1024 * 1024  # 10 GB default
    target_size_bytes: int = 8 * 1024 * 1024 * 1024  # 8 GB target after GC
    min_blob_age_seconds: int = 3600  # Don't delete blobs younger than 1 hour
    dry_run: bool = False  # If True, don't actually delete


@dataclass
class GCResult:
    """Result of garbage collection."""

    blobs_deleted: int = 0
    bytes_freed: int = 0
    blobs_retained: int = 0
    bytes_retained: int = 0
    errors: list[str] = field(default_factory=list)
    dry_run: bool = False

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "blobs_deleted": self.blobs_deleted,
            "bytes_freed": self.bytes_freed,
            "blobs_retained": self.blobs_retained,
            "bytes_retained": self.bytes_retained,
            "errors": self.errors,
            "dry_run": self.dry_run,
        }


@dataclass
class BlobInfo:
    """Information about a blob for GC decisions."""

    digest: Digest
    path: Path
    size_bytes: int
    access_time: float  # Unix timestamp of last access
    modify_time: float  # Unix timestamp of last modification


class CASGarbageCollector:
    """LRU-based garbage collection for CAS blobs.

    Tracks access times and deletes unused blobs when storage
    exceeds configured threshold.
    """

    def __init__(
        self,
        cas: ContentAddressableStorage,
        config: GCConfig | None = None,
    ):
        self.cas = cas
        self.config = config or GCConfig()
        self._access_log_path = cas.root_path / "gc" / "access.jsonl"
        self._reference_roots: set[str] = set()  # Root digests to protect

    async def get_storage_stats(self) -> tuple[int, int]:
        """Get current storage statistics.

        Returns:
            Tuple of (total_bytes, blob_count).
        """
        total_bytes = 0
        blob_count = 0

        blobs_path = self.cas.blobs_path
        if not blobs_path.exists():
            return 0, 0

        for shard_dir in blobs_path.iterdir():
            if shard_dir.is_dir():
                for blob_file in shard_dir.iterdir():
                    if blob_file.is_file():
                        stat = blob_file.stat()
                        total_bytes += stat.st_size
                        blob_count += 1

        return total_bytes, blob_count

    async def list_blobs(self) -> list[BlobInfo]:
        """List all blobs with their metadata."""
        blobs: list[BlobInfo] = []
        blobs_path = self.cas.blobs_path

        if not blobs_path.exists():
            return []

        for shard_dir in blobs_path.iterdir():
            if not shard_dir.is_dir():
                continue

            for blob_file in shard_dir.iterdir():
                if not blob_file.is_file():
                    continue

                stat = blob_file.stat()
                digest = Digest(
                    hash=blob_file.name,
                    size_bytes=stat.st_size,
                )

                blobs.append(
                    BlobInfo(
                        digest=digest,
                        path=blob_file,
                        size_bytes=stat.st_size,
                        access_time=stat.st_atime,
                        modify_time=stat.st_mtime,
                    )
                )

        return blobs

    def add_reference_root(self, root_digest: str) -> None:
        """Add a root digest that should be protected from GC.

        All blobs reachable from this root will be retained.
        """
        self._reference_roots.add(root_digest)

    def remove_reference_root(self, root_digest: str) -> None:
        """Remove a root digest from protection."""
        self._reference_roots.discard(root_digest)

    async def collect_referenced_digests(self) -> set[str]:
        """Collect all digests reachable from reference roots.

        Returns:
            Set of digest hashes that should be retained.
        """
        referenced: set[str] = set()

        for root_hash in self._reference_roots:
            await self._collect_tree_digests_by_hash(root_hash, referenced)

        return referenced

    async def _collect_tree_digests_by_hash(self, hash: str, collected: set[str]) -> None:
        """Recursively collect all digests in a tree by hash."""
        if hash in collected:
            return

        collected.add(hash)

        # Try to load as directory and recurse
        try:
            content, _ = await self.cas.read_blob_by_hash(hash)
            directory = Directory.deserialize(content)

            # Add file digests
            for f in directory.files:
                collected.add(f.digest.hash)

            # Recurse into subdirectories
            for d in directory.directories:
                await self._collect_tree_digests_by_hash(d.digest.hash, collected)

        except (BlobNotFoundError, json.JSONDecodeError):
            # Not a directory or not found - that's OK
            pass

    async def should_run_gc(self) -> bool:
        """Check if GC should run based on storage threshold."""
        total_bytes, _ = await self.get_storage_stats()
        return total_bytes > self.config.max_size_bytes

    async def run_gc(self) -> GCResult:
        """Run garbage collection.

        Deletes unreferenced blobs using LRU policy until storage
        is below target threshold.
        """
        import time

        result = GCResult(dry_run=self.config.dry_run)
        current_time = time.time()

        # Get storage stats
        total_bytes, _ = await self.get_storage_stats()

        if total_bytes <= self.config.max_size_bytes:
            # No GC needed
            result.bytes_retained = total_bytes
            return result

        # Collect referenced digests
        referenced = await self.collect_referenced_digests()

        # List all blobs
        blobs = await self.list_blobs()

        # Sort by access time (oldest first - LRU)
        blobs.sort(key=lambda b: b.access_time)

        bytes_to_free = total_bytes - self.config.target_size_bytes
        bytes_freed = 0

        for blob in blobs:
            # Check if we've freed enough
            if bytes_freed >= bytes_to_free:
                break

            # Skip referenced blobs
            if blob.digest.hash in referenced:
                result.blobs_retained += 1
                result.bytes_retained += blob.size_bytes
                continue

            # Skip recently created/accessed blobs
            age = current_time - blob.modify_time
            if age < self.config.min_blob_age_seconds:
                result.blobs_retained += 1
                result.bytes_retained += blob.size_bytes
                continue

            # Delete the blob
            try:
                if not self.config.dry_run:
                    blob.path.unlink()

                result.blobs_deleted += 1
                bytes_freed += blob.size_bytes

            except OSError as e:
                result.errors.append(f"Failed to delete {blob.digest}: {e}")

        result.bytes_freed = bytes_freed

        # Count remaining blobs
        if not self.config.dry_run:
            remaining_bytes, remaining_count = await self.get_storage_stats()
            result.bytes_retained = remaining_bytes
            result.blobs_retained = remaining_count

        return result

    async def log_access(self, digest: Digest) -> None:
        """Log blob access for LRU tracking.

        Updates the access time by touching the file.
        """
        path = self.cas._blob_path(digest)
        if path.exists():
            # Update access time
            path.touch()

    def format_gc_result(self, result: GCResult) -> str:
        """Format GC result for display."""
        lines = [
            "CAS Garbage Collection Results",
            "=" * 40,
            f"Mode: {'Dry Run' if result.dry_run else 'Live'}",
            "",
            f"Blobs deleted: {result.blobs_deleted:,}",
            f"Bytes freed: {self._format_bytes(result.bytes_freed)}",
            f"Blobs retained: {result.blobs_retained:,}",
            f"Bytes retained: {self._format_bytes(result.bytes_retained)}",
        ]

        if result.errors:
            lines.append("")
            lines.append(f"Errors ({len(result.errors)}):")
            for error in result.errors[:5]:
                lines.append(f"  - {error}")
            if len(result.errors) > 5:
                lines.append(f"  ... and {len(result.errors) - 5} more")

        return "\n".join(lines)

    def _format_bytes(self, num_bytes: int) -> str:
        """Format bytes as human-readable string."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if abs(num_bytes) < 1024:
                return f"{num_bytes:.1f} {unit}"
            num_bytes //= 1024
        return f"{num_bytes:.1f} PB"
