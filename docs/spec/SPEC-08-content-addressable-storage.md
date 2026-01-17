# SPEC-08: Content-Addressable Storage (REAPI-Inspired)

**Status**: Draft
**Issue**: TBD
**Author**: Claude + rand
**Date**: 2026-01-16

## Overview

This specification defines a content-addressable storage (CAS) system inspired by the [Bazel Remote Execution API (REAPI)](https://github.com/bazelbuild/remote-apis). CAS provides significant efficiency gains for workspace checkpointing, plugin caching, and deterministic task caching.

## Goals

- [SPEC-08.01] Store workspace state using content-addressable Merkle trees
- [SPEC-08.02] Enable incremental workspace transfers (only changed files)
- [SPEC-08.03] Deduplicate common content across sessions/containers
- [SPEC-08.04] Cache deterministic task results (builds, tests, lints)
- [SPEC-08.05] Reduce checkpoint storage and transfer costs by 60-90%

## Non-Goals

- Full REAPI protocol compliance (too rigid for conversational AI)
- REAPI's strict action execution model (we need stateful sessions)
- gRPC protocol (HTTP/REST simpler for our use case)

---

## Background: Why REAPI Concepts Apply

### Problem with Current Approach (SPEC-03)

SPEC-03 uses tar.zst archives for workspace snapshots:
- **Full snapshots only**: Every checkpoint stores entire workspace
- **No deduplication**: Same node_modules across sessions stored repeatedly
- **Slow resume**: Must extract full archive even for small changes
- **High storage costs**: Large workspaces accumulate quickly

### REAPI Solution: Content-Addressable Storage

REAPI's CAS stores content by hash, enabling:
- **Incremental transfers**: Only upload/download changed files
- **Deduplication**: Identical content shares storage automatically
- **Fast diff**: Merkle trees make comparison O(changed files), not O(total files)
- **Action caching**: Deterministic operations return cached results

---

## Architecture

### Merkle Tree Directory Structure

```
                    ┌─────────────────────────┐
                    │ Root Directory          │
                    │ hash: abc123            │
                    └────────────┬────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│ src/          │      │ tests/        │      │ package.json  │
│ hash: def456  │      │ hash: ghi789  │      │ hash: jkl012  │
└───────┬───────┘      └───────────────┘      └───────────────┘
        │
   ┌────┴────┐
   │         │
   ▼         ▼
┌─────┐  ┌─────┐
│app.py│ │util.py│
│hash1│  │hash2│
└─────┘  └─────┘
```

**Key insight**: Changing one file only invalidates hashes on the path from that file to root. Siblings remain unchanged and don't need re-upload.

### CAS Storage Backend

```
/vol/parhelia/cas/
├── blobs/                    # Content-addressed blobs
│   ├── sha256/
│   │   ├── abc123...         # File content (any size)
│   │   ├── def456...         # Directory proto
│   │   └── ...
├── trees/                    # Merkle tree roots (aliases)
│   ├── workspace-{session}/{checkpoint-id} -> sha256:xxx
│   └── plugins/cc-polymath -> sha256:yyy
└── actions/                  # Action result cache
    └── sha256/
        ├── {action-digest} -> ActionResult
        └── ...
```

---

## Requirements

### [SPEC-08.10] Digest Format

All content MUST be addressed by digest:

```python
@dataclass(frozen=True)
class Digest:
    """Content identifier (REAPI-compatible)."""
    hash: str           # Lowercase hex SHA-256
    size_bytes: int     # Size for verification

    @classmethod
    def from_content(cls, content: bytes) -> "Digest":
        return cls(
            hash=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
        )

    def __str__(self) -> str:
        return f"sha256:{self.hash[:12]}:{self.size_bytes}"
```

### [SPEC-08.11] Directory Proto

Directories MUST be represented as sorted lists of entries:

```python
@dataclass
class FileNode:
    """File entry in directory."""
    name: str           # Filename (not path)
    digest: Digest      # Content hash
    is_executable: bool

@dataclass
class DirectoryNode:
    """Subdirectory entry."""
    name: str           # Directory name
    digest: Digest      # Hash of Directory proto

@dataclass
class Directory:
    """Merkle tree node (REAPI-compatible)."""
    files: list[FileNode]           # Sorted by name
    directories: list[DirectoryNode] # Sorted by name
    symlinks: list[SymlinkNode]     # Sorted by name

    def to_digest(self) -> Digest:
        """Compute digest of this directory."""
        # Serialize to canonical form and hash
        content = self.serialize()
        return Digest.from_content(content)
```

### [SPEC-08.12] Content-Addressable Storage API

The CAS MUST support these operations:

```python
class ContentAddressableStorage:
    """REAPI-inspired CAS implementation."""

    async def contains(self, digest: Digest) -> bool:
        """Check if blob exists in CAS."""
        path = self._blob_path(digest)
        return await aiofiles.os.path.exists(path)

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
        async with aiofiles.open(path, "rb") as f:
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
        await aiofiles.os.makedirs(os.path.dirname(path), exist_ok=True)

        # Atomic write
        temp_path = f"{path}.tmp.{os.getpid()}"
        async with aiofiles.open(temp_path, "wb") as f:
            await f.write(content)
        await aiofiles.os.rename(temp_path, path)

        return digest

    def _blob_path(self, digest: Digest) -> str:
        """Get storage path for digest."""
        # Shard by first 2 chars for filesystem efficiency
        return f"/vol/parhelia/cas/blobs/sha256/{digest.hash[:2]}/{digest.hash}"
```

### [SPEC-08.13] Merkle Tree Operations

The system MUST support efficient tree operations:

```python
class MerkleTreeBuilder:
    """Build Merkle trees from directories."""

    def __init__(self, cas: ContentAddressableStorage):
        self.cas = cas

    async def build_tree(self, path: str) -> Digest:
        """Build Merkle tree from directory, return root digest."""
        return await self._build_directory(path)

    async def _build_directory(self, path: str) -> Digest:
        """Recursively build directory tree."""
        files: list[FileNode] = []
        directories: list[DirectoryNode] = []

        for entry in sorted(os.scandir(path), key=lambda e: e.name):
            # Skip excluded patterns
            if self._should_exclude(entry.name):
                continue

            if entry.is_file(follow_symlinks=False):
                content = await aiofiles.read(entry.path, "rb")
                digest = await self.cas.write_blob(content)
                files.append(FileNode(
                    name=entry.name,
                    digest=digest,
                    is_executable=os.access(entry.path, os.X_OK),
                ))

            elif entry.is_dir(follow_symlinks=False):
                digest = await self._build_directory(entry.path)
                directories.append(DirectoryNode(
                    name=entry.name,
                    digest=digest,
                ))

        directory = Directory(files=files, directories=directories, symlinks=[])
        dir_content = directory.serialize()
        return await self.cas.write_blob(dir_content)

    EXCLUDE_PATTERNS = {
        "node_modules", "__pycache__", ".git", "target",
        ".venv", "venv", ".mypy_cache", ".pytest_cache",
        "dist", "build", ".next", ".nuxt",
    }

    def _should_exclude(self, name: str) -> bool:
        return name in self.EXCLUDE_PATTERNS


class MerkleTreeDiff:
    """Compare Merkle trees efficiently."""

    async def diff(self, old_root: Digest, new_root: Digest) -> TreeDiff:
        """Find differences between two trees."""
        added: list[str] = []
        modified: list[str] = []
        deleted: list[str] = []

        await self._diff_recursive(
            old_root, new_root, "",
            added, modified, deleted
        )

        return TreeDiff(added=added, modified=modified, deleted=deleted)

    async def _diff_recursive(
        self,
        old_digest: Digest | None,
        new_digest: Digest | None,
        path: str,
        added: list[str],
        modified: list[str],
        deleted: list[str],
    ):
        """Recursive tree comparison."""
        if old_digest == new_digest:
            # Subtrees identical - skip entirely
            return

        if old_digest is None:
            # New subtree - all files are added
            await self._collect_all(new_digest, path, added)
            return

        if new_digest is None:
            # Deleted subtree - all files are deleted
            await self._collect_all(old_digest, path, deleted)
            return

        # Both exist but different - compare contents
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
                added, modified, deleted,
            )
```

### [SPEC-08.14] Incremental Workspace Checkpoint

Checkpoints MUST use incremental updates:

```python
class IncrementalCheckpointManager:
    """Checkpoint manager using CAS for efficiency."""

    def __init__(self, cas: ContentAddressableStorage):
        self.cas = cas
        self.tree_builder = MerkleTreeBuilder(cas)

    async def create_checkpoint(
        self,
        session: Session,
        previous_checkpoint: Checkpoint | None = None,
    ) -> Checkpoint:
        """Create checkpoint with incremental storage."""

        # Build Merkle tree of current workspace
        new_root = await self.tree_builder.build_tree(session.working_directory)

        # Compute what changed (for metadata)
        if previous_checkpoint:
            diff = await MerkleTreeDiff().diff(
                previous_checkpoint.workspace_root,
                new_root,
            )
        else:
            diff = None

        return Checkpoint(
            id=generate_ulid(),
            session_id=session.id,
            workspace_root=new_root,  # Just a digest, not full archive!
            changed_files=diff.modified if diff else [],
            # ... other fields
        )

    async def restore_checkpoint(
        self,
        checkpoint: Checkpoint,
        target_dir: str,
        from_checkpoint: Checkpoint | None = None,
    ):
        """Restore workspace from checkpoint, incrementally if possible."""

        if from_checkpoint and from_checkpoint.workspace_root:
            # Incremental restore - only sync changes
            diff = await MerkleTreeDiff().diff(
                from_checkpoint.workspace_root,
                checkpoint.workspace_root,
            )

            # Delete removed files
            for path in diff.deleted:
                full_path = os.path.join(target_dir, path)
                if os.path.exists(full_path):
                    os.remove(full_path)

            # Restore added/modified files
            for path in diff.added + diff.modified:
                await self._restore_file(checkpoint.workspace_root, path, target_dir)

        else:
            # Full restore
            await self._restore_tree(checkpoint.workspace_root, target_dir)
```

### [SPEC-08.15] Action Cache

Deterministic tasks MUST be cached by action digest:

```python
@dataclass
class Action:
    """Cacheable action definition (REAPI-inspired)."""
    command: list[str]          # Command to execute
    working_directory: str      # Relative to workspace root
    input_root: Digest          # Merkle tree of inputs
    environment: dict[str, str] # Sorted env vars
    timeout_seconds: int

    def to_digest(self) -> Digest:
        """Compute action cache key."""
        canonical = json.dumps({
            "command": self.command,
            "working_directory": self.working_directory,
            "input_root": str(self.input_root),
            "environment": sorted(self.environment.items()),
        }, sort_keys=True)
        return Digest.from_content(canonical.encode())


@dataclass
class ActionResult:
    """Cached result of action execution."""
    exit_code: int
    stdout_digest: Digest | None
    stderr_digest: Digest | None
    output_files: list[tuple[str, Digest]]  # (path, digest) pairs
    execution_metadata: dict


class ActionCache:
    """Cache for deterministic task results."""

    CACHEABLE_COMMANDS = {
        # Build commands
        "npm run build", "yarn build", "pnpm build",
        "cargo build", "go build", "make",
        # Test commands
        "npm test", "pytest", "cargo test", "go test",
        # Lint commands
        "eslint", "ruff", "clippy", "golangci-lint",
    }

    async def lookup(self, action: Action) -> ActionResult | None:
        """Look up cached result for action."""
        action_digest = action.to_digest()
        cache_path = f"/vol/parhelia/cas/actions/sha256/{action_digest.hash}"

        if not os.path.exists(cache_path):
            return None

        async with aiofiles.open(cache_path, "r") as f:
            data = json.loads(await f.read())

        return ActionResult(**data)

    async def store(self, action: Action, result: ActionResult):
        """Store action result in cache."""
        action_digest = action.to_digest()
        cache_path = f"/vol/parhelia/cas/actions/sha256/{action_digest.hash}"

        await aiofiles.os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        async with aiofiles.open(cache_path, "w") as f:
            await f.write(json.dumps(asdict(result)))

    def is_cacheable(self, command: list[str]) -> bool:
        """Check if command output can be cached."""
        cmd_str = " ".join(command)
        return any(cacheable in cmd_str for cacheable in self.CACHEABLE_COMMANDS)
```

### [SPEC-08.16] Plugin CAS Storage

Plugins MUST be stored in CAS for deduplication:

```python
class PluginCASManager:
    """Store plugins in CAS for deduplication across containers."""

    async def sync_plugin(
        self,
        plugin_url: str,
        version: str,
    ) -> Digest:
        """Sync plugin and return CAS root digest."""

        # Check if this version already in CAS
        alias = f"plugins/{plugin_name}/{version}"
        existing = await self.cas.get_alias(alias)
        if existing:
            return existing

        # Clone to temp directory
        temp_dir = tempfile.mkdtemp()
        await run_command(["git", "clone", "--depth=1", plugin_url, temp_dir])

        # Build Merkle tree and store in CAS
        root_digest = await self.tree_builder.build_tree(temp_dir)

        # Create alias for easy lookup
        await self.cas.set_alias(alias, root_digest)

        # Cleanup temp
        shutil.rmtree(temp_dir)

        return root_digest

    async def materialize_plugin(
        self,
        plugin_name: str,
        version: str,
        target_dir: str,
    ):
        """Restore plugin from CAS to filesystem."""
        alias = f"plugins/{plugin_name}/{version}"
        root_digest = await self.cas.get_alias(alias)

        if not root_digest:
            raise PluginNotFoundError(f"{plugin_name}@{version} not in CAS")

        await self._restore_tree(root_digest, target_dir)
```

---

## Performance Analysis

### Storage Savings

| Scenario | tar.zst (SPEC-03) | CAS (SPEC-08) | Savings |
|----------|-------------------|---------------|---------|
| 10 checkpoints, same workspace | 10 × 50MB = 500MB | 50MB + ~5MB deltas | ~90% |
| 5 sessions, same project | 5 × 50MB = 250MB | 50MB (deduplicated) | ~80% |
| node_modules across sessions | Per-session copy | Single copy | ~95% |

### Transfer Time

| Operation | tar.zst | CAS | Improvement |
|-----------|---------|-----|-------------|
| Checkpoint (minor change) | Full archive | Changed files only | ~10x faster |
| Resume (same workspace) | Extract full | No transfer | ~100x faster |
| Resume (small diff) | Extract full | Sync changed | ~10x faster |

---

## Integration with Existing Specs

### SPEC-03: Checkpoint/Resume

**Replace**:
```python
# Old: Full tar.zst snapshot
workspace_snapshot: str  # Path to workspace.tar.zst
```

**With**:
```python
# New: CAS root digest
workspace_root: Digest   # Merkle tree root
previous_root: Digest | None  # For incremental restore
```

### SPEC-07: Plugin Sync

**Replace**:
```python
# Old: Clone to volume directory
git clone {url} /vol/parhelia/plugins/{name}
```

**With**:
```python
# New: Store in CAS, materialize on demand
digest = await plugin_cas.sync_plugin(url, version)
await plugin_cas.materialize_plugin(name, version, target_dir)
```

---

## Acceptance Criteria

- [ ] [SPEC-08.AC1] Workspace stored as Merkle tree in CAS
- [ ] [SPEC-08.AC2] Incremental checkpoint uses <10% storage of full archive
- [ ] [SPEC-08.AC3] Incremental restore transfers only changed files
- [ ] [SPEC-08.AC4] Deterministic actions (build, test, lint) cached
- [ ] [SPEC-08.AC5] Plugins deduplicated across containers
- [ ] [SPEC-08.AC6] CAS storage auto-cleans via LRU policy

---

## Open Questions

1. **Remote CAS**: Should CAS be shared across Modal containers via object storage (S3/R2)?
2. **Compression**: Should blobs be compressed in CAS, or rely on filesystem compression?
3. **Garbage Collection**: LRU based on access time, or reference counting?

---

## References

- [Bazel Remote Execution API](https://github.com/bazelbuild/remote-apis)
- [REAPI Content Addressable Storage](https://github.com/bazelbuild/remote-apis/blob/main/build/bazel/remote/execution/v2/remote_execution.proto)
- [Using the REAPI for Distributed Builds](https://www.codethink.co.uk/articles/2019/using-the-reapi-for-distributed-builds/)
- SPEC-03: Checkpoint and Resume System
- SPEC-07: Plugin and MCP Synchronization
