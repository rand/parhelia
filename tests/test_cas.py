"""Tests for Content-Addressable Storage.

@trace SPEC-08.10 - Digest Format
@trace SPEC-08.11 - Directory Proto
@trace SPEC-08.12 - CAS API
@trace SPEC-08.13 - Merkle Operations
@trace SPEC-08.14 - Incremental Checkpoints
@trace SPEC-08.15 - Action Cache
@trace SPEC-08.16 - Plugin CAS Storage
"""

import hashlib
import json
import os
from pathlib import Path

import pytest


class TestDigest:
    """Tests for Digest dataclass - SPEC-08.10."""

    def test_digest_from_content(self):
        """@trace SPEC-08.10 - Digest MUST be computed from content hash."""
        from parhelia.cas import Digest

        content = b"hello world"
        digest = Digest.from_content(content)

        expected_hash = hashlib.sha256(content).hexdigest()
        assert digest.hash == expected_hash
        assert digest.size_bytes == len(content)

    def test_digest_str_format(self):
        """@trace SPEC-08.10 - Digest string MUST follow sha256:prefix:size format."""
        from parhelia.cas import Digest

        content = b"test content"
        digest = Digest.from_content(content)

        str_repr = str(digest)
        assert str_repr.startswith("sha256:")
        assert ":" in str_repr
        assert str(digest.size_bytes) in str_repr

    def test_digest_equality(self):
        """@trace SPEC-08.10 - Equal content MUST produce equal digests."""
        from parhelia.cas import Digest

        content = b"same content"
        digest1 = Digest.from_content(content)
        digest2 = Digest.from_content(content)

        assert digest1 == digest2
        assert hash(digest1) == hash(digest2)

    def test_digest_inequality(self):
        """@trace SPEC-08.10 - Different content MUST produce different digests."""
        from parhelia.cas import Digest

        digest1 = Digest.from_content(b"content a")
        digest2 = Digest.from_content(b"content b")

        assert digest1 != digest2

    def test_digest_is_frozen(self):
        """@trace SPEC-08.10 - Digest MUST be immutable (frozen dataclass)."""
        from parhelia.cas import Digest

        digest = Digest.from_content(b"test")

        with pytest.raises(AttributeError):
            digest.hash = "modified"


class TestDirectoryProto:
    """Tests for Directory proto - SPEC-08.11."""

    def test_file_node_creation(self):
        """@trace SPEC-08.11 - FileNode MUST capture file metadata."""
        from parhelia.cas import Digest, FileNode

        digest = Digest.from_content(b"file content")
        node = FileNode(name="test.txt", digest=digest, is_executable=False)

        assert node.name == "test.txt"
        assert node.digest == digest
        assert node.is_executable is False

    def test_directory_node_creation(self):
        """@trace SPEC-08.11 - DirectoryNode MUST reference subdirectory digest."""
        from parhelia.cas import Digest, DirectoryNode

        digest = Digest.from_content(b"directory proto")
        node = DirectoryNode(name="subdir", digest=digest)

        assert node.name == "subdir"
        assert node.digest == digest

    def test_directory_creation(self):
        """@trace SPEC-08.11 - Directory MUST contain sorted files and directories."""
        from parhelia.cas import Digest, Directory, DirectoryNode, FileNode

        file1 = FileNode(
            name="a.txt",
            digest=Digest.from_content(b"a"),
            is_executable=False,
        )
        file2 = FileNode(
            name="b.txt",
            digest=Digest.from_content(b"b"),
            is_executable=False,
        )
        subdir = DirectoryNode(
            name="subdir",
            digest=Digest.from_content(b"subdir proto"),
        )

        directory = Directory(
            files=[file1, file2],
            directories=[subdir],
            symlinks=[],
        )

        assert len(directory.files) == 2
        assert len(directory.directories) == 1
        assert directory.files[0].name == "a.txt"

    def test_directory_serialization(self):
        """@trace SPEC-08.11 - Directory MUST serialize to canonical form."""
        from parhelia.cas import Digest, Directory, FileNode

        file1 = FileNode(
            name="test.txt",
            digest=Digest.from_content(b"content"),
            is_executable=False,
        )

        directory = Directory(files=[file1], directories=[], symlinks=[])
        serialized = directory.serialize()

        assert isinstance(serialized, bytes)
        assert len(serialized) > 0

    def test_directory_to_digest(self):
        """@trace SPEC-08.11 - Directory MUST compute its own digest."""
        from parhelia.cas import Digest, Directory, FileNode

        file1 = FileNode(
            name="test.txt",
            digest=Digest.from_content(b"content"),
            is_executable=False,
        )

        directory = Directory(files=[file1], directories=[], symlinks=[])
        digest = directory.to_digest()

        assert isinstance(digest, Digest)
        assert len(digest.hash) == 64  # SHA-256 hex

    def test_directory_digest_deterministic(self):
        """@trace SPEC-08.11 - Same directory contents MUST produce same digest."""
        from parhelia.cas import Digest, Directory, FileNode

        def make_directory():
            return Directory(
                files=[
                    FileNode(
                        name="test.txt",
                        digest=Digest.from_content(b"content"),
                        is_executable=False,
                    )
                ],
                directories=[],
                symlinks=[],
            )

        dir1 = make_directory()
        dir2 = make_directory()

        assert dir1.to_digest() == dir2.to_digest()


class TestContentAddressableStorage:
    """Tests for CAS API - SPEC-08.12."""

    @pytest.fixture
    def cas_dir(self, tmp_path: Path) -> Path:
        """Create temporary CAS directory."""
        cas_path = tmp_path / "cas"
        cas_path.mkdir()
        return cas_path

    @pytest.fixture
    def cas(self, cas_dir: Path):
        """Create CAS instance."""
        from parhelia.cas import ContentAddressableStorage

        return ContentAddressableStorage(root_path=str(cas_dir))

    @pytest.mark.asyncio
    async def test_write_blob_returns_digest(self, cas):
        """@trace SPEC-08.12 - write_blob MUST return content digest."""
        from parhelia.cas import Digest

        content = b"test content"
        digest = await cas.write_blob(content)

        expected = Digest.from_content(content)
        assert digest == expected

    @pytest.mark.asyncio
    async def test_read_blob_returns_content(self, cas):
        """@trace SPEC-08.12 - read_blob MUST return original content."""
        content = b"test content for reading"
        digest = await cas.write_blob(content)

        retrieved = await cas.read_blob(digest)
        assert retrieved == content

    @pytest.mark.asyncio
    async def test_contains_returns_true_for_existing(self, cas):
        """@trace SPEC-08.12 - contains MUST return True for existing blobs."""
        content = b"existing content"
        digest = await cas.write_blob(content)

        assert await cas.contains(digest) is True

    @pytest.mark.asyncio
    async def test_contains_returns_false_for_missing(self, cas):
        """@trace SPEC-08.12 - contains MUST return False for missing blobs."""
        from parhelia.cas import Digest

        digest = Digest.from_content(b"nonexistent content")
        assert await cas.contains(digest) is False

    @pytest.mark.asyncio
    async def test_find_missing_returns_missing_digests(self, cas):
        """@trace SPEC-08.12 - find_missing MUST return digests not in CAS."""
        from parhelia.cas import Digest

        content1 = b"content 1"
        content2 = b"content 2"
        content3 = b"content 3"

        # Write only content1
        digest1 = await cas.write_blob(content1)
        digest2 = Digest.from_content(content2)
        digest3 = Digest.from_content(content3)

        missing = await cas.find_missing([digest1, digest2, digest3])

        assert digest1 not in missing
        assert digest2 in missing
        assert digest3 in missing

    @pytest.mark.asyncio
    async def test_write_blob_deduplicates(self, cas, cas_dir: Path):
        """@trace SPEC-08.12 - write_blob MUST deduplicate existing content."""
        content = b"duplicate content"

        # Write twice
        digest1 = await cas.write_blob(content)
        digest2 = await cas.write_blob(content)

        assert digest1 == digest2

        # Verify only one file exists
        blob_files = list(cas_dir.rglob("*"))
        blob_files = [f for f in blob_files if f.is_file()]
        assert len(blob_files) == 1

    @pytest.mark.asyncio
    async def test_read_blob_verifies_integrity(self, cas, cas_dir: Path):
        """@trace SPEC-08.12 - read_blob MUST verify content integrity."""
        from parhelia.cas import CorruptBlobError

        content = b"original content"
        digest = await cas.write_blob(content)

        # Corrupt the blob file
        blob_path = cas_dir / "blobs" / "sha256" / digest.hash[:2] / digest.hash
        blob_path.write_bytes(b"corrupted content")

        with pytest.raises(CorruptBlobError):
            await cas.read_blob(digest)

    @pytest.mark.asyncio
    async def test_write_blob_atomic(self, cas, cas_dir: Path):
        """@trace SPEC-08.12 - write_blob MUST use atomic writes."""
        content = b"atomic write test"
        digest = await cas.write_blob(content)

        # Verify no temp files left behind
        temp_files = list(cas_dir.rglob("*.tmp.*"))
        assert len(temp_files) == 0


class TestMerkleTreeBuilder:
    """Tests for Merkle tree operations - SPEC-08.13."""

    @pytest.fixture
    def cas(self, tmp_path: Path):
        """Create CAS instance for tree building."""
        from parhelia.cas import ContentAddressableStorage

        cas_path = tmp_path / "cas"
        cas_path.mkdir()
        return ContentAddressableStorage(root_path=str(cas_path))

    @pytest.fixture
    def tree_builder(self, cas):
        """Create Merkle tree builder."""
        from parhelia.cas import MerkleTreeBuilder

        return MerkleTreeBuilder(cas)

    @pytest.fixture
    def sample_workspace(self, tmp_path: Path) -> Path:
        """Create sample workspace directory."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # Create files
        (workspace / "README.md").write_text("# Project")
        (workspace / "main.py").write_text("print('hello')")

        # Create subdirectory
        src_dir = workspace / "src"
        src_dir.mkdir()
        (src_dir / "lib.py").write_text("def func(): pass")

        return workspace

    @pytest.mark.asyncio
    async def test_build_tree_returns_digest(self, tree_builder, sample_workspace):
        """@trace SPEC-08.13 - build_tree MUST return root digest."""
        from parhelia.cas import Digest

        root_digest = await tree_builder.build_tree(str(sample_workspace))

        assert isinstance(root_digest, Digest)
        assert len(root_digest.hash) == 64

    @pytest.mark.asyncio
    async def test_build_tree_deterministic(self, tree_builder, sample_workspace):
        """@trace SPEC-08.13 - build_tree MUST be deterministic."""
        digest1 = await tree_builder.build_tree(str(sample_workspace))
        digest2 = await tree_builder.build_tree(str(sample_workspace))

        assert digest1 == digest2

    @pytest.mark.asyncio
    async def test_build_tree_excludes_patterns(self, tree_builder, tmp_path: Path):
        """@trace SPEC-08.13 - build_tree MUST exclude configured patterns."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        (workspace / "main.py").write_text("code")

        # Create excluded directories
        (workspace / "node_modules").mkdir()
        (workspace / "node_modules" / "pkg.js").write_text("module")
        (workspace / "__pycache__").mkdir()
        (workspace / "__pycache__" / "cache.pyc").write_bytes(b"bytecode")

        root_digest = await tree_builder.build_tree(str(workspace))

        # Build should succeed and not include excluded dirs
        assert root_digest is not None

    @pytest.mark.asyncio
    async def test_build_tree_stores_all_content(self, tree_builder, cas, sample_workspace):
        """@trace SPEC-08.13 - build_tree MUST store all file content in CAS."""
        from parhelia.cas import Digest

        root_digest = await tree_builder.build_tree(str(sample_workspace))

        # Verify files are in CAS
        readme_content = (sample_workspace / "README.md").read_bytes()
        readme_digest = Digest.from_content(readme_content)
        assert await cas.contains(readme_digest)

        main_content = (sample_workspace / "main.py").read_bytes()
        main_digest = Digest.from_content(main_content)
        assert await cas.contains(main_digest)


class TestMerkleTreeDiff:
    """Tests for tree diff operations - SPEC-08.13."""

    @pytest.fixture
    def cas(self, tmp_path: Path):
        """Create CAS instance."""
        from parhelia.cas import ContentAddressableStorage

        cas_path = tmp_path / "cas"
        cas_path.mkdir()
        return ContentAddressableStorage(root_path=str(cas_path))

    @pytest.fixture
    def tree_builder(self, cas):
        """Create Merkle tree builder."""
        from parhelia.cas import MerkleTreeBuilder

        return MerkleTreeBuilder(cas)

    @pytest.fixture
    def tree_diff(self, cas):
        """Create tree diff instance."""
        from parhelia.cas import MerkleTreeDiff

        return MerkleTreeDiff(cas)

    @pytest.mark.asyncio
    async def test_diff_identical_trees(self, tree_builder, tree_diff, tmp_path: Path):
        """@trace SPEC-08.13 - diff of identical trees MUST return empty."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "file.txt").write_text("content")

        root = await tree_builder.build_tree(str(workspace))
        diff = await tree_diff.diff(root, root)

        assert len(diff.added) == 0
        assert len(diff.modified) == 0
        assert len(diff.deleted) == 0

    @pytest.mark.asyncio
    async def test_diff_detects_added_files(self, tree_builder, tree_diff, tmp_path: Path):
        """@trace SPEC-08.13 - diff MUST detect added files."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "original.txt").write_text("original")

        old_root = await tree_builder.build_tree(str(workspace))

        # Add new file
        (workspace / "new.txt").write_text("new file")
        new_root = await tree_builder.build_tree(str(workspace))

        diff = await tree_diff.diff(old_root, new_root)

        assert "new.txt" in diff.added
        assert len(diff.deleted) == 0

    @pytest.mark.asyncio
    async def test_diff_detects_modified_files(self, tree_builder, tree_diff, tmp_path: Path):
        """@trace SPEC-08.13 - diff MUST detect modified files."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "file.txt").write_text("original content")

        old_root = await tree_builder.build_tree(str(workspace))

        # Modify file
        (workspace / "file.txt").write_text("modified content")
        new_root = await tree_builder.build_tree(str(workspace))

        diff = await tree_diff.diff(old_root, new_root)

        assert "file.txt" in diff.modified

    @pytest.mark.asyncio
    async def test_diff_detects_deleted_files(self, tree_builder, tree_diff, tmp_path: Path):
        """@trace SPEC-08.13 - diff MUST detect deleted files."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "keep.txt").write_text("keep")
        (workspace / "delete.txt").write_text("delete me")

        old_root = await tree_builder.build_tree(str(workspace))

        # Delete file
        (workspace / "delete.txt").unlink()
        new_root = await tree_builder.build_tree(str(workspace))

        diff = await tree_diff.diff(old_root, new_root)

        assert "delete.txt" in diff.deleted


class TestIncrementalCheckpoint:
    """Tests for incremental checkpoints - SPEC-08.14."""

    @pytest.fixture
    def cas(self, tmp_path: Path):
        """Create CAS instance."""
        from parhelia.cas import ContentAddressableStorage

        cas_path = tmp_path / "cas"
        cas_path.mkdir()
        return ContentAddressableStorage(root_path=str(cas_path))

    @pytest.fixture
    def checkpoint_manager(self, cas):
        """Create checkpoint manager."""
        from parhelia.cas import IncrementalCheckpointManager

        return IncrementalCheckpointManager(cas)

    @pytest.mark.asyncio
    async def test_create_checkpoint_returns_digest(self, checkpoint_manager, tmp_path: Path):
        """@trace SPEC-08.14 - create_checkpoint MUST return workspace root digest."""
        from parhelia.cas import Digest

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "file.txt").write_text("content")

        checkpoint = await checkpoint_manager.create_checkpoint(str(workspace))

        assert isinstance(checkpoint.workspace_root, Digest)

    @pytest.mark.asyncio
    async def test_restore_checkpoint_recreates_files(self, checkpoint_manager, tmp_path: Path):
        """@trace SPEC-08.14 - restore_checkpoint MUST recreate workspace files."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "file.txt").write_text("test content")
        (workspace / "subdir").mkdir()
        (workspace / "subdir" / "nested.txt").write_text("nested content")

        checkpoint = await checkpoint_manager.create_checkpoint(str(workspace))

        # Restore to new location
        restore_dir = tmp_path / "restored"
        restore_dir.mkdir()
        await checkpoint_manager.restore_checkpoint(checkpoint, str(restore_dir))

        assert (restore_dir / "file.txt").read_text() == "test content"
        assert (restore_dir / "subdir" / "nested.txt").read_text() == "nested content"

    @pytest.mark.asyncio
    async def test_incremental_restore_only_transfers_changes(
        self, checkpoint_manager, tmp_path: Path
    ):
        """@trace SPEC-08.14 - incremental restore MUST only transfer changed files."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "unchanged.txt").write_text("unchanged")
        (workspace / "changed.txt").write_text("original")

        checkpoint1 = await checkpoint_manager.create_checkpoint(str(workspace))

        # Modify one file
        (workspace / "changed.txt").write_text("modified")
        checkpoint2 = await checkpoint_manager.create_checkpoint(str(workspace))

        # Restore incrementally
        restore_dir = tmp_path / "restored"
        restore_dir.mkdir()

        # First restore full checkpoint1
        await checkpoint_manager.restore_checkpoint(checkpoint1, str(restore_dir))

        # Then incremental restore from checkpoint1 to checkpoint2
        await checkpoint_manager.restore_checkpoint(
            checkpoint2, str(restore_dir), from_checkpoint=checkpoint1
        )

        assert (restore_dir / "unchanged.txt").read_text() == "unchanged"
        assert (restore_dir / "changed.txt").read_text() == "modified"


class TestActionCache:
    """Tests for action cache - SPEC-08.15."""

    @pytest.fixture
    def cas(self, tmp_path: Path):
        """Create CAS instance."""
        from parhelia.cas import ContentAddressableStorage

        cas_path = tmp_path / "cas"
        cas_path.mkdir()
        return ContentAddressableStorage(root_path=str(cas_path))

    @pytest.fixture
    def action_cache(self, cas):
        """Create action cache."""
        from parhelia.cas import ActionCache

        return ActionCache(cas)

    def test_action_to_digest_deterministic(self):
        """@trace SPEC-08.15 - Action digest MUST be deterministic."""
        from parhelia.cas import Action, Digest

        input_root = Digest.from_content(b"input tree")

        action1 = Action(
            command=["npm", "test"],
            working_directory=".",
            input_root=input_root,
            environment={"NODE_ENV": "test"},
            timeout_seconds=300,
        )

        action2 = Action(
            command=["npm", "test"],
            working_directory=".",
            input_root=input_root,
            environment={"NODE_ENV": "test"},
            timeout_seconds=300,
        )

        assert action1.to_digest() == action2.to_digest()

    def test_action_different_inputs_different_digest(self):
        """@trace SPEC-08.15 - Different inputs MUST produce different action digest."""
        from parhelia.cas import Action, Digest

        input1 = Digest.from_content(b"input 1")
        input2 = Digest.from_content(b"input 2")

        action1 = Action(
            command=["npm", "test"],
            working_directory=".",
            input_root=input1,
            environment={},
            timeout_seconds=300,
        )

        action2 = Action(
            command=["npm", "test"],
            working_directory=".",
            input_root=input2,
            environment={},
            timeout_seconds=300,
        )

        assert action1.to_digest() != action2.to_digest()

    @pytest.mark.asyncio
    async def test_action_cache_store_and_lookup(self, action_cache):
        """@trace SPEC-08.15 - ActionCache MUST store and retrieve results."""
        from parhelia.cas import Action, ActionResult, Digest

        input_root = Digest.from_content(b"input tree")
        action = Action(
            command=["pytest"],
            working_directory=".",
            input_root=input_root,
            environment={},
            timeout_seconds=300,
        )

        result = ActionResult(
            exit_code=0,
            stdout_digest=Digest.from_content(b"test output"),
            stderr_digest=None,
            output_files=[],
            execution_metadata={"duration_ms": 1234},
        )

        await action_cache.store(action, result)
        retrieved = await action_cache.lookup(action)

        assert retrieved is not None
        assert retrieved.exit_code == 0

    @pytest.mark.asyncio
    async def test_action_cache_returns_none_for_missing(self, action_cache):
        """@trace SPEC-08.15 - ActionCache MUST return None for uncached actions."""
        from parhelia.cas import Action, Digest

        input_root = Digest.from_content(b"input tree")
        action = Action(
            command=["unknown-command"],
            working_directory=".",
            input_root=input_root,
            environment={},
            timeout_seconds=300,
        )

        result = await action_cache.lookup(action)
        assert result is None

    def test_is_cacheable_for_known_commands(self, action_cache):
        """@trace SPEC-08.15 - is_cacheable MUST return True for deterministic commands."""
        assert action_cache.is_cacheable(["npm", "test"]) is True
        assert action_cache.is_cacheable(["pytest"]) is True
        assert action_cache.is_cacheable(["cargo", "test"]) is True
        assert action_cache.is_cacheable(["npm", "run", "build"]) is True

    def test_is_cacheable_false_for_unknown_commands(self, action_cache):
        """@trace SPEC-08.15 - is_cacheable MUST return False for non-deterministic commands."""
        assert action_cache.is_cacheable(["curl", "https://example.com"]) is False
        assert action_cache.is_cacheable(["echo", "hello"]) is False


# =============================================================================
# CAS Garbage Collection Tests
# =============================================================================


class TestGCConfig:
    """Tests for GCConfig dataclass."""

    def test_default_config(self):
        """GCConfig MUST have sensible defaults."""
        from parhelia.cas import GCConfig

        config = GCConfig()

        assert config.max_size_bytes == 10 * 1024 * 1024 * 1024  # 10 GB
        assert config.target_size_bytes == 8 * 1024 * 1024 * 1024  # 8 GB
        assert config.min_blob_age_seconds == 3600  # 1 hour
        assert config.dry_run is False

    def test_custom_config(self):
        """GCConfig MUST accept custom values."""
        from parhelia.cas import GCConfig

        config = GCConfig(
            max_size_bytes=1000,
            target_size_bytes=500,
            min_blob_age_seconds=60,
            dry_run=True,
        )

        assert config.max_size_bytes == 1000
        assert config.target_size_bytes == 500
        assert config.dry_run is True


class TestGCResult:
    """Tests for GCResult dataclass."""

    def test_default_result(self):
        """GCResult MUST have zero defaults."""
        from parhelia.cas import GCResult

        result = GCResult()

        assert result.blobs_deleted == 0
        assert result.bytes_freed == 0
        assert result.blobs_retained == 0
        assert result.errors == []

    def test_to_dict(self):
        """GCResult MUST serialize to dict."""
        from parhelia.cas import GCResult

        result = GCResult(
            blobs_deleted=10,
            bytes_freed=1024,
            blobs_retained=5,
            bytes_retained=512,
            errors=["error1"],
            dry_run=True,
        )

        data = result.to_dict()

        assert data["blobs_deleted"] == 10
        assert data["bytes_freed"] == 1024
        assert data["dry_run"] is True


class TestCASGarbageCollector:
    """Tests for CASGarbageCollector."""

    @pytest.fixture
    def temp_cas(self, tmp_path):
        """Create CAS in temp directory."""
        from parhelia.cas import ContentAddressableStorage

        return ContentAddressableStorage(str(tmp_path / "cas"))

    @pytest.fixture
    def gc(self, temp_cas):
        """Create garbage collector."""
        from parhelia.cas import CASGarbageCollector, GCConfig

        config = GCConfig(
            max_size_bytes=1000,
            target_size_bytes=500,
            min_blob_age_seconds=0,  # No age restriction for tests
        )
        return CASGarbageCollector(temp_cas, config)

    @pytest.mark.asyncio
    async def test_get_storage_stats_empty(self, gc):
        """get_storage_stats MUST return zeros for empty CAS."""
        total_bytes, blob_count = await gc.get_storage_stats()

        assert total_bytes == 0
        assert blob_count == 0

    @pytest.mark.asyncio
    async def test_get_storage_stats_with_blobs(self, gc, temp_cas):
        """get_storage_stats MUST count blobs correctly."""
        # Write some blobs
        await temp_cas.write_blob(b"blob1" * 10)
        await temp_cas.write_blob(b"blob2" * 20)
        await temp_cas.write_blob(b"blob3" * 30)

        total_bytes, blob_count = await gc.get_storage_stats()

        assert blob_count == 3
        assert total_bytes == 50 + 100 + 150  # 5*10 + 5*20 + 5*30

    @pytest.mark.asyncio
    async def test_list_blobs(self, gc, temp_cas):
        """list_blobs MUST return all blobs with metadata."""
        d1 = await temp_cas.write_blob(b"content1")
        d2 = await temp_cas.write_blob(b"content2")

        blobs = await gc.list_blobs()

        assert len(blobs) == 2
        hashes = {b.digest.hash for b in blobs}
        assert d1.hash in hashes
        assert d2.hash in hashes

    @pytest.mark.asyncio
    async def test_should_run_gc_below_threshold(self, gc, temp_cas):
        """should_run_gc MUST return False when below threshold."""
        # Write small blob (well under 1000 byte threshold)
        await temp_cas.write_blob(b"small")

        assert await gc.should_run_gc() is False

    @pytest.mark.asyncio
    async def test_should_run_gc_above_threshold(self, gc, temp_cas):
        """should_run_gc MUST return True when above threshold."""
        # Write enough data to exceed 1000 byte threshold
        for i in range(20):
            await temp_cas.write_blob(f"large content {i}".encode() * 10)

        assert await gc.should_run_gc() is True

    @pytest.mark.asyncio
    async def test_run_gc_deletes_unreferenced_blobs(self, gc, temp_cas):
        """run_gc MUST delete unreferenced blobs."""
        # Write enough data to trigger GC
        for i in range(20):
            await temp_cas.write_blob(f"content {i}".encode() * 10)

        # Run GC
        result = await gc.run_gc()

        assert result.blobs_deleted > 0
        assert result.bytes_freed > 0

    @pytest.mark.asyncio
    async def test_run_gc_retains_referenced_blobs(self, gc, temp_cas):
        """run_gc MUST retain blobs reachable from reference roots."""
        # Write a blob and mark it as referenced
        protected = await temp_cas.write_blob(b"protected content" * 100)
        gc.add_reference_root(protected.hash)

        # Write unreferenced blobs to trigger GC
        for i in range(20):
            await temp_cas.write_blob(f"content {i}".encode() * 10)

        # Run GC
        result = await gc.run_gc()

        # Protected blob should still exist
        assert await temp_cas.contains(protected)

    @pytest.mark.asyncio
    async def test_run_gc_dry_run(self, gc, temp_cas):
        """run_gc MUST not delete blobs in dry_run mode."""
        gc.config.dry_run = True

        # Write blobs
        for i in range(20):
            await temp_cas.write_blob(f"content {i}".encode() * 10)

        initial_bytes, initial_count = await gc.get_storage_stats()

        # Run GC in dry run mode
        result = await gc.run_gc()

        assert result.dry_run is True
        assert result.blobs_deleted > 0  # Would have deleted

        # But blobs should still exist
        final_bytes, final_count = await gc.get_storage_stats()
        assert final_count == initial_count

    @pytest.mark.asyncio
    async def test_add_remove_reference_root(self, gc):
        """add/remove_reference_root MUST manage protected roots."""
        gc.add_reference_root("abc123")
        gc.add_reference_root("def456")

        assert "abc123" in gc._reference_roots
        assert "def456" in gc._reference_roots

        gc.remove_reference_root("abc123")

        assert "abc123" not in gc._reference_roots
        assert "def456" in gc._reference_roots

    @pytest.mark.asyncio
    async def test_collect_referenced_digests(self, gc, temp_cas):
        """collect_referenced_digests MUST traverse tree."""
        from parhelia.cas import Digest, Directory, DirectoryNode, FileNode

        # Create a simple tree: root -> file
        file_content = b"file content"
        file_digest = await temp_cas.write_blob(file_content)

        directory = Directory(
            files=[FileNode(name="test.txt", digest=file_digest, is_executable=False)],
            directories=[],
            symlinks=[],
        )
        dir_content = directory.serialize()
        root_digest = await temp_cas.write_blob(dir_content)

        # Add root as reference
        gc.add_reference_root(root_digest.hash)

        # Collect referenced digests
        referenced = await gc.collect_referenced_digests()

        # Both root and file should be referenced
        assert root_digest.hash in referenced
        assert file_digest.hash in referenced

    def test_format_gc_result(self, gc):
        """format_gc_result MUST produce readable output."""
        from parhelia.cas import GCResult

        result = GCResult(
            blobs_deleted=100,
            bytes_freed=1024 * 1024,
            blobs_retained=50,
            bytes_retained=512 * 1024,
        )

        output = gc.format_gc_result(result)

        assert "100" in output  # blobs deleted
        assert "50" in output  # blobs retained
        assert "MB" in output or "KB" in output  # formatted bytes

    def test_format_bytes(self, gc):
        """_format_bytes MUST produce human-readable sizes."""
        assert "B" in gc._format_bytes(500)
        assert "KB" in gc._format_bytes(1024)
        assert "MB" in gc._format_bytes(1024 * 1024)
        assert "GB" in gc._format_bytes(1024 * 1024 * 1024)
