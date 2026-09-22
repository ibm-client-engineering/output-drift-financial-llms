"""Store creation must not change the caller's existing directory permissions."""

from __future__ import annotations

import os
import stat

import pytest

from dfah import ArtifactError
from dfah.store import FileStore


def test_store_preserves_existing_directory_modes(tmp_path):
    root = tmp_path / "shared-output"
    root.mkdir()
    os.chmod(root, 0o755)
    starts = root / "starts"
    starts.mkdir()
    os.chmod(starts, 0o750)

    with FileStore(root) as store:
        assert stat.S_IMODE(root.stat().st_mode) == 0o755
        assert stat.S_IMODE(starts.stat().st_mode) == 0o750
        for directory in (
            store.dispatches,
            store.episodes,
            store.commits,
            store.stale_leases,
        ):
            assert stat.S_IMODE(directory.stat().st_mode) & 0o077 == 0
        assert stat.S_IMODE(store.lease_guard.stat().st_mode) == 0o600


def test_store_creates_new_private_root_and_managed_directories(tmp_path):
    with FileStore(tmp_path / "new-run") as store:
        for directory in (
            store.root,
            store.starts,
            store.dispatches,
            store.episodes,
            store.commits,
            store.stale_leases,
        ):
            assert stat.S_IMODE(directory.stat().st_mode) & 0o077 == 0
        assert stat.S_IMODE(store.lease_guard.stat().st_mode) == 0o600


def test_store_rejects_existing_managed_symlink_without_chmod_target(tmp_path):
    target = tmp_path / "shared-target"
    target.mkdir()
    os.chmod(target, 0o755)
    root = tmp_path / "run"
    root.mkdir()
    (root / "starts").symlink_to(target, target_is_directory=True)

    with pytest.raises(ArtifactError, match="artifact directory is unsafe"):
        FileStore(root)
    assert stat.S_IMODE(target.stat().st_mode) == 0o755
