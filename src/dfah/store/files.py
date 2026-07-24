"""Private-by-default, append-only filesystem episode store."""

from __future__ import annotations

import hashlib
import os
import random
import socket
import stat
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .._canonical import canonical_bytes, sha256
from .._frozen import FrozenJsonMap
from ..exceptions import ArtifactError, EpisodeConflictError, IncompleteEpisodeError
from ..models import Episode, Record


class EpisodeStart(Record):
    """Durable episode plan written before budget reservation or dispatch."""

    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    suite_id: str
    suite_version: str
    case_id: str
    task: str
    replay_index: int = Field(ge=0)
    started_at: datetime
    idempotency_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    event: Literal["episode_planned"] = "episode_planned"

    @property
    def key(self) -> tuple[str, str, int]:
        """Shared durable episode key."""

        return self.manifest_hash, self.case_id, self.replay_index


class DispatchIntent(Record):
    """Durable boundary proving an external call may have been dispatched."""

    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_id: str
    replay_index: int = Field(ge=0)
    marked_at: datetime
    reserved_cost_usd: float = Field(default=0.0, ge=0.0)
    event: Literal["dispatching"] = "dispatching"

    @property
    def key(self) -> tuple[str, str, int]:
        """Shared durable episode key."""

        return self.manifest_hash, self.case_id, self.replay_index


class RunPlan(Record):
    """Immutable replay design bound to one artifact directory."""

    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    suite_id: str
    suite_version: str
    fixture_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    tool_schema_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    replays: int = Field(ge=2)
    seed: int
    sample_rate: float = Field(gt=0.0, le=1.0)
    episode_timeout_s: float | None = Field(default=None, gt=0.0)
    case_tasks: FrozenJsonMap
    schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    suite_cases_total: int = Field(gt=0)
    episodes_planned: int = Field(gt=0)
    event: Literal["run_plan"] = "run_plan"

    @model_validator(mode="after")
    def _coherent_schedule(self) -> RunPlan:
        if not self.case_tasks or any(
            not isinstance(case_id, str) or not isinstance(task, str) or not task
            for case_id, task in self.case_tasks.items()
        ):
            raise ValueError("run plan case_tasks must map case IDs to task names")
        if len(self.case_tasks) * self.replays != self.episodes_planned:
            raise ValueError("run plan episode count differs from cases times replays")
        if len(self.case_tasks) > self.suite_cases_total:
            raise ValueError("run plan selects more cases than the suite contains")
        schedule = [
            {
                "case_id": case_id,
                "task": task,
                "replay_index": replay_index,
            }
            for case_id, task in sorted(self.case_tasks.items())
            for replay_index in range(self.replays)
        ]
        random.Random(self.seed).shuffle(schedule)
        if sha256(tuple(schedule)) != self.schedule_sha256:
            raise ValueError("run plan schedule hash differs from its declared design")
        return self

    @property
    def hash(self) -> str:
        """Canonical run-plan commitment."""

        return sha256(self)


class RunLease(Record):
    """Exclusive writer identity retained if a process ends unexpectedly."""

    token: str = Field(pattern=r"^[0-9a-f]{32}$")
    pid: int = Field(gt=0)
    hostname: str = Field(min_length=1)
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    acquired_at: datetime
    event: Literal["run_lease"] = "run_lease"


def _key_digest(manifest_hash: str, case_id: str, replay_index: int) -> str:
    return sha256(
        {"manifest_hash": manifest_hash, "case_id": case_id, "replay_index": replay_index}
    )


class FileStore:
    """Append-only local artifact store with hash sidecars and commit markers.

    A terminal episode is visible only after its JSON, SHA-256 sidecar, and
    zero-content commit marker have all been durably written. Existing content
    is never overwritten.
    """

    def __init__(self, root: str | Path, *, create: bool = True):
        requested = Path(os.path.abspath(Path(root).expanduser()))
        if requested.is_symlink():
            raise ArtifactError(f"artifact run root is a symlink: {requested}")
        # Canonicalize standard ancestor aliases such as macOS /tmp -> /private/tmp.
        # The explicitly requested run root and every managed descendant must still
        # be real directories, not links.
        self.root = requested.parent.resolve(strict=False) / requested.name
        self.plan = self.root / "run-plan.json"
        self.lease = self.root / "writer.lock"
        self.lease_guard = self.root / "writer.guard"
        self.stale_leases = self.root / "stale-leases"
        self.starts = self.root / "starts"
        self.dispatches = self.root / "dispatches"
        self.episodes = self.root / "episodes"
        self.commits = self.root / "commits"
        for directory in (
            self.root,
            self.starts,
            self.dispatches,
            self.episodes,
            self.commits,
            self.stale_leases,
        ):
            if create:
                directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            if directory.is_symlink() or not directory.is_dir():
                raise ArtifactError(f"artifact directory is unsafe: {directory}")
            if create:
                os.chmod(directory, 0o700)
        directory_flags = (
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        )
        self._directory_fds = {
            directory: os.open(directory, directory_flags)
            for directory in (
                self.root,
                self.starts,
                self.dispatches,
                self.episodes,
                self.commits,
                self.stale_leases,
            )
        }
        self._lease_guard_fd: int | None = None
        if create:
            with suppress(FileExistsError):
                self._exclusive_write(self.lease_guard, b"")
        if self.lease_guard.is_symlink():
            raise ArtifactError("run writer guard path is unsafe")
        if self.lease_guard.exists():
            self._lease_guard_fd = self._open_pinned_guard()

    def close(self) -> None:
        """Release pinned directory descriptors; safe to call more than once."""

        guard = getattr(self, "_lease_guard_fd", None)
        if guard is not None:
            with suppress(OSError):
                os.close(guard)
            self._lease_guard_fd = None
        descriptors = getattr(self, "_directory_fds", {})
        for descriptor in set(descriptors.values()):
            with suppress(OSError):
                os.close(descriptor)
        descriptors.clear()

    def __enter__(self) -> FileStore:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def _directory_fd(self, path: Path) -> int:
        try:
            descriptor = self._directory_fds[path]
        except KeyError as exc:
            raise ArtifactError(f"artifact path is outside the pinned store: {path}") from exc
        try:
            named = os.stat(path, follow_symlinks=False)
            opened = os.fstat(descriptor)
        except OSError as exc:
            raise ArtifactError(f"artifact directory is no longer reachable: {path}") from exc
        if not stat.S_ISDIR(named.st_mode) or (named.st_dev, named.st_ino) != (
            opened.st_dev,
            opened.st_ino,
        ):
            raise ArtifactError(f"artifact directory identity changed: {path}")
        return descriptor

    def _open_pinned_guard(self) -> int:
        """Open and bind the persistent writer guard to its current inode."""

        root_directory = self._directory_fd(self.root)
        try:
            named = os.stat(
                self.lease_guard.name,
                dir_fd=root_directory,
                follow_symlinks=False,
            )
            if not stat.S_ISREG(named.st_mode):
                raise ArtifactError("run writer guard path is unsafe")
            descriptor = os.open(
                self.lease_guard.name,
                os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=root_directory,
            )
            opened = os.fstat(descriptor)
            if (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino):
                os.close(descriptor)
                raise ArtifactError("run writer guard changed while opening")
            return descriptor
        except ArtifactError:
            raise
        except OSError as exc:
            raise ArtifactError("run writer guard is unreadable") from exc

    def _open_verified_guard(self) -> int:
        """Open a distinct lock descriptor after checking the pinned identity."""

        pinned_descriptor = self._lease_guard_fd
        if pinned_descriptor is None:
            raise ArtifactError("run writer guard is unavailable")
        root_directory = self._directory_fd(self.root)
        try:
            pinned = os.fstat(pinned_descriptor)
            descriptor = os.open(
                self.lease_guard.name,
                os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=root_directory,
            )
            opened = os.fstat(descriptor)
            if (pinned.st_dev, pinned.st_ino) != (opened.st_dev, opened.st_ino):
                os.close(descriptor)
                raise ArtifactError("run writer guard identity changed")
            return descriptor
        except ArtifactError:
            raise
        except OSError as exc:
            raise ArtifactError("run writer guard is unreadable") from exc

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _read_regular_bytes(self, path: Path) -> bytes:
        """Read one regular artifact without following a final symlink."""

        directory = self._directory_fd(path.parent)
        try:
            metadata = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode):
                raise ArtifactError(f"artifact file is unsafe: {path.name}")
            descriptor = os.open(
                path.name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory,
            )
            try:
                opened = os.fstat(descriptor)
                if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                    raise ArtifactError(f"artifact file changed while opening: {path.name}")
                with os.fdopen(descriptor, "rb", closefd=False) as handle:
                    return handle.read()
            finally:
                os.close(descriptor)
        except ArtifactError:
            raise
        except OSError as exc:
            raise ArtifactError(f"artifact file is unreadable: {path.name}") from exc

    def _read_regular_text(self, path: Path, *, encoding: str) -> str:
        return self._read_regular_bytes(path).decode(encoding)

    def read_lease(self) -> RunLease:
        """Read the current single-writer lease without changing it."""

        try:
            return RunLease.model_validate_json(self._read_regular_bytes(self.lease))
        except (OSError, ValueError) as exc:
            raise ArtifactError("run writer lease is unreadable") from exc

    def _archive_current_lease(self, name: str, expected_payload: bytes) -> Path:
        """Preserve stale metadata using atomic no-overwrite hard-link semantics."""

        archive = self.stale_leases / name
        root_directory = self._directory_fd(self.root)
        archive_directory = self._directory_fd(self.stale_leases)
        try:
            source = os.open(
                self.lease.name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=root_directory,
            )
            try:
                source_stat = os.fstat(source)
                with os.fdopen(source, "rb", closefd=False) as handle:
                    observed_payload = handle.read()
            finally:
                os.close(source)
            if observed_payload != expected_payload:
                raise EpisodeConflictError(
                    "writer lease changed during stale recovery; retry after inspection"
                )
            os.link(
                self.lease.name,
                archive.name,
                src_dir_fd=root_directory,
                dst_dir_fd=archive_directory,
                follow_symlinks=False,
            )
            archived_stat = os.stat(
                archive.name, dir_fd=archive_directory, follow_symlinks=False
            )
            current_stat = os.stat(
                self.lease.name, dir_fd=root_directory, follow_symlinks=False
            )
            identity = (source_stat.st_dev, source_stat.st_ino)
            if (
                archived_stat.st_dev,
                archived_stat.st_ino,
            ) != identity or (current_stat.st_dev, current_stat.st_ino) != identity:
                os.unlink(archive.name, dir_fd=archive_directory)
                raise EpisodeConflictError(
                    "writer lease changed during stale recovery; retry after inspection"
                )
            os.unlink(self.lease.name, dir_fd=root_directory)
            os.fsync(root_directory)
            os.fsync(archive_directory)
        except FileExistsError as exc:
            raise EpisodeConflictError("stale-lease archive already exists") from exc
        except FileNotFoundError as exc:
            raise EpisodeConflictError(
                "writer lease changed during stale recovery; retry after inspection"
            ) from exc
        return archive

    def _remove_lease_if_token(self, token: str) -> None:
        """Remove only the exact metadata lease owned by this invocation."""

        root_directory = self._directory_fd(self.root)
        try:
            try:
                descriptor = os.open(
                    self.lease.name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=root_directory,
                )
            except FileNotFoundError:
                return
            try:
                opened = os.fstat(descriptor)
                with os.fdopen(descriptor, "rb", closefd=False) as handle:
                    payload = handle.read()
            finally:
                os.close(descriptor)
            try:
                current = RunLease.model_validate_json(payload)
            except ValueError:
                return
            named = os.stat(self.lease.name, dir_fd=root_directory, follow_symlinks=False)
            if current.token != token or (named.st_dev, named.st_ino) != (
                opened.st_dev,
                opened.st_ino,
            ):
                return
            os.unlink(self.lease.name, dir_fd=root_directory)
            os.fsync(root_directory)
        finally:
            pass

    @contextmanager
    def _metadata_lease(
        self,
        *,
        manifest_hash: str,
        recover_stale: bool = False,
    ) -> Iterator[RunLease]:
        """Maintain reviewable writer metadata while the OS guard is held.

        A lock left by a dead local process is preserved under ``stale-leases``
        only when the caller explicitly requests recovery. A live or remote-host
        lease is never displaced automatically.
        """

        record = RunLease(
            token=uuid.uuid4().hex,
            pid=os.getpid(),
            hostname=socket.gethostname(),
            manifest_hash=manifest_hash,
            acquired_at=datetime.now(timezone.utc),
        )
        payload = canonical_bytes(record, redact=True) + b"\n"
        if self.lease.is_symlink():
            raise ArtifactError("run writer lease path is unsafe")
        try:
            self._exclusive_write(self.lease, payload)
        except FileExistsError:
            if not recover_stale:
                raise EpisodeConflictError(
                    "artifact directory already has a writer lease; use explicit stale-lease "
                    "recovery only after confirming no runner is active"
                ) from None
            stale_payload = self._read_regular_bytes(self.lease)
            try:
                existing = RunLease.model_validate_json(stale_payload)
            except ValueError:
                self._archive_current_lease(f"corrupt-{uuid.uuid4().hex}.bin", stale_payload)
            else:
                if existing.hostname != record.hostname:
                    raise EpisodeConflictError(
                        "writer lease belongs to another host and cannot be recovered safely"
                    ) from None
                if self._pid_is_alive(existing.pid):
                    raise EpisodeConflictError(
                        "writer lease belongs to a live local process and cannot be recovered"
                    ) from None
                self._archive_current_lease(
                    f"{existing.acquired_at:%Y%m%dT%H%M%S}-{existing.token}-"
                    f"{uuid.uuid4().hex}.json",
                    stale_payload,
                )
            try:
                self._exclusive_write(self.lease, payload)
            except FileExistsError as exc:
                raise EpisodeConflictError(
                    "another writer acquired the run directory during stale recovery"
                ) from exc
        try:
            yield record
        finally:
            self._remove_lease_if_token(record.token)

    @contextmanager
    def run_lease(
        self,
        *,
        manifest_hash: str,
        recover_stale: bool = False,
    ) -> Iterator[RunLease]:
        """Hold one kernel-enforced writer lease for a replay invocation.

        The persistent guard inode is protected with a non-blocking POSIX
        advisory lock. The kernel releases it if the process ends, while the
        separate metadata lease remains available for explicit stale recovery.
        """

        try:
            import fcntl
        except ImportError as exc:
            raise ArtifactError(
                "the local FileStore writer lease requires POSIX advisory locks"
            ) from exc
        descriptor = self._open_verified_guard()
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise EpisodeConflictError(
                    "artifact directory already has an active writer process"
                ) from exc
            with self._metadata_lease(
                manifest_hash=manifest_hash,
                recover_stale=recover_stale,
            ) as record:
                yield record
        finally:
            with suppress(OSError):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def bind_plan(self, plan: RunPlan) -> RunPlan:
        """Write one immutable run design or verify its byte-identical predecessor."""

        payload = canonical_bytes(plan, redact=True) + b"\n"
        try:
            self._exclusive_write(self.plan, payload)
        except FileExistsError:
            existing = self.read_plan()
            if existing != plan:
                raise EpisodeConflictError(
                    "artifact directory is already bound to a different replay design"
                ) from None
        return plan

    def read_plan(self) -> RunPlan:
        """Read the immutable design for this artifact directory."""

        try:
            return RunPlan.model_validate_json(self._read_regular_bytes(self.plan))
        except (OSError, ValueError) as exc:
            raise ArtifactError("run plan is missing or unreadable") from exc

    def assert_schedule_inventory(
        self,
        *,
        manifest_hash: str,
        suite_id: str,
        suite_version: str,
        expected: Mapping[tuple[str, int], str],
    ) -> None:
        """Reject foreign or outside-design durable attempts before new dispatch."""

        for path in sorted(self.starts.glob("*.json")):
            try:
                start = EpisodeStart.model_validate_json(self._read_regular_bytes(path))
            except (OSError, ValueError) as exc:
                raise ArtifactError("episode-start inventory is unreadable") from exc
            if start.manifest_hash != manifest_hash:
                raise ArtifactError("artifact directory mixes multiple manifests")
            identity = (start.case_id, start.replay_index)
            if identity not in expected:
                raise EpisodeConflictError(
                    "artifact directory contains an attempt outside the bound replay design"
                )
            if (
                start.suite_id,
                start.suite_version,
                start.task,
            ) != (suite_id, suite_version, expected[identity]):
                raise EpisodeConflictError(
                    "episode-start metadata differs from the bound replay design"
                )
        for path in sorted(self.dispatches.glob("*.json")):
            try:
                dispatch = DispatchIntent.model_validate_json(self._read_regular_bytes(path))
            except (OSError, ValueError) as exc:
                raise ArtifactError("dispatch inventory is unreadable") from exc
            if dispatch.manifest_hash != manifest_hash:
                raise ArtifactError("artifact directory mixes multiple manifests")
            if (dispatch.case_id, dispatch.replay_index) not in expected:
                raise EpisodeConflictError(
                    "artifact directory contains a dispatch outside the bound replay design"
                )

    def _paths(self, manifest_hash: str, case_id: str, replay_index: int) -> tuple[Path, ...]:
        digest = _key_digest(manifest_hash, case_id, replay_index)
        return (
            self.starts / f"{digest}.json",
            self.dispatches / f"{digest}.json",
            self.episodes / f"{digest}.json",
            self.episodes / f"{digest}.json.sha256",
            self.commits / f"{digest}.commit",
        )

    def _fsync_directory(self, path: Path) -> None:
        os.fsync(self._directory_fd(path))

    def _exclusive_write(self, path: Path, payload: bytes) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        directory = self._directory_fd(path.parent)
        try:
            descriptor = os.open(path.name, flags, 0o600, dir_fd=directory)
            try:
                with os.fdopen(descriptor, "wb", closefd=False) as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                os.close(descriptor)
            os.fsync(directory)
        finally:
            pass

    def start(self, record: EpisodeStart) -> EpisodeStart:
        """Claim one episode key or return its byte-identical prior start."""

        start_path, _, _, _, commit_path = self._paths(*record.key)
        if commit_path.exists():
            return self.read_start(*record.key)
        payload = canonical_bytes(record, redact=True) + b"\n"
        try:
            self._exclusive_write(start_path, payload)
        except FileExistsError:
            existing = self.read_start(*record.key)
            if existing != record:
                raise EpisodeConflictError("episode key has a different start record") from None
        return record

    def mark_dispatching(self, record: DispatchIntent) -> DispatchIntent:
        """Persist the no-automatic-retry boundary immediately before dispatch."""

        start_path, dispatch_path, _, _, commit_path = self._paths(*record.key)
        if not start_path.exists():
            raise ArtifactError("dispatch intent has no durable episode plan")
        if commit_path.exists():
            raise EpisodeConflictError("cannot dispatch a committed episode")
        payload = canonical_bytes(record, redact=True) + b"\n"
        try:
            self._exclusive_write(dispatch_path, payload)
        except FileExistsError:
            try:
                existing = DispatchIntent.model_validate_json(
                    self._read_regular_bytes(dispatch_path)
                )
            except (OSError, ValueError) as exc:
                raise ArtifactError("dispatch intent is unreadable") from exc
            if existing != record:
                raise EpisodeConflictError(
                    "episode key has a different dispatch intent"
                ) from None
        return record

    def commit(self, episode: Episode) -> Episode:
        """Commit one terminal episode exactly once."""

        start_path, dispatch_path, episode_path, sidecar_path, commit_path = self._paths(
            *episode.key
        )
        if not start_path.exists():
            raise ArtifactError("terminal episode has no durable start")
        if not dispatch_path.exists():
            raise ArtifactError("terminal episode has no durable dispatch intent")
        payload = canonical_bytes(episode, redact=True) + b"\n"
        digest = hashlib.sha256(payload).hexdigest()
        if commit_path.exists():
            existing = self.read(*episode.key)
            if existing != episode:
                raise EpisodeConflictError("episode key has different committed content")
            return existing
        try:
            self._exclusive_write(episode_path, payload)
        except FileExistsError:
            if self._read_regular_bytes(episode_path) != payload:
                raise EpisodeConflictError("uncommitted episode content conflicts") from None
        try:
            self._exclusive_write(sidecar_path, f"{digest}\n".encode("ascii"))
        except FileExistsError:
            if self._read_regular_text(sidecar_path, encoding="ascii").strip() != digest:
                raise EpisodeConflictError("episode sidecar conflicts") from None
        with suppress(FileExistsError):
            self._exclusive_write(commit_path, b"")
        return self.read(*episode.key)

    def inspect(
        self, manifest_hash: str, case_id: str, replay_index: int
    ) -> Literal["absent", "started", "dispatching", "committed", "corrupt"]:
        """Return a durable state without mutating it."""

        start, dispatch, episode, sidecar, commit = self._paths(
            manifest_hash, case_id, replay_index
        )
        exists = (
            start.exists(),
            dispatch.exists(),
            episode.exists(),
            sidecar.exists(),
            commit.exists(),
        )
        if not any(exists):
            return "absent"
        if exists == (True, False, False, False, False):
            return "started"
        if exists == (True, True, False, False, False):
            return "dispatching"
        if exists[0] and exists[2] and exists[3] and exists[4]:
            try:
                self.read(manifest_hash, case_id, replay_index)
            except ArtifactError:
                return "corrupt"
            return "committed"
        return "corrupt"

    def read_start(self, manifest_hash: str, case_id: str, replay_index: int) -> EpisodeStart:
        """Read a durable start record."""

        start, _, _, _, _ = self._paths(manifest_hash, case_id, replay_index)
        try:
            return EpisodeStart.model_validate_json(self._read_regular_bytes(start))
        except (OSError, ValueError) as exc:
            raise ArtifactError("episode start is unreadable") from exc

    def read_dispatch(
        self, manifest_hash: str, case_id: str, replay_index: int
    ) -> DispatchIntent:
        """Read the durable post-admission dispatch boundary."""

        _, dispatch, _, _, _ = self._paths(manifest_hash, case_id, replay_index)
        try:
            return DispatchIntent.model_validate_json(self._read_regular_bytes(dispatch))
        except (OSError, ValueError) as exc:
            raise ArtifactError("dispatch intent is unreadable") from exc

    def read(self, manifest_hash: str, case_id: str, replay_index: int) -> Episode:
        """Verify and read one committed episode."""

        _, _, episode_path, sidecar_path, commit_path = self._paths(
            manifest_hash, case_id, replay_index
        )
        if not commit_path.exists():
            raise IncompleteEpisodeError("episode is not committed")
        try:
            if not stat.S_ISREG(os.lstat(commit_path).st_mode):
                raise ArtifactError("episode commit marker is unsafe")
            payload = self._read_regular_bytes(episode_path)
            expected = self._read_regular_text(sidecar_path, encoding="ascii").strip()
        except OSError as exc:
            raise ArtifactError("episode artifact is incomplete") from exc
        observed = hashlib.sha256(payload).hexdigest()
        if expected != observed:
            raise ArtifactError("episode SHA-256 mismatch")
        try:
            episode = Episode.model_validate_json(payload)
        except ValueError as exc:
            raise ArtifactError("episode schema validation failed") from exc
        if episode.key != (manifest_hash, case_id, replay_index):
            raise ArtifactError("episode content does not match its durable key")
        return episode

    def recover_partial(
        self, manifest_hash: str, case_id: str, replay_index: int
    ) -> Episode | None:
        """Finish a crash-interrupted commit when terminal JSON is already valid.

        Recovery never invents an episode. It only adds a missing hash sidecar
        and/or commit marker after validating the exact terminal JSON and key.
        """

        start_path, dispatch_path, episode_path, sidecar_path, commit_path = self._paths(
            manifest_hash, case_id, replay_index
        )
        if commit_path.exists():
            return self.read(manifest_hash, case_id, replay_index)
        if not start_path.exists() or not dispatch_path.exists() or not episode_path.exists():
            return None
        try:
            payload = self._read_regular_bytes(episode_path)
            episode = Episode.model_validate_json(payload)
        except (OSError, ValueError) as exc:
            raise ArtifactError("partial terminal episode is unreadable") from exc
        if episode.key != (manifest_hash, case_id, replay_index):
            raise ArtifactError("partial terminal episode key mismatch")
        digest = hashlib.sha256(payload).hexdigest()
        if sidecar_path.exists():
            if self._read_regular_text(sidecar_path, encoding="ascii").strip() != digest:
                raise ArtifactError("partial terminal episode SHA-256 mismatch")
        else:
            self._exclusive_write(sidecar_path, f"{digest}\n".encode("ascii"))
        with suppress(FileExistsError):
            self._exclusive_write(commit_path, b"")
        return self.read(manifest_hash, case_id, replay_index)

    def iter_episodes(self, *, manifest_hash: str | None = None) -> Iterator[Episode]:
        """Yield verified committed episodes in stable artifact order."""

        for commit in sorted(self.commits.glob("*.commit")):
            digest = commit.stem
            episode_path = self.episodes / f"{digest}.json"
            sidecar_path = self.episodes / f"{digest}.json.sha256"
            try:
                if not stat.S_ISREG(os.lstat(commit).st_mode):
                    raise ArtifactError(f"committed episode {digest} marker is unsafe")
                payload = self._read_regular_bytes(episode_path)
                expected = self._read_regular_text(sidecar_path, encoding="ascii").strip()
            except OSError as exc:
                raise ArtifactError(f"committed episode {digest} is incomplete") from exc
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ArtifactError(f"committed episode {digest} failed SHA-256")
            record = Episode.model_validate_json(payload)
            if manifest_hash is None or record.manifest_hash == manifest_hash:
                yield record

    def list(self, *, manifest_hash: str | None = None) -> tuple[Episode, ...]:
        """Read every committed episode, optionally filtering by manifest."""

        return tuple(
            sorted(
                self.iter_episodes(manifest_hash=manifest_hash),
                key=lambda item: (item.case_id, item.replay_index),
            )
        )

    def commitment(self, *, manifest_hash: str) -> tuple[str, int]:
        """Return a canonical root over every verified committed episode."""

        episodes = self.list(manifest_hash=manifest_hash)
        entries = tuple(
            {
                "manifest_hash": episode.manifest_hash,
                "case_id": episode.case_id,
                "replay_index": episode.replay_index,
                "episode_sha256": sha256(episode),
            }
            for episode in episodes
        )
        return sha256(entries), len(entries)
