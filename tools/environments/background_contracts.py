"""Shared background-process contracts for environment backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, kw_only=True)
class _DetachedCommandOutputResult:
    """Private shared fields for detached-command result variants."""

    pid: int | None = None
    output: str | None = None
    output_delta: str | None = None


@dataclass(frozen=True, kw_only=True)
class DetachedCommandSpawnResult(_DetachedCommandOutputResult):
    """Initial result for a detached background command spawn."""

    handle: Any = None


@dataclass(frozen=True, kw_only=True)
class DetachedCommandRunningResult(_DetachedCommandOutputResult):
    """Running-state result for a detached background command."""


@dataclass(frozen=True, kw_only=True)
class _DetachedCommandTerminalResult(_DetachedCommandOutputResult):
    """Private shared fields for terminal detached-command result variants."""

    exit_code: int


@dataclass(frozen=True, kw_only=True)
class DetachedCommandExitedResult(_DetachedCommandTerminalResult):
    """Terminal result for a normally exited detached background command."""


@dataclass(frozen=True, kw_only=True)
class DetachedCommandKilledResult(_DetachedCommandTerminalResult):
    """Terminal result for a killed detached background command."""

    exit_code: int = -15


@dataclass(frozen=True, kw_only=True)
class DetachedCommandErrorResult(_DetachedCommandTerminalResult):
    """Terminal result for a detached background command failure."""

    error: str
    exit_code: int = -1


DetachedCommandResult = (
    DetachedCommandSpawnResult
    | DetachedCommandRunningResult
    | DetachedCommandExitedResult
    | DetachedCommandKilledResult
    | DetachedCommandErrorResult
)


@dataclass(frozen=True)
class BackendBackgroundCheckpoint:
    """Parsed backend-owned background checkpoint shared across backends."""

    backend: str
    command: str

    def to_json(self) -> dict[str, Any]:
        """Serialize the parsed checkpoint back to JSON."""
        return {"backend": self.backend, "command": self.command}


def get_backend_background_checkpoint_backend(payload: Any) -> str | None:
    """Return the backend discriminator for a raw checkpoint payload."""
    if not isinstance(payload, dict):
        return None
    backend = payload.get("backend")
    return backend if isinstance(backend, str) and backend else None


class BackgroundProcessAdapter(Protocol):
    """Common non-local background-process contract for environment backends."""

    def spawn(self, timeout: int) -> DetachedCommandResult:
        """Start the detached process and return initial metadata/output."""

    def poll(self, timeout: int) -> DetachedCommandResult:
        """Refresh status and output without blocking indefinitely."""

    def wait(self, timeout: int) -> DetachedCommandResult:
        """Block for up to timeout seconds and return updated status/output."""

    def kill(self, timeout: int) -> DetachedCommandResult:
        """Terminate the detached process and return final status/output."""
