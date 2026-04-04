"""Compatibility exports for background-process adapter types."""

from __future__ import annotations

from tools.environments.background_contracts import (
    BackgroundProcessAdapter,
    DetachedCommandErrorResult,
    DetachedCommandExitedResult,
    DetachedCommandKilledResult,
    DetachedCommandResult,
    DetachedCommandRunningResult,
    DetachedCommandSpawnResult,
)
from tools.environments.shell_background import ShellEnvironmentBackgroundAdapter

__all__ = [
    "BackgroundProcessAdapter",
    "DetachedCommandErrorResult",
    "DetachedCommandExitedResult",
    "DetachedCommandKilledResult",
    "DetachedCommandResult",
    "DetachedCommandRunningResult",
    "DetachedCommandSpawnResult",
    "ShellEnvironmentBackgroundAdapter",
]
