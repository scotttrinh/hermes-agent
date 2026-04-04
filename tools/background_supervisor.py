"""Supervisor for non-local detached background sessions.

Owns background monitoring loops and serialized adapter access so the process
registry can stay focused on session storage and query APIs.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

from tools.environments.background_contracts import DetachedCommandResult

if TYPE_CHECKING:
    from tools.process_registry import ProcessSession

logger = logging.getLogger(__name__)


@dataclass
class _SessionRuntime:
    adapter_lock: threading.Lock
    poller_thread: threading.Thread | None = None


class BackgroundSupervisor:
    """Own monitoring threads for background adapters outside the registry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runtimes: dict[str, _SessionRuntime] = {}

    def _get_runtime(self, session_id: str) -> _SessionRuntime:
        with self._lock:
            runtime = self._runtimes.get(session_id)
            if runtime is None:
                runtime = _SessionRuntime(adapter_lock=threading.Lock())
                self._runtimes[session_id] = runtime
            return runtime

    def call_adapter(
        self,
        session: "ProcessSession",
        action: str,
        *,
        timeout: int,
    ) -> Optional[DetachedCommandResult]:
        adapter = session.background_adapter
        if adapter is None:
            return None

        runtime = self._get_runtime(session.id)
        with runtime.adapter_lock:
            if action == "poll":
                return adapter.poll(timeout=timeout)
            if action == "wait":
                return adapter.wait(timeout=timeout)
            if action == "kill":
                return adapter.kill(timeout=timeout)
        raise ValueError(f"Unsupported background adapter action: {action}")

    def monitor_session(
        self,
        session: "ProcessSession",
        *,
        get_session: Callable[[str], "ProcessSession" | None],
        sync_result: Callable[["ProcessSession", Optional[DetachedCommandResult]], None],
        handle_failure: Callable[["ProcessSession", Exception], None],
        interval_seconds: float = 1.0,
    ) -> None:
        if session.background_adapter is None or session.exited:
            return

        runtime = self._get_runtime(session.id)
        with self._lock:
            existing = self._runtimes.get(session.id)
            if existing and existing.poller_thread and existing.poller_thread.is_alive():
                return

            poller = threading.Thread(
                target=self._monitor_loop,
                args=(session.id, get_session, sync_result, handle_failure, interval_seconds),
                daemon=True,
                name=f"proc-poller-{session.id}",
            )
            runtime.poller_thread = poller
            self._runtimes[session.id] = runtime

        poller.start()

    def unregister(self, session_id: str) -> None:
        with self._lock:
            self._runtimes.pop(session_id, None)

    def _monitor_loop(
        self,
        session_id: str,
        get_session: Callable[[str], "ProcessSession" | None],
        sync_result: Callable[["ProcessSession", Optional[DetachedCommandResult]], None],
        handle_failure: Callable[["ProcessSession", Exception], None],
        interval_seconds: float,
    ) -> None:
        try:
            while True:
                session = get_session(session_id)
                if session is None or session.exited:
                    return

                try:
                    adapter_result = self.call_adapter(session, "poll", timeout=5)
                    sync_result(session, adapter_result)
                except Exception as exc:
                    handle_failure(session, exc)
                    return

                if session.exited:
                    return

                time.sleep(interval_seconds)
        finally:
            self.unregister(session_id)
