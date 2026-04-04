"""Vercel Sandbox execution environment.

Uses Vercel's sync sandbox client as a Hermes terminal backend. Commands run
through ``sh -lc`` so file tools keep working through shell semantics. The
backend owns native detached-process adapters for ``process_registry`` spawn
and checkpoint recovery.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from tools.credential_files import get_credential_file_mounts, iter_skills_files
from tools.environments.background_contracts import (
    BackendBackgroundCheckpoint,
    DetachedCommandErrorResult,
    DetachedCommandExitedResult,
    DetachedCommandKilledResult,
    DetachedCommandResult,
    DetachedCommandRunningResult,
    DetachedCommandSpawnResult,
)
from tools.environments.base import BaseEnvironment
from tools.interrupt import is_interrupted

logger = logging.getLogger(__name__)

_SNAPSHOT_STORE = get_hermes_home() / "vercel_sandbox_snapshots.json"
_DEFAULT_SHARED_CONTAINER_DISK_MB = 51200
_DEFAULT_GENERIC_CONTAINER_MEMORY_MB = 5120
_VERCEL_MEMORY_PER_VCPU_MB = 2048


def _load_snapshots() -> dict[str, str]:
    if not _SNAPSHOT_STORE.exists():
        return {}
    try:
        data = json.loads(_SNAPSHOT_STORE.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _save_snapshots(data: dict[str, str]) -> None:
    _SNAPSHOT_STORE.parent.mkdir(parents=True, exist_ok=True)
    _SNAPSHOT_STORE.write_text(json.dumps(data, indent=2))


def _snapshot_key(task_id: str) -> str:
    return f"vercel_sandbox:{task_id}"


def _get_snapshot_restore_candidate(task_id: str) -> str | None:
    snapshot_id = _load_snapshots().get(_snapshot_key(task_id))
    return snapshot_id if isinstance(snapshot_id, str) and snapshot_id else None


def _store_snapshot(task_id: str, snapshot_id: str) -> None:
    snapshots = _load_snapshots()
    snapshots[_snapshot_key(task_id)] = snapshot_id
    _save_snapshots(snapshots)


def _delete_snapshot(task_id: str, snapshot_id: str | None = None) -> None:
    snapshots = _load_snapshots()
    key = _snapshot_key(task_id)
    value = snapshots.get(key)
    if value is None:
        return
    if snapshot_id is None or value == snapshot_id:
        snapshots.pop(key, None)
        _save_snapshots(snapshots)


@dataclass(frozen=True)
class VercelResourceConstraints:
    """Typed Vercel resource contract persisted across checkpoint recovery."""

    cpu: int = 1
    disk: int = _DEFAULT_SHARED_CONTAINER_DISK_MB

    @classmethod
    def from_inputs(
        cls,
        *,
        cpu: float | int | None,
        memory: Any = None,
        disk: Any = None,
    ) -> VercelResourceConstraints:
        if cpu in (None, 0):
            normalized_cpu = 1
        else:
            if cpu < 0:
                logger.warning(
                    "Vercel Sandbox cpu %s is invalid; using default 1 vCPU",
                    cpu,
                )
                normalized_cpu = 1
            elif int(cpu) != cpu:
                logger.warning(
                    "Vercel Sandbox cpu %s is not a whole number; using default 1 vCPU",
                    cpu,
                )
                normalized_cpu = 1
            else:
                normalized_cpu = int(cpu)
                if normalized_cpu <= 0:
                    logger.warning(
                        "Vercel Sandbox cpu %s must be greater than zero; using default 1 vCPU",
                        cpu,
                    )
                    normalized_cpu = 1

        effective_vcpus = normalized_cpu

        if memory not in (None, 0):
            logger.warning(
                "Vercel Sandbox ignores requested memory=%s MB; "
                "Vercel derives memory from cpu=%s and will use %s MB",
                memory,
                effective_vcpus,
                effective_vcpus * _VERCEL_MEMORY_PER_VCPU_MB,
            )

        if disk in (None, 0):
            normalized_disk = _DEFAULT_SHARED_CONTAINER_DISK_MB
        else:
            normalized_disk = int(disk)
            if normalized_disk <= 0:
                logger.warning(
                    "Vercel Sandbox disk %s is invalid; using shared default %s MB",
                    disk,
                    _DEFAULT_SHARED_CONTAINER_DISK_MB,
                )
                normalized_disk = _DEFAULT_SHARED_CONTAINER_DISK_MB
            elif normalized_disk != _DEFAULT_SHARED_CONTAINER_DISK_MB:
                logger.warning(
                    "Vercel Sandbox does not support configurable container_disk=%s; using shared default %s MB",
                    normalized_disk,
                    _DEFAULT_SHARED_CONTAINER_DISK_MB,
                )
                normalized_disk = _DEFAULT_SHARED_CONTAINER_DISK_MB

        return cls(
            cpu=normalized_cpu,
            disk=normalized_disk,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: dict[str, Any] | None,
    ) -> VercelResourceConstraints:
        payload = checkpoint or {}
        return cls.from_inputs(
            cpu=payload.get("cpu"),
            memory=payload.get("memory"),
            disk=payload.get("disk"),
        )

    @property
    def effective_vcpus(self) -> int:
        return self.cpu

    @property
    def effective_memory_mb(self) -> int:
        return self.effective_vcpus * _VERCEL_MEMORY_PER_VCPU_MB

    def to_vercel_resources(self) -> dict[str, int]:
        return {
            "vcpus": self.effective_vcpus,
            "memory": self.effective_memory_mb,
        }

    def to_checkpoint(self) -> dict[str, int]:
        return {
            "cpu": self.cpu,
            "disk": self.disk,
        }


@dataclass
class VercelBackgroundHandle:
    """Native detached command handle tracked by ProcessRegistry."""

    command: Any
    sandbox_id: str = ""
    command_id: str = ""


@dataclass
class VercelBackgroundProcessAdapter:
    """Vercel-owned adapter mapping native detached commands to Hermes' protocol."""

    env: VercelSandboxEnvironment
    command: str
    cwd: str
    handle: VercelBackgroundHandle | None = None

    def spawn(self, timeout: int) -> DetachedCommandResult:
        try:
            result = self.env.spawn_background_process(self.command, cwd=self.cwd, timeout=timeout)
            self.handle = result.handle
            if self.handle is None:
                return DetachedCommandErrorResult(
                    error="Vercel background spawn did not return a handle",
                )
            return result
        except Exception as exc:
            return DetachedCommandErrorResult(error=f"Vercel background spawn failed: {exc}")

    def poll(self, timeout: int) -> DetachedCommandResult:
        try:
            if self.handle is None:
                raise ValueError("Vercel background adapter is missing a handle")
            return self.env.poll_background_process(self.handle, timeout=timeout)
        except Exception as exc:
            return DetachedCommandErrorResult(error=f"Vercel background poll failed: {exc}")

    def wait(self, timeout: int) -> DetachedCommandResult:
        try:
            if self.handle is None:
                raise ValueError("Vercel background adapter is missing a handle")
            return self.env.wait_background_process(self.handle, timeout=timeout)
        except Exception as exc:
            return DetachedCommandErrorResult(error=f"Vercel background wait failed: {exc}")

    def kill(self, timeout: int) -> DetachedCommandResult:
        try:
            if self.handle is None:
                raise ValueError("Vercel background adapter is missing a handle")
            return self.env.kill_background_process(self.handle, timeout=timeout)
        except Exception as exc:
            return DetachedCommandErrorResult(error=f"Vercel background kill failed: {exc}")

    def checkpoint_data(self) -> VercelBackgroundCheckpoint | None:
        if self.handle is None:
            return None
        return self.env.serialize_background_handle(self.handle, command=self.command)


@dataclass(frozen=True)
class VercelBackgroundCheckpoint(BackendBackgroundCheckpoint):
    """Parsed Vercel background checkpoint with strict JSON boundaries."""

    command: str
    sandbox_id: str
    command_id: str
    runtime: str | None
    cwd: str
    timeout: int
    task_id: str
    persistent_filesystem: bool
    resource_constraints: VercelResourceConstraints

    @classmethod
    def from_json(cls, payload: Any) -> VercelBackgroundCheckpoint:
        if not isinstance(payload, dict):
            raise ValueError("Vercel background checkpoint must be a JSON object")

        backend = payload.get("backend")
        if backend != "vercel_sandbox":
            raise ValueError("Checkpoint does not belong to Vercel Sandbox")

        command = payload.get("command")
        sandbox_id = payload.get("sandbox_id")
        command_id = payload.get("command_id")
        cwd = payload.get("cwd")
        timeout = payload.get("timeout")
        task_id = payload.get("task_id")
        persistent_filesystem = payload.get("persistent_filesystem")

        if not isinstance(command, str) or not command:
            raise ValueError("Vercel background checkpoint is missing command")
        if not isinstance(sandbox_id, str) or not sandbox_id:
            raise ValueError("Vercel background checkpoint is missing sandbox_id")
        if not isinstance(command_id, str) or not command_id:
            raise ValueError("Vercel background checkpoint is missing command_id")
        if not isinstance(cwd, str) or not cwd:
            raise ValueError("Vercel background checkpoint is missing cwd")
        if not isinstance(timeout, int):
            raise ValueError("Vercel background checkpoint is missing timeout")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("Vercel background checkpoint is missing task_id")
        if not isinstance(persistent_filesystem, bool):
            raise ValueError(
                "Vercel background checkpoint is missing persistent_filesystem"
            )

        runtime = payload.get("runtime")
        if runtime is not None and not isinstance(runtime, str):
            raise ValueError("Vercel background checkpoint runtime must be a string or null")
        resource_constraints = payload.get("resource_constraints")
        if not isinstance(resource_constraints, dict):
            raise ValueError(
                "Vercel background checkpoint is missing resource_constraints"
            )

        return cls(
            backend="vercel_sandbox",
            command=command,
            sandbox_id=sandbox_id,
            command_id=command_id,
            runtime=runtime,
            cwd=cwd,
            timeout=timeout,
            task_id=task_id,
            persistent_filesystem=persistent_filesystem,
            resource_constraints=VercelResourceConstraints.from_checkpoint(
                resource_constraints
            ),
        )

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["resource_constraints"] = self.resource_constraints.to_checkpoint()
        return payload


class VercelSandboxEnvironment(BaseEnvironment):
    """Hermes terminal backend backed by a Vercel sandbox."""

    _status_poll_seconds = 0.25
    _sandbox_timeout_floor_seconds = 900
    _sandbox_timeout_grace_seconds = 60

    def create_background_process_adapter(
        self,
        *,
        command: str,
        session_id: str,
        cwd: str | None,
    ) -> VercelBackgroundProcessAdapter:
        del session_id
        return VercelBackgroundProcessAdapter(
            env=self,
            command=command,
            cwd=cwd or "",
        )

    @classmethod
    def parse_background_checkpoint(cls, payload: Any) -> VercelBackgroundCheckpoint:
        return VercelBackgroundCheckpoint.from_json(payload)

    @classmethod
    def recover_background_session(
        cls,
        checkpoint: BackendBackgroundCheckpoint,
    ) -> tuple[VercelSandboxEnvironment, VercelBackgroundProcessAdapter]:
        if not isinstance(checkpoint, VercelBackgroundCheckpoint):
            raise TypeError("Vercel background recovery requires a parsed checkpoint")

        env, handle = cls.recover_background_handle(checkpoint)
        adapter = VercelBackgroundProcessAdapter(
            env=env,
            command=checkpoint.command,
            cwd=checkpoint.cwd,
            handle=handle,
        )
        return env, adapter

    def __init__(
        self,
        runtime: str | None = None,
        cwd: str = "/",
        timeout: int = 60,
        cpu: float = 1,
        memory: int = _VERCEL_MEMORY_PER_VCPU_MB,
        disk: int = _DEFAULT_SHARED_CONTAINER_DISK_MB,
        persistent_filesystem: bool = True,
        task_id: str = "default",
    ):
        super().__init__(cwd=cwd, timeout=timeout)
        self._runtime = runtime or None
        self._requested_cwd = cwd
        self._persistent = persistent_filesystem
        self._task_id = task_id
        self._resource_constraints = VercelResourceConstraints.from_inputs(
            cpu=cpu,
            memory=memory,
            disk=disk,
        )
        self._lock = threading.Lock()
        self._sandbox = None
        self._remote_home = "/root"
        self._synced_files: dict[str, tuple[float, int]] = {}
        self._closed = False

        from vercel.sandbox import (
            Sandbox,
            SandboxAuthError,
            SandboxPermissionError,
            SandboxRateLimitError,
            SandboxServerError,
        )

        self._auth_errors = (
            SandboxAuthError,
            SandboxPermissionError,
            SandboxRateLimitError,
            SandboxServerError,
        )

        restore_snapshot = _get_snapshot_restore_candidate(self._task_id) if self._persistent else None
        self._attach_initial_sandbox(restore_snapshot)

    def execute(
        self,
        command: str,
        cwd: str = "",
        *,
        timeout: int | None = None,
        stdin_data: str | None = None,
    ) -> dict:
        effective_timeout = timeout or self.timeout

        try:
            prepared = self._prepare_shell_command(command, stdin_data=stdin_data)
            result = self._exec_in_thread(
                prepared,
                cwd=self._resolve_cwd(cwd),
                timeout=effective_timeout,
            )
            if "error" in result:
                raise result["error"]
            return result
        except self._auth_errors as exc:
            return {"output": self._format_sdk_error(exc), "returncode": 1}
        except Exception as exc:
            return {"output": f"Vercel Sandbox execution error: {exc}", "returncode": 1}

    def spawn_background_process(
        self,
        command: str,
        *,
        cwd: str = "",
        timeout: int = 10,
    ) -> DetachedCommandSpawnResult:
        prepared = self._prepare_shell_command(command)
        with self._lock:
            self._ensure_sandbox_ready()
            self._sync_skills_and_credentials()
            self._ensure_timeout_budget(max(timeout, self.timeout))
            handle = self._run_detached_command_with_retry(
                prepared,
                cwd=self._resolve_cwd(cwd),
                timeout=max(timeout, self.timeout),
            )
        return DetachedCommandSpawnResult(
            handle=VercelBackgroundHandle(
                command=handle,
                sandbox_id=getattr(self._sandbox, "sandbox_id", "") or getattr(getattr(self._sandbox, "sandbox", None), "id", ""),
                command_id=getattr(handle, "cmd_id", ""),
            ),
            output="",
        )

    def poll_background_process(
        self, handle: VercelBackgroundHandle, *, timeout: int = 5
    ) -> DetachedCommandRunningResult | DetachedCommandExitedResult:
        with self._lock:
            self._refresh_background_observation(timeout)
            current = self._sandbox.get_command(handle.command.cmd_id)
        exit_code = current.cmd.exit_code
        output = current.output()
        if exit_code is None:
            return DetachedCommandRunningResult(output=output)
        return DetachedCommandExitedResult(exit_code=exit_code, output=output)

    def wait_background_process(
        self, handle: VercelBackgroundHandle, *, timeout: int = 5
    ) -> DetachedCommandRunningResult | DetachedCommandExitedResult:
        deadline = time.monotonic() + max(timeout, 1)
        while True:
            result = self.poll_background_process(handle, timeout=timeout)
            if isinstance(result, DetachedCommandExitedResult):
                return result
            if time.monotonic() >= deadline:
                return result
            time.sleep(self._status_poll_seconds)

    def kill_background_process(
        self, handle: VercelBackgroundHandle, *, timeout: int = 5
    ) -> DetachedCommandKilledResult:
        del timeout
        try:
            handle.command.kill()
        except Exception:
            pass
        try:
            output = handle.command.output()
        except Exception:
            output = ""
        return DetachedCommandKilledResult(output=output)

    def serialize_background_handle(
        self,
        handle: VercelBackgroundHandle,
        *,
        command: str,
    ) -> VercelBackgroundCheckpoint:
        sandbox_id = handle.sandbox_id or getattr(self._sandbox, "sandbox_id", "") or getattr(getattr(self._sandbox, "sandbox", None), "id", "")
        command_id = handle.command_id or getattr(handle.command, "cmd_id", "")
        if not isinstance(command, str) or not command:
            raise ValueError("Vercel background checkpoint requires a command string")
        if not sandbox_id or not command_id:
            raise ValueError("Vercel background checkpoint requires sandbox and command identifiers")
        return VercelBackgroundCheckpoint(
            backend="vercel_sandbox",
            command=command,
            sandbox_id=sandbox_id,
            command_id=command_id,
            runtime=self._runtime,
            cwd=self.cwd,
            timeout=self.timeout,
            task_id=self._task_id,
            persistent_filesystem=self._persistent,
            resource_constraints=self._resource_constraints,
        )

    @classmethod
    def recover_background_handle(
        cls,
        checkpoint: VercelBackgroundCheckpoint,
    ) -> tuple[VercelSandboxEnvironment, VercelBackgroundHandle]:
        from vercel.sandbox import (
            Sandbox,
            SandboxAuthError,
            SandboxPermissionError,
            SandboxRateLimitError,
            SandboxServerError,
        )

        sandbox_id = checkpoint.sandbox_id
        command_id = checkpoint.command_id

        sandbox = Sandbox.get(sandbox_id=sandbox_id)

        env = cls.__new__(cls)
        BaseEnvironment.__init__(
            env,
            cwd=checkpoint.cwd,
            timeout=checkpoint.timeout,
        )
        env._runtime = checkpoint.runtime
        env._requested_cwd = checkpoint.cwd
        env._persistent = checkpoint.persistent_filesystem
        env._task_id = checkpoint.task_id
        env._resource_constraints = checkpoint.resource_constraints
        env._lock = threading.Lock()
        env._sandbox = sandbox
        env._remote_home = "/root"
        env._synced_files = {}
        env._closed = False
        env._auth_errors = (
            SandboxAuthError,
            SandboxPermissionError,
            SandboxRateLimitError,
            SandboxServerError,
        )
        env._attach_sandbox(sandbox, requested_cwd=checkpoint.cwd)

        command = sandbox.get_command(command_id)
        return env, VercelBackgroundHandle(
            command=command,
            sandbox_id=sandbox_id,
            command_id=command_id,
        )

    def cleanup(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            sandbox = self._sandbox
            self._sandbox = None

        if sandbox is None:
            return

        if self._persistent:
            snapshot_id = self._snapshot_sandbox(sandbox)
            if snapshot_id:
                _store_snapshot(self._task_id, snapshot_id)
                logger.info(
                    "Vercel sandbox snapshot %s saved for task %s",
                    snapshot_id[:20],
                    self._task_id,
                )
            else:
                _delete_snapshot(self._task_id)
        else:
            _delete_snapshot(self._task_id)

        try:
            sandbox.stop(blocking=True, timeout=15.0, poll_interval=0.5)
        except Exception as exc:
            logger.debug("Vercel sandbox stop failed for task %s: %s", self._task_id, exc)
        finally:
            try:
                sandbox.client.close()
            except Exception:
                pass

    def _attach_initial_sandbox(self, restore_snapshot: str | None) -> None:
        from vercel.sandbox import Sandbox

        sandbox_timeout_seconds = max(self.timeout, self._sandbox_timeout_floor_seconds)
        sandbox_timeout_ms = int((sandbox_timeout_seconds + self._sandbox_timeout_grace_seconds) * 1000)

        if restore_snapshot:
            try:
                sandbox = Sandbox.create(
                    timeout=sandbox_timeout_ms,
                    runtime=self._runtime,
                    resources=self._resource_constraints.to_vercel_resources(),
                    source={
                        "type": "snapshot",
                        "snapshot_id": restore_snapshot,
                    },
                )
            except Exception:
                logger.warning(
                    "Vercel sandbox restore from snapshot %s failed for task %s; retrying fresh sandbox",
                    restore_snapshot[:20],
                    self._task_id,
                )
                _delete_snapshot(self._task_id, restore_snapshot)
                sandbox = Sandbox.create(
                    timeout=sandbox_timeout_ms,
                    runtime=self._runtime,
                    resources=self._resource_constraints.to_vercel_resources(),
                )
        else:
            sandbox = Sandbox.create(
                timeout=sandbox_timeout_ms,
                runtime=self._runtime,
                resources=self._resource_constraints.to_vercel_resources(),
            )
        self._attach_sandbox(sandbox, requested_cwd=self._requested_cwd)

    def _attach_sandbox(self, sandbox: Any, *, requested_cwd: str) -> None:
        self._closed = False
        self._sandbox = sandbox
        self._synced_files = {}
        self._wait_for_running()
        self._remote_home = self._detect_remote_home()

        default_cwds = {"", "/", "/root", "~"}
        if requested_cwd in default_cwds:
            self.cwd = sandbox.sandbox.cwd or self._remote_home or "/"
        else:
            self.cwd = requested_cwd

    def _ensure_sandbox_ready(self) -> None:
        if self._sandbox is None:
            restore_snapshot = _get_snapshot_restore_candidate(self._task_id) if self._persistent else None
            self._attach_initial_sandbox(restore_snapshot)
            return

        try:
            self._sandbox.refresh()
        except Exception as exc:
            logger.warning("Vercel sandbox refresh failed for task %s: %s; recreating", self._task_id, exc)
            try:
                self._sandbox.client.close()
            except Exception:
                pass
            self._attach_initial_sandbox(None)
            return

        status = str(getattr(self._sandbox, "status", "")).lower()
        if status in {"stopped", "failed", "aborted"}:
            try:
                self._sandbox.client.close()
            except Exception:
                pass
            self._attach_initial_sandbox(None)
            return

        self._wait_for_running()

    def _wait_for_running(self, timeout_seconds: int = 30) -> None:
        deadline = time.monotonic() + max(timeout_seconds, 1)
        while True:
            status = str(getattr(self._sandbox, "status", "")).lower()
            if not status or status == "running":
                return
            if status in {"stopped", "failed", "aborted"}:
                raise RuntimeError(f"Sandbox entered terminal state: {status}")
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Sandbox did not reach running state (last status: {status})")
            time.sleep(self._status_poll_seconds)
            self._sandbox.refresh()

    def _detect_remote_home(self) -> str:
        try:
            result = self._sandbox.run_command("sh", ["-lc", "printf %s \"$HOME\""], cwd=self._sandbox.sandbox.cwd)
            output = result.output().strip()
            if output:
                return output
        except Exception as exc:
            logger.debug("Vercel sandbox home detection failed for task %s: %s", self._task_id, exc)
        return "/root"

    def _ensure_timeout_budget(self, timeout_seconds: int) -> None:
        desired_ms = int((timeout_seconds + self._sandbox_timeout_grace_seconds) * 1000)
        current_ms = int(getattr(self._sandbox, "timeout", 0) or 0)
        if desired_ms <= current_ms:
            return
        self._sandbox.extend_timeout(desired_ms - current_ms)

    def _refresh_background_observation(self, timeout_seconds: int) -> None:
        if self._sandbox is None:
            raise RuntimeError("Vercel background observation requires an attached sandbox")
        self._sandbox.refresh()
        self._wait_for_running()
        self._ensure_timeout_budget(max(timeout_seconds, self.timeout))

    def _resolve_cwd(self, cwd: str) -> str:
        effective_cwd = cwd or self.cwd or self._sandbox.sandbox.cwd
        if effective_cwd == "~":
            return self._remote_home
        return effective_cwd

    def _prepare_shell_command(self, command: str, *, stdin_data: str | None = None) -> str:
        exec_command = command
        if stdin_data is not None:
            marker = f"HERMES_EOF_{uuid.uuid4().hex[:8]}"
            while marker in stdin_data:
                marker = f"HERMES_EOF_{uuid.uuid4().hex[:8]}"
            exec_command = f"{exec_command} << '{marker}'\n{stdin_data}\n{marker}"

        exec_command, sudo_stdin = self._prepare_command(exec_command)
        if sudo_stdin is not None:
            exec_command = f"printf '%s\\n' {shlex.quote(sudo_stdin.rstrip())} | {exec_command}"
        return exec_command

    def _build_command_env(self) -> dict[str, str]:
        from tools.environments.local import _sanitize_subprocess_env

        return _sanitize_subprocess_env(os.environ, self.env)

    def _exec_in_thread(self, command: str, *, cwd: str, timeout: int) -> dict[str, Any]:
        """Run a foreground command off-thread so Hermes keeps timeout control."""
        result_holder: dict[str, Any] = {"value": None, "error": None}
        command_holder: dict[str, Any] = {"handle": None}
        command_env = self._build_command_env()

        def _run() -> None:
            try:
                with self._lock:
                    self._ensure_sandbox_ready()
                    self._sync_skills_and_credentials()
                    self._ensure_timeout_budget(timeout)
                    handle = self._run_detached_command_with_retry(
                        command,
                        cwd=cwd,
                        timeout=timeout,
                        env=command_env,
                    )
                    command_holder["handle"] = handle

                finished = handle.wait()
                exit_code = getattr(finished, "exit_code", None)
                if exit_code is None:
                    exit_code = getattr(getattr(finished, "cmd", None), "exit_code", None)
                result_holder["value"] = {
                    "output": finished.output(),
                    "returncode": exit_code if exit_code is not None else 1,
                }
            except Exception as exc:
                result_holder["error"] = exc

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        deadline = time.monotonic() + timeout
        while thread.is_alive():
            if is_interrupted():
                handle = command_holder.get("handle")
                if handle is not None:
                    try:
                        handle.kill()
                    except Exception:
                        pass
                return {"output": "[Command interrupted]", "returncode": 130}

            if time.monotonic() >= deadline:
                handle = command_holder.get("handle")
                if handle is not None:
                    try:
                        handle.kill()
                    except Exception:
                        pass
                return self._timeout_result(timeout)

            thread.join(timeout=self._status_poll_seconds)

        if result_holder["error"] is not None:
            return {"error": result_holder["error"]}
        return result_holder["value"]

    def _run_detached_command(
        self,
        command: str,
        *,
        cwd: str,
        env: dict[str, str] | None = None,
    ):
        return self._sandbox.run_command_detached(
            "sh",
            ["-lc", command],
            cwd=cwd,
            env=env if env is not None else self._build_command_env(),
        )

    def _run_detached_command_with_retry(
        self,
        command: str,
        *,
        cwd: str,
        timeout: int,
        env: dict[str, str] | None = None,
    ):
        try:
            return self._run_detached_command(command, cwd=cwd, env=env)
        except self._auth_errors:
            raise
        except Exception as exc:
            logger.warning(
                "Vercel detached command startup failed for task %s: %s; retrying once after refresh",
                self._task_id,
                exc,
            )
            self._sandbox.refresh()
            self._wait_for_running()
            self._ensure_timeout_budget(timeout)
            return self._run_detached_command(command, cwd=cwd, env=env)

    def _sync_skills_and_credentials(self) -> None:
        container_base = f"{self._remote_home.rstrip('/')}/.hermes"
        pending_files: list[dict[str, Any]] = []
        entries = list(get_credential_file_mounts()) + list(iter_skills_files(container_base=container_base))

        for entry in entries:
            host_path = Path(entry["host_path"])
            try:
                stat = host_path.stat()
            except OSError:
                continue

            remote_path = entry["container_path"]
            fingerprint = (stat.st_mtime, stat.st_size)
            if self._synced_files.get(remote_path) == fingerprint:
                continue

            try:
                content = host_path.read_bytes()
            except OSError:
                continue

            pending_files.append({"path": remote_path, "content": content})
            self._synced_files[remote_path] = fingerprint

        if pending_files:
            self._sandbox.write_files(pending_files)

    @staticmethod
    def _format_sdk_error(exc: Exception) -> str:
        return f"Vercel Sandbox error: {exc}"

    @staticmethod
    def _extract_snapshot_id(snapshot: Any) -> str | None:
        if isinstance(snapshot, str) and snapshot:
            return snapshot
        for attr in ("id", "snapshot_id", "object_id"):
            value = getattr(snapshot, attr, None)
            if isinstance(value, str) and value:
                return value
        nested = getattr(snapshot, "snapshot", None)
        if nested is not None:
            return VercelSandboxEnvironment._extract_snapshot_id(nested)
        return None

    def _snapshot_sandbox(self, sandbox: Any) -> str | None:
        try:
            snapshot = sandbox.snapshot()
        except Exception as exc:
            logger.warning("Vercel sandbox snapshot failed for task %s: %s", self._task_id, exc)
            return None
        return self._extract_snapshot_id(snapshot)
