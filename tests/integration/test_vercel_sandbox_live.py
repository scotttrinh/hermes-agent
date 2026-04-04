"""Integration tests for the Vercel Sandbox terminal backend."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import uuid
from pathlib import Path

import pytest

from tools.process_registry import process_registry
import tools.process_registry as process_registry_module

pytestmark = pytest.mark.integration

if not os.getenv("VERCEL_OIDC_TOKEN") and not all(
    os.getenv(name) for name in ("VERCEL_TOKEN", "VERCEL_PROJECT_ID", "VERCEL_TEAM_ID")
):
    pytest.skip("Vercel Sandbox credentials not set", allow_module_level=True)

pytest.importorskip("vercel.sandbox")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

spec = importlib.util.spec_from_file_location(
    "terminal_tool", REPO_ROOT / "tools" / "terminal_tool.py"
)
terminal_module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(terminal_module)

terminal_tool = terminal_module.terminal_tool
cleanup_vm = terminal_module.cleanup_vm


@pytest.fixture(autouse=True)
def _force_vercel(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "vercel_sandbox")
    monkeypatch.setenv("TERMINAL_VERCEL_RUNTIME", os.getenv("TERMINAL_VERCEL_RUNTIME", "python3.13"))
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "true")


def _make_task_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _run_terminal(command, task_id, **kwargs):
    return json.loads(terminal_tool(command, task_id=task_id, **kwargs))


def _wait_for_running_process(session_id, *, attempts=20, delay=1.0):
    poll_result = None
    for _ in range(attempts):
        poll_result = process_registry.poll(session_id)
        output = poll_result.get("output_preview", "")
        if "bg-ready" in output and poll_result["status"] == "running":
            return poll_result
        time.sleep(delay)
    pytest.fail(f"background process {session_id} never reached a running state: {poll_result}")


def _snapshot_registry_state():
    return (
        dict(process_registry._running),
        dict(process_registry._finished),
        list(process_registry.pending_watchers),
    )


def _restore_registry_state(state):
    running, finished, pending_watchers = state
    with process_registry._lock:
        process_registry._running.clear()
        process_registry._running.update(running)
        process_registry._finished.clear()
        process_registry._finished.update(finished)
    process_registry.pending_watchers[:] = pending_watchers


class TestVercelBasic:
    def test_smoke_persistence_cycle(self):
        task_id = _make_task_id("vercel_live_persist")
        marker_path = "/tmp/hermes_vercel_live.txt"
        try:
            pwd_result = _run_terminal("pwd && python3 --version", task_id, timeout=120)
            assert pwd_result["exit_code"] == 0, pwd_result
            assert "Python" in pwd_result["output"]

            write_result = _run_terminal(f"printf 'live-smoke' > {marker_path}", task_id)
            assert write_result["exit_code"] == 0, write_result

            read_result = _run_terminal(f"cat {marker_path}", task_id)
            assert read_result["exit_code"] == 0, read_result
            assert "live-smoke" in read_result["output"]

            cleanup_vm(task_id)

            restored_result = _run_terminal(f"cat {marker_path}", task_id, timeout=120)
            assert restored_result["exit_code"] == 0, restored_result
            assert "live-smoke" in restored_result["output"]
        finally:
            cleanup_vm(task_id)

    def test_foreground_nonzero_exit_propagates(self):
        task_id = _make_task_id("vercel_live_nonzero")
        command = (
            "python3 -c \"import sys; "
            "print('about-to-fail'); sys.stdout.flush(); sys.exit(42)\""
        )
        try:
            result = _run_terminal(command, task_id, timeout=120)
            assert result["exit_code"] == 42, result
            assert "about-to-fail" in result["output"]
        finally:
            cleanup_vm(task_id)


class TestVercelBackgroundProcesses:
    def test_background_process_lifecycle(self):
        task_id = _make_task_id("vercel_live_bg")
        command = (
            "python3 -c \"import sys,time; "
            "print('bg-ready'); sys.stdout.flush(); time.sleep(30)\""
        )
        session_id = None
        try:
            spawn_result = _run_terminal(command, task_id, background=True, timeout=120)
            assert spawn_result["exit_code"] == 0, spawn_result
            session_id = spawn_result["session_id"]

            for _ in range(10):
                poll_result = process_registry.poll(session_id)
                output = poll_result.get("output_preview", "")
                if "bg-ready" in output or poll_result["status"] == "running":
                    break
            else:
                pytest.fail("background process never reached a running state")

            assert poll_result["status"] == "running"

            kill_result = process_registry.kill_process(session_id)
            assert kill_result["status"] == "killed"

            final_result = process_registry.poll(session_id)
            assert final_result["status"] == "exited"
            assert final_result["exit_code"] == -15
        finally:
            if session_id:
                process_registry.kill_process(session_id)
            cleanup_vm(task_id)

    def test_background_process_recovery_across_restart(self, tmp_path, monkeypatch):
        task_id = _make_task_id("vercel_live_bg_recover")
        checkpoint_path = tmp_path / "vercel-processes.json"
        command = (
            "python3 -c \"import sys,time; "
            "print('bg-ready'); sys.stdout.flush(); time.sleep(30)\""
        )
        session_id = None
        original_state = _snapshot_registry_state()
        try:
            spawn_result = _run_terminal(command, task_id, background=True, timeout=120)
            assert spawn_result["exit_code"] == 0, spawn_result
            session_id = spawn_result["session_id"]

            initial_poll = _wait_for_running_process(session_id)
            assert initial_poll["status"] == "running"
            assert "bg-ready" in initial_poll.get("output_preview", "")

            monkeypatch.setattr(process_registry_module, "CHECKPOINT_PATH", checkpoint_path)
            process_registry._write_checkpoint()
            assert checkpoint_path.exists()

            with process_registry._lock:
                process_registry._running.clear()
                process_registry._finished.clear()
            process_registry.pending_watchers.clear()

            recovered = process_registry.recover_from_checkpoint()
            assert recovered == 1
            assert session_id in process_registry._running

            recovered_poll = process_registry.poll(session_id)
            assert recovered_poll["status"] == "running"
            assert "bg-ready" in recovered_poll.get("output_preview", "")

            kill_result = process_registry.kill_process(session_id)
            assert kill_result["status"] == "killed"

            final_result = process_registry.poll(session_id)
            assert final_result["status"] == "exited"
            assert final_result["exit_code"] == -15
        finally:
            if session_id:
                process_registry.kill_process(session_id)
            _restore_registry_state(original_state)
            cleanup_vm(task_id)


class TestVercelIsolation:
    def test_different_task_ids_are_isolated(self):
        task_a = _make_task_id("vercel_live_iso_a")
        task_b = _make_task_id("vercel_live_iso_b")
        marker_path = "/tmp/hermes_vercel_isolated.txt"
        marker = uuid.uuid4().hex
        try:
            write_result = _run_terminal(
                f"printf '{marker}' > {marker_path}",
                task_a,
                timeout=120,
            )
            assert write_result["exit_code"] == 0, write_result

            read_result = _run_terminal(
                f"cat {marker_path} 2>&1 || echo NOT_FOUND",
                task_b,
                timeout=120,
            )
            assert read_result["exit_code"] == 0, read_result
            assert "NOT_FOUND" in read_result["output"] or "No such file" in read_result["output"]
            assert marker not in read_result["output"]
        finally:
            cleanup_vm(task_a)
            cleanup_vm(task_b)
