"""Tests for tools/process_registry.py — ProcessRegistry query methods, pruning, checkpoint."""

import json
import os
import signal
import subprocess
import sys
import time
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tools.environments.background_contracts import (
    DetachedCommandErrorResult,
    DetachedCommandExitedResult,
    DetachedCommandKilledResult,
    DetachedCommandRunningResult,
    DetachedCommandSpawnResult,
)
from tools.environments.local import _HERMES_PROVIDER_ENV_FORCE_PREFIX
from tools.background_process_adapters import (
    ShellEnvironmentBackgroundAdapter,
)
from tools.process_registry import (
    ProcessRegistry,
    ProcessSession,
    MAX_OUTPUT_CHARS,
    FINISHED_TTL_SECONDS,
    MAX_PROCESSES,
)


@pytest.fixture()
def registry():
    """Create a fresh ProcessRegistry."""
    return ProcessRegistry()


def _make_session(
    sid="proc_test123",
    command="echo hello",
    task_id="t1",
    exited=False,
    exit_code=None,
    output="",
    started_at=None,
) -> ProcessSession:
    """Helper to create a ProcessSession for testing."""
    s = ProcessSession(
        id=sid,
        command=command,
        task_id=task_id,
        started_at=started_at or time.time(),
        exited=exited,
        exit_code=exit_code,
        output_buffer=output,
    )
    return s


def _spawn_python_sleep(seconds: float) -> subprocess.Popen:
    """Spawn a portable short-lived Python sleep process."""
    return subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep({seconds})"],
    )


def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.05) -> bool:
    """Poll a predicate until it returns truthy or the timeout elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# =========================================================================
# Get / Poll
# =========================================================================

class TestGetAndPoll:
    def test_get_not_found(self, registry):
        assert registry.get("nonexistent") is None

    def test_get_running(self, registry):
        s = _make_session()
        registry._running[s.id] = s
        assert registry.get(s.id) is s

    def test_get_finished(self, registry):
        s = _make_session(exited=True, exit_code=0)
        registry._finished[s.id] = s
        assert registry.get(s.id) is s

    def test_poll_not_found(self, registry):
        result = registry.poll("nonexistent")
        assert result["status"] == "not_found"

    def test_poll_running(self, registry):
        s = _make_session(output="some output here")
        registry._running[s.id] = s
        result = registry.poll(s.id)
        assert result["status"] == "running"
        assert "some output" in result["output_preview"]
        assert result["command"] == "echo hello"

    def test_poll_exited(self, registry):
        s = _make_session(exited=True, exit_code=0, output="done")
        registry._finished[s.id] = s
        result = registry.poll(s.id)
        assert result["status"] == "exited"
        assert result["exit_code"] == 0


# =========================================================================
# Read log
# =========================================================================

class TestReadLog:
    def test_not_found(self, registry):
        result = registry.read_log("nonexistent")
        assert result["status"] == "not_found"

    def test_read_full_log(self, registry):
        lines = "\n".join([f"line {i}" for i in range(50)])
        s = _make_session(output=lines)
        registry._running[s.id] = s
        result = registry.read_log(s.id)
        assert result["total_lines"] == 50

    def test_read_with_limit(self, registry):
        lines = "\n".join([f"line {i}" for i in range(100)])
        s = _make_session(output=lines)
        registry._running[s.id] = s
        result = registry.read_log(s.id, limit=10)
        # Default: last 10 lines
        assert "10 lines" in result["showing"]

    def test_read_with_offset(self, registry):
        lines = "\n".join([f"line {i}" for i in range(100)])
        s = _make_session(output=lines)
        registry._running[s.id] = s
        result = registry.read_log(s.id, offset=10, limit=5)
        assert "5 lines" in result["showing"]


# =========================================================================
# Stdin helpers
# =========================================================================

class TestStdinHelpers:
    def test_close_stdin_not_found(self, registry):
        result = registry.close_stdin("nonexistent")
        assert result["status"] == "not_found"

    def test_close_stdin_pipe_mode(self, registry):
        proc = MagicMock()
        proc.stdin = MagicMock()
        s = _make_session()
        s.process = proc
        registry._running[s.id] = s

        result = registry.close_stdin(s.id)

        proc.stdin.close.assert_called_once()
        assert result["status"] == "ok"

    def test_close_stdin_pty_mode(self, registry):
        pty = MagicMock()
        s = _make_session()
        s._pty = pty
        registry._running[s.id] = s

        result = registry.close_stdin(s.id)

        pty.sendeof.assert_called_once()
        assert result["status"] == "ok"

    def test_close_stdin_allows_eof_driven_process_to_finish(self, registry, tmp_path):
        session = registry.spawn_local(
            'python3 -c "import sys; print(sys.stdin.read().strip())"',
            cwd=str(tmp_path),
            use_pty=False,
        )

        try:
            time.sleep(0.5)
            assert registry.submit_stdin(session.id, "hello")["status"] == "ok"
            assert registry.close_stdin(session.id)["status"] == "ok"

            deadline = time.time() + 5
            while time.time() < deadline:
                poll = registry.poll(session.id)
                if poll["status"] == "exited":
                    assert poll["exit_code"] == 0
                    assert "hello" in poll["output_preview"]
                    return
                time.sleep(0.2)

            pytest.fail("process did not exit after stdin was closed")
        finally:
            registry.kill_process(session.id)


# =========================================================================
# List sessions
# =========================================================================

class TestListSessions:
    def test_empty(self, registry):
        assert registry.list_sessions() == []

    def test_lists_running_and_finished(self, registry):
        s1 = _make_session(sid="proc_1", task_id="t1")
        s2 = _make_session(sid="proc_2", task_id="t1", exited=True, exit_code=0)
        registry._running[s1.id] = s1
        registry._finished[s2.id] = s2
        result = registry.list_sessions()
        assert len(result) == 2

    def test_filter_by_task_id(self, registry):
        s1 = _make_session(sid="proc_1", task_id="t1")
        s2 = _make_session(sid="proc_2", task_id="t2")
        registry._running[s1.id] = s1
        registry._running[s2.id] = s2
        result = registry.list_sessions(task_id="t1")
        assert len(result) == 1
        assert result[0]["session_id"] == "proc_1"

    def test_list_entry_fields(self, registry):
        s = _make_session(output="preview text")
        registry._running[s.id] = s
        entry = registry.list_sessions()[0]
        assert "session_id" in entry
        assert "command" in entry
        assert "status" in entry
        assert "pid" in entry
        assert "output_preview" in entry


# =========================================================================
# Active process queries
# =========================================================================

class TestActiveQueries:
    def test_has_active_processes(self, registry):
        s = _make_session(task_id="t1")
        registry._running[s.id] = s
        assert registry.has_active_processes("t1") is True
        assert registry.has_active_processes("t2") is False

    def test_has_active_for_session(self, registry):
        s = _make_session()
        s.session_key = "gw_session_1"
        registry._running[s.id] = s
        assert registry.has_active_for_session("gw_session_1") is True
        assert registry.has_active_for_session("other") is False

    def test_exited_not_active(self, registry):
        s = _make_session(task_id="t1", exited=True, exit_code=0)
        registry._finished[s.id] = s
        assert registry.has_active_processes("t1") is False


# =========================================================================
# Pruning
# =========================================================================

class TestPruning:
    def test_prune_expired_finished(self, registry):
        old_session = _make_session(
            sid="proc_old",
            exited=True,
            started_at=time.time() - FINISHED_TTL_SECONDS - 100,
        )
        registry._finished[old_session.id] = old_session
        registry._prune_if_needed()
        assert "proc_old" not in registry._finished

    def test_prune_keeps_recent(self, registry):
        recent = _make_session(sid="proc_recent", exited=True)
        registry._finished[recent.id] = recent
        registry._prune_if_needed()
        assert "proc_recent" in registry._finished

    def test_prune_over_max_removes_oldest(self, registry):
        # Fill up to MAX_PROCESSES
        for i in range(MAX_PROCESSES):
            s = _make_session(
                sid=f"proc_{i}",
                exited=True,
                started_at=time.time() - i,  # older as i increases
            )
            registry._finished[s.id] = s

        # Add one more running to trigger prune
        s = _make_session(sid="proc_new")
        registry._running[s.id] = s
        registry._prune_if_needed()

        total = len(registry._running) + len(registry._finished)
        assert total <= MAX_PROCESSES


# =========================================================================
# Spawn env sanitization
# =========================================================================

class TestSpawnEnvSanitization:
    def test_spawn_local_strips_blocked_vars_from_background_env(self, registry):
        captured = {}

        def fake_popen(cmd, **kwargs):
            captured["env"] = kwargs["env"]
            proc = MagicMock()
            proc.pid = 4321
            proc.stdout = iter([])
            proc.stdin = MagicMock()
            proc.poll.return_value = None
            return proc

        fake_thread = MagicMock()

        with patch.dict(os.environ, {
            "PATH": "/usr/bin:/bin",
            "HOME": "/home/user",
            "USER": "tester",
            "TELEGRAM_BOT_TOKEN": "bot-secret",
            "FIRECRAWL_API_KEY": "fc-secret",
        }, clear=True), \
            patch("tools.process_registry._find_shell", return_value="/bin/bash"), \
            patch("subprocess.Popen", side_effect=fake_popen), \
            patch("threading.Thread", return_value=fake_thread), \
            patch.object(registry, "_write_checkpoint"):
            registry.spawn_local(
                "echo hello",
                cwd="/tmp",
                env_vars={
                    "MY_CUSTOM_VAR": "keep-me",
                    "TELEGRAM_BOT_TOKEN": "drop-me",
                    f"{_HERMES_PROVIDER_ENV_FORCE_PREFIX}TELEGRAM_BOT_TOKEN": "forced-bot-token",
                },
            )

        env = captured["env"]
        assert env["MY_CUSTOM_VAR"] == "keep-me"
        assert env["TELEGRAM_BOT_TOKEN"] == "forced-bot-token"
        assert "FIRECRAWL_API_KEY" not in env
        assert f"{_HERMES_PROVIDER_ENV_FORCE_PREFIX}TELEGRAM_BOT_TOKEN" not in env
        assert env["PYTHONUNBUFFERED"] == "1"

# =========================================================================
# Native environment background hooks
# =========================================================================

class _NativeEnv:
    def __init__(self):
        self.execute = MagicMock()
        self.spawn_background_process = MagicMock(return_value=DetachedCommandSpawnResult(
            handle="cmd_123",
            pid=321,
            output="booting\n",
        ))
        self.poll_background_process = MagicMock(return_value=DetachedCommandRunningResult(
            output="booting\nstill running\n",
        ))
        self.wait_background_process = MagicMock(return_value=DetachedCommandExitedResult(
            exit_code=0,
            output="booting\ndone\n",
        ))
        self.kill_background_process = MagicMock(return_value=DetachedCommandKilledResult(
            output="terminated\n",
        ))
        self.serialize_background_handle = MagicMock(
            side_effect=lambda handle, *, command: type(
                "Checkpoint",
                (),
                {
                    "backend": "vercel_sandbox",
                    "command": command,
                    "to_json": staticmethod(
                        lambda: {
                            "backend": "vercel_sandbox",
                            "command": command,
                            "sandbox_id": "sb_123",
                            "command_id": "cmd_123",
                            "runtime": "python3.13",
                            "cwd": "/workspace",
                            "timeout": 30,
                            "task_id": "task-1",
                            "persistent_filesystem": True,
                            "resource_constraints": {"cpu": 1, "disk": 51200},
                        }
                    ),
                },
            )()
        )
        self.create_background_process_adapter = MagicMock(
            side_effect=lambda *, command, session_id, cwd: _EnvOwnedBackgroundAdapter(
                env=self,
                command=command,
                cwd=cwd or "",
            )
        )


class _EnvOwnedBackgroundAdapter:
    def __init__(self, *, env, command, cwd, handle=None):
        self.env = env
        self.command = command
        self.cwd = cwd
        self.handle = handle

    def spawn(self, timeout):
        result = self.env.spawn_background_process(self.command, cwd=self.cwd, timeout=timeout)
        self.handle = result.handle
        return result

    def poll(self, timeout):
        return self.env.poll_background_process(self.handle, timeout=timeout)

    def wait(self, timeout):
        return self.env.wait_background_process(self.handle, timeout=timeout)

    def kill(self, timeout):
        return self.env.kill_background_process(self.handle, timeout=timeout)

    def checkpoint_data(self):
        if self.handle is None:
            return None
        return self.env.serialize_background_handle(self.handle, command=self.command)


class TestNativeBackgroundProcesses:
    def test_spawn_via_env_uses_environment_owned_adapter(self, registry):
        env = _NativeEnv()

        with patch.object(registry, "_write_checkpoint"):
            session = registry.spawn_via_env(env, "pytest -q", cwd="/workspace", task_id="task-1")

        assert isinstance(session.background_adapter, _EnvOwnedBackgroundAdapter)
        assert session.background_adapter.handle == "cmd_123"
        assert session.pid == 321
        assert session.output_buffer.startswith("booting\n")
        assert registry.get(session.id) is session
        env.create_background_process_adapter.assert_called_once_with(
            command="pytest -q",
            session_id=session.id,
            cwd="/workspace",
        )
        env.spawn_background_process.assert_called_once_with("pytest -q", cwd="/workspace", timeout=10)
        env.execute.assert_not_called()

    def test_poll_native_process_refreshes_output(self, registry):
        env = _NativeEnv()
        session = _make_session(output="old")
        session.env_ref = env
        session.background_adapter = _EnvOwnedBackgroundAdapter(
            env=env,
            command="pytest -q",
            cwd="/workspace",
            handle="cmd_123",
        )
        registry._running[session.id] = session

        result = registry.poll(session.id)

        assert result["status"] == "running"
        assert "still running" in result["output_preview"]
        env.poll_background_process.assert_called_once_with("cmd_123", timeout=5)

    def test_wait_native_process_uses_backend_wait_hook(self, registry):
        env = _NativeEnv()
        session = _make_session(output="old")
        session.env_ref = env
        session.background_adapter = _EnvOwnedBackgroundAdapter(
            env=env,
            command="pytest -q",
            cwd="/workspace",
            handle="cmd_123",
        )
        registry._running[session.id] = session

        result = registry.wait(session.id, timeout=2)

        assert result["status"] == "exited"
        assert result["exit_code"] == 0
        assert "done" in result["output"]
        env.wait_background_process.assert_called()
        assert session.id in registry._finished

    def test_kill_native_process_uses_backend_kill_hook(self, registry):
        env = _NativeEnv()
        session = _make_session()
        session.env_ref = env
        session.background_adapter = _EnvOwnedBackgroundAdapter(
            env=env,
            command="pytest -q",
            cwd="/workspace",
            handle="cmd_123",
        )
        registry._running[session.id] = session

        result = registry.kill_process(session.id)

        assert result["status"] == "killed"
        env.kill_background_process.assert_called_once_with("cmd_123", timeout=5)
        assert session.exit_code == -15
        assert session.id in registry._finished

    def test_native_spawn_failure_is_captured(self, registry):
        env = _NativeEnv()
        env.spawn_background_process.side_effect = RuntimeError("backend unavailable")

        with patch.object(registry, "_write_checkpoint"):
            session = registry.spawn_via_env(env, "pytest -q")

        assert session.exited is True
        assert session.exit_code == -1
        assert "backend unavailable" in session.output_buffer
        assert session.id in registry._finished

    def test_ensure_background_monitor_starts_automatic_poller_for_non_local_sessions(
        self, registry, monkeypatch
    ):
        env = _NativeEnv()
        env.poll_background_process.side_effect = [
            DetachedCommandRunningResult(output="booting\nstill running\n"),
            DetachedCommandExitedResult(exit_code=0, output="booting\ndone\n"),
        ]
        monkeypatch.setattr("tools.process_registry.time.sleep", lambda _seconds: None)

        class _ImmediateThread:
            def __init__(self, *, target, args, **_kwargs):
                self._target = target
                self._args = args
                self.started = False

            def start(self):
                self.started = True
                self._target(*self._args)

            def is_alive(self):
                return False

        created_threads = []

        def _make_thread(*_args, **kwargs):
            thread = _ImmediateThread(**kwargs)
            created_threads.append(thread)
            return thread

        with patch("threading.Thread", side_effect=_make_thread), \
            patch.object(registry, "_write_checkpoint"):
            session = registry.spawn_via_env(env, "pytest -q", cwd="/workspace", task_id="task-1")
            started = registry.ensure_background_monitor(session.id)

        assert started is True
        assert created_threads
        assert session.exited is True
        assert session.exit_code == 0
        assert "done" in session.output_buffer
        assert session.id in registry._finished
        assert env.poll_background_process.call_count == 2


class TestShellBackgroundAdapterResults:
    def test_spawn_returns_typed_running_result(self):
        env = MagicMock()
        env.execute.return_value = {"output": "4321\n", "returncode": 0}
        adapter = ShellEnvironmentBackgroundAdapter(env=env, command="pytest -q", session_id="proc_123")

        result = adapter.spawn(timeout=10)

        assert isinstance(result, DetachedCommandSpawnResult)
        assert result.pid == 4321

    def test_spawn_returns_typed_error_when_pid_missing(self):
        env = MagicMock()
        env.execute.return_value = {"output": "not-a-pid\n", "returncode": 0}
        adapter = ShellEnvironmentBackgroundAdapter(env=env, command="pytest -q", session_id="proc_123")

        result = adapter.spawn(timeout=10)

        assert isinstance(result, DetachedCommandErrorResult)
        assert result.error == "Shell background spawn did not return a pid"

    def test_poll_returns_typed_exit_result(self):
        env = MagicMock()
        env.execute.side_effect = [
            {"output": "done\n", "returncode": 0},
            {"output": "1\n", "returncode": 0},
            {"output": "0\n", "returncode": 0},
        ]
        adapter = ShellEnvironmentBackgroundAdapter(
            env=env,
            command="pytest -q",
            session_id="proc_123",
            pid=4321,
            log_path="/tmp/hermes_bg_proc_123.log",
            pid_path="/tmp/hermes_bg_proc_123.pid",
        )

        result = adapter.poll(timeout=10)

        assert isinstance(result, DetachedCommandExitedResult)
        assert result.exit_code == 0

    def test_kill_returns_typed_error_when_pid_missing(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        adapter = ShellEnvironmentBackgroundAdapter(env=env, command="pytest -q", session_id="proc_123")

        result = adapter.kill(timeout=10)

        assert isinstance(result, DetachedCommandErrorResult)
        assert result.error == "Shell background kill could not resolve pid"

    def test_spawn_via_env_falls_back_to_shell_emulation_without_native_hooks(self, registry):
        class _FallbackEnv:
            def __init__(self):
                self.execute = MagicMock(return_value={"output": "4321\n", "returncode": 0})

            def create_background_process_adapter(self, *, command, session_id, cwd):
                return ShellEnvironmentBackgroundAdapter(
                    env=self,
                    command=command,
                    session_id=session_id,
                )

        env = _FallbackEnv()
        fake_thread = MagicMock()

        with patch("threading.Thread", return_value=fake_thread), \
            patch.object(registry, "_write_checkpoint"):
            session = registry.spawn_via_env(env, "pytest -q", cwd="/workspace")

        assert isinstance(session.background_adapter, ShellEnvironmentBackgroundAdapter)
        assert session.pid == 4321
        env.execute.assert_called_once()
        assert "nohup bash -c" in env.execute.call_args.args[0]

    def test_native_recovery_uses_vercel_recovery_contract(self, monkeypatch):
        recover_mock = MagicMock(return_value=("env", {"owner": "VercelSandboxEnvironment"}))
        fake_backend = type("FakeBackend", (), {"recover_background_session": recover_mock})
        monkeypatch.setattr(
            "tools.environments.vercel_sandbox.VercelSandboxEnvironment",
            fake_backend,
        )

        checkpoint = {"backend": "vercel_sandbox", "command_id": "cmd_123"}
        env, adapter = fake_backend.recover_background_session(checkpoint)

        assert env == "env"
        assert adapter == {"owner": "VercelSandboxEnvironment"}
        recover_mock.assert_called_once_with(checkpoint)


# =========================================================================
# Checkpoint
# =========================================================================

class TestCheckpoint:
    def test_write_checkpoint(self, registry, tmp_path):
        with patch("tools.process_registry.CHECKPOINT_PATH", tmp_path / "procs.json"):
            s = _make_session()
            registry._running[s.id] = s
            registry._write_checkpoint()

            data = json.loads((tmp_path / "procs.json").read_text())
            assert len(data) == 1
            assert data[0]["session_id"] == s.id

    def test_recover_no_file(self, registry, tmp_path):
        with patch("tools.process_registry.CHECKPOINT_PATH", tmp_path / "missing.json"):
            assert registry.recover_from_checkpoint() == 0

    def test_recover_dead_pid(self, registry, tmp_path):
        checkpoint = tmp_path / "procs.json"
        checkpoint.write_text(json.dumps([{
            "session_id": "proc_dead",
            "command": "sleep 999",
            "pid": 999999999,  # almost certainly not running
            "task_id": "t1",
        }]))
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            recovered = registry.recover_from_checkpoint()
            assert recovered == 0

    def test_write_checkpoint_includes_watcher_metadata(self, registry, tmp_path):
        with patch("tools.process_registry.CHECKPOINT_PATH", tmp_path / "procs.json"):
            s = _make_session()
            s.watcher_platform = "telegram"
            s.watcher_chat_id = "999"
            s.watcher_user_id = "u123"
            s.watcher_user_name = "alice"
            s.watcher_thread_id = "42"
            s.watcher_interval = 60
            registry._running[s.id] = s
            registry._write_checkpoint()

            data = json.loads((tmp_path / "procs.json").read_text())
            assert len(data) == 1
            assert data[0]["watcher_platform"] == "telegram"
            assert data[0]["watcher_chat_id"] == "999"
            assert data[0]["watcher_user_id"] == "u123"
            assert data[0]["watcher_user_name"] == "alice"
            assert data[0]["watcher_thread_id"] == "42"
            assert data[0]["watcher_interval"] == 60

    def test_write_checkpoint_includes_native_background_checkpoint(self, registry, tmp_path):
        with patch("tools.process_registry.CHECKPOINT_PATH", tmp_path / "procs.json"):
            env = _NativeEnv()
            session = _make_session()
            session.env_ref = env
            session.background_adapter = _EnvOwnedBackgroundAdapter(
                env=env,
                command="pytest -q",
                cwd="/workspace",
                handle="cmd_123",
            )
            registry._running[session.id] = session

            registry._write_checkpoint()

            data = json.loads((tmp_path / "procs.json").read_text())
            assert data[0]["background_checkpoint"]["backend"] == "vercel_sandbox"
            assert data[0]["background_checkpoint"]["command"] == "pytest -q"
            assert data[0]["background_checkpoint"]["command_id"] == "cmd_123"
            assert data[0]["background_checkpoint"]["resource_constraints"] == {
                "cpu": 1,
                "disk": 51200,
            }

    def test_recover_enqueues_watchers(self, registry, tmp_path):
        checkpoint = tmp_path / "procs.json"
        checkpoint.write_text(json.dumps([{
            "session_id": "proc_live",
            "command": "sleep 999",
            "pid": os.getpid(),  # current process — guaranteed alive
            "task_id": "t1",
            "session_key": "sk1",
            "watcher_platform": "telegram",
            "watcher_chat_id": "123",
            "watcher_user_id": "u123",
            "watcher_user_name": "alice",
            "watcher_thread_id": "42",
            "watcher_interval": 60,
        }]))
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            recovered = registry.recover_from_checkpoint()
            assert recovered == 1
            assert len(registry.pending_watchers) == 1
            w = registry.pending_watchers[0]
            assert w["session_id"] == "proc_live"
            assert w["platform"] == "telegram"
            assert w["chat_id"] == "123"
            assert w["user_id"] == "u123"
            assert w["user_name"] == "alice"
            assert w["thread_id"] == "42"
            assert w["check_interval"] == 60

    def test_recover_skips_watcher_when_no_interval(self, registry, tmp_path):
        checkpoint = tmp_path / "procs.json"
        checkpoint.write_text(json.dumps([{
            "session_id": "proc_live",
            "command": "sleep 999",
            "pid": os.getpid(),
            "task_id": "t1",
            "watcher_interval": 0,
        }]))
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            recovered = registry.recover_from_checkpoint()
            assert recovered == 1
            assert len(registry.pending_watchers) == 0

    def test_recovery_keeps_live_checkpoint_entries(self, registry, tmp_path):
        checkpoint = tmp_path / "procs.json"
        checkpoint.write_text(json.dumps([{
            "session_id": "proc_live",
            "command": "sleep 999",
            "pid": os.getpid(),
            "task_id": "t1",
            "session_key": "sk1",
        }]))

        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            recovered = registry.recover_from_checkpoint()
            assert recovered == 1
            assert registry.get("proc_live") is not None

            data = json.loads(checkpoint.read_text())
            assert len(data) == 1
            assert data[0]["session_id"] == "proc_live"
            assert data[0]["pid"] == os.getpid()
            assert data != []

    def test_recovery_skips_explicit_sandbox_backed_entries(self, registry, tmp_path):
        checkpoint = tmp_path / "procs.json"
        original = [{
            "session_id": "proc_remote",
            "command": "sleep 999",
            "pid": os.getpid(),
            "task_id": "t1",
            "pid_scope": "sandbox",
        }]
        checkpoint.write_text(json.dumps(original))

        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            recovered = registry.recover_from_checkpoint()
            assert recovered == 0
            assert registry.get("proc_remote") is None

            data = json.loads(checkpoint.read_text())
            assert data == []

    def test_detached_recovered_process_eventually_exits(self, registry, tmp_path):
        proc = _spawn_python_sleep(0.4)
        checkpoint = tmp_path / "procs.json"
        checkpoint.write_text(json.dumps([{
            "session_id": "proc_live",
            "command": "python -c 'import time; time.sleep(0.4)'",
            "pid": proc.pid,
            "task_id": "t1",
            "session_key": "sk1",
        }]))

        try:
            with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
                recovered = registry.recover_from_checkpoint()
                assert recovered == 1

                session = registry.get("proc_live")
                assert session is not None
                assert session.detached is True

                proc.wait(timeout=5)

                assert _wait_until(
                    lambda: registry.get("proc_live") is not None
                    and registry.get("proc_live").exited,
                    timeout=5,
                )

                poll_result = registry.poll("proc_live")
                assert poll_result["status"] == "exited"

                wait_result = registry.wait("proc_live", timeout=1)
                assert wait_result["status"] == "exited"
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
                    proc.wait(timeout=5)

    def test_recover_native_background_process(self, registry, tmp_path):
        checkpoint = tmp_path / "procs.json"
        checkpoint.write_text(json.dumps([{
            "session_id": "proc_native",
            "command": "python server.py",
            "cwd": "/workspace",
            "task_id": "task-1",
            "session_key": "sk1",
            "watcher_platform": "telegram",
            "watcher_chat_id": "123",
            "watcher_thread_id": "42",
            "watcher_interval": 60,
                "background_checkpoint": {
                    "backend": "vercel_sandbox",
                    "command": "python server.py",
                    "sandbox_id": "sb_123",
                    "command_id": "cmd_123",
                    "runtime": "python3.13",
                    "cwd": "/workspace",
                    "timeout": 30,
                    "task_id": "task-1",
                    "persistent_filesystem": True,
                    "resource_constraints": {"cpu": 1, "disk": 51200},
                },
            }]))
        recovered_adapter = _EnvOwnedBackgroundAdapter(
            env=MagicMock(),
            command="python server.py",
            cwd="/workspace",
            handle="cmd_123",
        )
        recovered_env = recovered_adapter.env
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint), \
            patch(
                "tools.environments.recovery_registry.VercelSandboxEnvironment.recover_background_session",
                return_value=(recovered_env, recovered_adapter),
            ) as recover_mock:
            recovered = registry.recover_from_checkpoint()

        assert recovered == 1
        recover_mock.assert_called_once()
        session = registry.get("proc_native")
        assert session is not None
        assert session.background_adapter is recovered_adapter
        assert session.env_ref is recovered_env
        assert len(registry.pending_watchers) == 1

    def test_recover_native_background_process_falls_back_to_pid_recovery(self, registry, tmp_path):
        checkpoint = tmp_path / "procs.json"
        checkpoint.write_text(json.dumps([{
            "session_id": "proc_native_fallback",
            "command": "sleep 999",
            "pid": os.getpid(),
            "task_id": "task-1",
            "background_checkpoint": {
                "backend": "unsupported_backend",
            },
        }]))

        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint), \
            patch("tools.process_registry.logger") as logger_mock:
            recovered = registry.recover_from_checkpoint()

        assert recovered == 1
        session = registry.get("proc_native_fallback")
        assert session is not None
        assert session.detached is True
        assert session.background_adapter is None
        logger_mock.warning.assert_called_once()

    def test_recover_invalid_native_background_checkpoint_falls_back_to_pid_recovery(
        self, registry, tmp_path
    ):
        checkpoint = tmp_path / "procs.json"
        checkpoint.write_text(json.dumps([{
            "session_id": "proc_native_invalid",
            "command": "sleep 999",
            "pid": os.getpid(),
            "task_id": "task-1",
            "background_checkpoint": {
                "sandbox_id": "sb_missing_backend",
            },
        }]))

        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint), \
            patch("tools.process_registry.logger") as logger_mock:
            recovered = registry.recover_from_checkpoint()

        assert recovered == 1
        session = registry.get("proc_native_invalid")
        assert session is not None
        assert session.detached is True
        assert session.background_adapter is None
        logger_mock.warning.assert_called_once()

    def test_recover_native_background_process_with_missing_command_falls_back_to_pid_recovery(
        self, registry, tmp_path
    ):
        checkpoint = tmp_path / "procs.json"
        checkpoint.write_text(json.dumps([{
            "session_id": "proc_native_missing_command",
            "command": "sleep 999",
            "pid": os.getpid(),
            "task_id": "task-1",
            "background_checkpoint": {
                "backend": "vercel_sandbox",
                "sandbox_id": "sb_123",
                "command_id": "cmd_123",
                "cwd": "/workspace",
                "timeout": 30,
                "task_id": "task-1",
                "persistent_filesystem": True,
                "resource_constraints": {"cpu": 1, "disk": 51200},
            },
        }]))

        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint), \
            patch("tools.process_registry.logger") as logger_mock:
            recovered = registry.recover_from_checkpoint()

        assert recovered == 1
        session = registry.get("proc_native_missing_command")
        assert session is not None
        assert session.detached is True
        assert session.background_adapter is None
        logger_mock.warning.assert_called_once()


# =========================================================================
# Kill process
# =========================================================================

class TestKillProcess:
    def test_kill_not_found(self, registry):
        result = registry.kill_process("nonexistent")
        assert result["status"] == "not_found"

    def test_kill_already_exited(self, registry):
        s = _make_session(exited=True, exit_code=0)
        registry._finished[s.id] = s
        result = registry.kill_process(s.id)
        assert result["status"] == "already_exited"

    def test_kill_detached_session_uses_host_pid(self, registry):
        s = _make_session(sid="proc_detached", command="sleep 999")
        s.pid = 424242
        s.detached = True
        registry._running[s.id] = s

        calls = []

        def fake_kill(pid, sig):
            calls.append((pid, sig))

        try:
            with patch("tools.process_registry.os.kill", side_effect=fake_kill):
                result = registry.kill_process(s.id)

            assert result["status"] == "killed"
            assert (424242, 0) in calls
            assert (424242, signal.SIGTERM) in calls
        finally:
            registry._running.pop(s.id, None)


# =========================================================================
# Tool handler
# =========================================================================

class TestProcessToolHandler:
    def test_list_action(self):
        from tools.process_registry import _handle_process
        result = json.loads(_handle_process({"action": "list"}))
        assert "processes" in result

    def test_poll_missing_session_id(self):
        from tools.process_registry import _handle_process
        result = json.loads(_handle_process({"action": "poll"}))
        assert "error" in result

    def test_unknown_action(self):
        from tools.process_registry import _handle_process
        result = json.loads(_handle_process({"action": "unknown_action"}))
        assert "error" in result
