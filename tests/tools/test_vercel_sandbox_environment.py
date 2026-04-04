"""Unit tests for the Vercel Sandbox terminal backend."""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.env_passthrough import clear_env_passthrough, register_env_passthrough
from tools.environments.background_contracts import (
    DetachedCommandExitedResult,
    DetachedCommandKilledResult,
    DetachedCommandRunningResult,
    DetachedCommandSpawnResult,
)


# ---------------------------------------------------------------------------
# Helpers to model the Vercel SDK
# ---------------------------------------------------------------------------


class _FakeCommand:
    def __init__(self, cmd_id: str, exit_code=None, output=""):
        self.cmd_id = cmd_id
        self.cmd = SimpleNamespace(exit_code=exit_code)
        self._output = output
        self.kill_calls = 0

    def output(self):
        return self._output

    def wait(self):
        while self.cmd.exit_code is None:
            time.sleep(0.01)
        return self

    def kill(self):
        self.kill_calls += 1
        if self.cmd.exit_code is None:
            self.cmd.exit_code = -15


class _FakeSandbox:
    def __init__(
        self,
        *,
        cwd="/workspace",
        timeout=900000,
        status="running",
        snapshot_value="snap_123",
    ):
        self.sandbox = SimpleNamespace(cwd=cwd, timeout=timeout, id="sb_123")
        self.status = status
        self.closed = 0
        self.client = SimpleNamespace(close=self._close)
        self.created_with = {}
        self.run_command_calls = []
        self.run_command_detached_calls = []
        self.get_command_calls = []
        self.extend_timeout_calls = []
        self.write_files_calls = []
        self.stop_calls = []
        self.refresh_calls = 0
        self.snapshot_calls = 0
        self.snapshot_value = snapshot_value
        self.run_command_detached_side_effects = []
        self.refresh_side_effects = []
        self._detached = _FakeCommand("cmd_fg", exit_code=0, output="done\n")

    @property
    def timeout(self):
        return self.sandbox.timeout

    def _close(self):
        self.closed += 1

    def refresh(self):
        self.refresh_calls += 1
        if self.refresh_side_effects:
            effect = self.refresh_side_effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            if callable(effect):
                effect(self)
        return None

    def run_command(self, cmd, args=None, *, cwd=None, env=None, sudo=False):
        self.run_command_calls.append((cmd, args, cwd, env, sudo))
        return _FakeCommand("cmd_home", exit_code=0, output="/remote-home")

    def run_command_detached(self, cmd, args=None, *, cwd=None, env=None, sudo=False):
        self.run_command_detached_calls.append((cmd, args, cwd, env, sudo))
        if self.run_command_detached_side_effects:
            effect = self.run_command_detached_side_effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            if callable(effect):
                return effect(cmd, args, cwd, env, sudo)
            return effect
        return self._detached

    def get_command(self, cmd_id):
        self.get_command_calls.append(cmd_id)
        return self._detached

    def extend_timeout(self, duration):
        self.extend_timeout_calls.append(duration)
        self.sandbox.timeout += duration

    def write_files(self, files):
        self.write_files_calls.append(files)

    def stop(self, *, blocking=False, timeout=30.0, poll_interval=0.5):
        self.stop_calls.append((blocking, timeout, poll_interval))

    def snapshot(self):
        self.snapshot_calls += 1
        if isinstance(self.snapshot_value, Exception):
            raise self.snapshot_value
        return self.snapshot_value


class _FakeSDK:
    def __init__(self):
        self.boxes: list[_FakeSandbox] = []
        self.create_kwargs: list[dict] = []
        self.create_side_effects: list[object] = []

    @property
    def current(self) -> _FakeSandbox:
        return self.boxes[-1]

    def create(self, **kwargs):
        self.create_kwargs.append(kwargs)
        if self.create_side_effects:
            effect = self.create_side_effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            if isinstance(effect, _FakeSandbox):
                effect.created_with = kwargs
                self.boxes.append(effect)
                return effect
        box = _FakeSandbox()
        box.created_with = kwargs
        self.boxes.append(box)
        return box


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_passthrough_state():
    clear_env_passthrough()
    yield
    clear_env_passthrough()


@pytest.fixture()
def vercel_sdk(monkeypatch):
    fake_sdk = _FakeSDK()
    sandbox_cls = types.SimpleNamespace(create=fake_sdk.create)
    sandbox_mod = types.ModuleType("vercel.sandbox")
    sandbox_mod.Sandbox = sandbox_cls
    sandbox_mod.SandboxAuthError = type("SandboxAuthError", (Exception,), {})
    sandbox_mod.SandboxPermissionError = type("SandboxPermissionError", (Exception,), {})
    sandbox_mod.SandboxRateLimitError = type("SandboxRateLimitError", (Exception,), {})
    sandbox_mod.SandboxServerError = type("SandboxServerError", (Exception,), {})
    vercel_mod = types.ModuleType("vercel")
    vercel_mod.sandbox = sandbox_mod
    monkeypatch.setitem(sys.modules, "vercel", vercel_mod)
    monkeypatch.setitem(sys.modules, "vercel.sandbox", sandbox_mod)
    return fake_sdk


@pytest.fixture()
def vercel_module(vercel_sdk, monkeypatch, tmp_path):
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)
    from tools.environments import vercel_sandbox as module

    monkeypatch.setattr(module, "get_credential_file_mounts", lambda: [])
    monkeypatch.setattr(module, "iter_skills_files", lambda **kwargs: [])
    monkeypatch.setattr(module, "_SNAPSHOT_STORE", Path(tmp_path) / "vercel_sandbox_snapshots.json")
    return module


@pytest.fixture()
def make_env(vercel_module):
    def _factory(**kwargs):
        kwargs.setdefault("runtime", "node22")
        kwargs.setdefault("cwd", "/")
        kwargs.setdefault("timeout", 30)
        kwargs.setdefault("task_id", "task-123")
        return vercel_module.VercelSandboxEnvironment(**kwargs)

    return _factory


@pytest.fixture()
def env(make_env):
    return make_env()


# ---------------------------------------------------------------------------
# Constructor / cwd resolution
# ---------------------------------------------------------------------------


class TestCwdResolution:
    def test_default_root_uses_remote_cwd(self, env, vercel_sdk):
        assert env.cwd == "/workspace"
        assert vercel_sdk.current.run_command_calls[0][0] == "sh"


class TestConstructor:
    def test_restores_from_saved_snapshot(self, vercel_sdk, vercel_module, make_env):
        vercel_module._store_snapshot("task-restore", "snap_saved")

        make_env(task_id="task-restore", persistent_filesystem=True)

        assert vercel_sdk.create_kwargs[0]["source"] == {
            "type": "snapshot",
            "snapshot_id": "snap_saved",
        }

    def test_maps_resources_to_vercel_shape(self, vercel_sdk, make_env):
        make_env(task_id="task-resources", cpu=2, memory=4096)

        assert vercel_sdk.create_kwargs[0]["resources"] == {
            "vcpus": 2,
            "memory": 4096,
        }

    def test_warns_and_ignores_explicit_memory_input(self, vercel_sdk, make_env, caplog):
        make_env(task_id="task-memory-ignored", cpu=2, memory=6144)

        assert vercel_sdk.create_kwargs[0]["resources"] == {
            "vcpus": 2,
            "memory": 4096,
        }
        assert "ignores requested memory=6144 MB" in caplog.text

    def test_defaults_memory_to_2048_per_vcpu(self, vercel_sdk, make_env):
        make_env(task_id="task-default-memory")

        assert vercel_sdk.create_kwargs[0]["resources"] == {
            "vcpus": 1,
            "memory": 2048,
        }

    def test_normalizes_shared_generic_memory_default(self, vercel_sdk, make_env):
        make_env(task_id="task-normalized-memory", cpu=2, memory=5120)

        assert vercel_sdk.create_kwargs[0]["resources"] == {
            "vcpus": 2,
            "memory": 4096,
        }

    def test_treats_shared_default_disk_as_unset(self, vercel_sdk, make_env):
        make_env(task_id="task-default-disk", disk=51200)

        assert vercel_sdk.create_kwargs[0]["resources"] == {
            "vcpus": 1,
            "memory": 2048,
        }

    def test_caps_non_default_disk_request_to_shared_default(self, make_env, vercel_sdk, caplog):
        make_env(task_id="task-disk-request", disk=8192)

        assert vercel_sdk.create_kwargs[0]["resources"] == {
            "vcpus": 1,
            "memory": 2048,
        }
        assert "does not support configurable container_disk=8192" in caplog.text

    def test_normalizes_fractional_vcpu_to_default_with_warning(self, make_env, vercel_sdk, caplog):
        make_env(task_id="task-fractional-cpu", cpu=1.5)

        assert vercel_sdk.create_kwargs[0]["resources"] == {
            "vcpus": 1,
            "memory": 2048,
        }
        assert "is not a whole number; using default 1 vCPU" in caplog.text

    def test_waits_for_running_status(self, vercel_sdk, make_env, monkeypatch):
        pending = _FakeSandbox(status="pending")
        pending.refresh_side_effects.append(lambda box: setattr(box, "status", "running"))
        vercel_sdk.create_side_effects.append(pending)
        monotonic = iter([0.0, 0.1, 0.2, 0.3])
        monkeypatch.setattr("tools.environments.vercel_sandbox.time.monotonic", lambda: next(monotonic))
        monkeypatch.setattr("tools.environments.vercel_sandbox.time.sleep", lambda _s: None)

        make_env(task_id="task-pending")

        assert pending.refresh_calls >= 1
        assert pending.status == "running"


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------


class TestExecute:
    def test_runs_shell_command_and_returns_output(self, env, vercel_sdk):
        vercel_sdk.current._detached = _FakeCommand("cmd_exec", exit_code=0, output="hello\n")
        vercel_sdk.current.sandbox.timeout = 1000

        result = env.execute("echo hello", cwd="/tmp", timeout=45)

        assert result == {"output": "hello\n", "returncode": 0}
        cmd, args, cwd, _, _ = vercel_sdk.current.run_command_detached_calls[-1]
        assert cmd == "sh"
        assert args == ["-lc", "echo hello"]
        assert cwd == "/tmp"
        assert vercel_sdk.current.extend_timeout_calls

    def test_wraps_stdin_as_heredoc(self, env, vercel_sdk):
        vercel_sdk.current._detached = _FakeCommand("cmd_stdin", exit_code=0, output="stdin ok\n")

        env.execute("python3", stdin_data="print('hi')")

        args = vercel_sdk.current.run_command_detached_calls[-1][1]
        assert "HERMES_EOF_" in args[1]
        assert "print('hi')" in args[1]

    def test_retries_one_transient_detached_startup_failure(self, env, vercel_sdk):
        retry_command = _FakeCommand("cmd_retry", exit_code=0, output="ok\n")

        def _retry_success(_cmd, _args, _cwd, _env, _sudo):
            vercel_sdk.current._detached = retry_command
            return retry_command

        vercel_sdk.current.run_command_detached_side_effects = [
            RuntimeError("transient"),
            _retry_success,
        ]

        result = env.execute("echo retry")

        assert result == {"output": "ok\n", "returncode": 0}
        assert vercel_sdk.current.refresh_calls >= 1
        assert len(vercel_sdk.current.run_command_detached_calls) == 2

    def test_forwards_only_allowlisted_passthrough_env(self, env, vercel_sdk, monkeypatch):
        from tools.environments.local import _HERMES_PROVIDER_ENV_BLOCKLIST

        blocked_var = next(iter(_HERMES_PROVIDER_ENV_BLOCKLIST))
        register_env_passthrough([blocked_var])
        monkeypatch.setenv(blocked_var, "allowed-secret")
        monkeypatch.setenv("OPENAI_API_KEY", "blocked-secret")

        env.execute("echo hello")

        forwarded_env = vercel_sdk.current.run_command_detached_calls[-1][3]
        assert forwarded_env[blocked_var] == "allowed-secret"
        assert "OPENAI_API_KEY" not in forwarded_env

    def test_timeout_kills_command(self, vercel_sdk, make_env, monkeypatch):
        env = make_env(timeout=1, task_id="task-timeout")
        never_finishes = _FakeCommand("cmd_timeout", exit_code=None, output="")
        vercel_sdk.current._detached = never_finishes
        monkeypatch.setattr("tools.environments.vercel_sandbox.time.sleep", lambda _s: None)
        monotonic = iter([0.0, 0.2, 1.2])
        monkeypatch.setattr("tools.environments.vercel_sandbox.time.monotonic", lambda: next(monotonic))

        result = env.execute("sleep 10", timeout=1)

        assert result["returncode"] == 124
        assert never_finishes.kill_calls == 1

    def test_interrupt_kills_command(self, vercel_sdk, make_env, monkeypatch):
        env = make_env(timeout=10, task_id="task-interrupt")
        long_running = _FakeCommand("cmd_interrupt", exit_code=None, output="")
        vercel_sdk.current._detached = long_running
        interrupted = iter([False, True])
        monkeypatch.setattr("tools.environments.vercel_sandbox.is_interrupted", lambda: next(interrupted))
        monkeypatch.setattr("tools.environments.vercel_sandbox.time.sleep", lambda _s: None)

        result = env.execute("sleep 10", timeout=10)

        assert result["returncode"] == 130
        assert long_running.kill_calls == 1


# ---------------------------------------------------------------------------
# Background processes
# ---------------------------------------------------------------------------


class TestBackgroundProcesses:
    def test_background_lifecycle_uses_native_handles(self, env, vercel_sdk, monkeypatch):
        from tools.environments.local import _HERMES_PROVIDER_ENV_BLOCKLIST

        blocked_var = next(iter(_HERMES_PROVIDER_ENV_BLOCKLIST))
        register_env_passthrough([blocked_var])
        monkeypatch.setenv(blocked_var, "bg-secret")
        monkeypatch.setenv("OPENAI_API_KEY", "blocked-secret")

        running = _FakeCommand("cmd_bg", exit_code=None, output="booting\n")
        vercel_sdk.current._detached = running

        spawned = env.spawn_background_process("python server.py", cwd="/srv")
        assert isinstance(spawned, DetachedCommandSpawnResult)
        handle = spawned.handle
        forwarded_env = vercel_sdk.current.run_command_detached_calls[-1][3]
        assert forwarded_env[blocked_var] == "bg-secret"
        assert "OPENAI_API_KEY" not in forwarded_env

        poll_result = env.poll_background_process(handle)
        assert isinstance(poll_result, DetachedCommandRunningResult)
        assert poll_result.output == "booting\n"

        running.cmd.exit_code = 0
        running._output = "booting\nready\n"
        monkeypatch.setattr("tools.environments.vercel_sandbox.time.sleep", lambda _s: None)
        wait_result = env.wait_background_process(handle, timeout=1)
        assert isinstance(wait_result, DetachedCommandExitedResult)
        assert wait_result.exit_code == 0

        kill_result = env.kill_background_process(handle)
        assert isinstance(kill_result, DetachedCommandKilledResult)
        assert running.kill_calls == 1

    def test_poll_background_process_refreshes_sandbox_timeout(self, env, vercel_sdk):
        running = _FakeCommand("cmd_bg", exit_code=None, output="booting\n")
        vercel_sdk.current._detached = running

        handle = env.spawn_background_process("python server.py", cwd="/srv").handle
        vercel_sdk.current.sandbox.timeout = 1000
        extend_call_count = len(vercel_sdk.current.extend_timeout_calls)
        refresh_call_count = vercel_sdk.current.refresh_calls

        result = env.poll_background_process(handle, timeout=45)

        assert isinstance(result, DetachedCommandRunningResult)
        assert vercel_sdk.current.refresh_calls == refresh_call_count + 1
        assert len(vercel_sdk.current.extend_timeout_calls) == extend_call_count + 1

    def test_wait_background_process_refreshes_sandbox_timeout_while_running(
        self, env, vercel_sdk, monkeypatch
    ):
        running = _FakeCommand("cmd_bg", exit_code=None, output="booting\n")
        vercel_sdk.current._detached = running

        handle = env.spawn_background_process("python server.py", cwd="/srv").handle
        vercel_sdk.current.sandbox.timeout = 1000
        extend_call_count = len(vercel_sdk.current.extend_timeout_calls)
        refresh_call_count = vercel_sdk.current.refresh_calls

        def _sleep(_seconds):
            running.cmd.exit_code = 0
            running._output = "booting\ndone\n"

        monkeypatch.setattr("tools.environments.vercel_sandbox.time.sleep", _sleep)

        result = env.wait_background_process(handle, timeout=1)

        assert isinstance(result, DetachedCommandExitedResult)
        assert vercel_sdk.current.refresh_calls >= refresh_call_count + 2
        assert len(vercel_sdk.current.extend_timeout_calls) >= extend_call_count + 1

    def test_background_handle_includes_sandbox_and_command_identity(self, env, vercel_sdk):
        running = _FakeCommand("cmd_bg", exit_code=None, output="booting\n")
        vercel_sdk.current._detached = running

        spawned = env.spawn_background_process("python server.py", cwd="/srv")
        handle = spawned.handle

        assert handle.sandbox_id == "sb_123"
        assert handle.command_id == "cmd_bg"

    def test_serialize_background_handle_captures_recovery_metadata(self, env, vercel_sdk):
        running = _FakeCommand("cmd_bg", exit_code=None, output="booting\n")
        vercel_sdk.current._detached = running
        handle = env.spawn_background_process("python server.py", cwd="/srv").handle

        checkpoint = env.serialize_background_handle(handle, command="python server.py")

        assert checkpoint.to_json() == {
            "backend": "vercel_sandbox",
            "command": "python server.py",
            "sandbox_id": "sb_123",
            "command_id": "cmd_bg",
            "runtime": "node22",
            "cwd": "/workspace",
            "timeout": 30,
            "task_id": "task-123",
            "persistent_filesystem": True,
            "resource_constraints": {"cpu": 1, "disk": 51200},
        }

    def test_create_background_process_adapter_returns_vercel_direct_adapter(self, env):
        adapter = env.create_background_process_adapter(
            command="python server.py",
            session_id="proc_123",
            cwd="/srv",
        )

        assert adapter.__class__.__name__ == "VercelBackgroundProcessAdapter"
        assert adapter.env is env
        assert adapter.command == "python server.py"
        assert adapter.cwd == "/srv"

    def test_recover_background_handle_reattaches_to_existing_sandbox(
        self, vercel_sdk, vercel_module, monkeypatch
    ):
        recovered_sandbox = _FakeSandbox(cwd="/workspace")
        recovered_sandbox.sandbox.id = "sb_reattach"
        recovered_sandbox._detached = _FakeCommand("cmd_recovered", exit_code=None, output="booting\n")

        sandbox_get_calls = []

        def fake_get(*, sandbox_id):
            sandbox_get_calls.append(sandbox_id)
            return recovered_sandbox

        monkeypatch.setattr(
            sys.modules["vercel.sandbox"].Sandbox,
            "get",
            staticmethod(fake_get),
            raising=False,
        )

        env, handle = vercel_module.VercelSandboxEnvironment.recover_background_handle(
            vercel_module.VercelBackgroundCheckpoint.from_json(
                {
                    "backend": "vercel_sandbox",
                    "command": "python server.py",
                    "sandbox_id": "sb_reattach",
                    "command_id": "cmd_recovered",
                    "runtime": "python3.13",
                    "cwd": "/workspace",
                    "timeout": 45,
                    "task_id": "task-restore",
                    "persistent_filesystem": True,
                    "resource_constraints": {"cpu": 1, "disk": 51200},
                }
            )
        )

        assert sandbox_get_calls == ["sb_reattach"]
        assert env._sandbox is recovered_sandbox
        assert handle.sandbox_id == "sb_reattach"
        assert handle.command_id == "cmd_recovered"

    def test_recover_background_session_returns_matching_env_and_adapter(
        self, vercel_sdk, vercel_module, monkeypatch
    ):
        recovered_sandbox = _FakeSandbox(cwd="/workspace")
        recovered_sandbox.sandbox.id = "sb_reattach"
        recovered_sandbox._detached = _FakeCommand("cmd_recovered", exit_code=None, output="booting\n")

        monkeypatch.setattr(
            sys.modules["vercel.sandbox"].Sandbox,
            "get",
            staticmethod(lambda *, sandbox_id: recovered_sandbox),
            raising=False,
        )

        env, adapter = vercel_module.VercelSandboxEnvironment.recover_background_session(
            vercel_module.VercelBackgroundCheckpoint.from_json(
                {
                    "backend": "vercel_sandbox",
                    "command": "python server.py",
                    "sandbox_id": "sb_reattach",
                    "command_id": "cmd_recovered",
                    "runtime": "python3.13",
                    "cwd": "/workspace",
                    "timeout": 45,
                    "task_id": "task-restore",
                    "persistent_filesystem": True,
                    "resource_constraints": {"cpu": 1, "disk": 51200},
                }
            )
        )

        assert adapter.__class__.__name__ == "VercelBackgroundProcessAdapter"
        assert env._sandbox is recovered_sandbox
        assert adapter.env is env
        assert adapter.handle.command_id == "cmd_recovered"

    def test_recover_background_handle_uses_structured_resource_constraints(
        self, vercel_sdk, vercel_module, monkeypatch
    ):
        recovered_sandbox = _FakeSandbox(cwd="/workspace")
        recovered_sandbox.sandbox.id = "sb_legacy"
        recovered_sandbox._detached = _FakeCommand("cmd_legacy", exit_code=None, output="booting\n")

        monkeypatch.setattr(
            sys.modules["vercel.sandbox"].Sandbox,
            "get",
            staticmethod(lambda *, sandbox_id: recovered_sandbox),
            raising=False,
        )

        env, handle = vercel_module.VercelSandboxEnvironment.recover_background_handle(
            vercel_module.VercelBackgroundCheckpoint.from_json(
                {
                    "backend": "vercel_sandbox",
                    "command": "python worker.py",
                    "sandbox_id": "sb_legacy",
                    "command_id": "cmd_legacy",
                    "runtime": "python3.13",
                    "cwd": "/workspace",
                    "timeout": 45,
                    "task_id": "task-restore",
                    "persistent_filesystem": True,
                    "resource_constraints": {"cpu": 2, "disk": 51200},
                }
            )
        )

        assert env._resource_constraints.to_checkpoint() == {"cpu": 2, "disk": 51200}
        assert env._resource_constraints.to_vercel_resources() == {"vcpus": 2, "memory": 4096}
        assert handle.command_id == "cmd_legacy"

    def test_checkpoint_parse_rejects_missing_command(self, vercel_module):
        with pytest.raises(ValueError, match="missing command"):
            vercel_module.VercelBackgroundCheckpoint.from_json(
                {
                    "backend": "vercel_sandbox",
                    "sandbox_id": "sb_legacy",
                    "command_id": "cmd_legacy",
                    "runtime": "python3.13",
                    "cwd": "/workspace",
                    "timeout": 45,
                    "task_id": "task-restore",
                    "persistent_filesystem": True,
                    "resource_constraints": {"cpu": 2, "disk": 51200},
                }
            )


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_saves_snapshot_when_persistent(self, vercel_sdk, vercel_module, make_env):
        env = make_env(task_id="task-persist", persistent_filesystem=True)

        env.cleanup()

        assert vercel_module._get_snapshot_restore_candidate("task-persist") == "snap_123"
        assert vercel_sdk.current.stop_calls == [(True, 15.0, 0.5)]

    def test_stops_and_clears_snapshot_when_snapshotting_fails(
        self, vercel_sdk, vercel_module, make_env
    ):
        vercel_module._store_snapshot("task-persist", "snap_old")
        sandbox = _FakeSandbox(snapshot_value=RuntimeError("snapshot failed"))
        vercel_sdk.create_side_effects.append(sandbox)

        env = make_env(task_id="task-persist", persistent_filesystem=True)

        env.cleanup()

        assert vercel_module._get_snapshot_restore_candidate("task-persist") is None
        assert sandbox.stop_calls == [(True, 15.0, 0.5)]
        assert sandbox.closed == 1

    def test_deletes_stale_snapshot_when_ephemeral(self, vercel_sdk, vercel_module, make_env):
        vercel_module._store_snapshot("task-ephemeral", "snap_old")

        env = make_env(task_id="task-ephemeral", persistent_filesystem=False)

        env.cleanup()

        assert vercel_module._get_snapshot_restore_candidate("task-ephemeral") is None
        assert vercel_sdk.current.snapshot_calls == 0
