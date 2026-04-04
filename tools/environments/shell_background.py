"""Shell-backed background adapter owned by the environment subsystem."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any

from tools.environments.background_contracts import (
    DetachedCommandErrorResult,
    DetachedCommandExitedResult,
    DetachedCommandKilledResult,
    DetachedCommandResult,
    DetachedCommandRunningResult,
    DetachedCommandSpawnResult,
)


@dataclass
class ShellEnvironmentBackgroundAdapter:
    """Adapter for the existing PID/log-file shell emulation path."""

    env: Any
    command: str
    session_id: str
    pid: int | None = None
    log_path: str = ""
    pid_path: str = ""

    def spawn(self, timeout: int) -> DetachedCommandResult:
        self.log_path = f"/tmp/hermes_bg_{self.session_id}.log"
        self.pid_path = f"/tmp/hermes_bg_{self.session_id}.pid"
        quoted_command = shlex.quote(self.command)
        bg_command = (
            f"nohup bash -c {quoted_command} > {self.log_path} 2>&1 & "
            f"echo $! > {self.pid_path} && cat {self.pid_path}"
        )
        result = self.env.execute(bg_command, timeout=timeout)
        output = result.get("output", "").strip()
        for line in output.splitlines():
            line = line.strip()
            if line.isdigit():
                self.pid = int(line)
                break
        if self.pid is None:
            return DetachedCommandErrorResult(
                error="Shell background spawn did not return a pid",
                output=output,
            )
        return DetachedCommandSpawnResult(pid=self.pid, output="")

    def poll(self, timeout: int) -> DetachedCommandResult:
        result = self.env.execute(f"cat {self.log_path} 2>/dev/null", timeout=max(timeout, 10))
        check = self.env.execute(
            f"kill -0 $(cat {self.pid_path} 2>/dev/null) 2>/dev/null; echo $?",
            timeout=min(timeout, 5) or 5,
        )
        check_output = check.get("output", "").strip()
        running = not check_output or check_output.splitlines()[-1].strip() == "0"
        output = result.get("output", "")

        if running:
            return DetachedCommandRunningResult(pid=self.pid, output=output)

        exit_result = self.env.execute(
            f"wait $(cat {self.pid_path} 2>/dev/null) 2>/dev/null; echo $?",
            timeout=min(timeout, 5) or 5,
        )
        exit_str = exit_result.get("output", "").strip()
        try:
            exit_code = int(exit_str.splitlines()[-1].strip())
        except (ValueError, IndexError):
            exit_code = -1
        if exit_code < 0:
            return DetachedCommandErrorResult(
                error="Shell background exit status could not be determined",
                pid=self.pid,
                output=output,
            )
        return DetachedCommandExitedResult(exit_code=exit_code, pid=self.pid, output=output)

    def wait(self, timeout: int) -> DetachedCommandResult:
        return self.poll(timeout=timeout)

    def kill(self, timeout: int) -> DetachedCommandResult:
        if self.pid is None:
            self.pid = self._read_pid(timeout=timeout)
        if self.pid is None:
            return DetachedCommandErrorResult(
                error="Shell background kill could not resolve pid",
            )
        self.env.execute(f"kill {self.pid} 2>/dev/null", timeout=min(timeout, 5) or 5)
        output = self.env.execute(
            f"cat {self.log_path} 2>/dev/null",
            timeout=min(timeout, 5) or 5,
        ).get("output", "")
        return DetachedCommandKilledResult(pid=self.pid, output=output)

    def _read_pid(self, timeout: int) -> int | None:
        result = self.env.execute(
            f"cat {self.pid_path} 2>/dev/null",
            timeout=min(timeout, 5) or 5,
        )
        output = result.get("output", "").strip()
        for line in output.splitlines():
            line = line.strip()
            if line.isdigit():
                return int(line)
        return None
