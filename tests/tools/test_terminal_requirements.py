import importlib
import logging

terminal_tool_module = importlib.import_module("tools.terminal_tool")


def _clear_terminal_env(monkeypatch):
    """Remove terminal env vars that could affect requirements checks."""
    keys = [
        "HERMES_ENABLE_NOUS_MANAGED_TOOLS",
        "TERMINAL_ENV",
        "TERMINAL_MODAL_MODE",
        "TERMINAL_SSH_HOST",
        "TERMINAL_SSH_USER",
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "VERCEL_OIDC_TOKEN",
        "VERCEL_PROJECT_ID",
        "VERCEL_TEAM_ID",
        "VERCEL_TOKEN",
        "HOME",
        "USERPROFILE",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)


def test_local_terminal_requirements(monkeypatch, caplog):
    """Local backend uses Hermes' own LocalEnvironment wrapper."""
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "local")

    with caplog.at_level(logging.ERROR):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is True
    assert "Terminal requirements check failed" not in caplog.text


def test_unknown_terminal_env_logs_error_and_returns_false(monkeypatch, caplog):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "unknown-backend")

    with caplog.at_level(logging.ERROR):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is False
    assert any(
        "Unknown TERMINAL_ENV 'unknown-backend'" in record.getMessage()
        for record in caplog.records
    )


def test_ssh_backend_without_host_or_user_logs_and_returns_false(monkeypatch, caplog):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "ssh")

    with caplog.at_level(logging.ERROR):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is False
    assert any(
        "SSH backend selected but TERMINAL_SSH_HOST and TERMINAL_SSH_USER" in record.getMessage()
        for record in caplog.records
    )


def test_modal_backend_without_token_or_config_logs_specific_error(monkeypatch, caplog, tmp_path):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "modal")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(terminal_tool_module, "is_managed_tool_gateway_ready", lambda _vendor: False)
    monkeypatch.setattr(terminal_tool_module.importlib.util, "find_spec", lambda _name: object())

    with caplog.at_level(logging.ERROR):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is False
    assert any(
        "Modal backend selected but no direct Modal credentials/config was found" in record.getMessage()
        for record in caplog.records
    )


def test_modal_backend_with_managed_gateway_does_not_require_direct_creds_or_minisweagent(monkeypatch, tmp_path):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("HERMES_ENABLE_NOUS_MANAGED_TOOLS", "1")
    monkeypatch.setenv("TERMINAL_ENV", "modal")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("TERMINAL_MODAL_MODE", "managed")
    monkeypatch.setattr(terminal_tool_module, "is_managed_tool_gateway_ready", lambda _vendor: True)
    monkeypatch.setattr(
        terminal_tool_module.importlib.util,
        "find_spec",
        lambda _name: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    assert terminal_tool_module.check_terminal_requirements() is True


def test_modal_backend_auto_mode_prefers_managed_gateway_over_direct_creds(monkeypatch, tmp_path):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("HERMES_ENABLE_NOUS_MANAGED_TOOLS", "1")
    monkeypatch.setenv("TERMINAL_ENV", "modal")
    monkeypatch.setenv("MODAL_TOKEN_ID", "tok-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "tok-secret")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(terminal_tool_module, "is_managed_tool_gateway_ready", lambda _vendor: True)
    monkeypatch.setattr(
        terminal_tool_module.importlib.util,
        "find_spec",
        lambda _name: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    assert terminal_tool_module.check_terminal_requirements() is True


def test_modal_backend_direct_mode_does_not_fall_back_to_managed(monkeypatch, caplog, tmp_path):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "modal")
    monkeypatch.setenv("TERMINAL_MODAL_MODE", "direct")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(terminal_tool_module, "is_managed_tool_gateway_ready", lambda _vendor: True)

    with caplog.at_level(logging.ERROR):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is False
    assert any(
        "TERMINAL_MODAL_MODE=direct" in record.getMessage()
        for record in caplog.records
    )


def test_modal_backend_managed_mode_does_not_fall_back_to_direct(monkeypatch, caplog, tmp_path):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "modal")
    monkeypatch.setenv("TERMINAL_MODAL_MODE", "managed")
    monkeypatch.setenv("MODAL_TOKEN_ID", "tok-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "tok-secret")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(terminal_tool_module, "is_managed_tool_gateway_ready", lambda _vendor: False)

    with caplog.at_level(logging.ERROR):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is False
    assert any(
        "HERMES_ENABLE_NOUS_MANAGED_TOOLS is not enabled" in record.getMessage()
        for record in caplog.records
    )


def test_modal_backend_managed_mode_without_feature_flag_logs_clear_error(monkeypatch, caplog, tmp_path):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "modal")
    monkeypatch.setenv("TERMINAL_MODAL_MODE", "managed")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(terminal_tool_module, "is_managed_tool_gateway_ready", lambda _vendor: False)

    with caplog.at_level(logging.ERROR):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is False
    assert any(
        "HERMES_ENABLE_NOUS_MANAGED_TOOLS is not enabled" in record.getMessage()
        for record in caplog.records
    )


def test_vercel_sandbox_backend_without_sdk_logs_clear_error(monkeypatch, caplog):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "vercel_sandbox")
    monkeypatch.setattr(terminal_tool_module.importlib.util, "find_spec", lambda _name: None)

    with caplog.at_level(logging.ERROR):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is False
    assert any(
        "vercel is required for the Vercel Sandbox terminal backend" in record.getMessage()
        for record in caplog.records
    )


def test_vercel_sandbox_backend_without_credentials_logs_clear_error(monkeypatch, caplog):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "vercel_sandbox")
    monkeypatch.setattr(terminal_tool_module.importlib.util, "find_spec", lambda _name: object())

    with caplog.at_level(logging.ERROR):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is False
    assert any(
        "Vercel Sandbox backend selected but credentials are missing" in record.getMessage()
        for record in caplog.records
    )


def test_vercel_sandbox_backend_with_token_credentials_passes(monkeypatch):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "vercel_sandbox")
    monkeypatch.setenv("VERCEL_TOKEN", "tok")
    monkeypatch.setenv("VERCEL_PROJECT_ID", "prj")
    monkeypatch.setenv("VERCEL_TEAM_ID", "team")
    monkeypatch.setattr(terminal_tool_module.importlib.util, "find_spec", lambda _name: object())

    assert terminal_tool_module.check_terminal_requirements() is True
