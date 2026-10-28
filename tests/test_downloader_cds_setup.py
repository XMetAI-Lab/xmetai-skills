"""Exercise local CDS setup using fake tokens and isolated home directories.

Requires pytest, but no real CDS credentials, Datakit repository, or network.
Run from the xmetai-skills repository root:

    conda run -n xmetai-skills \\
      python -m pytest tests/test_downloader_cds_setup.py -q -rs

Platform-specific cases intentionally skip on other operating systems:
POSIX permission, atomic saving, and terminal cases require POSIX; the two
real Windows ACL cases require Windows and are skipped on Linux.
"""

import errno
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import warnings

import pytest


ROOT = Path(__file__).resolve().parents[1]
ENTRY = (
    ROOT / "skills" / "dataset-downloader"
    / "scripts" / "setup_cds.py"
)
FAKE_TOKEN = "test-only-not-a-real-token"


@pytest.fixture
def setup(tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("cds_setup_test", ENTRY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for name in ("CDSAPI_RC", "CDSAPI_URL", "CDSAPI_KEY"):
        monkeypatch.delenv(name, raising=False)
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        monkeypatch.setattr(type(stream), "isatty", lambda self: True)
    monkeypatch.setattr(sys, "argv", [str(ENTRY)])
    monkeypatch.setattr(module.getpass, "getpass", lambda prompt: FAKE_TOKEN)
    return module


def answers(monkeypatch, *values):
    responses = iter(values)
    monkeypatch.setattr("builtins.input", lambda prompt: next(responses))


def assert_saved(path, token=FAKE_TOKEN):
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "url: https://cds.climate.copernicus.eu/api"
    assert json.loads(lines[1].removeprefix("key: ")) == token
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize("arguments", [["--manual"], []])
def test_noninteractive_modes(tmp_path, arguments):
    result = subprocess.run(
        [sys.executable, str(ENTRY), *arguments], input=FAKE_TOKEN,
        capture_output=True, text=True, check=False,
        env={**os.environ, "HOME": str(tmp_path)},
    )
    assert result.returncode == (0 if arguments else 2)
    assert FAKE_TOKEN not in result.stdout + result.stderr
    assert not list(tmp_path.iterdir())
    if arguments:
        assert "<YOUR_PERSONAL_ACCESS_TOKEN>" in result.stdout
    else:
        assert "private interactive terminal" in result.stderr


def test_menu_manual_does_not_prompt_for_token(setup, tmp_path, monkeypatch, capsys):
    answers(monkeypatch, "2")
    monkeypatch.setattr(
        setup.getpass, "getpass", lambda prompt: pytest.fail("must not request token")
    )
    assert setup.main() == 0
    assert "manual configuration" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission contract")
@pytest.mark.parametrize("existing", [False, True])
def test_interactive_save(setup, tmp_path, monkeypatch, capsys, existing):
    path = tmp_path / ".cdsapirc"
    if existing:
        path.write_text("existing configuration", encoding="utf-8")
        path.chmod(0o644)
    answers(monkeypatch, *("1", "yes") if existing else ("1",))
    assert setup.main() == 0
    assert_saved(path)
    assert [item.name for item in tmp_path.iterdir()] == [".cdsapirc"]
    captured = capsys.readouterr()
    assert FAKE_TOKEN not in captured.out + captured.err
    assert "No authentication check" in captured.out


@pytest.mark.skipif(os.name != "posix", reason="POSIX setup")
def test_overwrite_declined(setup, tmp_path, monkeypatch):
    path = tmp_path / ".cdsapirc"
    path.write_text("unchanged", encoding="utf-8")
    answers(monkeypatch, "1", "")
    monkeypatch.setattr(
        setup.getpass, "getpass", lambda prompt: pytest.fail("must not request token")
    )
    assert setup.main() == 0
    assert path.read_text(encoding="utf-8") == "unchanged"


@pytest.mark.skipif(os.name != "posix", reason="POSIX setup")
@pytest.mark.parametrize("kind", ["symlink", "directory"])
def test_refuse_nonregular_config(setup, tmp_path, monkeypatch, kind):
    path = tmp_path / ".cdsapirc"
    target = tmp_path / "target"
    if kind == "symlink":
        target.write_text("unchanged", encoding="utf-8")
        path.symlink_to(target)
    else:
        path.mkdir()
    answers(monkeypatch, "1")
    assert setup.main() == 2
    if kind == "symlink":
        assert target.read_text(encoding="utf-8") == "unchanged"
        assert path.is_symlink()
    else:
        assert path.is_dir()


@pytest.mark.skipif(os.name != "posix", reason="POSIX setup")
@pytest.mark.parametrize("name", ["CDSAPI_RC", "CDSAPI_URL", "CDSAPI_KEY"])
def test_overrides_refused_without_values(setup, tmp_path, monkeypatch, capsys, name):
    monkeypatch.setenv(name, FAKE_TOKEN)
    answers(monkeypatch, "1")
    assert setup.main() == 2
    captured = capsys.readouterr()
    assert name in captured.err
    assert FAKE_TOKEN not in captured.out + captured.err
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(os.name != "posix", reason="POSIX setup")
@pytest.mark.parametrize("token", ["", "has spaces", "line\nbreak", "\x1b[31m", "\u4e2d"])
def test_invalid_token_is_not_saved_or_printed(setup, tmp_path, monkeypatch, capsys, token):
    answers(monkeypatch, "1")
    monkeypatch.setattr(setup.getpass, "getpass", lambda prompt: token)
    assert setup.main() == 2
    assert not list(tmp_path.iterdir())
    captured = capsys.readouterr()
    assert "Token must be" in captured.err
    assert FAKE_TOKEN not in captured.out + captured.err


@pytest.mark.skipif(os.name != "posix", reason="POSIX setup")
def test_no_echo_fallback(setup, tmp_path, monkeypatch, capsys):
    def unavailable(prompt):
        warnings.warn("Cannot hide input", setup.getpass.GetPassWarning)
        pytest.fail("warning must stop fallback")

    answers(monkeypatch, "1")
    monkeypatch.setattr(setup.getpass, "getpass", unavailable)
    assert setup.main() == 2
    assert "cannot hide token input" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(os.name != "posix", reason="POSIX setup")
@pytest.mark.parametrize("error", [EOFError, KeyboardInterrupt])
def test_cancel_before_saving(setup, tmp_path, monkeypatch, error):
    def cancelled(prompt):
        raise error()

    answers(monkeypatch, "1")
    monkeypatch.setattr(setup.getpass, "getpass", cancelled)
    assert setup.main() == 130
    assert not list(tmp_path.iterdir())


def test_invalid_menu_choice(setup, tmp_path, monkeypatch, capsys):
    answers(monkeypatch, "bad choice")
    assert setup.main() == 2
    assert "Choose 1 or 2" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(os.name != "posix", reason="POSIX atomic saving")
@pytest.mark.parametrize("originally_exists", [False, True])
def test_concurrent_config_not_overwritten(setup, tmp_path, originally_exists):
    path = tmp_path / ".cdsapirc"
    if originally_exists:
        path.write_text("original", encoding="utf-8")
    original = setup.existing_config(path)
    path.write_text("concurrent update with different size", encoding="utf-8")
    with pytest.raises((FileExistsError, setup.SetupError)):
        setup.save_config(path, FAKE_TOKEN, original)
    assert path.read_text(encoding="utf-8") == "concurrent update with different size"
    assert [item.name for item in tmp_path.iterdir()] == [".cdsapirc"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX saving")
def test_failed_write_cleans_temporary_file(setup, tmp_path, monkeypatch, capsys):
    def failed(*args):
        raise OSError(errno.ENOSPC, FAKE_TOKEN)

    answers(monkeypatch, "1")
    monkeypatch.setattr(setup.os, "fsync", failed)
    assert setup.main() == 1
    assert not list(tmp_path.iterdir())
    captured = capsys.readouterr()
    assert FAKE_TOKEN not in captured.out + captured.err
    assert "errno=" in captured.err


@pytest.mark.skipif(os.name != "posix", reason="POSIX saving")
def test_yaml_scalar_quoted_without_injection(setup, tmp_path):
    path = tmp_path / ".cdsapirc"
    token = 'test-only:#"{}[]'
    setup.save_config(path, token, None)
    assert_saved(path, token)

@pytest.mark.parametrize("existing", [False, True])
def test_windows_flow_protects_empty_file_first(
    setup, tmp_path, monkeypatch, capsys, existing
):
    monkeypatch.setattr(setup, "PLATFORM", "nt")
    path = tmp_path / ".cdsapirc"
    if existing:
        path.write_text("original", encoding="utf-8")
    protected = []

    def protect(temporary):
        assert temporary.parent == tmp_path
        assert temporary.read_bytes() == b""
        protected.append(temporary)

    monkeypatch.setattr(setup, "restrict_windows_access", protect)
    answers(monkeypatch, *("1", "yes") if existing else ("1",))
    assert setup.main() == 0
    assert len(protected) == 1
    content = path.read_text(encoding="utf-8")
    assert content == f"url: {setup.CDS_URL}\nkey: {json.dumps(FAKE_TOKEN)}\n"
    assert [item.name for item in tmp_path.iterdir()] == [".cdsapirc"]
    captured = capsys.readouterr()
    assert "Windows current-user-only ACL" in captured.out
    assert str(path) in captured.out
    assert FAKE_TOKEN not in captured.out + captured.err


@pytest.mark.parametrize("failure", ["missing", "acl", "timeout"])
def test_windows_permission_failure_preserves_config(
    setup, tmp_path, monkeypatch, capsys, failure
):
    monkeypatch.setattr(setup, "PLATFORM", "nt")
    path = tmp_path / ".cdsapirc"
    path.write_text("original", encoding="utf-8")
    monkeypatch.setattr(
        setup.shutil, "which", lambda name: None if failure == "missing" else "powershell.exe"
    )

    def failed(command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 30)
        return subprocess.CompletedProcess(command, 1, stderr=FAKE_TOKEN)

    monkeypatch.setattr(setup.subprocess, "run", failed)
    answers(monkeypatch, "1", "yes")
    assert setup.main() == 2
    assert path.read_text(encoding="utf-8") == "original"
    assert [item.name for item in tmp_path.iterdir()] == [".cdsapirc"]
    captured = capsys.readouterr()
    assert FAKE_TOKEN not in captured.out + captured.err
    assert "Error:" in captured.err


def test_windows_acl_invocation_contains_no_token(setup, tmp_path, monkeypatch):
    path = tmp_path / "empty config with spaces"
    path.touch()
    monkeypatch.setattr(setup.shutil, "which", lambda name: "powershell.exe")

    def run(command, **kwargs):
        assert command == [
            "powershell.exe", "-NoProfile", "-NonInteractive", "-Command", setup.WINDOWS_ACL
        ]
        assert kwargs["env"]["XMETAI_CDS_CONFIG_PATH"] == str(path)
        assert kwargs["capture_output"] is True
        assert kwargs["timeout"] == 30
        assert FAKE_TOKEN not in repr(command)
        assert FAKE_TOKEN not in repr(kwargs)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(setup.subprocess, "run", run)
    setup.restrict_windows_access(path)


def test_unsupported_platform_refused(setup, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(setup, "PLATFORM", "unsupported")
    answers(monkeypatch, "1")
    assert setup.main() == 2
    assert "supports Windows and POSIX" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(os.name != "nt", reason="Requires real Windows ACLs")
@pytest.mark.parametrize("existing", [False, True])
def test_native_windows_save(setup, tmp_path, monkeypatch, existing):
    path = tmp_path / ".cdsapirc"
    if existing:
        path.write_text("original", encoding="utf-8")
    answers(monkeypatch, *("1", "yes") if existing else ("1",))
    assert setup.main() == 0
    assert path.read_text(encoding="utf-8") == (
        f"url: {setup.CDS_URL}\nkey: {json.dumps(FAKE_TOKEN)}\n"
    )
    # Check the final published file, not just the temporary file.
    verify = setup.WINDOWS_ACL[setup.WINDOWS_ACL.index("$saved ="):]
    command = (
        "$ErrorActionPreference = 'Stop'; "
        "$path = $env:XMETAI_CDS_CONFIG_PATH; "
        "$sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User; "
        + verify
    )
    result = subprocess.run(
        [setup.shutil.which("powershell.exe"), "-NoProfile", "-NonInteractive",
         "-Command", command],
        env={**os.environ, "XMETAI_CDS_CONFIG_PATH": str(path)},
        capture_output=True, check=False, timeout=30,
    )
    assert result.returncode == 0


@pytest.mark.skipif(os.name != "posix", reason="Real POSIX terminal test")
def test_real_terminal_hides_token(tmp_path):
    import pty
    import select
    import time

    master, slave = pty.openpty()
    environment = {
        name: value for name, value in os.environ.items()
        if name not in {"CDSAPI_RC", "CDSAPI_URL", "CDSAPI_KEY"}
    }
    environment["HOME"] = str(tmp_path)
    process = subprocess.Popen(
        [sys.executable, str(ENTRY)], stdin=slave, stdout=slave, stderr=slave,
        env=environment, start_new_session=True,
    )
    os.close(slave)
    output = bytearray()

    def wait_for(marker):
        deadline = time.monotonic() + 10
        while marker not in output:
            remaining = deadline - time.monotonic()
            assert remaining > 0, "Interactive prompt timed out"
            ready, _, _ = select.select([master], [], [], remaining)
            assert ready, "Interactive prompt timed out"
            output.extend(os.read(master, 4096))

    try:
        wait_for(b"Choose [1/2]: ")
        os.write(master, b"1\n")
        wait_for(b"CDS Personal Access Token: ")
        os.write(master, FAKE_TOKEN.encode() + b"\n")
        wait_for(b"No authentication check")
        assert process.wait(timeout=10) == 0
        assert FAKE_TOKEN.encode() not in output
        assert_saved(tmp_path / ".cdsapirc")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        os.close(master)
