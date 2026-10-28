"""Configure CDS credentials in a private local terminal, or show manual steps."""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import warnings


CDS_URL = "https://cds.climate.copernicus.eu/api"
PLATFORM = os.name
WINDOWS_ACL = r"""
$ErrorActionPreference = 'Stop'
$path = $env:XMETAI_CDS_CONFIG_PATH
$sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
$acl = New-Object System.Security.AccessControl.FileSecurity
$acl.SetOwner($sid)
$acl.SetAccessRuleProtection($true, $false)
$rule = [System.Security.AccessControl.FileSystemAccessRule]::new(
    $sid, 'FullControl', 'Allow')
$acl.AddAccessRule($rule)
Set-Acl -LiteralPath $path -AclObject $acl
$saved = Get-Acl -LiteralPath $path
if (-not $saved.AreAccessRulesProtected) { throw 'Unprotected ACL' }
$rules = @($saved.GetAccessRules(
    $true, $true, [System.Security.Principal.SecurityIdentifier]))
if ($rules.Count -ne 1 -or $rules[0].IdentityReference.Value -ne $sid.Value -or
    $rules[0].AccessControlType -ne 'Allow' -or
    $rules[0].FileSystemRights -ne 'FullControl') { throw 'Unexpected ACL' }
"""
MANUAL = """CDS manual configuration (no files are changed):
1. Sign in at https://cds.climate.copernicus.eu/profile and obtain your
   Personal Access Token. Never paste it into chat or a shell command.
2. On the machine and under the user that will run Datakit, use a private
   local editor to create the user-home .cdsapirc with these two fields:
   url: https://cds.climate.copernicus.eu/api
   key: <YOUR_PERSONAL_ACCESS_TOKEN>
   Replace the placeholder privately. Do not put credentials in the repository,
   Profile YAML, Batch YAML, command arguments, screenshots or shared logs.
3. On Linux/macOS, restrict access: chmod 600 ~/.cdsapirc
   On Windows, use %USERPROFILE%\\.cdsapirc (not .cdsapirc.txt).
   In file Properties > Security > Advanced, disable inherited permissions
   and restrict access to your current user. The wizard can do this for you.
4. Accept the terms for each target dataset on the CDS website.
5. Install the Datakit CDS downloader requirements in the download environment.
CDSAPI_RC, CDSAPI_URL and CDSAPI_KEY can override client configuration.
Check those privately if using a non-default configuration; never print them.
Saving a file or passing --dry-run does NOT verify authentication.
No network request or download has been made."""


class SetupError(Exception):
    """A safe-to-display configuration error."""


def existing_config(path: Path) -> os.stat_result | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode) or (
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    ):
        raise SetupError("Refusing to replace a symlink or non-regular ~/.cdsapirc.")
    return info


def restrict_windows_access(path: Path) -> None:
    powershell = shutil.which("powershell.exe")
    if powershell is None:
        raise SetupError("Windows PowerShell is required to restrict file access; use --manual.")
    try:
        result = subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-Command", WINDOWS_ACL],
            env={**os.environ, "XMETAI_CDS_CONFIG_PATH": str(path)},
            capture_output=True, check=False, timeout=30,
        )
    except subprocess.TimeoutExpired:
        raise SetupError("Windows file permission setup timed out; no token saved.") from None
    if result.returncode:
        raise SetupError("Cannot set owner-only Windows file permissions; no token saved.")


def save_config(path: Path, token: str, original: os.stat_result | None) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".cdsapirc-",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            if PLATFORM == "posix":
                os.fchmod(stream.fileno(), 0o600)
        if PLATFORM == "nt":
            # Protect the empty file before writing any credential bytes.
            restrict_windows_access(temporary)
        elif PLATFORM != "posix":
            raise SetupError("Unsupported operating system; use --manual.")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            # JSON quoting also produces a YAML scalar without allowing new fields.
            stream.write(f"url: {CDS_URL}\nkey: {json.dumps(token)}\n")
            stream.flush()
            os.fsync(stream.fileno())
        if original is None:
            # Do not overwrite a file created while the user was entering the token.
            os.link(temporary, path)
        else:
            current = existing_config(path)
            if current != original:
                raise SetupError("Configuration changed during setup; no overwrite performed.")
            os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def interactive_setup() -> int:
    if PLATFORM not in {"posix", "nt"}:
        raise SetupError("Interactive saving supports Windows and POSIX systems; use --manual.")
    overrides = [
        name for name in ("CDSAPI_RC", "CDSAPI_URL", "CDSAPI_KEY")
        if name in os.environ
    ]
    if overrides:
        raise SetupError(
            "Client overrides are set (" + ", ".join(overrides)
            + "). Resolve them privately before using default setup, or use --manual."
        )
    path = Path.home() / ".cdsapirc"
    print(f"Configuration target: {path}")
    original = existing_config(path)
    if original is not None:
        answer = input("~/.cdsapirc exists. Replace the ENTIRE file? [y/N]: ")
        if answer.strip().lower() not in {"y", "yes"}:
            print("Cancelled; existing configuration was not changed.")
            return 0
    print("Obtain your Personal Access Token: https://cds.climate.copernicus.eu/profile")
    print("Enter it only here in your private terminal; input will be hidden.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        token = getpass.getpass("CDS Personal Access Token: ")
    if not token or any(not 33 <= ord(character) <= 126 for character in token):
        raise SetupError("Token must be non-empty printable ASCII without whitespace.")
    save_config(path, token, original)
    permissions = "Windows current-user-only ACL" if PLATFORM == "nt" else "0600"
    print(f"Saved {path} with owner-only permissions ({permissions}).")
    print("No authentication check or download was made; accept target dataset terms.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual", action="store_true", help="Print guidance without prompting")
    args = parser.parse_args()
    if args.manual:
        print(MANUAL)
        return 0
    if not sys.stdin.isatty() or not sys.stdout.isatty() or not sys.stderr.isatty():
        print(
            "Error: run this command yourself in a private interactive terminal, "
            "not a chat tool, pipe or captured task. Use --manual for guidance.",
            file=sys.stderr,
        )
        return 2
    try:
        print("CDS credential setup")
        print("1. Local interactive setup (hidden token input)")
        print("2. Manual configuration guidance (no changes)")
        choice = input("Choose [1/2]: ").strip()
        if choice == "2":
            print(MANUAL)
            return 0
        if choice != "1":
            raise SetupError("Choose 1 or 2; no configuration was changed.")
        return interactive_setup()
    except SetupError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    except getpass.GetPassWarning:
        print("Error: cannot hide token input; configuration was not changed.", file=sys.stderr)
        return 2
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled; setup did not complete.", file=sys.stderr)
        return 130
    except OSError as error:
        # Do not print exception details that could expose user-supplied data.
        print(f"Error: local configuration I/O failed (errno={error.errno}).", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
