"""Launch an installed CueMol3 without modifying its bundle or normal profile."""

from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

from pymol_imd import constants

from .cdp import CDP, CDPError


def default_app() -> Path:
    """Locate the platform's default CueMol3 installation."""
    if sys.platform == "darwin":
        return Path("/Applications/CueMol3.app")
    names = ("CueMol3.exe", "cuemol3.exe") if sys.platform == "win32" else ("cuemol3", "CueMol3")
    for name in names:
        executable = shutil.which(name)
        if executable is not None:
            return Path(executable)
    if sys.platform == "win32":
        for variable, suffix in (
            ("ProgramFiles", Path("CueMol3")),
            ("ProgramFiles(x86)", Path("CueMol3")),
            ("LOCALAPPDATA", Path("Programs") / "CueMol3"),
        ):
            prefix = os.environ.get(variable)
            if prefix:
                executable = Path(prefix) / suffix / "CueMol3.exe"
                if executable.is_file():
                    return executable
    raise FileNotFoundError(
        f"CueMol3 was not found on PATH ({', '.join(names)}) or in a default installation; "
        "install it or pass --app with its executable path"
    )


def close_launched_app(process: subprocess.Popen, endpoint: str | None) -> None:
    """Close only the debug instance created by this launcher."""
    if endpoint:
        connection = None
        try:
            connection = CDP(endpoint, timeout=constants.CUEMOL_CDP_TIMEOUT_S)
            connection.request("Browser.close")
        except (CDPError, OSError, TimeoutError):
            pass
        finally:
            if connection is not None:
                connection.close()
        try:
            process.wait(timeout=constants.CUEMOL_CDP_TIMEOUT_S)
            return
        except subprocess.TimeoutExpired:
            pass
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=constants.CUEMOL_CDP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=constants.CUEMOL_CDP_TIMEOUT_S)


def launch(
    app: Path | None,
    profile: Path,
    log_path: Path,
    timeout: float = constants.CUEMOL_LAUNCH_TIMEOUT_S,
) -> tuple[subprocess.Popen, str]:
    app = (default_app() if app is None else app).expanduser().resolve()
    bundle = None
    if app.suffix == ".app":
        if not (app / "Contents" / "Info.plist").is_file():
            raise FileNotFoundError(
                f"CueMol app bundle not found: {app}; pass --app with its installed path"
            )
        bundle = app
        with (app / "Contents" / "Info.plist").open("rb") as stream:
            executable = plistlib.load(stream)["CFBundleExecutable"]
        app = app / "Contents" / "MacOS" / executable
    if not app.is_file():
        raise FileNotFoundError(
            f"CueMol executable not found: {app}; pass --app with its installed path"
        )
    profile = profile.resolve()
    profile.mkdir(parents=True, exist_ok=True)
    active_port = profile / "DevToolsActivePort"
    if active_port.exists():
        raise RuntimeError("Use a new profile directory for each test launch")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("CUEMOL_FRESH_PREFS", None)
    arguments = [
        "--remote-debugging-port=0",
        "--remote-debugging-address=127.0.0.1",
        f"--user-data-dir={profile}",
    ]
    if bundle is not None and sys.platform == "darwin":
        command = [
            "open",
            "-n",
            "-W",
            "-a",
            str(bundle),
            "--stdout",
            str(log_path.resolve()),
            "--stderr",
            str(log_path.resolve()),
            "--args",
            *arguments,
        ]
    else:
        command = [str(app), *arguments]
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    endpoint = None
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"CueMol exited with code {process.returncode}; inspect {log_path}"
                )
            if active_port.exists():
                lines = active_port.read_text().splitlines()
                if lines and lines[0].isdigit() and 1 <= int(lines[0]) <= 65535:
                    endpoint = f"http://127.0.0.1:{int(lines[0])}"
                    return process, endpoint
            time.sleep(0.1)
        raise TimeoutError(
            f"CueMol did not publish a debug endpoint; inspect {log_path}. "
            "Close its separate test window if it remains open."
        )
    except BaseException:
        close_launched_app(process, endpoint)
        raise
