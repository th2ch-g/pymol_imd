"""Launch an installed CueMol3 without modifying its bundle or normal profile."""

from __future__ import annotations

import os
import plistlib
import subprocess
import sys
import time
from pathlib import Path

from pymol_imd import constants

from .cdp import CDP, CDPError


def close_launched_app(process: subprocess.Popen, endpoint: str | None) -> None:
    """Close only the debug instance created by this launcher."""
    if process.poll() is not None:
        return
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
    process.terminate()
    try:
        process.wait(timeout=constants.CUEMOL_CDP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=constants.CUEMOL_CDP_TIMEOUT_S)


def launch(
    app: Path,
    profile: Path,
    log_path: Path,
    timeout: float = constants.CUEMOL_LAUNCH_TIMEOUT_S,
) -> tuple[subprocess.Popen, str]:
    app = app.expanduser().resolve()
    bundle = None
    if app.suffix == ".app":
        bundle = app
        with (app / "Contents" / "Info.plist").open("rb") as stream:
            executable = plistlib.load(stream)["CFBundleExecutable"]
        app = app / "Contents" / "MacOS" / executable
    if not app.is_file():
        raise FileNotFoundError(f"CueMol executable not found: {app}")
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
