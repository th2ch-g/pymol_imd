"""Headless regression checks for the command-line and coercion surfaces."""

from __future__ import annotations

import contextlib
import io
import tempfile
from pathlib import Path

from pymol_imd import commands
from pymol_imd.cli import _build_parser, _build_pymol_statement, _connect_statement, main
from pymol_imd.commands import _as_bool
from pymol_imd.imd import IMDConfig


def _expect_system_exit(callable_, code: int = 2) -> None:
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            callable_()
        except SystemExit as exc:
            assert exc.code == code
        else:
            raise AssertionError(f"expected SystemExit({code})")


def main_test() -> int:
    statement = _build_pymol_statement(
        "aa",
        "/tmp/input.pdb",
        {},
        minimize=True,
        run=True,
        minimize_kwargs={"max_chunks": 2},
    )
    assert "imd_minimize(max_chunks=2, run_after=True)" in statement
    assert "; imd_run()" not in statement

    connect = _connect_statement("/tmp/prepared.pdb", "127.0.0.1", 8888, "obj", {})
    assert (
        connect.index("retain_order")
        < connect.index("load_prepared(cmd,")
        < connect.index("imd_connect")
    )

    _expect_system_exit(lambda: _build_parser("serve").parse_args(["x.pdb", "--const-k", "1"]))
    try:
        _as_bool("truthy")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid boolean spelling was accepted")
    try:
        IMDConfig(integrator="typo")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid integrator was accepted")
    for invalid_config in (
        {"nsteps_per_frame": 0},
        {"frame_interval_ms": 0},
        {"mouse_force": "false"},
    ):
        try:
            IMDConfig(**invalid_config)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid IMDConfig was accepted: {invalid_config}")

    class _Signal:
        callback = None

        def connect(self, callback):
            self.callback = callback

        def disconnect(self):
            self.callback = None

    class _Timer:
        def __init__(self):
            self.timeout = _Signal()

        def stop(self):
            return None

        def start(self, _interval):
            self.timeout.callback()

    class _FailingSystem:
        running = False
        config = type("Config", (), {"frame_interval_ms": 1})()

        def reset_minimization(self):
            return None

        def minimize_step(self, **_kwargs):
            raise RuntimeError("synthetic minimizer failure")

    previous = (commands._imd_system, commands._timer_minimize, commands.imd_run)
    started = []
    try:
        commands._imd_system = _FailingSystem()
        commands._timer_minimize = _Timer()
        commands.imd_run = lambda: started.append(True)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            commands.imd_minimize(run_after=True)
        assert not started, "production started after minimization failure"
    finally:
        commands._imd_system, commands._timer_minimize, commands.imd_run = previous

    with tempfile.TemporaryDirectory() as tmp:
        pdb = Path(tmp) / "input.pdb"
        pdb.write_text("END\n")

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            assert main([str(pdb)]) == 2
        assert "old `pymol_imd <PDB>` form was removed" in stderr.getvalue()

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            rc = main(
                [
                    "launch",
                    str(pdb),
                    "--pymol",
                    "/bin/echo",
                    "--print-only",
                    "--no-record",
                    "--minimize",
                    "--run",
                ]
            )
        assert rc == 0
        output = stdout.getvalue()
        assert "imd_minimize(run_after=True)" in output
        assert "pymol_imd 0." not in output

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            rc = main(["local", str(pdb), "--pymol", "/bin/echo", "--print-only", "--no-record"])
        assert rc == 0 and "imd_run()" in stdout.getvalue()

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rc = main(
                [
                    "launch",
                    str(pdb),
                    "--level",
                    "aa",
                    "--ff",
                    "martini3001",
                    "--pymol",
                    "/bin/echo",
                ]
            )
        assert rc == 2 and "not valid for --level aa" in stderr.getvalue()

    print("CLI smoke test OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_test())
