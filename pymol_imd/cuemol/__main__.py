"""Command-line entry point for the experimental installed-CueMol3 adapter."""

from __future__ import annotations

import argparse
import json
import logging
import math
import tempfile
from pathlib import Path

from pymol_imd import constants
from pymol_imd.client import IMDClient

from .bridge import CueMolBridge, CueMolIMD
from .launch import close_launched_app, launch


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    launch_parser = commands.add_parser(
        "launch", help="Launch an installed CueMol3 with a separate profile"
    )
    launch_parser.add_argument(
        "--app",
        type=Path,
        help="Installed app bundle or executable (default: platform installation)",
    )
    connect_parser = commands.add_parser(
        "connect", help="Display the server's prepared PDB and stream IMDv3"
    )
    connect_parser.add_argument(
        "prepared", type=Path, help="The exact prepared= PDB printed by IMD_READY"
    )
    source = connect_parser.add_mutually_exclusive_group()
    source.add_argument(
        "--app",
        type=Path,
        help="Installed app bundle or executable (default: platform installation)",
    )
    source.add_argument("--endpoint", help="Existing CueMol loopback HTTP debug endpoint")
    connect_parser.add_argument("--host", default=constants.IMD_DEFAULT_HOST)
    connect_parser.add_argument("--port", type=int, default=constants.IMD_DEFAULT_PORT)
    connect_parser.add_argument("--const-k", type=float, default=constants.AA_RECOMMENDED_CONST_K)
    connect_parser.add_argument(
        "--max-force-kj",
        type=float,
        help="Optional per-atom pull cap (kJ/mol/nm). Default: no cap.",
    )
    connect_parser.add_argument(
        "--drag-level",
        choices=constants.SUPPORTED_DRAG_LEVELS,
        default=constants.DEFAULT_DRAG_LEVEL,
    )
    connect_parser.add_argument(
        "--frame-interval-ms", type=int, default=constants.CUEMOL_FRAME_INTERVAL_MS
    )
    connect_parser.add_argument(
        "--duration", type=float, help="Stop after this many wall-clock seconds"
    )
    connect_parser.add_argument("--scene-id", type=int)
    connect_parser.add_argument("--view-id", type=int)
    connect_parser.add_argument("--name", default="IMD molecule")
    for subparser in (launch_parser, connect_parser):
        subparser.add_argument("--output-dir", type=Path, default=Path(".pixi/cuemol3-runs"))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "connect":
        for name in ("const_k", "max_force_kj", "frame_interval_ms", "duration"):
            value = getattr(args, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                parser.error(f"--{name.replace('_', '-')} must be finite and positive")
        if not 1 <= args.port <= 65535:
            parser.error("--port must be between 1 and 65535")
        if not args.prepared.is_file():
            parser.error("The prepared PDB does not exist")
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=args.output_dir))
    process = None
    bridge = None
    try:
        endpoint = getattr(args, "endpoint", None)
        if endpoint is None:
            process, endpoint = launch(args.app, run_dir / "profile", run_dir / "app.log")
            metadata = {"endpoint": endpoint, "launcher_pid": process.pid}
            (run_dir / "launch.json").write_text(json.dumps(metadata, indent=2) + "\n")
            print("CUEMOL_READY " + json.dumps(metadata), flush=True)
        if args.command == "launch":
            print("Keep this process running; use --endpoint in a second terminal.", flush=True)
            return process.wait()
        bridge = CueMolBridge(
            endpoint,
            args.prepared,
            scene_id=args.scene_id,
            view_id=args.view_id,
            name=args.name,
        )
        client = IMDClient(args.host, args.port)
        runner = CueMolIMD(
            bridge,
            client,
            const_k=args.const_k,
            max_force_kj=args.max_force_kj,
            drag_level=args.drag_level,
            frame_interval_ms=args.frame_interval_ms,
        )
        evidence = run_dir / "evidence.json"
        stats = runner.run(duration=args.duration, evidence=evidence)
        summary = {key: value for key, value in stats.items() if key not in {"initial", "final"}}
        print("CUEMOL_IMD_STOPPED " + json.dumps(summary), flush=True)
        print(f"Evidence: {evidence}", flush=True)
        return 130 if stats.get("interrupted") else 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logging.error("%s", exc)
        return 1
    finally:
        if bridge is not None:
            bridge.close()
        if process is not None:
            close_launched_app(process, endpoint)


if __name__ == "__main__":
    raise SystemExit(main())
