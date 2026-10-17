"""Create a locked native PyMOL launcher for CUA discovery on macOS."""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--startup", type=Path)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("This helper is macOS-only; use the platform's existing viewer launcher.")
    import fcntl

    project = args.project.resolve()
    if not (project / "pymol_imd" / "commands.py").is_file():
        parser.error("Run from the pymol_imd checkout or supply --project.")
    output = project / ".pixi" / "cua"
    output.mkdir(parents=True, exist_ok=True)
    app = output / "PyMOL IMD CUA.app"
    executable = app / "Contents" / "MacOS" / "launcher"
    bundle_id = "org.pymol.imd.cua"
    lock_path = output / "viewer.lock"
    pid_path = output / "viewer.pid"
    log_path = output / "viewer.log"
    info = {
        "app": str(app.relative_to(project)),
        "executable": str(executable.relative_to(project)),
        "bundle_id": bundle_id,
        "log": str(log_path.relative_to(project)),
    }
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            info["status"] = "running"
            if pid_path.exists():
                info["pid"] = pid_path.read_text().strip()
            print(json.dumps(info))
            if args.status:
                return 0
            parser.error("The viewer lock is held; reuse the running app instead of rebuilding.")
        if args.status:
            info["status"] = "stopped"
            print(json.dumps(info))
            return 0
        compiler = shutil.which("clang")
        if compiler is None:
            parser.error(
                "A macOS C compiler is required; use an existing PyMOL app if unavailable."
            )
        include = Path(sysconfig.get_path("include"))
        library_dir = Path(sysconfig.get_config_var("LIBDIR"))
        version = (
            sysconfig.get_config_var("LDVERSION")
            or f"{sys.version_info.major}.{sys.version_info.minor}{sys.abiflags}"
        )
        library = library_dir / f"libpython{version}.dylib"
        if not library.is_file() or not (include / "Python.h").is_file():
            parser.error("Run this helper with the pixi environment's Python development files.")
        python_args = ["-m", "pymol", "-k", "-d", "from pymol_imd import *"]
        if args.startup:
            startup = args.startup if args.startup.is_absolute() else project / args.startup
            if not startup.is_file():
                parser.error(f"Startup script does not exist: {args.startup}")
            python_args += ["-r", str(startup.resolve())]
        environment = {
            "PATH": str(Path(sys.prefix) / "bin") + os.pathsep + os.environ.get("PATH", ""),
            "PYTHONHOME": sys.prefix,
            "PYTHONUTF8": "1",
            "QT_MAC_WANTS_LAYER": "1",
            "OPENMM_CPU_THREADS": os.environ.get("OPENMM_CPU_THREADS", "2"),
            "WARP_CACHE_PATH": str(project / ".pixi" / "warp-cache"),
            "AIMNET_CACHE_DIR": str(project / ".pixi" / "aimnet-cache"),
        }
        setters = "\n".join(
            f"setenv({json.dumps(key)}, {json.dumps(value)}, 1);"
            for key, value in environment.items()
        )
        arguments = ", ".join(json.dumps(value) for value in python_args)
        source = output / "launcher.c"
        source.write_text(f"""#include <Python.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/file.h>
#include <unistd.h>

int main(int argc, char **argv) {{
    if (chdir({json.dumps(str(project))})) return 1;
    int lock_fd = open({json.dumps(str(lock_path))}, O_CREAT | O_RDWR, 0600);
    if (lock_fd < 0 || flock(lock_fd, LOCK_EX | LOCK_NB)) {{
        fputs("PyMOL CUA viewer is already running or its lock is unavailable.\\n", stderr);
        return 1;
    }}
    FILE *pid = fopen({json.dumps(str(pid_path))}, "w");
    if (!pid) return 1;
    fprintf(pid, "%d\\n", getpid());
    fclose(pid);
    {setters}
    if (!freopen({json.dumps(str(log_path))}, "w", stdout)) return 1;
    if (!freopen({json.dumps(str(log_path))}, "a", stderr)) return 1;
    setvbuf(stdout, NULL, _IOLBF, 0);
    char *python_argv[] = {{argv[0], {arguments}, NULL}};
    return Py_BytesMain({len(python_args) + 1}, python_argv);
}}
""")
        executable.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                compiler,
                str(source),
                "-I",
                str(include),
                str(library),
                f"-Wl,-rpath,{library_dir}",
                "-o",
                str(executable),
            ],
            check=True,
        )
        metadata = {
            "CFBundleDisplayName": "PyMOL IMD CUA",
            "CFBundleExecutable": "launcher",
            "CFBundleIdentifier": bundle_id,
            "CFBundleName": "PyMOL IMD CUA",
            "CFBundlePackageType": "APPL",
            "CFBundleVersion": "1",
            "NSHighResolutionCapable": True,
        }
        with (app / "Contents" / "Info.plist").open("wb") as stream:
            plistlib.dump(metadata, stream)
        info["status"] = "prepared"
        print(json.dumps(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
