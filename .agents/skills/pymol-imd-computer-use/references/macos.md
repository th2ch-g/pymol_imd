# macOS viewer discovery

An unbundled conda Python process may open PyMOL successfully while CUA cannot
select it by `PyMOL` or by the Python executable. Use an existing recognizable
PyMOL app when available. Otherwise the helper below creates a local native
bundle that embeds the current pixi Python. It requires the existing macOS C
compiler; it installs nothing and does not launch the viewer by itself.

From the repository root:

```bash
pixi run --locked python .agents/skills/pymol-imd-computer-use/scripts/prepare_macos_app.py
```

To run a reviewed initialization/observation script at startup, add
`--startup .pixi/cua/startup.py`. The helper prints the generated app path,
executable path, bundle identifier, lock status, and log path. Paths are derived
from the checkout and active pixi environment. Generated files stay under
`.pixi/cua/` and must remain ignored.

After confirming there is no existing task-owned GUI, start the printed
executable using a native command execution tool. In a sandbox that cannot
open a window, request the appropriate tool escalation for this single launch.
Verify one process and a live window before selecting the printed bundle ID
with CUA. **Do not use a cold `getApp` call as the launcher**: some integrations
may launch or retry while discovery is delayed.

The native executable holds an exclusive lock for its entire lifetime. A second
invocation fails before Python or a window starts. This lock covers this
helper's launches only; inspect for viewers started by other routes as well.

```bash
pixi run --locked python .agents/skills/pymol-imd-computer-use/scripts/prepare_macos_app.py --status
```

`--status` reports whether that lock is held. A leftover PID file alone does
not establish that a viewer is running. Rebuilding while the lock is held is
refused. If a startup fails, inspect the generated log and process state before
any retry; do not create another app bundle to evade the lock.

Set a moderate viewport through PyMOL if screen capture is stale. Read the
current CUA state again after resize or fullscreen changes. Keyboard shortcuts
such as Control-Command-F are not reliable across PyMOL builds and may trigger
an unrelated PyMOL command. Prefer observed window controls over guessed keys.
