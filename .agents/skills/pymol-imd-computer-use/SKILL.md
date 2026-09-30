---
name: pymol-imd-computer-use
description: "Operate and debug pymol_imd in a live PyMOL viewer with computer-use, including real mouse pulling, setup/minimize/run/stop, recording, and IMDv3 connections. Use when physical UI verification is requested; headless simulation tests alone do not need this skill."
---

# PyMOL IMD Computer Use

Use the existing viewer when possible. Keep at most one task-owned PyMOL GUI
process alive, and run expensive backend tests sequentially. Inspect running
processes and the CUA app inventory before launching; a delayed app-selection
call is not evidence that another launch is needed. Preserve unrelated sessions.

## Prepare the session

- Follow the checkout's `AGENTS.md` and current command signatures. Run Python
  through `pixi run --locked`; add no dependencies for UI automation.
- Put generated structures, logs, sessions, and trajectories in an ignored
  task directory, such as `.pixi/cua/`. Use distinct recording names for each
  restart because local recording overwrites existing paths.
- Start with a small peptide for interaction checks. Increase model size or
  load an MLIP only after the basic viewer path works. Use the requested model
  when the user asks for a scientific demonstration.
- Use `pymol -k` for a fresh, task-owned debug viewer when personal startup
  scripts are irrelevant. Do not alter the user's startup configuration.
- For a macOS Python process that CUA cannot identify as PyMOL, read
  [the macOS launcher procedure](references/macos.md). It supplies a native
  app bundle with a process lock; it does not synthesize UI events.

## Operate through computer-use

Use only the documented CUA APIs for UI input and screenshots. Select an
already-running app from observed inventory or its verified bundle identity.
Read current accessibility state before choosing element indices; those indices
change after window or title changes. A terminal window is not a substitute for
an unavailable CUA target.

The external PyMOL **Command Input Area** usually supports `setValue` followed
by `pressKey('Return')`. Use the index from the current accessibility tree.
Commands can load a prepared structure, configure the simulation, start/stop it,
and install a read-only observer. Inspect the resulting UI before the next action.

For physical pulling, use CUA `drag([x, y], [x2, y2])` on an atom identified in
the latest screenshot. Small drags are enough to verify force transmission.
If the available drag API cannot hold PyMOL's editing modifiers, temporarily
map plain left drag to `MovA` in this task-owned viewer:

```python
cmd.config_mouse("three_button_editing")
cmd.button("left", "none", "mova")
cmd.unpick()
```

Afterward restore `cmd.config_mouse("three_button_editing")`. Do not describe
`alter_state`, `load_coordset`, a scripted force, or a molecular PNG export as
physical mouse input or a UI screenshot. Avoid dragging a bond or background;
verify PyMOL's clicked-atom message and the recorded atom index.

If the screenshot is stale, request fresh state and a fresh screenshot. A
`viewport` resize can refresh a stuck capture. If coordinate actions report
`noWindowsAvailable`, inspect the actual window and fullscreen state. Use an
observed `Raise` secondary action when available, then reacquire the screenshot.
Reuse or restore that window; if a restart is necessary, close the task-owned viewer
and verify its process has exited before launching exactly one replacement.
If capture still fails after reconnecting CUA and one controlled restart,
record the concrete CUA error and finish unaffected verification. Do not retry
app launches in a loop or describe unperformed gestures as passed.

## Verify the requested behavior

For a comprehensive check, cover the relevant modes below. For a narrower
request, use only the checks needed to establish its outcome.

| Operation | Evidence |
| --- | --- |
| Setup, minimize, run, stop | Correct object/atom count; MD steps advance; stop clears forces and the Qt timer |
| Atom / residue / chain drag | Actual CUA gesture; matching nonzero force rows; correct expanded atom set |
| Multi-chain model | Chain drag affects only the selected chain, including when residue numbers repeat |
| Force cap | Lower the cap for a small test drag; the frame is skipped and MD keeps running |
| Recording off / restart | Files stop growing while MD continues; restart uses new paths; outputs close and read back |
| Object/atom deletion | A clear error, stopped timer, released pull forces, closed recording/connection |
| Socket reconnect / server loss | Fresh session files; clean reconnect; disconnected viewer releases its timer |
| Distance measurements | Visible measurement points agree with current coordinates during MD |

Use a short-lived observer or recording to capture events that a screenshot
may miss. The force CSV uses **zero-based** atom indices; PyMOL's `index` is
**one-based**. Local positions are nm in OpenMM and angstrom in PyMOL. Socket
forces use kJ/mol/angstrom on the wire and kJ/mol/nm at the engine boundary.
Keep observations read-only; do not let an observer apply artificial drags.

For socket checks, load exactly the server's `IMD_READY prepared=...` PDB and
use its announced port. `local` is the same-process Qt mode, not socket mode.
Use a small localhost server for transport checks; follow the user's SSH and
scheduler requirements when actual remote GPU computation is requested.

For ML/MM reaction demonstrations, verify the selected prepared atom identities,
charge, multiplicity, link atoms, and any ligand/water membership. Identify
formed/broken bonds using that model's atom names. Inspect recorded coordinates
rather than static PyMOL bonds; do not infer a reaction or barrier from a drag,
a transient distance, or the linear correction force's raw OpenMM energy.
State when conversion was not observed, and keep runs with different model
regions or pulling parameters separate. Respect any rejected parameter change.

## Finish

Stop the simulation, disconnect the client, and close task-owned output streams.
Check the final visible state and any remaining task-owned processes. Preserve
requested artifacts and close the task-owned viewer if it is no longer needed.
Report what was physically operated, what the recordings establish, any fixes
and tests, and what remains unverified. Never equate a passing smoke test with
physical mouse verification.
