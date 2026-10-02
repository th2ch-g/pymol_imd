---
name: pymol-imd-mouse-experiments
description: Run physical mouse-drag experiments for chorismate mutase reactions and QBP domain opening in a live pymol_imd viewer, and assess the recorded structures. Use for these molecular demonstrations and their reproducibility; ordinary headless regressions use the project's smoke tasks.
---

# Molecular Mouse Experiments

Use the versioned inputs in `examples/`. These are manual computer-use
experiments separate from `pymol_imd/tests/`, with generated evidence under an
ignored task directory such as `.pixi/cua/`. Read `examples/README.md` and
`examples/sources.json` before interpreting the models.

Read the existing [viewer skill](../pymol-imd-computer-use/SKILL.md) for launch,
single-window ownership, CUA input, recording and cleanup. Use its macOS helper
when needed. Physical gestures must go through computer-use; helper scripts may
observe or analyze coordinates but must not generate pulling forces or alter
coordinates during a mouse trial.

- For QBP, read [the AA and CG procedure](references/qbp.md).
- For chorismate mutase, read [the ML/MM procedure](references/chorismate.md).
- Use [the trajectory analyzer](scripts/analyze_trajectory.py) after stopping
  recording; `--help` describes its inputs. It reports geometry, not chemical
  mechanism or a success certificate.

## Evidence required for each trial

Record the input and prepared atom identities, backend/forcefield, solvent and
box, timestep, platform, pulling scale/cap/granularity, elapsed wall time and MD
time. Save the initial state and use it for a comparable no-drag control. Keep
changed parameters and repeated attempts in separate directories.

Capture a current screenshot, identify a visible atom, perform a small CUA
drag, then confirm the clicked atom (or active domain selection) and nonzero
recorded forces. A view rotation,
background drag, or cap-rejected gesture is not a successful force application.
Record gesture coordinates and timing separately from the force CSV. Stop and
flush recording before concluding that an empty CSV means no force was applied.

Keep a release segment after the final drag. Save XTC plus its matching prepared
PDB, force/state CSVs, console recovery warnings and the final structure. A
rollback can discard a genuine mouse attempt from the accepted force CSV;
report that rejection explicitly. Do not raise the divergence threshold or
change a model merely to obtain a pass.

Assess control drift, reference agreement, local deformation, energy failures
and persistence after release. Report `observed`, `not observed`, or `invalid
trial` with the conditions and evidence. A failed trial is useful evidence;
do not replace it with an automated steering result and call it mouse success.

Stop the simulation, restore the normal editing mouse map, close task-owned
recorders and check the visible state. Close the task-owned viewer when no
further interaction is needed.
