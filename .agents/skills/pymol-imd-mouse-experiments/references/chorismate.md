# Chorismate mutase with ML/MM

Use `examples/chorismate_mutase.pdb` plus `examples/chorismate.sdf`. Read the
fixture provenance: this is a wild-type trimer with a transplanted substrate,
one bridging water, trimmed unresolved termini and completed hydrogens.

```python
imd_setup_mlmm(
    'examples/chorismate_mutase.pdb',
    ligand_sdf='examples/chorismate.sdf',
    ml_selection=('resn ISJ or resn HOH or (chain A and resi 63+74+75) '
                  'or (chain B and resi 7+78+90+108)'),
    ml_charge=0, ml_mult=1, ml_backend='aimnet2',
    work_dir='.pixi/cua/chorismate/build')
imd_minimize()
```

Check the current command signature before execution. Expected reference counts
are 5727 full atoms, 160 selected ML atoms and 12 link H atoms. Verify prepared
identities rather than reusing old numeric indices; ensure all ligand atoms are
in the ML region and the reactant is intact after minimization. Record solvent,
constraints, charge/multiplicity and MLIP/device. AA and CG cannot model this
bond rearrangement through their fixed bonded terms.

Make the substrate pickable with element-colored sticks/spheres and labels on
the six reactive atoms. Use the same starting checkpoint for the no-drag control
and each mouse attempt. Orient for visible motion of C16 toward C3 and extension
of the C1-O11 linkage. Apply actual small CUA drags to selected ligand atoms,
confirm accepted forces, and leave an unforced release segment. Never substitute
scripted distance restraints for the physical mouse trial.

## Physical input with manual MD stepping

If CUA latency leaves many unforced MD frames between gestures, use a
manual-step mouse trial. Keep `nsteps_per_frame=1` so the ML/MM correction is
refreshed every integration step. Increasing that setting can destabilize this
reactive model even without input; check such a condition separately before
interpreting any reaction.

Start recording at every step, call `imd_run`, then pause only its Qt timer:

```python
from pymol_imd import commands as imd_commands
imd_commands._timer.stop()
```

The driver's `running` flag and recording stay active. Select atoms using
PyMOL's native `cmd.select`, `cmd.drag` and `dgmv`/`dgrt` mouse mapping. Perform
actual CUA gestures, then consume their resulting viewer displacement with the
normal driver step on the GUI thread:

```python
imd_commands._imd_system.step()
```

A useful paired input selects `C16+C1` and pulls toward C3, then selects
`C3+O11` and pulls oppositely before advancing that one MD step. Derive the
screen directions and coordinates from the current camera and screenshot.
Record both gestures for the shared MD step and verify all four force rows.
A native rotation of `C16+O11` can also move the two reaction coordinates.
Keep the force cap unchanged and reduce a gesture if the whole frame is rejected.

This diagnostic procedure uses the current internal timer and driver. Do not
call `run_drag`, supply synthetic offsets, or assign molecular coordinates in
this mode. The displacement must come from the actual CUA mouse events.
Analyze the MD coordinates after integration, not the temporary cursor-displaced
geometry shown between the drag and the MD step. Record the manual clock mode,
physical gesture count, accepted force frames and simulated duration separately.

On release, clear pull forces, finish the native drag selection and resume the
existing Qt timer without reopening the recorder:

```python
imd_commands._imd_system.clear_pull_forces()
cmd.drag()
cmd.set_wizard()
cmd.config_mouse("three_button_editing")
imd_commands._timer.start(imd_commands._imd_system.config.frame_interval_ms)
```

Continue an unforced segment and classify every recorded release frame. Keep
a control from the same checkpoint when possible; label a control reinitialized
from the same coordinates and seed as such. Save both failed continuous-clock
attempts and the manual-step attempt instead of pooling their force exposure.

## Assess the formal [3,3] rearrangement

In this ISJ atom naming, all six changes are required:

| Pair | Reactant | Product-like geometry |
| --- | --- | --- |
| C1-O11 | Single bond | Broken |
| C3-C16 | Nonbonded | Single bond |
| C1-C2 | Single bond | Double bond |
| C2-C3 | Double bond | Single bond |
| C12-C16 | Double bond | Single bond |
| O11-C12 | Single bond | Carbonyl double bond |

```sh
pixi run --locked python .agents/skills/pymol-imd-mouse-experiments/scripts/analyze_trajectory.py \
  chorismate --topology .pixi/cua/chorismate/mouse/initial.pdb \
  --trajectory .pixi/cua/chorismate/mouse/trajectory.xtc \
  --output .pixi/cua/chorismate/mouse/analysis
```

Inspect all six distance traces, representative conformations and release
frames. Distance thresholds alone cannot assign bond orders reliably. Check
the inferred ligand connectivity, formal charge and stereochemistry separately
with the existing RDKit installation. RDKit bond inference is a geometric
interpretation, not an electronic bond-order calculation.

PyMOL retains the initial PDB bond table, so a new bond may not appear and an old
bond may remain drawn after a coordinate change. Do not equate that display with
the force model: ONIOM subtracts the selected region's matching MM interactions.
Check the model construction if the selected ligand is constrained or excluded.

Call a stable result externally driven prephenate-like geometry unless stronger
evidence establishes the mechanism. A concerted transition state, IRC, barrier
height and spontaneous enzymatic catalysis are not proved by this experiment.
Frozen linear correction energies are not an ONIOM reaction-energy profile.
If release returns to reactant, only one linkage changes, or the geometry
deforms, report that outcome rather than claiming the Claisen reaction occurred.
