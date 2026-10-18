# QBP opening with AA and CG

Use `examples/qbp.pdb`, the exact requested mature sequence in a closed-like
AlphaFold conformation. References and their sequence/terminal differences are
documented in `examples/README.md`. Test AA and CG sequentially in one viewer.

## Prepare and observe

```python
imd_setup_aa('examples/qbp.pdb', work_dir='.pixi/cua/qbp/aa/build',
             drag_level='atom', nsteps_per_frame=1)
# In a separate trial, replace imd_setup_aa with imd_setup_cg.
imd_minimize()
```

Wait for minimization to finish, then save the prepared topology and checkpoint
before control and mouse trials. A checkpoint restore is allowed between trials;
it is not allowed as coordinate steering within a mouse trial. Record the actual
configuration because per-backend defaults differ.

Select and drag the complete domain, rather than pulling a single residue.
PyMOL's native drag selection moves exactly those atoms; keep the plugin's
`drag_level='atom'` to avoid expanding the selection to the whole chain.

```python
cmd.select('domain_B', 'qbp_imd and resi 90-180')
cmd.drag('domain_B')
cmd.button('left', 'none', 'dgmv')
```

Use the actual setup object's name if it differs. Verify the dragging wizard's
selected-atom count and force-CSV membership: the prepared AA fixture has 1394
atoms in domain B, and the default CG mapping has 204 beads. The selected set
must cover residues 90-180 and exclude domain
A. This native operation requires no additional public plugin command.

Color domain A (`resi 5-84 or resi 186-224`) and B differently. Use ribbons for
AA or sticks for CG; residue 131 (`CA` or `BB`) can serve as a visible landmark.
Use a fresh screenshot before each physical CUA drag. `dgmv` moves the activated
selection even without a fresh atom-pick message; verify the selected set and
recorded forces instead. At cleanup, call `cmd.set_wizard()` / `cmd.drag()` and
restore `cmd.config_mouse('three_button_editing')`.

Align the open reference to domain A of the initial structure to determine the
opening direction. Orient the camera so the desired motion lies in the screen
plane. Verify the first recorded force direction: PyMOL's view matrix convention
can reverse or transpose an assumed projection. Do not infer opening from screen
translation alone. Begin with short drags; use existing `imd_change` settings
to reduce force if the model deforms or integration fails.
An illustrative gentle AA trial uses `const_k=1000` and one step per frame;
record any stronger trial separately. A single CUA drag may produce only one
accepted force frame, so report the actual force duty cycle rather than assuming
the force was held throughout the wall-clock gesture.
Keep the altered trial distinct from the default-parameter attempt.

Translation alone can leave the opened domain twisted relative to the reference.
Use `cmd.button("left", "none", "dgrt")` to rotate the same native drag
selection with physical gestures, then switch back to `dgmv` for translation.
Repeat bounded gestures while following both distance and reference RMSD;
one or two gestures may be insufficient. Realign the display-only reference
to domain A after global motion before judging screen overlap.

AA uses Amber and GBn2 by default. CG has Martini's periodic bounding box and
elastic network; the network can constrain an interdomain transition. Do not
silently disable it, change secondary structure or add solvent to force opening.

## Decide what happened

The source restraint's zero-based residue sets become one-based PDB domains
A: 5-84 plus 186-224; B: 90-180. Its distance uses all-heavy-atom geometric
centroids, approximately 25.0 angstrom closed and 29.1 angstrom open. CG bead
centroids and CA/BB distances are different observables.

```sh
pixi run --locked python .agents/skills/pymol-imd-mouse-experiments/scripts/analyze_trajectory.py \
  qbp --topology .pixi/cua/qbp/aa/mouse/initial.pdb \
  --trajectory .pixi/cua/qbp/aa/mouse/trajectory.xtc \
  --output .pixi/cua/qbp/aa/mouse/analysis
```

Run the same analysis for control and CG. For CG, the analyzer compares BB beads
to reference CA atoms as an explicitly approximate backbone comparison.
Compare distance, whole-backbone RMSDs to open/closed references, domain B after
fitting A, and each domain's internal RMSD to the initial structure. Inspect
representative frames and the release segment. Require coherent hinge motion
with preserved folds and improved open-reference agreement. A large centroid
distance caused by stretching/unfolding, periodic wrapping or an energy failure
is not an open-state transition.

A short negative result means the transition was not observed under that
duration and pulling protocol; it does not establish that AA or CG can never
reach the open state.
