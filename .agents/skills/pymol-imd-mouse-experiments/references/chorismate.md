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
