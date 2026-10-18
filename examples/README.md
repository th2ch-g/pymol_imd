# Molecular mouse experiments

These are versioned input fixtures for manual computer-use experiments. Run
the experiments with the
[mouse experiment skill](../.agents/skills/pymol-imd-mouse-experiments/SKILL.md).
Generated preparation files, recordings, screenshots and reports belong under
an ignored directory such as `.pixi/cua/`, outside `tests/`.
For code-driven experiments, follow the
[scripted molecular protocols](../.agents/skills/pymol-imd-mouse-experiments/references/scripted.md).

## Videos

The six MP4s replay verified trajectories from physical mouse experiments in
PyMOL. Each trajectory is rendered independently with PyMOL and CueMol3; the
CueMol3 clips do not represent a separate CueMol mouse experiment. They are
14-second, silent H.264 videos at 960 × 540 and 20 fps. The displayed MD clock
gives the simulation time; playback is resampled without coordinate interpolation.

| Level | PyMOL | CueMol3 | Observed result |
|---|---|---|---|
| AA | [MP4](pymol_aa.mp4) | [MP4](cuemol_aa.mp4) | QBP partial opening: heavy-atom domain distance 25.04 → 30.10 Å; open-reference RMSD 5.27 → 3.77 Å |
| CG | [MP4](pymol_cg.mp4) | [MP4](cuemol_cg.mp4) | QBP transient opening and return: BB-centroid distance approximately 24.4 → 29.3 → 24.5 Å |
| ML/MM | [MP4](pymol_mlmm.mp4) | [MP4](cuemol_mlmm.mp4) | Driven chorismate-to-prephenate-like conversion; product connectivity persists through all 2,000 unforced release frames |

AA uses Amber/GBn2 at 2 fs and shows 0–21.694 ps. CG uses Martini 3 at 20 fs
with its elastic network retained and shows 136–146 ps of the recorded run.
The domains are fitted on domain A for display; the protein folds remain
substantially preserved. The CG distance is a BB proxy and its opening is
transient, rather than a persistent open-state transition.

The enzyme run uses AIMNet2/Amber ONIOM with GBn2, 160 ML atoms and 12 link H,
ML charge 0 and multiplicity 1, and a 0.5 fs step. Its 126 physical gestures
were consumed by 85 manually advanced MD steps, followed by 2,000 unforced
steps (1 ps). All six formal Claisen bond changes and the product's connectivity
and stereochemistry were checked using coordinate-based RDKit inference.
The movie updates displayed ligand bonds from that inference; it is not an
electronic bond-order or spontaneous-catalysis calculation. The initial and
final C1–O11/C3–C16 distances distinguish bond breaking from bond formation.

All three recordings used a 300 K target and an explicit 5,000 kJ/mol/nm pull
cap, with no automatic recovery in the shown runs. The current plugin's default
is uncapped. Raw trajectories, analysis and intermediate frames stay in ignored
work directories; the MP4s and their JPEG previews are the published artifacts.

## QBP

`qbp.pdb` is the 226-residue mature sequence specified by the
[RGI input](https://github.com/cddlab/rgi_toolkit/blob/main/examples/distance/alphafold3/qbp_25.00.json).
It comes from AlphaFold DB P0AEQ3 model v6 with the 22-residue signal peptide
removed and residues renumbered from 1.
This is a closed-like prediction with no ligand or water.

`qbp_open_reference.pdb` (1GGG) and `qbp_closed_reference.pdb` (1WDN) contain
protein chain A only. Both references have L23 where the requested sequence
has K23; the open reference lacks residues 1-4 and 225-226 and the closed
reference lacks residues 1-3. Use common backbone atoms for structural fits.

The [RGI paper](https://doi.org/10.1021/acs.jctc.6c00199) uses a geometric
distance between domains A (residues 5-84 and 186-224) and B (90-180), in this
PDB's one-based numbering. The all-heavy-atom centroid distance is about
25.0 angstrom in the closed reference and 29.1 angstrom in the open reference.
CA-only distances and CG bead distances are proxies, not that same observable.
Check reference RMSDs and within-domain deformation as well as distance.

```sh
pixi run pymol_imd launch examples/qbp.pdb --level aa
pixi run pymol_imd launch examples/qbp.pdb --level cg
```

AA defaults to Amber with GBn2 implicit solvent. CG uses Martini 3, an automatic
periodic box and an elastic network; record those constraints when judging
whether a hinge opening was possible. Minimize before starting production.

## Chorismate mutase

`chorismate_mutase.pdb` is an assembled, minimized reactant model: the wild-type
3ZO8 trimer, chorismate transferred from the aligned Arg90Cit complex 3ZP7,
and one bridging water. Protein chains A/B/C contain resolved residues 1-117;
unresolved residues 118-127 are omitted and residue 117 has a terminal OXT.
It is not an unmodified experimental structure.

The model has 5727 atoms. The 24-atom chorismate dianion is `resn ISJ`, chain D;
the bridging water is `resn HOH`, chain E. Supply `chorismate.sdf` for the
ligand's atom order, stereochemistry and formal charge (-2).

```sh
pixi run pymol_imd launch examples/chorismate_mutase.pdb --level mlmm \
  --ligand-sdf examples/chorismate.sdf \
  --ml-selection 'resn ISJ or resn HOH or (chain A and resi 63+74+75) or (chain B and resi 7+78+90+108)' \
  --ml-charge 0 --ml-mult 1 --ml-backend aimnet2
```

The reference preparation has 160 selected atoms and 12 link hydrogens. Verify
these counts after preparation. AA and CG have fixed bonded potentials and
are unsuitable for demonstrating this chemical reaction. In ML/MM, inspect
recorded coordinates and all six bond changes of the formal Claisen
rearrangement. PyMOL's initial bond table does not update automatically.

An externally driven product-like geometry does not establish an enzymatic
mechanism, transition state, barrier, or spontaneous catalysis. Do not count
automated distance restraints as a successful physical mouse experiment.
