# pymol_imd

Multiscale Interactive Molecular Dynamics inside PyMOL.

`pymol_imd` lets you drag atoms around in the PyMOL viewer and feel the
molecule push back through a live OpenMM simulation. Three resolution
levels are supported from the same plugin:

| Level | Forcefield | Use when |
|---|---|---|
| `aa`   | Amber ff14SB / GAFF (AM1-BCC ligand charges) | Atomistic protein / DNA / RNA / lipid / ligand systems |
| `cg`   | Martini 3 via `vermouth` + `martini_openmm`  | Large biomolecules, fast 20 fs timestep |
| `mlmm` | ONIOM-style: full-system Amber MM + MLIP delta on a selected region | Reactive / electronic-effect-sensitive regions |

Supported MLIP backends for `--level mlmm` (in order of "ready to use"):

| Backend | Default | Login | Size | Notes |
|---|---|---|---|---|
| [AIMNet2](https://github.com/zubatyuk/aimnet) | **yes** | no | ~10 MB (bundled) | Fastest CPU; charge / mult go to the constructor |
| [MACE-OMOL-0](https://github.com/ACEsuit/mace-foundations) | no | no | ~400 MB | ASL license (academic only) |
| [ORB](https://github.com/orbital-materials/orb-models) | no | no | ~100 MB | `atoms.info["spin"]` driven |
| [UMA](https://huggingface.co/facebook/UMA) | no | **yes** | gated | Needs HF auth + `pixi run install-uma` |

---

## Install

Requires [pixi](https://pixi.sh). Native support: `linux-64`, `osx-arm64`.
Intel Mac (`osx-64`) and Windows users should run via a Linux VM / WSL2.

```bash
git clone <this-repo>
cd pymol_imd
pixi install   # covers AA + CG + ML-MM (AIMNet2 default + MACE) in one step
```

The default `pixi install` brings in:

- AA / CG backends with all their dependencies
- **AIMNet2** (`aimnet`) — default ML/MM backend, ~10 MB
- **MACE** (`mace-torch`) — alternative ML/MM backend

For the remaining two MLIP backends, run their pip task once. They are
kept out of the default lock because:

- **ORB** (`orb-models`): the chain `orb-models → cached-path` requests
  older `rich` / `filelock` than the conda lock picks
- **UMA** (`fairchem-core`): pins `e3nn>=0.5`, while `mace-torch` pins
  `e3nn==0.4.4` — the two cannot coexist in one resolver run; pip's
  looser resolver installs UMA after the fact

```bash
pixi run install-orb        # orb-models
pixi run install-uma        # fairchem-core; UMA also needs `huggingface-cli login`
```

---

## Quick start (CLI)

```bash
# Atomistic IMD on ubiquitin
pixi run pymol_imd 1ubq.pdb --level aa --minimize --run

# CG (Martini 3, 20 fs timestep)
pixi run pymol_imd 1ubq.pdb --level cg --minimize --run

# ONIOM ML/MM with AIMNet2 (default) on residue 1
pixi run pymol_imd 1ubq.pdb --level mlmm \
    --ml-selection "resi 1 and resn MET" --ml-charge 0 --ml-mult 1 --run
```

PyMOL opens with Editing mode preselected, so **Ctrl + left-drag** on an
atom pulls it; the IMD loop converts the drag into a per-atom force on
OpenMM each frame.

Show the generated PyMOL command without launching:

```bash
pixi run pymol_imd 1ubq.pdb --level aa --print-only
```

### CLI flags

All numeric / string knobs are optional; each falls back to its single-source
default in `pymol_imd/constants.py` (per-backend where noted).

| Flag | Levels | Description |
|---|---|---|
| `--level {aa,cg,mlmm}` | all | Simulation backend (default: `aa`) |
| `--pbc` | all | Enable periodic box / NPT (default: NVT vacuum, CG auto-boxes) |
| `--temperature K` | all | Langevin thermostat temperature (default 300) |
| `--nsteps-per-frame N` | all | MD steps per GUI frame (default 1) |
| `--const-k K` | all | Pull-force scale (kJ/mol/nm); default per backend (AA/MLMM 50000, CG 10000) |
| `--max-force-kj F` | all | Per-atom \|F\| cap on the pull force (kJ/mol/nm, default 5000) |
| `--integrator {langevin,langevin_middle}` | all | Langevin scheme; default per backend (AA `langevin_middle`, CG/MLMM `langevin`) |
| `--timestep-ps PS` | all | MD timestep; default per backend (AA 0.002 / CG 0.020 / MLMM 0.0005) |
| `--friction PER_PS` | all | Langevin friction 1/ps (default 1.0) |
| `--pressure BAR` | all | Barostat pressure with `--pbc` (default 1.0) |
| `--energy-cutoff-kj E` | all | \|U\| above this halts the loop as diverged (default 1e9) |
| `--frame-interval-ms MS` | all | Qt timer tick interval for the IMD / minimize loops (default 1) |
| `--nonbonded-cutoff-nm NM` | all | Nonbonded cutoff; default per backend (AA/MLMM 1.0, CG 1.1) |
| `--box-padding-nm NM` | `aa`, `cg` | Bounding-box padding when a box is built (default 1.0) |
| `--no-mouse-force` | all | Disable user-pull |
| `--minimize` | all | Run `imd_minimize` after setup |
| `--min-max-iter N` | all | Minimizer iterations per chunk, with `--minimize` (default 50) |
| `--min-max-chunks N` | all | Max minimize chunks, with `--minimize` (default 200) |
| `--min-tol-kj E` | all | Minimize convergence tolerance kJ/mol, with `--minimize` (default 1e-3) |
| `--run` | all | Run `imd_run` after setup |
| `--traj-out PATH` | all | Record a trajectory during `imd_run` (`.xtc`/`.dcd`/`.pdb` by extension); needs `--run` |
| `--traj-interval N` | all | Trajectory: write every N GUI frames (default 1) |
| `--force-csv PATH` | all | Record user pull forces to CSV (`frame,step,time_ps,atom_index,fx,fy,fz,fmag`, atom_index 0-based); needs `--run` |
| `--force-csv-interval N` | all | Force CSV: write every N GUI frames (default 1) |
| `--no-record` | all | Disable the default trajectory + force-CSV recording |
| `--topology-dir DIR` | all | Output dir for intermediate / topology files (default `./<stem>_topology`) |
| `--ligand-sdf PATH` | `aa` | Ligand SDF for GAFF + AM1-BCC parameterisation |
| `--ph PH` | `aa` | `Modeller.addHydrogens` protonation pH (default 7.4) |
| `--gaff-version VER` | `aa` | GAFF version for ligand templates (default `gaff-2.2.20`) |
| `--ff NAME` | `cg`  | Martini forcefield label (default: `martini3001`) |
| `--no-elastic-network` | `cg` | Skip martinize2 `-elastic` |
| `--martinize-maxwarn N` | `cg` | martinize2 `-maxwarn` value (default 99999) |
| `--ml-selection SEL` | `mlmm` | **Required.** PyMOL selection language (`resi 1 and resn MET`, `chain A and not resn HOH`, …) |
| `--ml-backend {uma,orb,mace,aimnet2}` | `mlmm` | MLIP backend (default: `aimnet2`) |
| `--ml-device {auto,cuda,mps,cpu}` | `mlmm` | Torch device; `auto` prefers cuda → mps → cpu |
| `--ml-charge Z`, `--ml-mult M` | `mlmm` | **Required.** ML region net charge / spin multiplicity |
| `--link-atom-method {scaled,fixed,none}` | `mlmm` | QM/MM boundary link-atom placement (default `scaled`) |
| `--ml-compile` | `mlmm` | Opt-in `torch.compile` for the MLIP (numerically ~identical) |
| `--enable-cueq` | `mlmm` | Opt-in cuEquivariance for MACE (CUDA; `pixi run install-cueq`) |
| `--pymol PATH` | all | Override PyMOL executable |
| `--print-only` | all | Show the underlying `pymol -d ...` and exit |

---

## Use from inside PyMOL

`pymol_imd` registers the same commands at the PyMOL prompt:

```
PyMOL> from pymol_imd import *
PyMOL> imd_setup_aa /path/to/input.pdb
PyMOL> imd_minimize             # converges on stable structure
PyMOL> imd_record traj=traj.xtc, force_csv=forces.csv  # arm recording (optional)
PyMOL> imd_run                  # production loop (recording starts here)
PyMOL> imd_change const_k=20000 # hot-patch on the fly
PyMOL> imd_stop                 # recording flushed + closed here
```

CG / ML-MM analogues:

```
PyMOL> imd_setup_cg input.pdb timestep_ps=0.020
PyMOL> imd_setup_mlmm input.pdb ml_selection='resi 30-50' ml_charge=0 ml_mult=1
```

After `setup_*`:

- A new PyMOL object `<basename>_imd` appears with backend-appropriate
  representation (cartoon + sticks for AA / ML-MM, spheres + bonds for CG).
- Mouse mode switches to **3-Button Editing** automatically.

---

## Recording trajectories and forces

`imd_record` arms output for the **next** `imd_run`. Recording starts when
`imd_run` is called and stops — files flushed + closed — on `imd_stop` (or on
an automatic divergence / error halt). Minimization is never recorded.

```
PyMOL> imd_setup_aa input.pdb
PyMOL> imd_record traj=run.xtc, force_csv=pull.csv
PyMOL> imd_run                  # recording begins
PyMOL> imd_stop                 # files finalized
PyMOL> imd_record off           # disarm (optional)
```

- **Trajectory** format is chosen by file extension: `.xtc`, `.dcd`, `.pdb`.
- **Force CSV** logs only the user pull force, one row per dragged atom per
  recorded frame, columns `frame,step,time_ps,atom_index,fx,fy,fz,fmag`
  (forces in kJ/mol/nm, after the `max_force_kj` cap). `atom_index` is
  **0-based** (OpenMM / trajectory convention) — add 1 to match a PyMOL
  `index` selection.
- `traj_interval` / `force_csv_interval` are in **GUI frames** (default 1):
  `traj_interval=5` writes every 5th frame.

From the CLI the same is available via flags (recording requires `--run`):

```bash
pixi run pymol_imd input.pdb --level aa --run \
    --traj-out run.xtc --traj-interval 5 \
    --force-csv pull.csv
```

> Changing `nsteps_per_frame` with `imd_change` mid-run does **not** retune an
> active trajectory reporter's interval (fixed at `imd_run` time). Run
> `imd_stop` then `imd_run` again to resync.

---

## ML/MM region selection

`imd_setup_mlmm` requires `ml_selection`. The string is **PyMOL's full selection
language** (`and`, `or`, `not`, `within`, `byres`, `resi`, `resn`, `chain`,
`name`, …) — resolved through `cmd.iterate` against a temporary load of the
prepared PDB:

```bash
# A single residue
--ml-selection 'resi 1 and resn MET'

# A bound ligand and everything within 5 Å (shell residues whole)
--ml-selection 'byres (resn LIG) within 5 of chain A'

# Active-site residues only, no water
--ml-selection 'chain A and resi 30-50 and not resn HOH'

# Whole system (pure MLIP, slow on big proteins)
--ml-selection 'all'
```

Headless / no-PyMOL contexts fall back to a strict mini-grammar
(`all`, `resi <range>`, `chain <id>`, `resn <name>`, comma-separated
indices) without combiners.

### Link atoms at the QM/MM boundary

When the ML selection cuts across a covalent bond, a hydrogen **link atom**
caps the dangling QM bond so the selection no longer has to be molecularly
closed. `--link-atom-method` (`imd_setup_mlmm link_atom_method=...`) picks the
placement:

- `scaled` (default): Morokuma/Dapprich g-factor — the link H sits at
  `r_Q + g·(r_M − r_Q)` and stretches with the broken bond.
- `fixed`: keeps the Q–H distance at a tabulated bond length (1.09 Å for C,
  1.01 Å for N, …) via the IMOMM projector Jacobian.
- `none`: disables link atoms — only for molecularly closed selections.

The link H's force is redistributed back onto the real QM- and MM-side atoms
through the placement Jacobian.

---

## How it works

1. **Build**: each backend turns the input PDB into an OpenMM `System`,
   `Topology`, and `Positions`. AA goes through `Modeller.addHydrogens`;
   CG runs `martinize2` then `martini_openmm.MartiniTopFile`; MLMM
   constructs a full-system Amber `System` and an MM-only subsystem for
   the ONIOM subtraction.
2. **Prepare visualisation**: a `*_prepared.pdb` is written so PyMOL is
   re-loaded against exactly the same atom order OpenMM uses (critical:
   `Modeller.addHydrogens` reorders atoms).
3. **Attach forces**: a `CustomExternalForce("- fx*x - fy*y - fz*z")` is
   attached for the user pull; ML/MM also attaches a second
   `CustomExternalForce` for the per-frame ONIOM correction
   `F_ML - F_MM_subset`.
4. **Loop**: a Qt timer steps the simulation, reads PyMOL atom positions,
   computes `delta = pymol_xyz - openmm_xyz` for each atom, multiplies by
   `const_k`, then caps the resulting force magnitude at `max_force_kj` (a
   |F| cap) and pushes it into the pull `CustomExternalForce`. Then OpenMM
   integrates one frame's worth of steps and the new coordinates are
   written back into PyMOL via `cmd.alter_state`.
5. **Safety**: every step potential energy is checked; NaN / |U| above
   `energy_cutoff_kj` halts the loop with a console message.

---

## Troubleshooting

### MLIP / Apple Silicon (MPS)
MPS cannot allocate float64 storage, which **MACE / ORB / AIMNet2 all
need at model-load time** (their weight files are float64). The plugin
detects this and silently falls back to CPU with a notice like
`AIMNet2 weights are float64 but MPS cannot allocate float64 storage;
running on CPU instead`. CPU MLIPs are usable for small ML regions
(tens of atoms); pick a larger backend only if you have CUDA.

### UMA model gated
The UMA backend pulls weights from `facebook/UMA`, a HuggingFace gated
repo. After `pixi run install-uma`, run `pixi run huggingface-cli login`
and request access at https://huggingface.co/facebook/UMA.

### CG initial Martini download
First `imd_setup_cg` downloads `martini_v3.0.0.itp` (~2 MB) into
`~/.cache/pymol_imd/martini3/`. Subsequent runs are offline.

### Diverging simulation
If the structure flies apart, `imd_run` stops automatically with a hint:
> `imd_run: stopped (divergence or error). Adjust const_k / max_force_kj.`

Try `imd_change const_k=10000 max_force_kj=2000` and re-run, or shorten
`--timestep-ps`. The loop also self-halts when |U| exceeds `--energy-cutoff-kj`.

---

## Project layout

```
pymol_imd/
├── __init__.py        Re-exports setup_* / imd_*; sets PYTORCH_ENABLE_MPS_FALLBACK
├── cli.py             pymol_imd executable (argparse → pymol -d ...)
├── commands.py        PyMOL command surface (setup_*, imd_*)
├── constants.py       Single source of truth: tunable defaults + fixed physical constants
├── imd.py             IMDSystem driver: CustomExternalForce + Qt timer + ONIOM hook
├── prep.py            RCSB fetch / Modeller.addHydrogens / AM1-BCC / prepared-PDB
├── backends/
│   ├── base.py        Backend Protocol + SystemBundle dataclass
│   ├── aa.py          Amber backend
│   ├── cg.py          Martini 3 backend (vermouth + martini_openmm)
│   └── mlmm.py        ONIOM ML/MM backend (UMA / ORB / MACE / AIMNet2) + link atoms
└── tests/
    ├── smoke.py        Headless AA + CG smoke
    ├── smoke_mlmm.py   Headless ML/MM ONIOM delta regression (vs refs/*.npy)
    ├── smoke_record.py Headless trajectory + force-CSV recording smoke
    └── refs/           Pinned coordinates + reference ML/MM delta arrays
```

---

## License

GPLv3 (see [LICENSE](LICENSE)).

Acknowledgements: this project sits on top of OpenMM, PyMOL, vermouth /
martini_openmm, and the MACE / UMA / ORB / AIMNet2 MLIPs. See their
respective licences (MACE-OMOL-0 is academic-use only).
