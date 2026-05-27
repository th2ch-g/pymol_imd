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

| Flag | Levels | Description |
|---|---|---|
| `--level {aa,cg,mlmm}` | all | Simulation backend (default: `aa`) |
| `--pbc` | all | Enable periodic box / NPT (default: NVT vacuum, CG auto-boxes) |
| `--temperature K` | all | Langevin thermostat temperature |
| `--nsteps-per-frame N` | all | MD steps per GUI frame |
| `--const-k K` | all | Pull-force scale (kJ/mol/nm); default comes from each backend |
| `--no-mouse-force` | all | Disable user-pull |
| `--minimize` | all | Run `imd_minimize` after setup |
| `--run` | all | Run `imd_run` after setup |
| `--ligand-sdf PATH` | `aa` | Ligand SDF for GAFF + AM1-BCC parameterisation |
| `--ff NAME` | `cg`  | Martini forcefield label (default: `martini3001`) |
| `--no-elastic-network` | `cg` | Skip martinize2 `-elastic` |
| `--ml-selection SEL` | `mlmm` | **Required.** PyMOL selection language (`resi 1 and resn MET`, `chain A and not resn HOH`, …) |
| `--ml-backend {uma,orb,mace,aimnet2}` | `mlmm` | MLIP backend (default: `aimnet2`) |
| `--ml-device {auto,cuda,mps,cpu}` | `mlmm` | Torch device; `auto` prefers cuda → mps → cpu |
| `--ml-charge Z`, `--ml-mult M` | `mlmm` | ML region net charge / spin multiplicity |
| `--pymol PATH` | all | Override PyMOL executable |
| `--print-only` | all | Show the underlying `pymol -d ...` and exit |

---

## Use from inside PyMOL

`pymol_imd` registers the same commands at the PyMOL prompt:

```
PyMOL> from pymol_imd import *
PyMOL> setup_aa /path/to/input.pdb
PyMOL> imd_minimize             # converges on stable structure
PyMOL> imd_run                  # production loop
PyMOL> imd_change const_k=20000 # hot-patch on the fly
PyMOL> imd_stop
```

CG / ML-MM analogues:

```
PyMOL> setup_cg input.pdb timestep_ps=0.020
PyMOL> setup_mlmm input.pdb ml_selection='resi 30-50' ml_charge=0 ml_mult=1
```

After `setup_*`:

- A new PyMOL object `<basename>_imd` appears with backend-appropriate
  representation (cartoon + sticks for AA / ML-MM, spheres + bonds for CG).
- Mouse mode switches to **3-Button Editing** automatically.

---

## ML/MM region selection

`setup_mlmm` requires `ml_selection`. The string is **PyMOL's full selection
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
   computes `delta = pymol_xyz - openmm_xyz` for each atom, caps the
   magnitude at `max_pull_nm`, multiplies by `const_k` and pushes the
   resulting force into the pull `CustomExternalForce`. Then OpenMM
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
First `setup_cg` downloads `martini_v3.0.0.itp` (~2 MB) into
`~/.cache/pymol_imd/martini3/`. Subsequent runs are offline.

### Diverging simulation
If the structure flies apart, `imd_run` stops automatically with a hint:
> `imd_run: stopped (divergence or error). Adjust const_k / max_pull_nm.`

Try `imd_change const_k=10000 max_pull_nm=0.3` and re-run.

---

## Project layout

```
pymol_imd/
├── __init__.py        Re-exports setup_* / imd_*; sets PYTORCH_ENABLE_MPS_FALLBACK
├── cli.py             pymol_imd executable (argparse → pymol -d ...)
├── commands.py        PyMOL command surface (setup_*, imd_*)
├── imd.py             IMDSystem driver: CustomExternalForce + Qt timer + ONIOM hook
├── prep.py            RCSB fetch / Modeller.addHydrogens / AM1-BCC / prepared-PDB
├── backends/
│   ├── base.py        Backend Protocol + SystemBundle dataclass
│   ├── aa.py          Amber backend
│   ├── cg.py          Martini 3 backend (vermouth + martini_openmm)
│   └── mlmm.py        ONIOM ML/MM backend (UMA / ORB / MACE / AIMNet2)
└── tests/
    └── smoke.py       Headless AA + CG smoke
```

---

## License

GPLv3 (see [LICENSE](LICENSE)).

Acknowledgements: this project sits on top of OpenMM, PyMOL, vermouth /
martini_openmm, and the MACE / UMA / ORB / AIMNet2 MLIPs. See their
respective licences (MACE-OMOL-0 is academic-use only).
