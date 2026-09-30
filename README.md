# pymol_imd

Multiscale Interactive Molecular Dynamics inside PyMOL.

`pymol_imd` lets you drag atoms around in the PyMOL viewer and feel the
molecule push back through a live OpenMM simulation. Three resolution
levels are supported from the same plugin:

| Level | Forcefield | Use when |
|---|---|---|
| `aa`   | Amber ff14SB / GAFF (AM1-BCC ligand charges) | Atomistic protein / DNA / RNA / lipid / ligand systems |
| `cg`   | Martini 3 via `vermouth` + `martini_openmm`  | Protein coarse graining, fast 20 fs timestep |
| `mlmm` | ONIOM-style: full-system Amber MM + MLIP delta on a selected region | Reactive / electronic-effect-sensitive regions |

Supported MLIP backends for `--level mlmm` (in order of "ready to use"):

| Backend | Default | Login | Size | Notes |
|---|---|---|---|---|
| [AIMNet2](https://github.com/zubatyuk/aimnet) | **yes** | no | ~10 MB (cached on first use) | Fastest CPU; charge / mult go to the constructor |
| [MACE-OMOL-0](https://github.com/ACEsuit/mace-foundations) | no | no | ~400 MB | ASL license (academic only) |
| [ORB](https://github.com/orbital-materials/orb-models) | no | no | ~100 MB | Charge and spin multiplicity from `atoms.info` |
| [UMA](https://huggingface.co/facebook/UMA) | no | **yes** | gated | Needs HF auth + `pixi run install-uma` |

**Solvent:** `aa` and `mlmm` default to **GBn2 generalized-Born implicit solvent**
(`NoCutoff`, NVT) — electrostatic screening without explicit waters or a box. Pass
`--implicit-solvent none` for the old in-vacuo behavior, or `--pbc` for an explicit
periodic box with PME/NPT (which disables GB). `cg` (Martini) keeps its own solvent
treatment. For ML/MM the GB term is applied to both the full system and the
subtracted MM region so the ONIOM energy stays consistent.

The minimization stage relaxes the Amber MM system. Frozen ML/MM corrections
and mouse-pull forces are cleared before minimization; the ML correction is
recomputed from the relaxed coordinates when production starts.

---

## Install

Requires [pixi](https://pixi.sh). Native support: `linux-64`, `osx-arm64`.
Intel Mac (`osx-64`) and Windows users should run via a Linux VM / WSL2.

```bash
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

### CUDA acceleration (Linux + NVIDIA)

A separate `gpu` environment resolves a CUDA torch build — the single biggest
ML/MM speedup (10-100x vs CPU). The default env stays CPU-only and unaffected.

```bash
pixi install -e gpu
pixi run -e gpu pymol_imd local input.pdb --level mlmm \
    --ml-selection "resi 30-50" --ml-charge 0 --ml-mult 1
```

AIMNet2's CUDA helper compilation needs CUDA Toolkit headers visible to the
host compiler. If it reports `cuda.h: No such file or directory`, load the
site's CUDA Toolkit and expose its existing `CUDA_HOME` include directory:

```bash
export CPATH="${CUDA_HOME:?Load the CUDA Toolkit first}/include${CPATH:+:$CPATH}"
```

`--ml-compile` is opt-in for every backend, including ORB and UMA. ORB uses
full float32 precision so CUDA does not silently enable TF32 matrix products.
For MACE, ORB, and UMA, `--ml-mult` is passed as spin multiplicity (`2S+1`);
a singlet is `1`.

---

## Quick start (CLI)

```bash
# Atomistic IMD on ubiquitin
pixi run pymol_imd local 1ubq.pdb --level aa --minimize

# CG (Martini 3, 20 fs timestep)
pixi run pymol_imd local 1ubq.pdb --level cg --minimize

# ONIOM ML/MM with AIMNet2 (default) on residue 1
pixi run pymol_imd local 1ubq.pdb --level mlmm \
    --ml-selection "resi 1 and resn MET" --ml-charge 0 --ml-mult 1
```

PyMOL opens with Editing mode preselected, so **Ctrl + left-drag** on an
atom pulls it; the IMD loop converts the drag into a per-atom force on
OpenMM each frame.
Picked atoms or bonds can change PyMOL's editing bindings. Click **Unpick**
to return to the usual controls, or follow the **MovA** entry in the mouse
grid at the lower right.

Show the generated PyMOL command without launching:

```bash
pixi run pymol_imd launch 1ubq.pdb --level aa --print-only
```

The CLI requires one of four subcommands:

- `launch`: configure and open PyMOL; add `--run` to start production.
- `local`: configure, open PyMOL, and start production automatically.
- `serve`: run the headless IMDv3 engine.
- `connect`: open PyMOL as an IMDv3 client.

The former implicit form `pymol_imd <PDB>` was removed in 0.2.0; use
`pymol_imd launch <PDB>` or `pymol_imd local <PDB>`.

### CLI flags

All numeric / string knobs are optional; each falls back to its single-source
default in `pymol_imd/constants.py` (per-backend where noted).
`setup` below means the `launch`, `local`, and `serve` subcommands.

| Flag | Scope | Description |
|---|---|---|
| `--level {aa,cg,mlmm}` | setup | Simulation backend (default: `aa`) |
| `--pbc` | setup | Enable periodic box / NPT. Without it (default): AA/MLMM use GBn2 implicit solvent, CG auto-boxes — all NVT |
| `--implicit-solvent {auto,none,gbn2,obc2,obc1,gbn,hct}` | setup: `aa`, `mlmm` | GB implicit-solvent model. Default `auto` = GBn2 (off under `--pbc`); `none` = in-vacuo |
| `--temperature K` | setup | Langevin thermostat temperature (default 300) |
| `--nsteps-per-frame N` | launch/local | MD steps per GUI frame (default 1); `serve` uses `--rate` |
| `--drag-level {atom,residue,chain}` | launch/local | Mouse-drag granularity; `residue`/`chain` grab and pull a whole residue/chain (default `atom`) |
| `--const-k K` | launch/local | Pull-force scale (kJ/mol/nm); default per backend (AA/MLMM 50000, CG 10000) |
| `--max-force-kj F` | setup/connect | Per-atom \|F\| safety limit (kJ/mol/nm, default 5000); over-limit pulls are skipped, not clamped |
| `--integrator {langevin,langevin_middle}` | setup | Langevin scheme; default per backend (AA `langevin_middle`, CG/MLMM `langevin`) |
| `--platform {auto,cuda,opencl,cpu,reference}` | setup | OpenMM compute platform for the MM engine; `auto` (default) picks the fastest available (CUDA > OpenCL > CPU). Separate from ML/MM `--ml-device` |
| `--timestep-ps PS` | setup | MD timestep; default per backend (AA 0.002 / CG 0.020 / MLMM 0.0005) |
| `--friction PER_PS` | setup | Langevin friction 1/ps (default 1.0) |
| `--pressure BAR` | setup | Barostat pressure with `--pbc` (default 1.0) |
| `--energy-cutoff-kj E` | setup | \|U\| above this halts the loop as diverged (default 1e9) |
| `--frame-interval-ms MS` | launch/local/connect | Qt/client timer interval |
| `--nonbonded-cutoff-nm NM` | setup | Nonbonded cutoff; default per backend (AA/MLMM 1.0, CG 1.1) |
| `--box-padding-nm NM` | setup | Bounding-box padding when a box is built (default 1.0) |
| `--no-mouse-force` | launch/local | Disable user-pull |
| `--minimize` | setup | Minimize after setup |
| `--min-max-iter N` | setup | Minimizer iterations per chunk, with `--minimize` (default 50) |
| `--min-max-chunks N` | setup | Max minimize chunks, with `--minimize` (default 200) |
| `--min-tol-kj E` | setup | Minimize convergence tolerance kJ/mol, with `--minimize` (default 1e-3) |
| `--run` | launch | Run `imd_run` after setup; `local` implies it and `serve` starts after GO |
| `--traj-out PATH` | launch/local/serve | Record a trajectory (`.xtc`/`.dcd`/`.pdb` by extension) |
| `--traj-interval N` | setup | Trajectory: write every N GUI/server frames (default 1) |
| `--force-csv PATH` | setup | Record user pull forces to CSV (`frame,step,time_ps,atom_index,fx,fy,fz,fmag`, atom index 0-based) |
| `--force-csv-interval N` | setup | Force CSV: write every N GUI/server frames (default 1) |
| `--state-csv PATH` | setup | Record thermodynamic state CSV (energies + `temperature_k`; `+volume_nm3,density_g_cm3` under `--pbc`) |
| `--state-csv-interval N` | setup | State CSV: write every N GUI/server frames (default 1) |
| `--no-record` | setup | Disable the default trajectory + force-CSV + state-CSV recording |
| `--topology-dir DIR` | setup | Output dir for intermediate / topology files (default `./<stem>_topology`) |
| `--ligand-sdf PATH` | setup: `aa` | Ligand SDF for GAFF + AM1-BCC parameterisation |
| `--ph PH` | setup: `aa` | `Modeller.addHydrogens` protonation pH (default 7.4) |
| `--gaff-version VER` | setup: `aa` | GAFF version for ligand templates (default `gaff-2.2.20`) |
| `--ff NAME` | setup: `cg`  | Martini forcefield label (default: `martini3001`) |
| `--no-elastic-network` | setup: `cg` | Skip martinize2 `-elastic` |
| `--martinize-maxwarn N` | setup: `cg` | martinize2 `-maxwarn` value (default 99999) |
| `--ml-selection SEL` | setup: `mlmm` | **Required.** PyMOL selection language (`resi 1 and resn MET`, `chain A and not resn HOH`, …) |
| `--ml-backend {uma,orb,mace,aimnet2}` | setup: `mlmm` | MLIP backend (default: `aimnet2`) |
| `--ml-device {auto,cuda,mps,cpu}` | setup: `mlmm` | Torch device; `auto` prefers cuda → mps → cpu |
| `--ml-charge Z`, `--ml-mult M` | setup: `mlmm` | **Required.** ML region net charge / spin multiplicity |
| `--link-atom-method {scaled,fixed,none}` | setup: `mlmm` | QM/MM boundary link-atom placement (default `scaled`) |
| `--ml-compile` | setup: `mlmm` | Opt-in `torch.compile` for the MLIP (numerically ~identical) |
| `--enable-cueq` | setup: `mlmm` | Opt-in cuEquivariance for MACE (CUDA; `pixi run install-cueq`) |
| `--pymol PATH` | launch/local/connect | Override PyMOL executable |
| `--print-only` | launch/local/connect | Show the underlying `pymol -d ...` and exit |

---

## IMDv3 socket mode (client / server)

By default MD and the viewer run in one process. Optionally you can split them
over a TCP socket using the **IMDv3 protocol**: run the MD on a (possibly
remote) headless compute server and drive it from a local PyMOL — dragging
atoms works exactly as in-process.

```bash
# Normal local work needs NO socket: `local` is the classic same-process mode
# (= launch + run) in one command — the most robust path for interactive drag.
pixi run pymol_imd local 1ubq.pdb --level aa

# Split across machines with the socket. On the compute host (no PyMOL needed):
pixi run pymol_imd serve 1ubq.pdb --level aa --port 8888
#   -> prints  IMD_READY port=8888 prepared=<path>/1ubq_prepared.pdb

# On your workstation, load that SAME prepared PDB and connect:
pixi run pymol_imd connect <path>/1ubq_prepared.pdb --host <server> --port 8888
```

Only `serve` + `connect` use the socket (for splitting compute across
machines); `local` runs everything in one process via the classic QTimer loop.
The `connect` client must load the server's **prepared** PDB (the
`prepared=<path>` from the `IMD_READY` line) so the atom order matches OpenMM's
topology. Inside PyMOL the same is available as `imd_connect host, port` /
`imd_disconnect`.

`serve` accepts system/backend, minimization, safety, and recording flags
(e.g. `--ml-selection` for `--level mlmm`, `--minimize`) plus server flags:
`--host`, `--port`, `--rate` (MD steps per streamed frame), `--send-velocities`,
`--send-forces`. GUI-only flags such as `--const-k`, `--drag-level`,
`--frame-interval-ms`, and `--pymol` are rejected. `local` takes only launch flags (it is **not** a socket
server, so it has no `--host/--port/--rate`). `connect` shares the setup view
(cartoon for AA/ML-MM, colored spheres for CG — auto-detected) and 3-Button
Editing, so Ctrl+left-drag pulls atoms just like in-process.

**Smoothness**: the same-process path (`local`) is the smoothest — zero
latency, memory-direct. Over a socket every frame and every drag makes a TCP
round-trip, so motion is slightly less fluid. Lower the client tick to tune it
(`imd_connect host, port, frame_interval_ms=4`; default 8 ms) or raise `--rate`
on the server (fewer, larger-step frames). For normal local interactive work
prefer `local`; reach for the socket only when compute must live elsewhere.

Wire units follow the IMDv3 spec (angstrom, kJ/mol/angstrom, ps). Only IMDv3 is
implemented (not VMD's older IMDv2).

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
PyMOL> imd_drag_level residue   # pull a whole residue/chain at once (atom|residue|chain)
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

## Recording trajectories, forces and state

`imd_record` arms output for the **next** `imd_run`. Recording starts when
`imd_run` is called and stops — files flushed + closed — on `imd_stop` (or on
an automatic divergence / error halt). `imd_record off` also closes active
outputs immediately without stopping MD. Minimization is never recorded.

```
PyMOL> imd_setup_aa input.pdb
PyMOL> imd_record traj=run.xtc, force_csv=pull.csv, state_csv=state.csv
PyMOL> imd_run                  # recording begins
PyMOL> imd_stop                 # files finalized
PyMOL> imd_record off           # disarm (optional)
```

- **Trajectory** format is chosen by file extension: `.xtc`, `.dcd`, `.pdb`.
- **Force CSV** logs the applied user pull force, one row per pulled atom per
  recorded frame, columns `frame,step,time_ps,atom_index,fx,fy,fz,fmag`
  (forces in kJ/mol/nm). `atom_index` is **0-based** (OpenMM / trajectory
  convention) — add 1 to match a PyMOL `index` selection. Frames where a drag
  exceeds `max_force_kj` are skipped (no row), so no row exceeds the limit.
- **State CSV** logs GROMACS-style thermodynamics, one row per recorded frame,
  columns `frame,step,time_ps,potential_energy_kj,kinetic_energy_kj,total_energy_kj,temperature_k`,
  plus `volume_nm3,density_g_cm3` under PBC. **Pressure is omitted**: OpenMM
  reports no instantaneous pressure and the IMD pull force contaminates the virial.
- All three default **on** — the CLI writes `<stem>.xtc` / `<stem>_forces.csv` /
  `<stem>_state.csv`; `--no-record` disables them together.
- `serve` creates collision-free files per successful client connection, for
  example `<stem>.session-0001.xtc`, and prints their paths in `IMD_SESSION`.
- CSV streams flush every 100 rows and always flush remaining rows on normal
  stop, disconnect, divergence, or error.
- `traj_interval` / `force_csv_interval` / `state_csv_interval` are in **GUI
  frames** (default 1): `traj_interval=5` writes every 5th frame.

From the CLI the same is available via flags; `local` runs immediately,
`launch` requires `--run`, and `serve` starts recording after each client's GO:

```bash
pixi run pymol_imd local input.pdb --level aa \
    --traj-out run.xtc --traj-interval 5 \
    --force-csv pull.csv --state-csv state.csv
```

Recording intervals are converted to MD steps at `imd_run` time and stay fixed
until the next run. Changing `nsteps_per_frame` keeps trajectory and CSV samples
aligned, including samples inside a GUI frame. Run `imd_stop` then `imd_run` to
derive new intervals from the current frame size.

---

## ML/MM region selection

`imd_setup_mlmm` requires `ml_selection`. The string is **PyMOL's full selection
language** (`and`, `or`, `not`, `within`, `byres`, `resi`, `resn`, `chain`,
`name`, …) — resolved through `cmd.iterate` against a temporary load of the
prepared PDB:

```bash
# A single residue
--ml-selection 'resi 1 and resn MET'

# Protein residues around a selected active-site residue
--ml-selection 'byres (chain A within 5 of (chain A and resi 42))'

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
   `const_k`. If any requested force exceeds `max_force_kj`, the whole frame's
   pull is skipped; otherwise it is pushed into the pull `CustomExternalForce`. Then OpenMM
   integrates one frame's worth of steps and the new coordinates are
   written back into PyMOL via `cmd.alter_state`.
5. **Safety**: every step potential energy is checked; NaN / |U| above
   `energy_cutoff_kj` halts the loop with a console message.

---

## Troubleshooting

### MLIP / Apple Silicon (MPS)
The currently bundled AIMNet2, MACE, and ORB paths do not run reliably on
MPS (for backend-specific Warp, dtype, or graph-operation reasons). The plugin
resolves this before model logging/thread setup and falls back to CPU with an
explicit notice. An explicitly requested unavailable MPS/CUDA device is an
error. CPU MLIPs are usable for small ML regions
(tens of atoms); pick a larger backend only if you have CUDA.

### UMA model gated
The UMA backend pulls weights from `facebook/UMA`, a HuggingFace gated
repo. After `pixi run install-uma`, run `pixi run huggingface-cli login`
and request access at https://huggingface.co/facebook/UMA.

### CG initial Martini download
First `imd_setup_cg` downloads `martini_v3.0.0.itp` (~15.3 MB) into
`~/.cache/pymol_imd/martini3/`. The download is atomic and SHA-256 checked;
subsequent runs are offline.

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
├── protocol.py        IMDv3 wire format: pack/unpack, header enum, endianness oracle
├── transport.py       TCP framing: recv_exact, IMDConnection, bind/accept/connect
├── server.py          IMDServer: headless OpenMM engine streaming IMDv3 frames
├── client.py          IMDClient (headless) + PyMOLIMDClient (Qt-timer viewer)
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
    ├── smoke_drag.py   Headless drag-level (atom/residue/chain) group-expansion smoke
    ├── smoke_socket.py Headless IMDv3 server<->client round-trip + protocol regression
    ├── smoke_cli.py    Headless CLI/subcommand/coercion regression
    ├── smoke_pymol.py  Real PyMOL command + Qt timer integration for all backends
    └── refs/           Pinned coordinates + reference ML/MM delta arrays
```

---

## Validation

`pixi run smoke-molecules` checks DNA, RNA, POPC/POPE lipids with and without
hydrogens, and an AM1-BCC ethanol ligand, including a bound ligand without
hydrogens. It verifies GBn2, preservation of
heavy-atom coordinates, atom counts, minimization, and finite MD forces.
Use `--platform cuda` in the GPU environment to check the same inputs on CUDA.

Nonstandard residues can gain missing hydrogens through `Modeller.addHydrogens`
when their atom names and heavy-atom bonds uniquely match a closed forcefield
template. Ambiguous protonation states require explicit input hydrogens.
With `--ligand-sdf`, a matching ligand already present in the PDB is
parameterized in place; otherwise, the SDF coordinates are appended.

`pixi run smoke-pymol` exercises AA, CG, and AIMNet2 ML/MM with a generated
tripeptide: setup, hydrogen completion, minimization followed by production,
recording, stop/restart, all drag levels, and cleanup after atom/object edits.
It uses real PyMOL commands and Qt timers, without requiring a visible window.

To run the same checks in a fresh GUI and export molecular renderings:

```bash
PYMOL_IMD_TEST_OUTPUT=.pixi/debug/gui PYTHONUTF8=1 \
    pixi run pymol -k -r pymol_imd/tests/smoke_pymol.py
```

The viewer exits after the checks. `-k` isolates the run from personal startup
scripts. Drag tests change atom coordinates through PyMOL; physical mouse
gestures still require a separate UI check.

`smoke-mlmm` uses the saved numerical references without replacing them. The
current references were verified with the OpenMM OpenCL platform and CPU
AIMNet2; a CPU-only OpenMM run can exceed the strict absolute tolerance because
its coordinates and forces use different precision. Check the logged main and
subset platforms before interpreting a numerical-reference failure.

`smoke-backends` checks a chosen MLIP with GBn2, link atoms, minimization,
production, trajectory/CSV recording, and optional CPU/CUDA force comparison:

```bash
pixi run -e gpu smoke-backends --backend aimnet2 --platform cuda --ml-device cuda --compare-cpu
pixi run -e gpu smoke-backends --backend mace --platform cuda --ml-device cuda --compare-cpu
```

`--molecule dna`, `rna`, or `popc` selects a whole-molecule ML region. The
`complex` and `complex_no_h` cases select a bound ligand and exercise the
ML/MM backend's `ligand_sdf` input, including missing-hydrogen completion.
The
cross-device tolerance applies only to these comparisons; pinned numerical
references keep their original tolerance. ORB/UMA require their optional
installation tasks. Use separate working environments when checking UMA and
MACE/ORB because their dependency versions conflict.

---

## License

MIT (see [LICENSE](LICENSE)).

Acknowledgements: this project sits on top of OpenMM, PyMOL, vermouth /
martini_openmm, and the MACE / UMA / ORB / AIMNet2 MLIPs. See their
respective licences (MACE-OMOL-0 is academic-use only).
