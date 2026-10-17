# Usage and development reference

[Back to the quick start](../README.md). Run all commands from the repository root.

- [Installation and MLIP backends](#installation-and-mlip-backends)
- [Solvent and minimization](#solvent-and-minimization)
- [CueMol3 details](#cuemol3-details)
- [CLI flags](#cli-flags)
- [IMDv3 socket mode](#imdv3-socket-mode-client--server)
- [PyMOL commands](#use-from-inside-pymol)
- [Scripted dragging](#scripted-dragging-without-a-viewer)
- [Recording](#recording-trajectories-forces-and-state)
- [ML/MM region selection](#mlmm-region-selection)
- [Force handling and recovery](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [Project layout](#project-layout)
- [Validation](#validation)

## Installation and MLIP backends

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

For UMA, use `pixi run --no-install` after installation so pixi does not restore
MACE's locked `e3nn` version before launching:

```bash
pixi run --no-install pymol_imd local input.pdb --level mlmm --ml-backend uma \
    --ml-selection "resi 1" --ml-charge 0 --ml-mult 1 --minimize
```

The optional UMA installation replaces dependencies in the active environment.
A normal `pixi run` restores the default locked versions; rerun `install-uma`
before the next UMA session. MACE and UMA require different `e3nn` versions.

### CUDA acceleration (Linux + NVIDIA)

The separate `gpu` environment provides a CUDA-enabled torch build for
ML/MM. The default environment keeps its CPU torch build.

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

### MLIP backend reference

Supported MLIP backends for `--level mlmm` (in order of "ready to use"):

| Backend | Default | Login | Size | Notes |
|---|---|---|---|---|
| [AIMNet2](https://github.com/zubatyuk/aimnet) | **yes** | no | ~10 MB (cached on first use) | Fastest CPU; charge / mult go to the constructor |
| [MACE-OMOL-0](https://github.com/ACEsuit/mace-foundations) | no | no | ~400 MB | ASL license (academic only) |
| [ORB](https://github.com/orbital-materials/orb-models) | no | no | ~100 MB | Charge and spin multiplicity from `atoms.info` |
| [UMA](https://huggingface.co/facebook/UMA) | no | **yes** | gated | Needs HF auth + `pixi run install-uma` |

## Solvent and minimization

**Solvent:** `aa` and `mlmm` default to **GBn2 generalized-Born implicit solvent**
(`NoCutoff`, NVT) — electrostatic screening without explicit waters or a box. Pass
`--implicit-solvent none` for the old in-vacuo behavior, or `--pbc` for an explicit
periodic box with PME/NPT (which disables GB). `cg` (Martini) keeps its own solvent
treatment. For ML/MM, both the full system and the subtracted MM region include GB.
The subset uses its own Born radii, an approximation to the full-system
environment.

The minimization stage relaxes the Amber MM system. Frozen ML/MM corrections
and mouse-pull forces are cleared before minimization; the ML correction is
recomputed from the relaxed coordinates when production starts.

## CueMol3 details

The `cuemol connect` client accepts `--const-k`, `--max-force-kj`,
`--drag-level atom|residue|chain`, and `--frame-interval-ms`. Set
`--max-force-kj` on both the engine and client; the smaller cap takes effect.
The client panel shows **(capped)** when it limits a pull. Trajectories,
force CSVs and state CSVs are written by the engine.

To keep the viewer open after disconnecting, launch it separately and connect
to the `endpoint` printed by `CUEMOL_READY`:

```bash
pixi run cuemol launch --app path/to/CueMol3.app
pixi run cuemol connect path/to/prepared.pdb \
    --endpoint http://127.0.0.1:PORT --port 8888
```

Use separate terminals for these commands. A normal CueMol launch without a
debug endpoint cannot be attached this way. The server may be remote via
`--host`; the viewer debug endpoint must remain on loopback. Each connection
writes client evidence to a unique directory under `.pixi/cuemol3-runs/`.

This prototype supports one visible viewport and one native worker. It loads
a new ball-and-stick object with CueMol's default secondary-structure painting
and element colors on each connection. CG uses explicit prepared-PDB bonds
and neutral bead coloring. It keeps the last object as a
snapshot after detaching, and uses orthographic projection for pulling.
Leave perspective disabled while pulling. Per-atom coordinate updates can
be slow for large systems. It depends on CueMol3's internal worker API;
CueMol3 2.3.21.544 was tested with AA QBP (3,536 atoms), Martini QBP (528
beads), and an AIMNet2 ML/MM peptide (33 atoms), using the engine's default
temperature and friction, with force capping verified at 1,000 kJ/mol/nm.
Physical mouse checks covered atom/residue/chain
pulling, force capping, pause/resume, reconnect, and server loss across
these sessions. Other CueMol versions and larger systems remain unverified.

Adapter regression checks:

```bash
pixi run smoke-cuemol
pixi run smoke-cuemol-input  # isolated input-routing checks; uses an existing Node.js
```

The input-routing check uses a DOM stub and does not replace physical mouse
verification. Node.js is only used for this optional development check.

## CLI flags

The `pymol_imd` CLI requires an explicit subcommand:

| Subcommand | Behavior |
|---|---|
| `launch` | Configure and open PyMOL; add `--run` to start production |
| `local` | Configure, open PyMOL and start production automatically |
| `serve` | Run the headless IMDv3 engine |
| `connect` | Open PyMOL as an IMDv3 client |

The former implicit form `pymol_imd <PDB>` was removed in 0.2.0. To inspect
the generated PyMOL command without launching:

```bash
pixi run pymol_imd launch input.pdb --level aa --print-only
```

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
| `--const-k K` | launch/local | Pull-force scale (kJ/mol/nm²); default per backend (AA/MLMM 50000, CG 10000) |
| `--max-force-kj F` | setup/connect | Per-atom \|F\| cap (kJ/mol/nm, default 5000); excessive pulls keep their direction at this magnitude |
| `--integrator {langevin,langevin_middle}` | setup | Langevin scheme; default per backend (AA `langevin_middle`, CG/MLMM `langevin`) |
| `--platform {auto,cuda,opencl,cpu,reference}` | setup | OpenMM compute platform for the MM engine; `auto` (default) picks the fastest available (CUDA > OpenCL > CPU). Separate from ML/MM `--ml-device` |
| `--timestep-ps PS` | setup | MD timestep; default per backend (AA 0.002 / CG 0.020 / MLMM 0.0005) |
| `--friction PER_PS` | setup | Langevin friction 1/ps (default 1.0) |
| `--pressure BAR` | setup | Barostat pressure with `--pbc` (default 1.0) |
| `--energy-cutoff-kj E` | setup | \|U\| or kinetic energy above this triggers rollback and recovery (default 1e9) |
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
machines); `local` uses one MD worker and a Qt viewer loop in the same process.
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

**Responsiveness**: local MD runs on a dedicated worker so OpenMM and MLIP
calculations do not block the Qt event loop. PyMOL coordinate reads and drawing
stay on the GUI thread, with at most one MD frame in flight. Mouse edits made
during computation are consumed once on the next frame. Each completed state
is drawn before the next MD frame is submitted, avoiding competing render and
compute workloads on the same GPU. Stop and live parameter changes wait for
the pending frame before touching OpenMM or recording streams.

The molecular update rate still depends on MD/MLIP and rendering costs. Smaller
`nsteps_per_frame` values shorten the wait for a new state without changing the
integration timestep. Large ML regions can remain expensive even with a
responsive viewer. Socket mode also adds network and buffering latency; its
client interval is tunable with `imd_connect host, port, frame_interval_ms=4`
(default 8 ms). Raising server `--rate` sends fewer frames with more MD steps
between them. Prefer `local` for computation on the viewer's machine.

Wire units follow the IMDv3 spec (angstrom, kJ/mol/angstrom, ps). Only IMDv3 is
implemented (not VMD's older IMDv2).

---

## Use from inside PyMOL

`pymol_imd` registers the same commands at the PyMOL prompt:

```
PyMOL> from pymol_imd import *
PyMOL> imd_setup_aa input.pdb
PyMOL> imd_minimize             # converges on stable structure
PyMOL> imd_record traj=traj.xtc, force_csv=forces.csv  # arm recording (optional)
PyMOL> imd_run                  # production loop (recording starts here)
PyMOL> imd_change const_k=20000, max_force_kj=2500  # change sensitivity and cap live
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

## Scripted dragging without a viewer

`pymol_imd.scripted.run_drag` repeats virtual mouse offsets at every MD step.
It uses the same force scale, per-atom force cap, group expansion,
ML/MM correction, recovery and recording as interactive dragging. It never
replaces the molecular coordinates. Select atoms by their zero-based indices
in the prepared OpenMM topology.

```python
from pathlib import Path

from pymol_imd.backends.aa import AABackend
from pymol_imd.imd import IMDConfig, IMDSystem
from pymol_imd.scripted import run_drag

backend = AABackend()
bundle = backend.build_system(
    Path("examples/qbp.pdb"), work_dir=Path(".pixi/scripted/qbp/build")
)
system = IMDSystem(bundle, IMDConfig(
    const_k=backend.recommended_const_k,
    integrator=backend.recommended_integrator,
    timestep_ps=backend.recommended_timestep_ps,
    nsteps_per_frame=1,
))
for _ in range(20):
    if system.minimize_step():
        break
else:
    raise RuntimeError("Minimization did not converge")

domain_b = [atom.index for atom in bundle.topology.atoms()
            if 90 <= int(atom.residue.id) <= 180]
system.set_recording(
    traj=".pixi/scripted/qbp/run.xtc",
    force_csv=".pixi/scripted/qbp/forces.csv",
    state_csv=".pixi/scripted/qbp/state.csv",
)
result = run_drag(system, domain_b, [0.05, 0.0, 0.0],
                  duration_ps=1.0, release_ps=1.0)
print(result)
```

The offset is in angstrom and is measured from the current coordinates each
step: `0.05` angstrom with `const_k=50000` requests `250` kJ/mol/nm per selected
atom. It is not a promised final displacement. `duration_ps` and `release_ps`
are simulated time and must be whole numbers of integration steps. A constant
three-vector applies to every selected atom; an `(n_selected, 3)` array supplies
individual vectors. A callback `offsets(elapsed_ps, positions_angstrom)` may
compute changing directions or targets from read-only full-system coordinates;
its return value follows the selected-index order.

This synchronous API requires a stopped system, closed recorders and
`nsteps_per_frame=1`. It opens the configured outputs, runs, releases forces and
closes outputs even if the callback raises. Use a new output path for each run.
`system.apply_drag(indices, offsets_angstrom)` is also available for code that
already owns the MD step loop; call it on the simulation thread before stepping
and call `clear_pull_forces()` on release. Scripted runs are recorded separately
from physical mouse experiments.

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
  convention) — add 1 to match a PyMOL `index` selection. Excessive pulls record
  their capped applied forces, so no row exceeds `max_force_kj`.
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
4. **Loop**: the Qt thread consumes each mouse edit once and displays accepted
   coordinates with `cmd.load_coordset`. One worker advances OpenMM, so the
   viewer stays responsive during MD and ML inference. No more than one frame
   is queued. The timestep and steps per frame are unchanged. In the PyMOL
   socket client, each input/display exchange also completes before another
   Qt callback can run it, so a redraw cannot turn a server frame into input.
5. **Safety**: each pull force is capped at `max_force_kj` while preserving its
   direction. Non-finite input rejects the whole pull frame.
   Invalid coordinates, excessive potential or kinetic energy, and invalid
   ML/MM forces reject the MD frame before recording. The engine restores its
   checkpoint and retries without pulling, then with zero velocities. A rejected
   gesture stays released until the client releases it. Only failed recovery
   stops MD. Between MD frames, socket pulls expire if no fresh force packet
   arrived for 250 ms. A fresh drag completed between frames is applied once;
   pause/resume discards pending input.

These checks do not monitor bond lengths or structural strain. Sustained pulling
can distort a molecule while its energy remains below the configured cutoff;
successful recovery and finite recorded coordinates do not establish structural
integrity.

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
repo. Request access at https://huggingface.co/facebook/UMA and log in with
`pixi run --no-install huggingface-cli login`. After `pixi run install-uma`,
launch UMA with `pixi run --no-install` to retain its required dependencies.

### CG initial Martini download
First `imd_setup_cg` downloads `martini_v3.0.0.itp` (~15.3 MB) into
`~/.cache/pymol_imd/martini3/`. The download is atomic and SHA-256 checked;
subsequent runs are offline.

### Diverging simulation
When energy exceeds `--energy-cutoff-kj`, a state becomes non-finite, or OpenMM
integration raises, the driver restores the checkpoint preceding that frame
and retries without the user's pull. A second retry also zeros velocities;
the Langevin thermostat restores thermal motion. The timestep stays unchanged.
These recovery events are logged and alter the trajectory's sampling, so they
must be considered when interpreting results.

Rejected frames are excluded from trajectories and CSVs. Both local and socket
runs continue after successful recovery. If both retries fail, the driver leaves
the restored coordinates in place and stops. Lower `const_k` / `max_force_kj`,
shorten `--timestep-ps`, or inspect the starting structure before restarting.

---

## Project layout

```
pymol_imd/
├── __init__.py        Re-exports setup_* / imd_*; sets PYTORCH_ENABLE_MPS_FALLBACK
├── cli.py             pymol_imd executable (argparse → pymol -d ...)
├── commands.py        PyMOL command surface (setup_*, imd_*)
├── constants.py       Single source of truth: tunable defaults + fixed physical constants
├── imd.py             IMDSystem: forces, stepping, recovery and recording
├── interactive.py     One local MD worker; Qt-only viewer coordinate exchange
├── scripted.py        Continuous Python-driven drag and release through the same force path
├── protocol.py        IMDv3 wire format: pack/unpack, header enum, endianness oracle
├── transport.py       TCP framing: recv_exact, IMDConnection, bind/accept/connect
├── server.py          IMDServer: headless OpenMM engine streaming IMDv3 frames
├── client.py          IMDClient (headless) + PyMOLIMDClient (Qt-timer viewer)
├── cuemol/            Experimental CueMol3 launcher, bridge and viewer input
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
    ├── smoke_scripted_drag.py Code-driven drag parity, timing, release and cleanup
    ├── smoke_socket.py Headless IMDv3 server<->client round-trip + protocol regression
    ├── smoke_cli.py    Headless CLI/subcommand/coercion regression
    ├── smoke_interactive.py Local worker responsiveness and lifecycle checks
    ├── smoke_cuemol.py CueMol mapping, force transport and launcher checks
    ├── smoke_cuemol_input.cjs Isolated viewer input-routing checks
    ├── smoke_pymol.py  Real PyMOL command + Qt timer integration for all backends
    └── refs/           Pinned coordinates + reference ML/MM delta arrays
```

---

## Validation

Versioned QBP and chorismate mutase inputs are in [examples/](../examples/README.md).
The [molecular mouse experiment skill](../.agents/skills/pymol-imd-mouse-experiments/SKILL.md)
describes physical domain dragging in AA/CG and ligand dragging in ML/MM, with
trajectory analysis and explicit structural criteria. These manual computer-use
experiments are separate from the headless smoke tests; save their generated
evidence under an ignored directory such as `.pixi/cua/`.

`pixi run smoke-recovery` checks rollback and bounded retry, preservation of the
ML/MM correction, and rejection of invalid frames from XTC/DCD/PDB/CSV outputs.

`pixi run smoke-scripted-drag` compares scripted and mouse-path trajectories
and force CSVs, and checks continuous input, force capping, timed release,
input validation and cleanup after callback failures.

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
default selection explicitly includes protein residues 1-3 and water residues
130-132 from 1AKI to retain the original reference atom set after preserving
PDB identifiers. The current references were verified with the OpenMM OpenCL platform and CPU
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
The cross-device tolerance applies only to these comparisons; pinned numerical
references keep their original tolerance. ORB/UMA require their optional
installation tasks. Use separate working environments when checking UMA and
MACE/ORB because their dependency versions conflict.
