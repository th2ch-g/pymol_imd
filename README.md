# pymol_imd

Interactive molecular dynamics with **PyMOL** and an experimental **CueMol3**
adapter. Drag atoms in the viewer while OpenMM advances the simulation.

| Level | Model | Typical use |
|---|---|---|
| `aa` | Amber ff14SB / GAFF with AM1-BCC ligand charges | Atomistic proteins, DNA, RNA, lipids and ligands |
| `cg` | Martini 3 | Coarse-grained proteins; 20 fs timestep |
| `mlmm` | Amber MM + an ONIOM MLIP correction | A selected region treated with AIMNet2, MACE, ORB or UMA |

[Install](#install) · [PyMOL](#pymol) · [CueMol3](#cuemol3-experimental) ·
[Pull settings](#pull-settings)

## Install

Requires [pixi](https://pixi.sh). Run commands from the repository root.
Native platforms are Linux (`linux-64`) and Apple Silicon (`osx-arm64`).
Use WSL2 on Windows or a Linux VM on Intel Mac.

```bash
pixi install
```

This installs all three simulation levels, with **AIMNet2** as the default
MLIP and **MACE** as an alternative. Install ORB or UMA with
`pixi run install-orb` or `pixi run install-uma`. UMA also requires model access;
use `pixi run --no-install` after `install-uma` to retain its dependencies.

## PyMOL

Replace `input.pdb` with your structure. Choose one simulation level:

```bash
# All-atom Amber
pixi run pymol_imd local input.pdb --level aa --minimize

# Coarse-grained Martini 3
pixi run pymol_imd local input.pdb --level cg --minimize

# ML/MM with AIMNet2 on residue 1
pixi run pymol_imd local input.pdb --level mlmm --minimize \
    --ml-selection "resi 1" --ml-charge 0 --ml-mult 1
```

For ML/MM, choose the selection, net charge and spin multiplicity to match
your molecule. These three arguments are required.

| Action | PyMOL control |
|---|---|
| Pull an atom | **Ctrl + left-drag** in the automatically selected Editing mode |
| Pull a residue or chain | `imd_drag_level residue` or `imd_drag_level chain` |
| Return to atom pulling | `imd_drag_level atom` |
| Stop / resume production | `imd_stop` / `imd_run` |
| Change sensitivity and cap | `imd_change const_k=20000, max_force_kj=2500` |

Picked atoms or bonds can change the editing bindings. Click **Unpick** to
restore the usual controls, or follow **MovA** in the lower-right mouse grid.

`local` starts production after setup and optional minimization. Use `launch`
to configure the viewer without starting production, or add `--print-only`
to inspect its generated command.

## CueMol3 (experimental)

Install the CueMol3 Electron app separately. The adapter uses its own profile.

**1. Start the engine in one terminal:**

```bash
pixi run pymol_imd serve input.pdb --level aa --minimize --port 8888
```

**2. Connect the viewer in another terminal:** use the **exact prepared PDB**
printed as `prepared=...` in the server's `IMD_READY` line. Replace the example
paths with that file and the installed app bundle (macOS) or executable (Linux).

```bash
pixi run cuemol connect path/to/prepared.pdb \
    --app path/to/CueMol3.app --port 8888
```

| Action | Upper-right IMD panel |
|---|---|
| Pull an atom | Turn **Pull atoms** on, then plain left-drag from an atom's center |
| Release | Release the mouse; **Escape** also returns to camera navigation |
| Pull a residue or chain | Choose **Drag level**; changing it releases the current pull |
| Move the camera | Turn pulling off for left-drag rotation; right/middle buttons, modifiers and wheel remain available |
| Pause / resume | **Pause / Resume**, which also releases active pulls |
| End the session | **Disconnect**; a viewer started with `connect --app` closes with its client |

Stop the engine with **Ctrl+C** in its terminal.

- **CG:** use `serve --level cg` and add `--const-k 10000` to `cuemol connect`.
- **ML/MM:** use `serve --level mlmm` with `--ml-selection`, `--ml-charge` and
  `--ml-mult`, as in the PyMOL example.
- **Remote engine:** add `--host` to the client. Trajectories and CSVs are
  recorded by the engine.

Keep perspective disabled while pulling. The adapter currently supports one
visible viewport and one native worker. CueMol3 **2.3.21.544** was physically
tested with AA QBP (3,536 atoms), Martini QBP (528 beads), and an AIMNet2 ML/MM
peptide (33 atoms). Other versions and larger systems remain unverified.

## Pull settings

Excessive finite forces keep their direction and are **capped per atom**.
The default cap is **5,000 kJ/mol/nm**. Change it at startup with
`--max-force-kj`, or during a local PyMOL run with `imd_change`:

```bash
pixi run pymol_imd local input.pdb --level aa --minimize --max-force-kj 2500
```

For socket connections, set `--max-force-kj` on both the engine and client;
the smaller cap takes effect. CueMol's panel shows **(capped)** when the client
limits a pull. `--const-k` controls the force response to mouse displacement.

Invalid MD frames are rolled back and retried without pulling, then with zero
velocities; successful recovery keeps MD running. Rejected frames are excluded
from recordings. Recovery changes
trajectory sampling. Sustained pulling may still distort the structure before
an energy limit is reached.

## Simulation defaults and output

- **AA / ML/MM:** GBn2 implicit solvent, NVT, no periodic box. Use
  `--implicit-solvent none` for in-vacuo MD or `--pbc` for PME/NPT without GB.
- **CG:** Martini's cutoff requires a periodic box, which is created
  automatically. `--pbc` additionally enables pressure coupling.
- **Minimization:** `--minimize` relaxes the MM system before production.
  ML/MM corrections are recomputed afterward; there is no equilibration stage.
- **Recording is on by default:** trajectory, applied pull-force CSV and state
  CSV. Use `--no-record` to disable them. Socket output names are unique per
  connection.

## Documentation and checks

Use `pixi run pymol_imd local --help`, `pixi run pymol_imd serve --help`,
or `pixi run cuemol connect --help` to list command options.
See [example structures](examples/README.md) and
[physical mouse experiments](.agents/skills/pymol-imd-mouse-experiments/SKILL.md).

Core checks include `pixi run smoke`, `pixi run smoke-mlmm`,
`pixi run smoke-interactive`, `pixi run smoke-pymol`, `pixi run smoke-socket`,
`pixi run smoke-cuemol` and `pixi run smoke-cuemol-input`.
Headless and scripted checks complement physical mouse verification.

## License

[MIT](LICENSE). OpenMM, PyMOL, CueMol3, vermouth, martini_openmm and the MLIP
models retain their respective licenses; MACE-OMOL-0 is for academic use only.
