# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`pymol_imd` is a PyMOL plugin that runs Interactive Molecular Dynamics (IMD) inside the PyMOL viewer using OpenMM as the integrator. It supports three simulation backends, selected at setup time:

- **CG (Martini)** — coarse-grained MD via `setup_cg`
- **AA (Amber)** — all-atom MD via `setup_aa`
- **ML/MM** — hybrid ML/MM MD via `setup_mlmm`, wrapping the [`mlmm_toolkit`](https://github.com/t-0hmura/mlmm_toolkit)

The plugin must accept proteins, DNA, RNA, lipids, and small-molecule ligands; complete missing hydrogens when possible; parameterize ligands with AM1-BCC charges via AmberTools/OpenFF; and run in vacuo by default (no solvent, no PBC, no box). After `setup_*`, the standard flow is `minimize` → production-run (no equilibration), and user-driven pulling of atoms in the PyMOL viewer is translated into per-atom forces fed into OpenMM each frame.

## Toolchain & Environment

- **Package manager**: `pixi` (already installed system-wide). Do NOT use `pip` / `uv` directly; declare all deps in `pyproject.toml` under `[tool.pixi.dependencies]` (conda-forge) or `[tool.pixi.pypi-dependencies]`.
- **Platforms**: native support is `linux-64`, `osx-arm64`, `osx-64`. Windows is **WSL2-only** (no native `win-64` pixi build) because `ambertools` has no conda-forge Windows build. Document this in README; do not list `win-64` under `[tool.pixi.project] platforms`.
- **Python**: ≥3.11
- **Approved libraries** (already greenlit by the user):
  - core: `openmm`, `pymol-open-source`, `ambertools`, `openmmforcefields`, `openff-toolkit`, `rdkit`
  - CG: `vermouth` (martinize2) + `martini-openmm` (git+URL, no conda-forge build)
  - ML/MM: `mlmm_toolkit` (git+URL) + `pytorch`, `pytorch_scatter`, `pytorch_geometric`, `ase`, `parmed`, `biopython` from conda-forge
  - hydrogen completion: **only** `openmm.app.Modeller.addHydrogens` — do NOT pull in `pdbfixer`, `propka`, etc.
- Any other dependency requires explicit user approval before being added.
- **pixi environments**: All three backends (AA / CG / ML-MM) live in the single `default` environment. `pixi install` covers AA + CG + MACE-based ML/MM out of the box.
- **mlmm_toolkit-free**: As of the latest revision, `pymol_imd/backends/mlmm.py` builds the MLIP ASE Calculators directly (`mace_omol`, `ORBCalculator`, `AIMNet2Calculator`, `FAIRChemCalculator`). The previous `mlmm_toolkit` dispatcher dependency was removed — no need to clone the fork at `../mlmm_toolkit`.
- **CG forcefield**: Martini 3 `.itp` is downloaded once into `~/.cache/pymol_imd/martini3/` on first `setup_cg`. Online required for first run; subsequent runs are offline.
- **CG forces PBC**: Martini uses a 1.1 nm cutoff so it cannot run in vacuo. CG backend auto-creates a bounding-box around the molecule + 1 nm padding when `pbc=False`. AA/MLMM still default to NoCutoff vacuo.
- **ML/MM = ONIOM**: `E_total = E_MM(full) + E_ML(region) - E_MM(region)`. Each frame the ML region delta force `F_ML - F_MM_subset` is pushed into a dedicated `CustomExternalForce`. Link-atom handling for cross-region bonds is **not implemented yet**; pick selections that are molecularly closed (entire residues / chains).
- **MLIP model access**:
  - **MACE-OMOL-0 (default)**: public ASL-licensed model on GitHub releases of `mace-foundations`. No login, ~400 MB on first call, cached to `~/.cache/mace/`. Already covered by `pixi install` via `mace-torch`.
  - **UMA**: HuggingFace gated repo `facebook/UMA`. Requires `huggingface-cli login` + access approval. Install fairchem-core via `pixi run install-uma`.
  - **ORB**: install via `pixi run install-orb` (`orb-models`). Public model.
  - **AIMNet2**: install via `pixi run install-aimnet2` (`aimnet`). Model is bundled with the package.
- **Apple Silicon (MPS) support**: `pymol_imd.backends.mlmm._auto_torch_device` selects `cuda → mps → cpu`. On MPS the MACE dtype is forced to `float32` because MPS does not support float64.
- **MPS fallback**: If a torch op is unimplemented on MPS, export `PYTORCH_ENABLE_MPS_FALLBACK=1` before launching PyMOL so the missing ops silently drop to CPU.

### Common commands

```bash
pixi install                      # create / sync env (+ editable installs the pymol_imd CLI)
pixi run pymol_imd input.pdb --level aa                  # launch PyMOL with setup_aa
pixi run pymol_imd input.pdb --level cg --minimize       # CG + auto-minimize
pixi run pymol_imd input.pdb --level mlmm \
    --ml-selection "resi 30-50" --ml-backend uma --run   # ONIOM ML/MM + auto-run
pixi run pymol_imd <pdb> --level <aa|cg|mlmm> --print-only   # show the underlying pymol -d command
pixi run pymol <pdb> -d "from pymol_imd import *; setup_aa <pdb>; imd_run"  # legacy interactive route
pixi run ruff-format
pixi run ruff-lint
pixi run r                        # format + lint
```

The `pymol_imd` CLI (`pymol_imd/cli.py`) is a thin wrapper: it builds a
`pymol -d "from pymol_imd import *; setup_<level>(...)"` invocation and
execs PyMOL. All backend-specific flags (`--ligand-sdf`, `--ml-backend`, etc.)
become kwargs on the matching `setup_*` call. The PyMOL-side commands
(`setup_aa`, `setup_cg`, `setup_mlmm`, `imd_run`, …) keep working unchanged
for users who prefer the interactive route.

There is currently no test target; if adding one, register it as a `[tool.pixi.tasks]` entry.

## Architecture

The codebase is being rebuilt; the previous monolithic `pymol_imd/__init__.py` (kept in the `/init` command args for reference) collapses the three backends into a single `SimulationSystem` class. The new design should separate:

1. **PyMOL command layer** (`pymol_imd/commands.py`): thin wrappers registered via `cmd.extend`. Argument coercion (string → bool/int/float) happens here because PyMOL passes everything as strings.
2. **Backend strategies** (`pymol_imd/backends/{cg,aa,mlmm}.py`): each owns its forcefield assembly, topology preparation, and `System` construction. They share an interface (e.g., `build_system(pdb, ligand_sdf, **kw) -> (System, Topology, Positions)`).
3. **IMD driver** (`pymol_imd/imd.py`): hosts the `CustomExternalForce` that turns the per-atom delta between PyMOL coordinates and OpenMM coordinates into a pulling force, plus the `QtCore.QTimer` loop that steps the simulation and pushes coordinates back into PyMOL via `cmd.alter_state`.
4. **Preparation utilities** (`pymol_imd/prep.py`): hydrogen completion (`Modeller.addHydrogens` only), ligand parameterization (AM1-BCC via AmberTools), CG mapping (martinize2), MM-region selection for ML/MM.

### Critical design constraints

- **Pull-force mechanism**: A single `CustomExternalForce("- fx*x - fy*y - fz*z")` with per-particle `(fx,fy,fz)` parameters is added to every System. Each frame, compare PyMOL coords vs OpenMM coords; non-zero deltas become forces scaled by `const_k`. This is the mechanism that lets the user drag atoms in the PyMOL window during MD.
- **Atom-index alignment**: PyMOL must keep atom order matching OpenMM's `Topology`. The plugin sets `cmd.set("retain_order", 1)` on import — preserve this.
- **GUI thread**: MD steps run on the Qt event loop via `QtCore.QTimer`. Heavy per-frame work (large `nsteps_per_frame`, ML inference) will freeze the viewer; expose `nsteps_per_frame` so users can tune throughput vs interactivity.
- **In-vacuo default**: Use `NoCutoff` for nonbonded; do not add solvent/ions unless the input PDB already contains them. Do not set a periodic box unless the user explicitly requests one.
- **Ensemble selection (resolved)**: Default to **NVT in vacuo** (Langevin integrator, no barostat, `NoCutoff`). When the user passes a setup-time flag requesting a periodic box (e.g., `pbc=True` or an explicit `box` argument), switch to **NPT with an implicit bounding box** by adding a `MonteCarloBarostat` and setting periodic vectors from the molecule's bounding box. The command surface MUST expose this toggle on `setup_aa` / `setup_cg` / `setup_mlmm`.

### Backend-specific notes

- **AA**: forcefield stack mirrors the reference impl — `amber14-all.xml`, `protein.ff14SB`, `tip3pfb` (only if water present), `lipid17`, `DNA.OL15`, `RNA.OL3`. Ligands go through `GAFFTemplateGenerator` (gaff-2.2.20). AM1-BCC charges are produced by AmberTools `antechamber` or OpenFF; verify the chosen path runs on all four platforms. Missing hydrogens are added with `Modeller.addHydrogens` only.
- **CG (Martini)**: two-step pipeline — `vermouth` (martinize2) converts the AA input into Martini topology files (`.top`/`.itp`/`.gro`), then `martini-openmm` parses those into an OpenMM `System`/`Topology`/`Positions`. Same `CustomExternalForce` is attached so IMD pulling still works on CG beads.
- **ML/MM**: wraps `mlmm_toolkit` (declared as `git+URL` in pixi pypi-dependencies). The user selects the ML region via a PyMOL selection string; the rest is handled by Amber-style MM. Cross-platform install must be verified after the first pixi sync.

## User Interaction Conventions

- All comments and docstrings in code: **English**.
- User-facing messages (errors, prints) should be clear and actionable; on failures (e.g., missing hydrogens that can't be auto-completed), emit a single explicit error rather than silently continuing.
- Debug / verify changes inside the plugin yourself (smoke-test with a small PDB) before reporting work as done; the user explicitly asked Claude to absorb debugging load.
- Ask before adding a non-approved dependency, before making a non-trivial spec interpretation, and before changing the public command surface (`setup_cg/setup_aa/setup_mlmm/imd_run/imd_minimize/imd_stop/imd_change`).
