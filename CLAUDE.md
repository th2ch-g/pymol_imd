# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`pymol_imd` is a PyMOL plugin that runs Interactive Molecular Dynamics (IMD) inside the PyMOL viewer using OpenMM as the integrator. It supports three simulation backends, selected at setup time:

- **CG (Martini)** — coarse-grained MD via `setup_cg`
- **AA (Amber)** — all-atom MD via `setup_aa`
- **ML/MM** — ONIOM-style hybrid MD via `setup_mlmm` (Amber MM full system + MLIP correction on a user-selected region)

The plugin must accept proteins, DNA, RNA, lipids, and small-molecule ligands; complete missing hydrogens when possible; parameterize ligands with AM1-BCC charges via AmberTools/OpenFF; and run in vacuo by default (no solvent, no PBC, no box). After `setup_*`, the standard flow is `minimize` → production-run (no equilibration), and user-driven pulling of atoms in the PyMOL viewer is translated into per-atom forces fed into OpenMM each frame.

## Toolchain & Environment

- **Package manager**: `pixi` (already installed system-wide). Do NOT use `pip` / `uv` directly; declare all deps in `pyproject.toml` under `[tool.pixi.dependencies]` (conda-forge) or `[tool.pixi.pypi-dependencies]`.
- **Platforms**: native support is `linux-64`, `osx-arm64`. Intel Mac (`osx-64`) is excluded because the default-bundled `aimnet` pulls `warp-lang` which lacks osx-64 wheels. Windows is **WSL2-only** because `ambertools` has no conda-forge Windows build. Intel Mac users should run via a Linux VM.
- **Python**: ≥3.11
- **Approved libraries** (already greenlit by the user):
  - core: `openmm`, `pymol-open-source`, `ambertools`, `openmmforcefields`, `openff-toolkit`, `rdkit`
  - CG: `vermouth` (martinize2) + `martini-openmm` (git+URL, no conda-forge build)
  - ML/MM: default backend is **AIMNet2** (~10 MB, bundled in the `aimnet` wheel via `pixi install`, fastest CPU inference). MACE (`mace-torch`) is also in the default deps. ORB (`orb-models`) and UMA (`fairchem-core`) sit behind `pixi run install-orb` / `install-uma` tasks because their dep chains can't co-resolve with the conda lock (ORB → cached-path → old rich; UMA's `e3nn>=0.5` vs mace's `e3nn==0.4.4`).
  - hydrogen completion: **only** `openmm.app.Modeller.addHydrogens` — do NOT pull in `pdbfixer`, `propka`, etc.
- Any other dependency requires explicit user approval before being added.
- **pixi environments**: All three backends (AA / CG / ML-MM) live in the single `default` environment. `pixi install` brings AA + CG + ML-MM with **AIMNet2 (default) + MACE** in one step. ORB and UMA are intentionally NOT in pypi-dependencies because their dep chains collide with the lock (ORB → cached-path → old rich; UMA's `e3nn>=0.5` vs mace's `e3nn==0.4.4`); they install via `pixi run install-orb` / `install-uma` whose pip's looser resolver handles them.
- **No mlmm_toolkit**: `pymol_imd/backends/mlmm.py` builds the MLIP ASE Calculators directly (`mace_omol`, `ORBCalculator`, `AIMNet2Calculator`, `FAIRChemCalculator`). No external dispatcher; everything ML-related lives inside this repo.
- **CG forcefield**: Martini 3 `.itp` is downloaded once into `~/.cache/pymol_imd/martini3/` on first `setup_cg`. Online required for first run; subsequent runs are offline.
- **CG forces PBC**: Martini uses a 1.1 nm cutoff so it cannot run in vacuo. CG backend auto-creates a bounding-box around the molecule + 1 nm padding when `pbc=False`. AA/MLMM still default to NoCutoff vacuo.
- **ML/MM = ONIOM**: `E_total = E_MM(full) + E_ML(region) - E_MM(region)`. Each frame the ML region delta force `F_ML - F_MM_subset` is pushed into a dedicated `CustomExternalForce`. Link-atom handling for cross-region bonds is **not implemented yet**; pick selections that are molecularly closed (entire residues / chains).
- **MLIP model access** (ordered by default-friendliness):
  - **AIMNet2 (default)**: ~10 MB, bundled inside the `aimnet` pip wheel. No login, near-instant warm-up. `setup_mlmm` picks this when `ml_backend` is unspecified. Notable: charge / mult go into the `AIMNet2ASE` constructor, not `atoms.info`.
  - **MACE-OMOL-0**: public ASL-licensed model on GitHub releases of `mace-foundations`. No login, ~400 MB on first call, cached to `~/.cache/mace/`. Available via `mace-torch` already in `pixi install`.
  - **ORB**: `orb-models` 0.7.x. Public model, ~100 MB DL on first call. Note: API requires `pretrained.<model>()` to return `(model, adapter)` tuple and `ORBCalculator` lives under `orb_models.forcefield.inference.calculator`. ORB reads charge/spin from `atoms.info`.
  - **UMA**: HuggingFace gated repo `facebook/UMA`. Requires `huggingface-cli login` + access approval. Install fairchem-core via `pixi run install-uma`.
- **Apple Silicon (MPS) support**: `pymol_imd.backends.mlmm._auto_torch_device` selects `cuda → mps → cpu`. **All three default-bundled MLIP backends (MACE / ORB / AIMNet2) hit float64-storage-on-MPS errors at model-load time**, so `_mps_to_cpu_fallback()` transparently downgrades to CPU and prints a notice. Do not remove that fallback — fixing the underlying issue is upstream's job (cast weights to float32 at load).
- **MPS fallback**: If a torch op is unimplemented on MPS, export `PYTORCH_ENABLE_MPS_FALLBACK=1` before launching PyMOL so the missing ops silently drop to CPU.

### Common commands

```bash
pixi install                      # create / sync env (+ editable installs the pymol_imd CLI)
pixi run pymol_imd input.pdb --level aa                  # launch PyMOL with setup_aa
pixi run pymol_imd input.pdb --level cg --minimize       # CG + auto-minimize
pixi run pymol_imd input.pdb --level mlmm \
    --ml-selection "resi 30-50" --ml-charge 0 --ml-mult 1 --run   # ONIOM ML/MM (default: aimnet2)
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
- **ML/MM**: ONIOM = full-system Amber MM + per-frame MLIP delta force on the ML region. The ML region is resolved through PyMOL's full selection language when running inside a session (`cmd.iterate` on a temp prepared-PDB copy), with a strict mini-DSL fallback for headless runs. **AIMNet2 is the default backend** because it's small + fast + login-free; MACE / ORB are already in the default env, UMA needs `pixi run install-uma` + HF auth. The CLI requires `--ml-selection`, `--ml-charge`, `--ml-mult` for `--level mlmm` (validated up front before PyMOL launches).

## User Interaction Conventions

- All comments and docstrings in code: **English**.
- User-facing messages (errors, prints) should be clear and actionable; on failures (e.g., missing hydrogens that can't be auto-completed), emit a single explicit error rather than silently continuing.
- Debug / verify changes inside the plugin yourself (smoke-test with a small PDB) before reporting work as done; the user explicitly asked Claude to absorb debugging load.
- Ask before adding a non-approved dependency, before making a non-trivial spec interpretation, and before changing the public command surface (`setup_cg/setup_aa/setup_mlmm/imd_run/imd_minimize/imd_stop/imd_change`).
