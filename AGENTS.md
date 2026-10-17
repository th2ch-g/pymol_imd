# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`pymol_imd` is a PyMOL plugin that runs Interactive Molecular Dynamics (IMD) inside the PyMOL viewer using OpenMM as the integrator. It supports three simulation backends, selected at setup time:

- **CG (Martini)** — coarse-grained MD via `imd_setup_cg`
- **AA (Amber)** — all-atom MD via `imd_setup_aa`
- **ML/MM** — ONIOM-style hybrid MD via `imd_setup_mlmm` (Amber MM full system + MLIP correction on a user-selected region)

The plugin must accept proteins, DNA, RNA, lipids, and small-molecule ligands; complete missing hydrogens when possible; parameterize ligands with AM1-BCC charges via AmberTools/OpenFF; and run with **GBn2 implicit solvent by default** for AA / ML-MM (in-vacuo selectable via `implicit_solvent=none`; no PBC/box unless requested). After `imd_setup_*`, the standard flow is `minimize` → production-run (no equilibration), and user-driven pulling of atoms in the PyMOL viewer is translated into per-atom forces fed into OpenMM each frame.

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
- **CG forcefield**: Martini 3 `.itp` is downloaded once into `~/.cache/pymol_imd/martini3/` on first `imd_setup_cg`. Online required for first run; subsequent runs are offline.
- **CG forces PBC**: Martini uses a 1.1 nm cutoff so it cannot run in vacuo. CG backend auto-creates a bounding-box around the molecule + 1 nm padding when `pbc=False`. AA/MLMM default to NoCutoff + **GBn2 implicit solvent** (in-vacuo via `implicit_solvent=none`).
- **ML/MM = ONIOM**: `E_total = E_MM(full) + E_ML(region) - E_MM(region)`. Each frame the ML region delta force `F_ML - F_MM_subset` is pushed into a dedicated `CustomExternalForce`. Link-atom handling for cross-region bonds **is implemented** (`link_atom_method`: `scaled` Morokuma/Dapprich g-factor [default], `fixed` IMOMM projector, or `none`); `_detect_link_atoms` enumerates boundary bonds and the link H force is redistributed to the QM/MM atoms via the placement Jacobian. Molecularly closed selections still work (`none`).
- **MLIP model access** (ordered by default-friendliness):
  - **AIMNet2 (default)**: ~10 MB, bundled inside the `aimnet` pip wheel. No login, near-instant warm-up. `imd_setup_mlmm` picks this when `ml_backend` is unspecified. Notable: charge / mult go into the `AIMNet2ASE` constructor, not `atoms.info`.
  - **MACE-OMOL-0**: public ASL-licensed model on GitHub releases of `mace-foundations`. No login, ~400 MB on first call, cached to `~/.cache/mace/`. Available via `mace-torch` already in `pixi install`.
  - **ORB**: `orb-models` 0.7.x. Public model, ~100 MB DL on first call. Note: API requires `pretrained.<model>()` to return `(model, adapter)` tuple and `ORBCalculator` lives under `orb_models.forcefield.inference.calculator`. ORB reads charge/spin from `atoms.info`.
  - **UMA**: HuggingFace gated repo `facebook/UMA`. Requires `huggingface-cli login` + access approval. Install fairchem-core via `pixi run install-uma`, then use `pixi run --no-install` for UMA commands. A normal pixi run restores MACE's locked `e3nn==0.4.4`; UMA requires `e3nn>=0.5`.
- **Apple Silicon (MPS) support**: `pymol_imd.backends.mlmm._auto_torch_device` selects `cuda → mps → cpu`. **All three default-bundled MLIP backends (MACE / ORB / AIMNet2) hit float64-storage-on-MPS errors at model-load time**, so `_mps_to_cpu_fallback()` transparently downgrades to CPU and prints a notice. Do not remove that fallback — fixing the underlying issue is upstream's job (cast weights to float32 at load).
- **MPS fallback**: If a torch op is unimplemented on MPS, export `PYTORCH_ENABLE_MPS_FALLBACK=1` before launching PyMOL so the missing ops silently drop to CPU.
- **OpenMM platform (MM engine, all backends)**: the MM compute platform is selected via the `platform` knob (`imd_setup_*(platform=...)` / `--platform`): `auto` [default → OpenMM picks the fastest: CUDA > OpenCL > CPU] / `cuda` / `opencl` / `cpu` / `reference`, validated by `constants.resolve_platform_name` and mapped to an `openmm.Platform` by `IMDSystem._resolve_platform` (an explicit name OpenMM can't provide raises — no silent CPU fallback — and the chosen platform is logged at setup). This is SEPARATE from the ML/MM MLIP device (`ml_device` / `_auto_torch_device`): on a GPU box you typically set BOTH `--platform cuda` (MM engine) and `pixi install -e gpu` + `--ml-device cuda` (MLIP). `auto` keeps the historical zero-config behavior (OpenMM auto-selects).

### Common commands

```bash
pixi install                      # create / sync env (+ editable installs the pymol_imd CLI)
pixi install -e gpu               # Linux+NVIDIA: separate CUDA-torch env (10-100x ML/MM speedup; default env stays CPU)
pixi run pymol_imd launch input.pdb --level aa            # launch PyMOL with imd_setup_aa
pixi run pymol_imd launch input.pdb --level cg --minimize # CG + auto-minimize
pixi run pymol_imd local input.pdb --level mlmm \
    --ml-selection "resi 30-50" --ml-charge 0 --ml-mult 1   # ONIOM ML/MM (default: aimnet2)
pixi run pymol_imd launch <pdb> --level <aa|cg|mlmm> --print-only
pixi run pymol <pdb> -d "from pymol_imd import *; imd_setup_aa <pdb>; imd_run"  # legacy interactive route
# --- IMDv3 socket mode (optional): MD on a headless server, PyMOL as the client ---
pixi run pymol_imd serve <pdb> --level aa --port 8888        # headless compute server (no PyMOL)
pixi run pymol_imd connect <prepared.pdb> --port 8888        # PyMOL client -> (remote) server
pixi run pymol_imd local <pdb> --level aa                    # one box: MD worker + Qt viewer (NO socket); = launch + run
pixi run ruff-format
pixi run ruff-lint
pixi run r                        # format + lint
pixi run smoke                    # headless AA + CG build/minimize/step + force-cap check
pixi run smoke-mlmm               # ML/MM ONIOM delta regression vs tests/refs/*.npy (atol 1e-4)
pixi run record-smoke             # trajectory (XTC/DCD/PDB) + force-CSV recording smoke
pixi run smoke-socket             # IMDv3 server<->client localhost round-trip + protocol/units regression
pixi run smoke-drag               # drag-level (atom/residue/chain) group-expansion regression
pixi run smoke-scripted-drag      # code-driven drag parity, timing, release, recovery, and recording
pixi run smoke-interactive        # local worker responsiveness, pending input, recording, and cleanup
pixi run smoke-cli                # CLI/subcommand/coercion regression
```

The `pymol_imd` CLI (`pymol_imd/cli.py`) requires an explicit
`launch` / `local` / `serve` / `connect` subcommand. Launch/local build a
`pymol -d "from pymol_imd import *; imd_setup_<level>(...)"` invocation and
execs PyMOL. All backend-specific flags (`--ligand-sdf`, `--ml-backend`, etc.)
become kwargs on the matching `imd_setup_*` call. The PyMOL-side commands
(`imd_setup_aa`, `imd_setup_cg`, `imd_setup_mlmm`, `imd_run`, …) keep working unchanged
for users who prefer the interactive route.

Tests are headless smoke scripts under `pymol_imd/tests/` (run via the
`smoke` / `smoke-mlmm` / `record-smoke` pixi tasks above; no pytest suite).
`smoke-mlmm` is a **numerical-regression gate**: it pins initial coordinates
and compares the ONIOM delta force against `tests/refs/mlmm_<backend>_delta.npy`
within `atol=1e-4`. It must pass WITHOUT setting `SMOKE_MLMM_SAVE_REF=1`
(that env var re-saves the reference; only do so for an intentional change).
Add new tests as `[tool.pixi.tasks]` entries.

## Architecture

The plugin is organized into separated layers (the earlier monolithic
`SimulationSystem` design has been split out):

1. **PyMOL command layer** (`pymol_imd/commands.py`): thin wrappers registered via `cmd.extend`. Argument coercion (string → bool/int/float) happens here because PyMOL passes everything as strings. Per-backend defaults are resolved via `_resolve_const_k` / `_resolve_integrator` / `_resolve_timestep_ps`; common IMD knobs flow through `_build_imd_kwargs` into `IMDConfig`.
2. **Backend strategies** (`pymol_imd/backends/{cg,aa,mlmm}.py`): each owns its forcefield assembly, topology preparation, and `System` construction. They share the `Backend` protocol in `base.py`: `build_system(pdb, **kw) -> SystemBundle`, plus `recommended_const_k` / `recommended_integrator` / `recommended_timestep_ps` class attrs.
3. **IMD driver** (`pymol_imd/imd.py`): hosts the pulling `CustomExternalForce`, OpenMM stepping, recovery, and recording. `IMDConfig` holds the runtime knobs. The local viewer loop (`pymol_imd/interactive.py`) keeps at most one MD frame on a dedicated worker and reads/updates PyMOL coordinates on the Qt thread. Accepted states are loaded with `cmd.load_coordset` in original atom order.
4. **Preparation utilities** (`pymol_imd/prep.py`): hydrogen completion (`Modeller.addHydrogens` only), ligand parameterization (AM1-BCC via AmberTools), CG mapping (martinize2), MM-region selection for ML/MM.
5. **Constants** (`pymol_imd/constants.py`): single source of truth for tunable defaults (`DEFAULT_*` global, `AA_*`/`CG_*`/`MLMM_*` per-backend) and fixed physical constants (eV→kJ/mol, covalent radii, link-atom bond lengths). **Do not hardcode magic numbers inline** — add the default here and expose it via an `imd_setup_*` kwarg + CLI flag. Must stay import-light (no `pymol`/`openmm` imports) since `cli.py` imports it.

### IMDv3 socket mode (optional)

A second, opt-in run mode splits MD (OpenMM) from the viewer (PyMOL) across a TCP socket using the **IMDv3 protocol**, so the compute can live on a (possibly remote) machine while PyMOL stays local. This is separate from the default same-process worker/Qt mode. Four modules sit alongside the layers above:

- **Protocol** (`pymol_imd/protocol.py`): pure pack/unpack of the IMDv3 wire format — 8-byte **big-endian** headers (`int32 type` + `int32 dataslot`) + **native-endian** bodies, including the 24-byte `TIME` body with an `int64` step. Stdlib + numpy only (no socket/openmm/pymol); unit-agnostic (NO unit conversion lives here).
- **Transport** (`pymol_imd/transport.py`): `recv_exact` framing, `IMDConnection`, `bind_listen` / `accept_one` / `connect`. Stdlib `socket` only.
- **Server** (`pymol_imd/server.py`): `IMDServer` wraps an `IMDSystem` and runs its own step loop. A **receiver thread** stages validated MDComm forces into a lock-guarded buffer; the **step thread is the ONLY thread that touches OpenMM**. Recording starts after GO and uses collision-free `.session-NNNN` files per connection.
- **Client** (`pymol_imd/client.py`): `IMDClient` uses a persistent incremental receive buffer, so header/body fragmentation and poll timeouts cannot discard bytes or block the Qt loop. `PyMOLIMDClient` displays frames and sends drag forces.

**Units**: the wire is IMDv3-fixed (length angstrom, force kJ/mol/angstrom, time ps). OpenMM is nm / kJ/mol/nm, so server/client convert at the OpenMM boundary using the named factors in `constants.py` (`NM_TO_ANGSTROM`, `KJ_PER_ANGSTROM_TO_KJ_PER_NM`, …) — `protocol.py` stays unit-free. **Atom order must match**: the server prints `IMD_READY port=<P> prepared=<path>`; the client MUST load that prepared PDB. CLI: `serve` (headless engine) + `connect` (PyMOL client) are the **split, cross-machine socket setup**; `local` is **NOT socket** — it is the same-process worker/Qt mode (`= launch + run`) in one command, so normal local interactive work never pays socket latency/backpressure. PyMOL commands: `imd_connect` / `imd_disconnect` — the client reuses the setup representation + 3-Button Editing via `_apply_imd_representation` (CG auto-detected from `BB` beads, so `connect` looks like `imd_setup_*`). **Smoothness**: local response still depends on MD and rendering costs; socket motion is tunable via `IMD_CLIENT_FRAME_INTERVAL_MS` (client Qt tick, default 8 ms) and `serve --rate` (steps per streamed frame). **Client freeze guard**: `PyMOLIMDClient._drain_to_latest` caps frames read per Qt tick (`IMD_CLIENT_MAX_FRAMES_PER_TICK`) — an unbounded drain wedges the Qt loop because the server streams a frame per step. The `smoke-socket` task is a headless localhost round-trip + protocol/unit/endianness + freeze-guard regression (no PyMOL, no VMD). Only IMDv3 is implemented (not VMD's older IMDv2).

### Critical design constraints

- **Pull-force mechanism**: A single `CustomExternalForce("- fx*x - fy*y - fz*z")` with per-particle `(fx,fy,fz)` parameters is added to every System. Consume each viewer edit once; deltas above `DELTA_NOISE_NM` become forces scaled by `const_k`. Cap each finite pull force at `max_force_kj` while preserving its direction, including direct driver calls; reject the whole pull frame only for non-finite input. The cap remains configurable through setup kwargs, CLI flags, and local `imd_change`. On excessive potential/kinetic energy, non-finite states, invalid proposed ML/MM forces, or an OpenMM integration exception, restore the preceding checkpoint and retry without pulling; the second retry also zeros velocities. Rejected gestures stay released until a release input. The timestep stays unchanged. Buffer reports until a frame is accepted so rolled-back steps never enter recordings. Stop only if both retries fail. Socket forces expire without a refresh for `IMD_FORCE_LEASE_S`; pause/resume clears pending input. This behavior is shared by local and socket runs; regression: `pixi run smoke-recovery`.
- **Atom-index alignment**: PyMOL must keep atom order matching OpenMM's `Topology`. The plugin sets `cmd.set("retain_order", 1)` on import — preserve this.
- **GUI thread**: Local production MD runs on one worker in `LocalIMDLoop`; only the Qt thread may call PyMOL. Never queue more than one frame. Consume mouse-coordinate deltas before overwriting the viewer, and compare against the last consumed/displayed coordinates so MD motion cannot become a pull force. Commands must wait for the pending frame before changing OpenMM or recording state; stop/setup must join the worker. Keep `nsteps_per_frame` and the physical timestep unchanged. `IMDSystem.step()` remains synchronous for explicit scripted stepping; minimization still uses Qt chunks. Regression: `pixi run smoke-interactive` and `pixi run smoke-pymol`.
- **Drag granularity**: `drag_level` (`atom` [default] / `residue` / `chain`) is exposed on every `imd_setup_*` and live-switchable via `imd_drag_level`. PyMOL's 3-Button Editing drag moves only a covalent fragment and ignores `mouse_selection_mode`, so granularity is realized plugin-side: `IMDSystem._expand_drag_groups` copies a dragged atom's mean pull force onto its whole residue / chain. Regression: `pixi run smoke-drag`.
- **Default solvent = GB implicit (AA / ML-MM)**: default to **GBn2 generalized-Born** with `NoCutoff`. The `implicit_solvent` knob (`auto` [default → GBn2 when non-periodic] / `none` [in-vacuo] / explicit model `gbn2`/`obc2`/`obc1`/`gbn`/`hct`) is resolved by `constants.resolve_implicit_solvent`. GB is enabled by **appending its implicit XML** (`implicit/gbn2.xml`, …) to the `ForceField` file list — NOT the prmtop-only `createSystem(implicitSolvent=)` kwarg. `implicit_solvent="none"` restores the old in-vacuo path. Do not add explicit solvent/ions unless the input PDB already contains them; do not set a periodic box unless `pbc=True`. GB and `pbc`/PME are mutually exclusive: an explicit GB model + `pbc=True` raises; `auto` yields to PME under `pbc`. CG is unaffected (Martini has its own solvent). NOTE: OpenMM's GB XMLs are protein-centric — exotic residue types may lack GB params and raise; users fall back to `implicit_solvent=none`.
- **Ensemble selection (resolved)**: Default to **NVT** (Langevin integrator, no barostat, `NoCutoff`) — with GBn2 implicit solvent for AA / ML-MM (in-vacuo via `implicit_solvent=none`). When the user passes a setup-time flag requesting a periodic box (e.g., `pbc=True` or an explicit `box` argument), switch to **NPT with an implicit bounding box** by adding a `MonteCarloBarostat` and setting periodic vectors from the molecule's bounding box (GB is disabled in this path). The command surface MUST expose this toggle on `imd_setup_aa` / `imd_setup_cg` / `imd_setup_mlmm`.
- **Recording outputs (default on)**: the production run writes a trajectory, force CSV, and state CSV. Server outputs are per connection (`.session-NNNN`). CSV files flush every 100 rows and on every stop/error path. All streams gate on cumulative `currentStep`; pressure remains omitted. Regression: `pixi run record-smoke` and `pixi run smoke-socket`.

### Backend-specific notes

- **AA**: forcefield stack mirrors the reference impl — `amber14-all.xml`, `protein.ff14SB`, `tip3pfb` (only if water present), `lipid17`, `DNA.OL15`, `RNA.OL3`. Ligands go through `GAFFTemplateGenerator` (gaff-2.2.20). AM1-BCC charges are produced by AmberTools `antechamber` or OpenFF; verify the chosen path runs on all four platforms. Missing hydrogens are added with `Modeller.addHydrogens` only. Default GBn2 implicit solvent is enabled by appending its XML to the `ForceField` (`createSystem` otherwise unchanged); `implicit_solvent="none"` → in-vacuo.
- **CG (Martini)**: two-step pipeline — `vermouth` (martinize2) converts the AA input into Martini topology files (`.top`/`.itp`/`.gro`), then `martini-openmm` parses those into an OpenMM `System`/`Topology`/`Positions`. Same `CustomExternalForce` is attached so IMD pulling still works on CG beads.
- **ML/MM**: ONIOM = full-system Amber MM + per-frame MLIP delta force on the ML region. The ML region is resolved through PyMOL's full selection language when running inside a session (`cmd.iterate` on a temp prepared-PDB copy), with a strict mini-DSL fallback for headless runs. **AIMNet2 is the default backend** because it's small + fast + login-free; MACE / ORB are already in the default env, UMA needs `pixi run install-uma` + HF auth. The CLI requires `--ml-selection`, `--ml-charge`, `--ml-mult` for `--level mlmm` (validated up front before PyMOL launches). Default GBn2 implicit solvent is appended to the **shared** `ForceField`, so BOTH `E_MM(full)` and `E_MM(subset)` carry GB and the ONIOM subtraction stays consistent — a documented Born-radii approximation of the same class as the existing PME-on-subset one. `smoke-mlmm` pins `implicit_solvent="none"` for its pinned numeric refs (kept bitwise-valid) and covers the GB path with a separate build+step check.

## User Interaction Conventions

- All comments and docstrings in code: **English**.
- User-facing messages (errors, prints) should be clear and actionable; on failures (e.g., missing hydrogens that can't be auto-completed), emit a single explicit error rather than silently continuing.
- Debug / verify changes inside the plugin yourself (smoke-test with a small PDB) before reporting work as done; the user explicitly asked Claude to absorb debugging load.
- Ask before adding a non-approved dependency, before making a non-trivial spec interpretation, and before changing the public command surface (`imd_setup_cg/imd_setup_aa/imd_setup_mlmm/imd_record/imd_minimize/imd_run/imd_stop/imd_change/imd_drag_level/imd_connect/imd_disconnect`).
