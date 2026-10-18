"""Map an existing CueMol worker to the existing headless IMDv3 client."""

from __future__ import annotations

import contextlib
import json
import logging
import time
from importlib.resources import files
from pathlib import Path

import numpy as np
from openmm.app import PDBFile

from pymol_imd import constants
from pymol_imd.client import IMDClient
from pymol_imd.imd import compute_pull_forces, expand_group_forces
from pymol_imd.particle_masses import read_particle_masses
from pymol_imd.protocol import IMDHeaderType

from .cdp import CDP, CDPError

logger = logging.getLogger(__name__)


def read_prepared_bonds(path: Path) -> list[list[int]]:
    """Map explicit PDB bonds to prepared atom order, including hybrid-36 IDs."""
    lines = path.read_text().splitlines()
    serials = [line[6:11] for line in lines if line.startswith(("ATOM  ", "HETATM"))]
    indices = {serial: index for index, serial in enumerate(serials)}
    if len(indices) != len(serials):
        raise ValueError("Prepared PDB has duplicate atom serials")
    bonds = set()
    for line in lines:
        if not line.startswith("CONECT"):
            continue
        source = indices.get(line[6:11])
        for offset in range(11, min(len(line), 31), 5):
            target = indices.get(line[offset : offset + 5])
            if source is not None and target is not None and source != target:
                bonds.add(tuple(sorted((source, target))))
    return [list(pair) for pair in sorted(bonds)]


def read_prepared_pdb(path: Path) -> tuple[list[list[str]], np.ndarray]:
    """Preserve prepared-PDB order and reject ambiguous atom identities."""
    keys = []
    positions = []
    models = 0
    for line in path.read_text().splitlines():
        if line.startswith("MODEL "):
            models += 1
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        keys.append(
            [
                line[21:22].strip(),
                line[22:27].strip(),
                line[17:20].strip(),
                line[12:16].strip(),
                line[16:17].strip(),
            ]
        )
        positions.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if not keys or models > 1:
        raise ValueError("Use the server's single-model prepared PDB")
    if len({tuple(key) for key in keys}) != len(keys):
        raise ValueError("The prepared PDB contains duplicate atom identities")
    coordinates = np.asarray(positions, dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError("The prepared PDB contains non-finite coordinates")
    return keys, coordinates


def target_forces(
    positions: np.ndarray,
    target: dict | None,
    atom_keys: list[list[str]],
    *,
    const_k: float,
    max_force_kj: float,
    drag_level: str = "atom",
    particle_masses: np.ndarray | None = None,
    group_ids: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Use the shared force cap and units, including residue/chain expansion."""
    delta = np.zeros_like(positions, dtype=float)
    if target is not None:
        index = target["index"]
        if not isinstance(index, int) or not 0 <= index < len(positions):
            raise ValueError("CueMol returned an invalid IMD atom index")
        position = np.asarray(target["position"], dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("CueMol returned an invalid drag target")
        delta[index] = (position - positions[index]) * constants.ANGSTROM_TO_NM
    active, forces, exceeded = compute_pull_forces(
        delta, const_k, max_force_kj, constants.DELTA_NOISE_NM
    )
    if drag_level not in constants.SUPPORTED_DRAG_LEVELS:
        raise ValueError(f"Invalid drag level: {drag_level}")
    if len(active) and drag_level != "atom":
        if particle_masses is None:
            raise ValueError("Group pulling needs particle masses; use a newly prepared server PDB")
        if group_ids is None:
            width = 1 if drag_level == "chain" else 2
            lookup = {}
            groups = np.array(
                [lookup.setdefault(tuple(key[:width]), len(lookup)) for key in atom_keys]
            )
        else:
            groups = group_ids[drag_level]
        active, forces, group_exceeded = expand_group_forces(
            active, forces, groups, particle_masses, max_force_kj
        )
        exceeded += group_exceeded
    return active, forces, exceeded


class CueMolBridge:
    """Attach to one live CueMol page and its already initialized native worker."""

    def __init__(
        self,
        endpoint: str,
        prepared: Path,
        *,
        scene_id: int | None = None,
        view_id: int | None = None,
        name: str = "IMD molecule",
    ):
        self.atom_keys, self.prepared_positions = read_prepared_pdb(prepared)
        self.particle_masses = read_particle_masses(prepared)
        atoms = list(PDBFile(str(prepared)).topology.atoms())
        if len(atoms) != len(self.atom_keys):
            raise ValueError("Prepared PDB topology does not match the coordinate atom order")
        self.group_ids = {
            "residue": np.array([atom.residue.index for atom in atoms]),
            "chain": np.array([atom.residue.chain.index for atom in atoms]),
        }
        is_cg = any(key[3] == "BB" for key in self.atom_keys)
        self.cdp = CDP(endpoint, timeout=constants.CUEMOL_CDP_TIMEOUT_S)
        self.page = None
        self.worker = None
        self.initial = None
        self._worker_owned = False
        self._ui_owned = False
        self._closed = False
        try:
            deadline = time.monotonic() + constants.CUEMOL_LAUNCH_TIMEOUT_S
            while True:
                targets = self.cdp.request("Target.getTargets")["targetInfos"]
                pages = [
                    t for t in targets if t["type"] == "page" and "/renderer/index.html" in t["url"]
                ]
                workers = [t for t in targets if t["type"] == "worker"]
                if len(pages) == 1 and len(workers) == 1:
                    break
                if len(pages) > 1 or len(workers) > 1:
                    raise CDPError("The experimental adapter needs one CueMol window and worker")
                if time.monotonic() >= deadline:
                    raise CDPError("CueMol did not create its view worker")
                time.sleep(0.1)
            self.page = self.cdp.attach(pages[0]["targetId"])
            self.cdp.request(
                "Target.setAutoAttach",
                {
                    "autoAttach": True,
                    "waitForDebuggerOnStart": False,
                    "flatten": True,
                },
                self.page,
            )
            while self.worker is None:
                self.cdp.request("Runtime.enable", session=self.page)
                attached = [
                    event["params"]["sessionId"]
                    for event in self.cdp.events
                    if event.get("method") == "Target.attachedToTarget"
                    and event.get("sessionId") == self.page
                    and event["params"]["targetInfo"]["targetId"] == workers[0]["targetId"]
                ]
                if attached:
                    self.worker = attached[-1]
                elif time.monotonic() >= deadline:
                    raise CDPError("CueMol did not attach its initialized view worker")
                else:
                    time.sleep(0.05)
            self.cdp.request("Runtime.enable", session=self.worker)
            ready = files(__package__).joinpath("worker_ready.js").read_text()
            while not self.cdp.evaluate(ready, self.worker):
                if time.monotonic() >= deadline:
                    raise CDPError("CueMol did not initialize its scene and viewport")
                time.sleep(0.05)
            self.cdp.evaluate(files(__package__).joinpath("worker.js").read_text(), self.worker)
            self._worker_owned = True
            self.initial = self._worker_call(
                "load",
                {
                    "path": str(prepared.resolve()),
                    "atomKeys": self.atom_keys,
                    "positions": self.prepared_positions.tolist() if is_cg else None,
                    "sceneId": scene_id,
                    "viewId": view_id,
                    "name": name,
                    "bonds": read_prepared_bonds(prepared) if is_cg else None,
                    "beadRadius": constants.CG_SPHERE_SCALE if is_cg else None,
                    "bondRadius": constants.CG_STICK_RADIUS if is_cg else None,
                },
            )
            if not np.allclose(
                self.initial["positions"], self.prepared_positions, atol=0.001, rtol=0
            ):
                raise CDPError("CueMol coordinates do not match the prepared PDB")
            self.cdp.evaluate(files(__package__).joinpath("renderer.js").read_text(), self.page)
            self._ui_owned = True
            self._page_call("configure", {"leaseMs": constants.CUEMOL_INPUT_LEASE_MS})
        except BaseException:
            self.close()
            raise

    def _worker_call(self, method: str, payload: object | None = None):
        arg = "" if payload is None else json.dumps(payload, allow_nan=False)
        return self.cdp.evaluate(f"__cuemolIMD.{method}({arg})", self.worker)

    def _page_call(self, method: str, payload: object | None = None):
        arg = "" if payload is None else json.dumps(payload, allow_nan=False)
        return self.cdp.evaluate(f"__cuemolIMDUI.{method}({arg})", self.page)

    def snapshot(self) -> dict:
        return self._worker_call("snapshot")

    def poll_input(self) -> dict:
        return self._page_call("poll")

    def configure_drag_level(self, level: str) -> None:
        self._page_call("configure", {"dragLevel": level})

    def tick(self, positions, step: int, events: list[dict]) -> dict:
        return self._worker_call(
            "tick",
            {
                "positions": positions.tolist() if positions is not None else None,
                "step": step,
                "events": events,
            },
        )

    def status(self, step: int, force: float, *, over_limit: bool = False) -> None:
        self._page_call("update", {"step": step, "force": force, "overLimit": over_limit})

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for owned, session, expression in (
            (self._ui_owned, self.page, "globalThis.__cuemolIMDUI?.detach()"),
            (self._worker_owned, self.worker, "globalThis.__cuemolIMD?.detach()"),
        ):
            if owned and session:
                with contextlib.suppress(CDPError, OSError, TimeoutError):
                    self.cdp.evaluate(expression, session)
        self.cdp.close()


class CueMolIMD:
    """Run an atom-mapped, bounded-rate CueMol display and mouse-force loop."""

    def __init__(
        self,
        bridge: CueMolBridge,
        client: IMDClient,
        *,
        const_k: float = constants.AA_RECOMMENDED_CONST_K,
        max_force_kj: float = constants.DEFAULT_MAX_FORCE_KJ,
        drag_level: str = constants.DEFAULT_DRAG_LEVEL,
        frame_interval_ms: int = constants.CUEMOL_FRAME_INTERVAL_MS,
    ):
        self.bridge = bridge
        self.client = client
        self.const_k = const_k
        self.max_force_kj = max_force_kj
        self.drag_level = drag_level
        self.interval = frame_interval_ms / 1000
        self.stats = {
            "frames_received": 0,
            "frames_displayed": 0,
            "last_step": 0,
            "force_packets": 0,
            "nonzero_force_packets": 0,
            "release_packets": 0,
            "peak_force_kj_per_nm": 0.0,
            "force_cap_clamps": 0,
            "pointer_counts": {},
            "paused": False,
            "drag_level": drag_level,
            "drag_level_changes": 0,
        }
        self.last_positions = None
        self.active_prev = False

    def _release(self) -> None:
        self.client.send_forces(np.empty(0, dtype=np.int32), np.empty((0, 3)))
        self.stats["force_packets"] += 1
        self.stats["release_packets"] += 1
        self.active_prev = False

    def _latest_frame(self, deadline: float):
        frame = None
        for _ in range(constants.CUEMOL_MAX_FRAMES_PER_TICK):
            incoming = self.client.poll_frame()
            if incoming is None:
                break
            frame = incoming
            self.stats["frames_received"] += 1
            if time.monotonic() >= deadline:
                break
        return frame

    def run(self, *, duration: float | None = None, evidence: Path | None = None) -> dict:
        started = time.monotonic()
        trace = None
        if evidence:
            evidence.parent.mkdir(parents=True, exist_ok=True)
            trace = evidence.with_suffix(".events.jsonl").open("w")
        try:
            self.client.connect()
            if not self.client.session.coords:
                raise ValueError("The IMD server does not stream coordinates")
            self.bridge.configure_drag_level(self.drag_level)
            print(
                "CUEMOL_IMD_READY "
                + json.dumps(
                    {
                        "atoms": len(self.bridge.atom_keys),
                        "scene": self.bridge.initial["sceneId"],
                        "object": self.bridge.initial["objectId"],
                    }
                ),
                flush=True,
            )
            while duration is None or time.monotonic() - started < duration:
                tick_started = time.monotonic()
                inputs = self.bridge.poll_input()
                self.stats["pointer_counts"] = inputs["counts"]
                if "disconnect" in inputs["controls"]:
                    break
                for control in inputs["controls"]:
                    if control == "pause":
                        self._release()
                        self.client.send_control(IMDHeaderType.PAUSE)
                        inputs["events"].append({"type": "release"})
                        self.stats["paused"] = True
                    elif control == "resume":
                        self.client.send_control(IMDHeaderType.RESUME)
                        self.stats["paused"] = False
                    elif control.startswith("drag-level:"):
                        level = control.removeprefix("drag-level:")
                        if level not in constants.SUPPORTED_DRAG_LEVELS:
                            raise ValueError(f"Invalid CueMol drag level: {level!r}")
                        if level != self.drag_level:
                            self._release()
                            inputs["events"].insert(0, {"type": "release"})
                            self.drag_level = level
                            self.stats["drag_level"] = level
                            self.stats["drag_level_changes"] += 1
                frame = self._latest_frame(tick_started + self.interval / 2)
                if self.client.closed:
                    raise ConnectionError("The IMD server disconnected")
                if frame is not None:
                    expected = (len(self.bridge.atom_keys), 3)
                    if frame.positions_ang is None or frame.positions_ang.shape != expected:
                        raise ValueError("The IMD frame does not match the prepared PDB atom count")
                    self.last_positions = frame.positions_ang
                    self.stats["last_step"] = frame.step
                    self.stats["frames_displayed"] += 1
                state = self.bridge.tick(
                    frame.positions_ang if frame is not None else None,
                    self.stats["last_step"],
                    inputs["events"],
                )
                target = None if self.stats["paused"] else state["target"]
                force_max = 0.0
                if self.last_positions is not None:
                    active, forces, exceeded = target_forces(
                        self.last_positions,
                        target,
                        self.bridge.atom_keys,
                        const_k=self.const_k,
                        max_force_kj=self.max_force_kj,
                        drag_level=self.drag_level,
                        particle_masses=self.bridge.particle_masses,
                        group_ids=self.bridge.group_ids,
                    )
                    force_max = float(np.linalg.norm(forces, axis=1).max())
                    self.stats["peak_force_kj_per_nm"] = max(
                        self.stats["peak_force_kj_per_nm"], force_max
                    )
                    if exceeded:
                        if not self.stats.get("over_limit", False):
                            logger.warning(
                                "Pull capped at %.1f kJ/mol/nm",
                                self.max_force_kj,
                            )
                        self.stats["force_cap_clamps"] += 1
                    self.stats["over_limit"] = bool(exceeded)
                    if len(active):
                        self.client.send_forces(
                            active, forces[active] * constants.KJ_PER_NM_TO_KJ_PER_ANGSTROM
                        )
                        self.stats["force_packets"] += 1
                        self.stats["nonzero_force_packets"] += 1
                        self.active_prev = True
                    elif self.active_prev:
                        self._release()
                if trace and (inputs["events"] or target):
                    trace.write(
                        json.dumps(
                            {
                                "elapsed_s": time.monotonic() - started,
                                "step": self.stats["last_step"],
                                "events": inputs["events"],
                                "controls": inputs["controls"],
                                "drag_level": self.drag_level,
                                "target": target,
                                "force_kj_per_nm": force_max,
                            }
                        )
                        + "\n"
                    )
                    trace.flush()
                self.bridge.status(
                    self.stats["last_step"],
                    force_max,
                    over_limit=self.stats.get("over_limit", False),
                )
                time.sleep(max(0, self.interval - (time.monotonic() - tick_started)))
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                self.stats["interrupted"] = True
            else:
                self.stats["error"] = str(exc)
                raise
        finally:
            try:
                if self.client.conn is not None and not self.client.closed:
                    with contextlib.suppress(OSError, RuntimeError):
                        self._release()
                self.client.disconnect()
                try:
                    self.stats["final"] = self.bridge.snapshot()
                    if self.last_positions is not None:
                        self.stats["coordinate_max_error_ang"] = float(
                            np.max(
                                np.abs(
                                    np.asarray(self.stats["final"]["positions"])
                                    - self.last_positions
                                )
                            )
                        )
                except (CDPError, OSError, TimeoutError) as exc:
                    self.stats["snapshot_error"] = str(exc)
                self.stats["elapsed_s"] = time.monotonic() - started
                self.stats["initial"] = self.bridge.initial
                if evidence:
                    evidence.write_text(json.dumps(self.stats, indent=2, allow_nan=False) + "\n")
            finally:
                try:
                    if trace:
                        trace.close()
                finally:
                    self.bridge.close()
        return self.stats
