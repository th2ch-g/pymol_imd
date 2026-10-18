"""Coordinate updates shared by the local and socket viewers."""

from pathlib import Path

import numpy as np

_prepared_masses: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def load_prepared(cmd, path: str, obj: str) -> None:
    """Load the server's prepared PDB and retain its particle mass metadata."""
    from .particle_masses import read_particle_masses

    masses = read_particle_masses(Path(path))
    cmd.load(path, obj)
    _prepared_masses.pop(obj, None)
    if masses is not None:
        _prepared_masses[obj] = (cmd.get_coordset(obj, state=1).copy(), masses)


def take_prepared_masses(cmd, obj: str) -> np.ndarray | None:
    """Consume metadata only for the unchanged object loaded for this connection."""
    saved = _prepared_masses.pop(obj, None)
    if saved is not None and np.array_equal(cmd.get_coordset(obj, state=1), saved[0]):
        return saved[1]
    return None


def load_coordinates(cmd, obj: str, positions_ang) -> None:
    cmd.load_coordset(positions_ang, obj, state=1)
    if cmd.get_setting_int("dynamic_measures") and cmd.get_names_of_type("object:measurement"):
        # Bulk coordinate loading does not invalidate cached measurement points.
        cmd.update(obj, obj, target_state=1, source_state=1, matchmaker=0)
