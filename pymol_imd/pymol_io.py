"""Coordinate updates shared by the local and socket viewers."""


def load_coordinates(cmd, obj: str, positions_ang) -> None:
    cmd.load_coordset(positions_ang, obj, state=1)
    if cmd.get_setting_int("dynamic_measures") and cmd.get_names_of_type("object:measurement"):
        # Bulk coordinate loading does not invalidate cached measurement points.
        cmd.update(obj, obj, target_state=1, source_state=1, matchmaker=0)
