"""ML/MM backend wrapping `mlmm_toolkit`.

This is a thin shim around the upstream toolkit:
    https://github.com/t-0hmura/mlmm_toolkit

Because ``mlmm_toolkit`` is officially Linux-only and pulled in via a pixi
feature task, this module imports it lazily and raises a clear error if the
toolkit is missing.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from .base import SystemBundle

logger = logging.getLogger(__name__)


class MLMMBackend:
    name = "mlmm"

    def build_system(
        self,
        pdb_path: Path,
        *,
        ligand_sdf: Path | None = None,
        pbc: bool = False,
        ml_selection: str = "all",
        **_: object,
    ) -> SystemBundle:
        try:
            import mlmm_toolkit  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "mlmm_toolkit is not installed in this environment. "
                "On Linux: `pixi install -e mlmm && pixi run -e mlmm install-mlmm`. "
                "macOS / Windows are not supported by the upstream toolkit."
            ) from exc

        if sys.platform != "linux":
            raise RuntimeError(f"setup_mlmm requires Linux; current platform is {sys.platform}.")

        # NOTE: Full wiring of mlmm_toolkit's System construction is a follow-up
        # task. The skeleton here exists so command registration and module
        # imports stay consistent across platforms.
        raise NotImplementedError(
            "MLMMBackend.build_system is a stub — wire up mlmm_toolkit.build "
            "(ml region selection by `ml_selection`, MM region by complement) "
            "once the upstream API is finalized."
        )
