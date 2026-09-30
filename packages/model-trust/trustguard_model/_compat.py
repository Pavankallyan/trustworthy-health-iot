"""Window type compatibility shim.

The canonical ``Window`` type is owned by Layer 1 and defined in
``trustguard_data.stream`` (built in parallel by a sibling package).  Per the
frozen SPEC this module must import it as ``from trustguard_data.stream import
Window`` and must NOT redefine it.

Because the sibling package may not exist yet at import time, this module
tries the canonical import first and only falls back to a local dataclass that
is structurally identical to the SPEC definition when the import fails.  Once
``trustguard_data.stream`` is importable, the fallback is never used.
"""

from __future__ import annotations

try:  # canonical Layer 1 definition (preferred)
    from trustguard_data.stream import Window  # type: ignore[import]
    _WINDOW_SOURCE = "trustguard_data.stream"
except Exception:  # sibling package not built yet: SPEC-identical fallback
    from dataclasses import dataclass, field

    import numpy as np

    @dataclass
    class Window:  # type: ignore[no-redef]
        """Fallback Window, field-for-field identical to SPEC v1.0."""

        device_id: str
        start_s: float
        end_s: float
        channels: dict[str, np.ndarray] = field(default_factory=dict)
        fs: dict[str, float] = field(default_factory=dict)

    _WINDOW_SOURCE = "fallback (trustguard_data.stream not importable yet)"

__all__ = ["Window", "WINDOW_SOURCE"]
WINDOW_SOURCE = _WINDOW_SOURCE
