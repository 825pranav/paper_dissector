"""Coercion helpers for turning loose LLM JSON into schema-valid values.

Models regularly emit a score of ``25`` where the schema wants 0-1, an enum
label that is close-but-not-exact, or a bare string where a list is expected.
Rather than let a single sloppy field abort a whole claim, we normalise here.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, TypeVar

log = logging.getLogger(__name__)

E = TypeVar("E", bound=Enum)


def clamp01(value: Any, default: float = 0.5) -> float:
    """Coerce to a float in [0, 1]. Values that look like percentages are rescaled."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    if 1.0 < number <= 100.0:
        number = number / 100.0
    return min(max(number, 0.0), 1.0)


def coerce_enum(value: Any, enum_cls: type[E], default: E) -> E:
    """Best-effort mapping of an arbitrary model string onto an enum member."""
    if isinstance(value, enum_cls):
        return value
    if value is None:
        return default

    key = str(value).strip().upper().replace(" ", "_").replace("-", "_")

    for member in enum_cls:
        if key == str(member.value).upper() or key == member.name.upper():
            return member

    # Accept unambiguous prefixes, e.g. "PARTIAL" -> PARTIALLY_SUPPORTED.
    matches = [
        m for m in enum_cls
        if str(m.value).upper().startswith(key) or key.startswith(str(m.value).upper())
    ]
    if len(matches) == 1:
        return matches[0]

    log.warning("unrecognised %s value %r; defaulting to %s", enum_cls.__name__, value, default)
    return default


def as_str_list(value: Any) -> list[str]:
    """Normalise a model's 'list of strings' field, which may arrive as a bare string."""
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                # e.g. [{"point": "..."}] — take the first string value.
                for v in item.values():
                    if isinstance(v, str) and v.strip():
                        out.append(v.strip())
                        break
            elif item is not None:
                out.append(str(item))
        return out
    return [str(value)]


def as_text(value: Any, default: str = "") -> str:
    """Normalise a field that should be prose but may arrive as a dict or list."""
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip() or default
    if isinstance(value, (list, tuple)):
        parts = [as_text(v) for v in value]
        return " ".join(p for p in parts if p) or default
    if isinstance(value, dict):
        parts = [as_text(v) for v in value.values()]
        return " ".join(p for p in parts if p) or default
    return str(value)


def as_bool(value: Any, default: bool = False) -> bool:
    """Coerce a model's boolean field, which may arrive as a yes/no string."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    key = str(value).strip().lower()
    if key in ("true", "yes", "y", "1", "present"):
        return True
    if key in ("false", "no", "n", "0", "absent", "none", "null"):
        return False
    return default
