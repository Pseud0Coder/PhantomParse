"""Configuration loading for PhantomParse.

Precedence, highest first:

1. An explicit path passed to :func:`load_config`.
2. The user configuration at ``~/.config/phantomparse/config.yaml``.
3. The bundled defaults in ``config/default_config.yaml``.
4. Field defaults declared on :class:`Config`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLED_CONFIG_PATH = REPO_ROOT / "config" / "default_config.yaml"
USER_CONFIG_PATH = Path.home() / ".config" / "phantomparse" / "config.yaml"


@dataclass
class Config:
    """Runtime configuration shared by the daemon, overlay, and benchmarks."""

    hotkey: str = "<ctrl>+<space>"
    ttl_seconds: float = 30.0
    similarity_threshold: float = 0.75
    max_staging_items: int = 64
    model: str = "llama3"
    endpoint: str = "http://localhost:11434"
    prefill_tokens_per_second: float = 240.0
    typing_wpm: int = 60
    max_ram_mb: float = 50.0
    max_chunk_tokens: int = 24

    def to_dict(self) -> Dict[str, Any]:
        """Return the configuration as a plain dictionary."""
        return asdict(self)

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Persist the configuration as YAML, creating parent directories."""
        target = Path(path) if path is not None else USER_CONFIG_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=True)
        return target

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]]) -> "Config":
        """Build a config from raw YAML data, coercing types and ignoring unknown keys."""
        defaults = cls()
        coerced: Dict[str, Any] = {}
        for field in fields(cls):
            if values and values.get(field.name) is not None:
                coerced[field.name] = _coerce(values[field.name], getattr(defaults, field.name))
        return cls(**coerced)


def _coerce(value: Any, default: Any) -> Any:
    """Coerce a YAML scalar to the type of its dataclass field."""
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, str):
        return str(value)
    return value


def _read_yaml(path: Path) -> Dict[str, Any]:
    """Read a YAML mapping, returning an empty mapping when the file is absent."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        try:
            data = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML configuration at {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration at {path} must be a YAML mapping")
    return data


def load_config(path: Optional[Union[str, Path]] = None) -> Config:
    """Load the effective configuration without raising on missing files."""
    merged: Dict[str, Any] = dict(_read_yaml(BUNDLED_CONFIG_PATH))
    user_path = Path(path) if path is not None else USER_CONFIG_PATH
    merged.update(_read_yaml(user_path))
    return Config.from_mapping(merged)


def ensure_user_config(config: Optional[Config] = None) -> Path:
    """Create the user configuration from defaults when it does not exist yet."""
    if USER_CONFIG_PATH.exists():
        return USER_CONFIG_PATH
    return (config or Config()).save(USER_CONFIG_PATH)
