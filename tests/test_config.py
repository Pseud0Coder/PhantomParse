"""Tests for the layered configuration loader."""

from __future__ import annotations

import pytest
import yaml

from phantomparse.config import (
    BUNDLED_CONFIG_PATH,
    Config,
    load_config,
    ensure_user_config,
)


def test_field_defaults_are_the_documented_runtime_values() -> None:
    config = Config()

    assert config.hotkey == "<ctrl>+<space>"
    assert config.ttl_seconds == 30.0
    assert config.similarity_threshold == 0.75
    assert config.max_staging_items == 64
    assert config.max_ram_mb == 50.0
    assert config.max_chunk_tokens == 24
    assert config.typing_wpm == 60


def test_bundled_yaml_mirrors_the_dataclass_defaults() -> None:
    assert BUNDLED_CONFIG_PATH.exists()

    bundled = yaml.safe_load(BUNDLED_CONFIG_PATH.read_text(encoding="utf-8"))
    assert bundled == Config().to_dict()


def test_explicit_path_overrides_bundled_defaults(tmp_path) -> None:
    override = tmp_path / "config.yaml"
    override.write_text(
        "ttl_seconds: 12.5\nmax_staging_items: 8\nmodel: mistral\n",
        encoding="utf-8",
    )

    config = load_config(override)

    assert config.ttl_seconds == 12.5
    assert config.max_staging_items == 8
    assert config.model == "mistral"
    assert config.hotkey == Config().hotkey


def test_missing_explicit_path_falls_back_to_defaults(tmp_path) -> None:
    config = load_config(tmp_path / "absent.yaml")
    assert config.to_dict() == Config().to_dict()


def test_from_mapping_coerces_scalar_types() -> None:
    config = Config.from_mapping(
        {
            "ttl_seconds": "45",
            "max_staging_items": "16",
            "typing_wpm": 90.0,
            "similarity_threshold": "0.6",
        }
    )

    assert config.ttl_seconds == 45.0
    assert isinstance(config.ttl_seconds, float)
    assert config.max_staging_items == 16
    assert isinstance(config.max_staging_items, int)
    assert config.typing_wpm == 90
    assert config.similarity_threshold == 0.6


def test_from_mapping_ignores_unknown_and_null_keys() -> None:
    config = Config.from_mapping({"unknown_key": "ignored", "model": None})
    assert config.model == Config().model
    assert not hasattr(config, "unknown_key")


def test_from_mapping_returns_defaults_for_none() -> None:
    assert Config.from_mapping(None).to_dict() == Config().to_dict()


def test_save_round_trips_through_load_config(tmp_path) -> None:
    custom = Config(model="phi3", ttl_seconds=9.0, max_chunk_tokens=12)
    path = custom.save(tmp_path / "nested" / "config.yaml")

    assert path.exists()
    assert load_config(path).to_dict() == custom.to_dict()


def test_invalid_yaml_raises_value_error(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("key: [unclosed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid YAML"):
        load_config(bad)


def test_non_mapping_yaml_raises_value_error(tmp_path) -> None:
    bad = tmp_path / "list.yaml"
    bad.write_text("- one\n- two\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_config(bad)


def test_ensure_user_config_writes_once(tmp_path, monkeypatch) -> None:
    target = tmp_path / "config.yaml"
    monkeypatch.setattr("phantomparse.config.USER_CONFIG_PATH", target)

    first = ensure_user_config(Config(model="qwen"))
    second = ensure_user_config(Config(model="other"))

    assert first == second == target
    assert load_config(target).model == "qwen"
