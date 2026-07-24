"""Configuration: defaults, YAML loading, CLI-style overrides."""
import tempfile
from pathlib import Path

from core.config import AppConfig


def test_default_config_is_complete():
    cfg = AppConfig()
    assert cfg.model.path.endswith(".pt")
    assert cfg.cameras.width > 0 and cfg.cameras.height > 0
    assert cfg.tracking.filter in ("kalman", "adaptive")
    assert 0.0 < cfg.model.conf <= 1.0


def test_yaml_values_override_defaults():
    yaml_text = """
model:
  conf: 0.75
cameras:
  width: 640
  height: 480
tracking:
  filter: "adaptive"
  reid: false
"""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cfg.yaml"
        p.write_text(yaml_text)
        cfg = AppConfig.from_yaml(str(p))
    assert cfg.model.conf == 0.75
    assert (cfg.cameras.width, cfg.cameras.height) == (640, 480)
    assert cfg.tracking.filter == "adaptive"
    assert cfg.tracking.reid is False
    # Untouched sections keep defaults
    assert cfg.model.path.endswith(".pt")


def test_missing_yaml_falls_back_to_defaults():
    cfg = AppConfig.from_yaml("does_not_exist_anywhere.yaml")
    assert cfg.cameras.width == AppConfig().cameras.width
