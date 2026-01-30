"""Load config/config.yaml."""
from pathlib import Path
import os
import yaml

_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _ROOT / "config" / "config.yaml"


def load_config():
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    for key in ("data_raw", "data_processed", "model_registry"):
        if key in cfg.get("paths", {}):
            p = cfg["paths"][key]
            if not os.path.isabs(p):
                cfg["paths"][key] = str(_ROOT / p)
    return cfg
