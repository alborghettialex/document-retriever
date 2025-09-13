from yaml import safe_load
from pathlib import Path

def load_config(config_path: str):
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = safe_load(f)
    return config