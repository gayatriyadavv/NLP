import os
import yaml
from typing import Dict, Any


class ConfigDict(dict):
    """
    Plain dict that additionally resolves dotted key paths in ``get``.

    Callers throughout the codebase do ``config.get("a.b.c", default)`` on the
    result of ``get_config()``. On a plain dict that lookup always misses and
    silently returns the default, so config values were never applied. Normal
    (non-dotted) keys behave exactly like ``dict.get``.
    """

    def get(self, key: Any, default: Any = None) -> Any:
        if isinstance(key, str) and "." in key and not dict.__contains__(self, key):
            val: Any = self
            for k in key.split("."):
                if isinstance(val, dict) and k in val:
                    val = val[k]
                else:
                    return default
            return val
        return dict.get(self, key, default)


class ConfigLoader:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self._config = self._load_config()

    def _load_config(self) -> ConfigDict:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")
        with open(self.config_path, "r") as f:
            return ConfigDict(yaml.safe_load(f) or {})

    @property
    def config(self) -> ConfigDict:
        return self._config

    def get(self, key_path: str, default: Any = None) -> Any:
        return self._config.get(key_path, default)


def get_config(config_path: str = "config/config.yaml") -> ConfigDict:
    return ConfigLoader(config_path).config
