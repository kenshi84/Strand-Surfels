"""YAML configuration with command line overrides.

Every entry of the configuration file becomes a command line option named by
its dotted path, e.g. ``--optimization.iterations 30000`` or
``--optimization.save_iterations 7000 15000``. Booleans take ``true`` /
``false``.
"""

from __future__ import annotations

import argparse
import copy
from typing import Any, Dict, List, Optional

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    """Configuration dictionary of a YAML file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_bool(text: str) -> bool:
    value = text.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"not a boolean: {text!r}")


def _flatten(cfg: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat = {}
    for key, value in cfg.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, path + "."))
        else:
            flat[path] = value
    return flat


def add_config_arguments(parser: argparse.ArgumentParser, cfg: Dict[str, Any]) -> None:
    """Add one ``--dotted.path`` option per configuration entry (default: unset)."""
    group = parser.add_argument_group("configuration overrides")
    for path, value in _flatten(cfg).items():
        dest = "cfg__" + path.replace(".", "__")
        if isinstance(value, bool):
            group.add_argument(f"--{path}", type=_parse_bool, dest=dest, metavar="BOOL")
        elif isinstance(value, list):
            item_type = type(value[0]) if value else str
            group.add_argument(f"--{path}", type=item_type, nargs="*", dest=dest)
        elif value is None:
            group.add_argument(f"--{path}", dest=dest)
        else:
            group.add_argument(f"--{path}", type=type(value), dest=dest)


def apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Copy of ``cfg`` with the options given on the command line applied."""
    cfg = copy.deepcopy(cfg)
    for dest, value in vars(args).items():
        if not dest.startswith("cfg__") or value is None:
            continue
        *parents, leaf = dest[len("cfg__") :].split("__")
        node = cfg
        for key in parents:
            node = node[key]
        node[leaf] = value
    return cfg


def merge_defaults(cfg: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Copy of ``cfg`` with the entries of ``defaults`` it lacks (recursively)."""
    out = copy.deepcopy(cfg)
    for key, value in defaults.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_defaults(out[key], value)
        elif key not in out:
            out[key] = copy.deepcopy(value)
    return out


def load_config(
    path: str,
    argv: Optional[List[str]] = None,
    parser: Optional[argparse.ArgumentParser] = None,
    defaults: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], argparse.Namespace]:
    """Configuration of ``path`` (completed with ``defaults``) with overrides
    from ``argv``; also returns the parsed arguments of ``parser`` (to which
    the override options are added)."""
    cfg = merge_defaults(load_yaml(path), defaults or {})
    parser = parser or argparse.ArgumentParser()
    add_config_arguments(parser, cfg)
    args = parser.parse_args(argv)
    return apply_overrides(cfg, args), args


def save_config(path: str, cfg: Dict[str, Any]) -> None:
    """Write a configuration dictionary as YAML."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
