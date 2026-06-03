"""Configuration objects and a simple YAML loader.

All hyperparameters live in ``configs/default.yaml``. We use a plain
``pyyaml`` loader (not Hydra) to keep the dependency surface small; the nested
dataclasses below give typed, autocomplete-friendly access and document every
knob. Override files can be merged on top of the default.

TODO(config): if the experiment matrix grows, swap this for Hydra/OmegaConf to
get composition and command-line overrides for free.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, Type, TypeVar, get_type_hints

import yaml

T = TypeVar("T")


@dataclass
class ModelConfig:
    """Transformer + state head sizes."""

    consciousness_dim: int = 32
    memory_dim: int = 32
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    max_seq_len: int = 160


@dataclass
class TrainConfig:
    """Optimization and loss-weighting hyperparameters."""

    learning_rate: float = 3e-4
    batch_size: int = 8
    epochs: int = 3
    seed: int = 0
    weight_lm: float = 1.0
    weight_answer: float = 1.0
    weight_consistency: float = 0.1
    grad_clip: float = 1.0


@dataclass
class DataConfig:
    """Toy-dataset generation and split settings."""

    num_examples: int = 100
    val_fraction: float = 0.2
    seed: int = 0


@dataclass
class Config:
    """Top-level configuration."""

    curriculum_phase: int = 1
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)


def _from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    """Recursively construct a (possibly nested) dataclass from a dict.

    Unknown keys raise, so typos in the YAML fail loudly instead of being
    silently ignored.
    """
    if not is_dataclass(cls):
        return data  # type: ignore[return-value]
    kwargs: Dict[str, Any] = {}
    field_names = {f.name for f in fields(cls)}
    # Resolve string annotations (PEP 563) to real types for nested dataclasses.
    hints = get_type_hints(cls)
    for key, value in (data or {}).items():
        if key not in field_names:
            raise KeyError(f"Unknown config key {key!r} for {cls.__name__}")
        field_type = hints.get(key)
        if is_dataclass(field_type) and isinstance(value, dict):
            kwargs[key] = _from_dict(field_type, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)  # type: ignore[arg-type]


def default_config_path() -> str:
    """Absolute path to the packaged ``configs/default.yaml``."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "configs", "default.yaml"))


def load_config(path: str | None = None) -> Config:
    """Load a :class:`Config` from a YAML file (defaults to the packaged one)."""
    path = path or default_config_path()
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return _from_dict(Config, raw)
