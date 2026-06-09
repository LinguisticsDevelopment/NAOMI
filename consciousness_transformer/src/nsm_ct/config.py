"""Configuration objects and a simple YAML loader.

All hyperparameters live in ``configs/default.yaml``. A plain ``pyyaml`` loader
keeps the dependency surface small; the nested dataclasses below give typed,
documented access. Unknown keys raise on load so typos fail loudly.

TODO(config): swap for Hydra/OmegaConf if the experiment matrix grows.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, Optional, Type, TypeVar, get_type_hints

import yaml

T = TypeVar("T")


@dataclass
class ModelConfig:
    """Transformer + state/memory sizes."""

    consciousness_dim: int = 32     # width of the abstract state vector
    memory_dim: int = 32            # width of memory slots / reads
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    max_sentence_len: int = 24      # max tokens per input sentence (and option/question)
    reasoning_hops: int = 1         # passes over memory at the question (1 = no multi-hop)
    use_long_term: bool = False     # enable persistent cross-episode long-term memory
    ltm_max_size: int = 10000       # cap on long-term entries (pruning placeholder)


@dataclass
class TrainConfig:
    """Optimization and loss-weighting hyperparameters."""

    learning_rate: float = 3e-4
    batch_size: int = 16
    epochs: int = 5
    seed: int = 0
    weight_answer: float = 1.0        # answer-correctness loss (the only task signal)
    weight_consistency: float = 0.05  # placeholder consciousness consistency weight
    grad_clip: float = 1.0


@dataclass
class DataConfig:
    """Episode source and split settings."""

    source: str = "curriculum"       # curriculum | babi | textbook
    answer_mode: str = "mc"          # mc | open
    num_episodes: int = 200
    val_fraction: float = 0.2
    seed: int = 0
    max_context: int = 6             # max statements per episode (padded)
    max_level: int = 3               # curriculum difficulty ceiling
    babi_task: int = 1
    babi_path: Optional[str] = None


@dataclass
class Config:
    """Top-level configuration."""

    curriculum_phase: int = 1
    input_encoder: str = "token"     # token | parser
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)


def _from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    """Recursively construct a (possibly nested) dataclass from a dict."""
    if not is_dataclass(cls):
        return data  # type: ignore[return-value]
    kwargs: Dict[str, Any] = {}
    field_names = {f.name for f in fields(cls)}
    hints = get_type_hints(cls)  # resolve PEP 563 string annotations
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
