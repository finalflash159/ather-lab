"""Validated G4 run configuration. No implicit long run or output directory."""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from ather_exploration.config import FrozenConfig

METHOD_NAMES = ("ppo", "recurrent_ppo", "ppo_curriculum", "recurrent_ppo_curriculum")
LEGACY_METHODS = dict(zip(("A", "B", "C", "D"), METHOD_NAMES, strict=True))


def normalize_method(value):
    if not isinstance(value, str):
        raise ValueError("method must be a descriptive method name")  # noqa: TRY004 -- Pydantic validator
    value = LEGACY_METHODS.get(value, value)
    if value not in METHOD_NAMES:
        raise ValueError(f"Unknown method {value!r}; choose {', '.join(METHOD_NAMES)}")
    return value


class TrackingConfig(FrozenConfig):
    mode: Literal["online", "offline", "disabled"] = "disabled"
    project: str = Field(default="ather-exploration", min_length=1)
    entity: str | None = None


class TrainingConfig(FrozenConfig):
    method: Literal["ppo", "recurrent_ppo", "ppo_curriculum", "recurrent_ppo_curriculum"] = "ppo"

    @field_validator("method", mode="before")
    @classmethod
    def descriptive_method(cls, value):
        return normalize_method(value)

    @property
    def recurrent(self):
        return self.method in ("recurrent_ppo", "recurrent_ppo_curriculum")

    @property
    def curriculum_enabled(self):
        return self.method in ("ppo_curriculum", "recurrent_ppo_curriculum")

    seed: int = Field(default=0, ge=0, lt=2**32)
    total_timesteps: int = Field(default=4096, gt=0)
    n_envs: int = Field(default=1, gt=0, le=64)
    n_steps: int = Field(default=256, gt=1)
    batch_size: int = Field(default=64, gt=1)
    n_epochs: int = Field(default=4, gt=0)
    learning_rate: float = Field(default=3e-4, gt=0, allow_inf_nan=False)
    final_learning_rate: float = Field(default=3e-5, gt=0, allow_inf_nan=False)
    device: str = "cpu"
    vec_backend: Literal["dummy", "subproc"] = "dummy"
    torch_threads: int = Field(default=1, gt=0)
    checkpoint_updates: int = Field(default=1, gt=0)
    trace_every: int = Field(default=0, ge=0)
    banks: dict[str, str]
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)

    @model_validator(mode="after")
    def coherent(self):
        rollout = self.n_envs * self.n_steps
        if self.total_timesteps % rollout or rollout % self.batch_size:
            raise ValueError(
                "Budget must divide into full rollouts; batch_size must divide rollout"
            )
        if set(self.banks) != {"small", "medium", "large"}:
            raise ValueError("Provide exactly small/medium/large READY banks")
        if self.final_learning_rate > self.learning_rate:
            raise ValueError("Final learning rate must not exceed initial")
        return self


def read_training_config(path):
    import json

    import yaml

    path = Path(path).resolve()
    data = (
        json.loads(path.read_text()) if path.suffix == ".json" else yaml.safe_load(path.read_text())
    )
    config = TrainingConfig.model_validate(data)
    return config.model_copy(
        update={"banks": {k: str((path.parent / v).resolve()) for k, v in config.banks.items()}}
    )
