"""Validated G4 run configuration. No implicit long run or output directory."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


class P1RewardConfig(FrozenConfig):
    area: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    discovery: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    activation: float = Field(default=0.5, gt=0, allow_inf_nan=False)
    step_cost: float = Field(default=0.005, ge=0, allow_inf_nan=False)


class P2RewardConfig(P1RewardConfig):
    wall_penalty: float = Field(default=0.02, ge=0, allow_inf_nan=False)
    area: float = Field(default=0.01, ge=0, allow_inf_nan=False)
    discovery: float = Field(default=0.05, ge=0, allow_inf_nan=False)


class P2GateConfig(FrozenConfig):
    # Pilot thresholds: calibrate before formal comparisons.
    approach_efficiency: float = Field(default=0.60, ge=0, le=1)
    discovery_success: float = Field(default=0.95, ge=0, le=1)
    coverage: float = Field(default=0.75, ge=0, le=1)
    coverage_auc: float = Field(default=0.50, ge=0, le=1)


class P3RewardConfig(FrozenConfig):
    area: float = Field(default=0.01, ge=0, allow_inf_nan=False)
    discovery: float = Field(default=0.05, ge=0, allow_inf_nan=False)
    activation: float = Field(default=0.5, gt=0, allow_inf_nan=False)
    wall_penalty: float = Field(default=0.02, ge=0, allow_inf_nan=False)
    room_exploration: float = Field(default=0.0, ge=0, le=2.0, allow_inf_nan=False)


class P3GateConfig(FrozenConfig):
    # Explicit pilot thresholds; report joint outcomes and every map stratum.
    success: float = Field(default=0.80, ge=0, le=1)
    stochastic_success: float = Field(default=0.75, ge=0, le=1)
    coverage: float = Field(default=0.75, ge=0, le=1)
    coverage_auc: float = Field(default=0.50, ge=0, le=1)
    room_coverage: float = Field(default=0.50, ge=0, le=1)
    room_coverage_auc: float = Field(default=0.50, ge=0, le=1)
    joint_success: float = Field(default=0.70, ge=0, le=1)
    subgroup_success: float = Field(default=0.60, ge=0, le=1)
    wall_block: float = Field(default=0.10, ge=0, le=1)


class P4Config(FrozenConfig):
    enabled: bool = False
    task_budget: int = Field(default=1048576, ge=65536)
    minimum: int = Field(default=65536, ge=32768)
    success: float = Field(default=0.75, ge=0, le=1)
    survival: float = Field(default=0.85, ge=0, le=1)
    joint: float = Field(default=0.65, ge=0, le=1)
    room_coverage: float = Field(default=0.50, ge=0, le=1)
    coverage_auc: float = Field(default=0.40, ge=0, le=1)


class SkillConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    enabled: bool = False
    stop_after: Literal["P1", "P2", "P3", "P4", "P5"] = "P5"
    train_count: int = Field(default=256, ge=8, le=256)
    validation_count: int = Field(default=64, ge=4, le=64)
    eval_interval: int = Field(default=16384, gt=0)
    activation: float = Field(default=0.5, gt=0, allow_inf_nan=False)
    death: float = Field(default=2.0, gt=0, allow_inf_nan=False)
    frontier: bool = False
    p4: P4Config = Field(default_factory=P4Config)
    p3_visit_bonus: float = Field(default=0.0, ge=0, le=0.01, allow_inf_nan=False)
    p3_visit_cap: float = Field(default=0.1, gt=0, le=0.5, allow_inf_nan=False)
    first_visit: bool = False
    wall_mask: bool = False
    p1_reward: P1RewardConfig = Field(default_factory=P1RewardConfig)
    p2_reward: P2RewardConfig = Field(default_factory=P2RewardConfig)
    p2_gates: P2GateConfig = Field(default_factory=P2GateConfig)
    p2c_horizon: int = Field(default=192, ge=32, le=1024)
    p2_task_budget: int = Field(default=262144, ge=32768)
    p3_reward: P3RewardConfig = Field(default_factory=P3RewardConfig)
    p3_gates: P3GateConfig = Field(default_factory=P3GateConfig)
    p3_horizon: int = Field(default=256, ge=192, le=1024)
    p3_minimum: int = Field(default=65536, ge=32768)
    p3_task_budget: int = Field(default=524288, ge=65536)


class LearningRateTrial(FrozenConfig):
    parent_steps: int = Field(default=655360, gt=0)
    additional_steps: int = Field(default=65536, gt=0)
    task: Literal["P3a"] = "P3a"


class UnfinishedTrial(FrozenConfig):
    """Fixed-task controlled branch; additional steps exclude prefix reconstruction."""

    parent_steps: int = Field(default=1212416, gt=0)
    additional_steps: int = Field(default=65536, gt=0)
    restart_probability: float = Field(default=0.25, ge=0, le=0.5)
    pool_per_band: int = Field(default=16, ge=1, le=64)
    minimum_remaining: int = Field(default=64, ge=32)
    mastery_episodes: int = Field(default=8, ge=4)
    mastery_rate: float = Field(default=0.75, gt=0, le=1)


class P3ResumeConfig(FrozenConfig):
    """Audited P3a-boundary continuation with P3b/P3c room-completion replay."""

    parent_steps: int = Field(default=737280, gt=0)
    restart_probability: float = Field(default=0.25, ge=0, le=0.5)
    pool_per_band: int = Field(default=16, ge=1, le=64)
    minimum_remaining: int = Field(default=64, ge=32)
    mastery_episodes: int = Field(default=8, ge=4)
    mastery_rate: float = Field(default=0.75, gt=0, le=1)


class RecoveryConfig(FrozenConfig):
    """Public-route supervision and stagnation replay; no inference helper."""

    sampling: Literal["stale_uniform", "disagreement_balanced", "aggregated_teaching"] = (
        "stale_uniform"
    )
    parent_steps: int = 1048576
    additional_steps: int = Field(default=131072, ge=65536)
    route_coefficient: float = Field(default=0.02, ge=0, le=0.1, allow_inf_nan=False)
    label_limit: int = Field(default=256, ge=1, le=1024)
    label_after: int = Field(default=8, ge=1)
    stagnation_steps: int = Field(default=16, ge=8)
    lookback: int = Field(default=8, ge=1)
    minimum_remaining: int = Field(default=64, ge=64)
    restart_probability: float = Field(default=0.25, ge=0, le=0.5)
    pool_per_kind: int = Field(default=24, ge=1, le=128)
    probe_count: int = Field(default=64, ge=1, le=256)


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

    skills: SkillConfig = Field(default_factory=SkillConfig)
    p3_restart: bool = False
    lr_trial: LearningRateTrial | None = None
    unfinished_trial: UnfinishedTrial | None = None
    p3_resume: P3ResumeConfig | None = None
    recovery: RecoveryConfig | None = None
    p4_transfer: bool = False
    seed: int = Field(default=0, ge=0, lt=2**32)
    total_timesteps: int = Field(default=4096, gt=0)
    n_envs: int = Field(default=1, gt=0, le=64)
    n_steps: int = Field(default=256, gt=1)
    batch_size: int = Field(default=64, gt=1)
    n_epochs: int = Field(default=4, gt=0)
    gamma: float = Field(default=0.999, gt=0, le=1)
    gae_lambda: float = Field(default=0.95, ge=0, le=1)
    ent_coef: float = Field(default=0.01, ge=0)
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
        if self.p4_transfer:
            if (
                not self.skills.enabled
                or not self.skills.p4.enabled
                or not self.skills.frontier
                or self.method != "ppo"
                or self.skills.stop_after != "P4"
                or self.recovery
                or self.p3_resume
                or self.lr_trial
                or self.unfinished_trial
                or self.p3_restart
                or self.skills.wall_mask
            ):
                raise ValueError(
                    "P4 transfer requires frontier PPO through P4 with no other transfer"
                )
            if (
                self.skills.p4.minimum > self.skills.p4.task_budget
                or self.skills.p4.task_budget % self.skills.eval_interval
                or self.skills.p4.minimum % self.skills.eval_interval
                or self.total_timesteps < 1638400 + 3 * self.skills.p4.task_budget
            ):
                raise ValueError("P4 budgets must cover all three stages and align evaluation")
        if self.skills.p4.enabled and not self.p4_transfer:
            raise ValueError("P4 design requires explicit audited transfer")
        if self.recovery:
            r = self.recovery
            if (
                self.p3_resume
                or self.p3_restart
                or self.lr_trial
                or self.unfinished_trial
                or not self.skills.enabled
                or not self.skills.frontier
                or self.skills.stop_after != "P3"
                or self.method != "ppo"
                or self.skills.wall_mask
            ):
                raise ValueError(
                    "Recovery requires frontier skill PPO through P3, no other transfer"
                )
            if self.learning_rate != self.final_learning_rate:
                raise ValueError("Recovery requires constant learning rate")
            if r.sampling == "aggregated_teaching" and r.probe_count < 32:
                raise ValueError("Teaching requires at least 32 stratified train probes")
            if (
                r.parent_steps != (1572864 if r.sampling != "stale_uniform" else 1048576)
                or r.parent_steps % self.skills.eval_interval
                or r.additional_steps % self.skills.eval_interval
                or r.additional_steps < self.skills.p3_minimum
                or r.parent_steps + r.additional_steps + self.skills.p3_task_budget
                > self.total_timesteps
                or self.skills.p3_task_budget % self.skills.eval_interval
                or r.minimum_remaining >= self.skills.p3_horizon
                or r.lookback >= r.stagnation_steps
                or r.probe_count > self.skills.train_count
            ):
                raise ValueError("Recovery boundaries, probe count or replay horizon invalid")
        if self.unfinished_trial:
            trial = self.unfinished_trial
            if (
                self.p3_restart
                or self.lr_trial
                or not self.skills.frontier
                or not self.skills.enabled
                or self.method != "ppo"
                or self.skills.wall_mask
            ):
                raise ValueError(
                    "Unfinished trial requires frontier skill PPO without other transfers"
                )
            if self.learning_rate != self.final_learning_rate:
                raise ValueError("Unfinished trial requires constant learning rate")
            if (
                trial.parent_steps % self.skills.eval_interval
                or trial.additional_steps % self.skills.eval_interval
                or trial.parent_steps + trial.additional_steps > self.total_timesteps
            ):
                raise ValueError("Unfinished trial boundaries must align with evaluation/budget")
            if trial.minimum_remaining >= self.skills.p3_horizon:
                raise ValueError("Restart must leave a nonempty prefix")

        if self.p3_resume:
            resume = self.p3_resume
            if (
                self.p3_restart
                or self.lr_trial
                or self.unfinished_trial
                or not self.skills.frontier
                or not self.skills.enabled
                or self.skills.stop_after != "P3"
                or self.method != "ppo"
                or self.skills.wall_mask
            ):
                raise ValueError(
                    "P3 resume requires frontier PPO through P3 without other transfer modes"
                )
            if resume.parent_steps % self.skills.eval_interval:
                raise ValueError("P3 resume checkpoint must align with evaluation interval")
            if resume.parent_steps >= self.total_timesteps:
                raise ValueError("P3 resume parent must be below total_timesteps")
            if resume.minimum_remaining >= self.skills.p3_horizon:
                raise ValueError("P3 replay must leave a nonempty episode suffix")

        if self.skills.frontier and (
            not self.skills.enabled or self.skills.stop_after != "P3" and not self.p4_transfer
        ):
            raise ValueError("Frontier currently supported for the P3 training family")
        if self.p3_restart and (
            self.lr_trial
            or not self.skills.frontier
            or self.method != "ppo"
            or self.skills.wall_mask
        ):
            raise ValueError("P3 restart requires unmasked frontier PPO and no LR trial")
        if self.lr_trial:
            if not self.skills.enabled or self.method != "ppo" or self.skills.wall_mask:
                raise ValueError("LR trial requires unmasked skill PPO")
            if self.learning_rate != self.final_learning_rate:
                raise ValueError("LR trial uses an explicit constant learning rate")
            if (
                self.lr_trial.additional_steps % self.skills.eval_interval
                or self.lr_trial.parent_steps % self.skills.eval_interval
            ):
                raise ValueError("LR trial boundaries must align with evaluation intervals")
            if self.lr_trial.parent_steps + self.lr_trial.additional_steps > self.total_timesteps:
                raise ValueError("LR trial exceeds total budget")
        if self.skills.p3_minimum > self.skills.p3_task_budget:
            raise ValueError("P3 minimum exceeds task budget")
        if self.skills.enabled:
            if self.method != "ppo":
                raise ValueError("Skill curriculum currently supports feedforward ppo only")
            if self.skills.eval_interval % rollout:
                raise ValueError("Skill eval interval must divide into full rollouts")
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
