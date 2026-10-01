"""One episode for interactive inspection; no display or training dependencies."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from ather_exploration.agents.baselines import make_baseline
from ather_exploration.config import EnvConfig, load_config, load_preset
from ather_exploration.environment.env import make_env, make_fixture_env
from ather_exploration.evaluation.metrics import EpisodeMetrics
from ather_exploration.seeds import stage_rng
from ather_exploration.types import Action, AgentState
from ather_exploration.worlds.scenarios import implementation_id, scenario_hash, write_record


@dataclass(frozen=True)
class SessionSpec:
    preset: str = "small"
    seed: int = 42
    agent: str = "manual"
    fixture: str | None = None
    config_path: str | None = None
    action_seed: int = 0
    checkpoint: str | None = None
    deterministic: bool = True
    replay: str | None = None


class EpisodeSession:
    """Owned by one worker. The UI only receives detached frames of committed state."""

    def __init__(self, spec: SessionSpec):
        if spec.agent not in (
            "manual",
            "random",
            "frontier",
            "ppo",
            "recurrent",
            "replay",
            "checkpoint",
        ):
            raise ValueError("Unknown controller")
        if spec.agent == "replay" and not spec.replay:
            raise ValueError("Replay mode requires --replay FILE")
        self.spec = spec
        self.agent = make_baseline(spec.agent) if spec.agent in ("random", "frontier") else None
        self.replay_steps = None
        self.state = AgentState()
        self.rng = stage_rng(spec.action_seed, "baseline-action")
        if spec.replay:
            from ather_exploration.evaluation.replay import replay_env, verify_replay
            from ather_exploration.worlds.scenarios import read_record

            verify_replay(spec.replay)
            payload = read_record(spec.replay)
            self.env = replay_env(payload)
            self.replay_steps = payload["steps"]
            self.replay_index = 0
        elif spec.fixture:
            self.env = make_fixture_env(spec.fixture)
        else:
            config = load_config(spec.config_path) if spec.config_path else load_preset(spec.preset)
            if spec.checkpoint and not spec.config_path:
                from ather_exploration.training.checkpoints import inspect_checkpoint

                _, metadata = inspect_checkpoint(spec.checkpoint)
                stored = metadata.get("environment_configs", {}).get(spec.preset)
                if stored is not None:
                    config = EnvConfig.model_validate(stored)
            skill_state = (
                metadata.get("skill_controller")
                if spec.checkpoint and not spec.config_path
                else None
            )
            viewer_task = metadata.get("viewer_task") if skill_state else None
            if skill_state and (viewer_task or "").startswith(("P1", "P2", "P3", "P4")):
                from ather_exploration.config import RewardConfig
                from ather_exploration.worlds.skill_tasks import make_skill_env

                skill_config = metadata["config"]["skills"]
                self.env = make_skill_env(
                    viewer_task,
                    spec.seed,
                    RewardConfig(
                        activation=skill_config["activation"], death=skill_config["death"]
                    ),
                    skill_config["first_visit"],
                )
            else:
                self.env = make_env(config)
        try:
            self.obs, _ = self.env.reset(seed=spec.seed)
            if spec.agent in ("ppo", "recurrent", "checkpoint"):
                if not spec.checkpoint:
                    raise ValueError("Select a READY checkpoint directory")
                from ather_exploration.training.checkpoints import load_agent

                self.agent = load_agent(spec.checkpoint, self.env.observation_space)
                if spec.agent != "checkpoint" and self.agent.name != spec.agent:
                    raise ValueError("Selected agent differs from checkpoint architecture")
            core = self.env.unwrapped
            self.metrics = EpisodeMetrics(
                core.evaluator_snapshot(),
                self.obs,
                core.reward_config,
                episode_id=f"ui-{scenario_hash(core.scenario)[:12]}",
                group=spec.preset,
                metadata={"agent": spec.agent, "scope": "interactive_debug", "seed": spec.seed},
            )
            self.done = self.replay_steps is not None and len(self.replay_steps) == 1
            self.result = None
            self.total_reward = 0.0
        except Exception:
            self.env.close()
            raise

    def frame(self):
        core = self.env.unwrapped
        diagnostics = dict(self.state.planner.diagnostics) if self.state.planner else {}
        return {
            "snapshot": core.evaluator_snapshot(),
            "observation": {k: v.copy() for k, v in self.obs.items()},
            "row": self.metrics.steps[-1],
            "diagnostics": diagnostics,
            "total_reward": self.total_reward,
            "done": self.done,
            "spec": self.spec,
            "checkpoint": getattr(self.agent, "metadata", {}).get("checkpoint"),
            "phase": getattr(self.agent, "metadata", {}).get("viewer_task"),
            "history": [(r["t"], r["coverage"], r["activation"] or 0) for r in self.metrics.steps],
        }

    def step(self, action=None):
        if self.done:
            raise ValueError("Episode ended; reset before stepping")
        if self.replay_steps is not None:
            if self.replay_index + 1 >= len(self.replay_steps):
                raise ValueError("End of recorded replay")
            self.replay_index += 1
            action = Action(self.replay_steps[self.replay_index]["action"])
        elif self.agent is None:
            if action is None:
                raise ValueError("Manual mode needs an explicit action")
            action = Action(action)
        else:
            if action is not None:
                raise ValueError("Manual input is disabled while a baseline controls the episode")
            action, self.state = self.agent.act(
                {k: v.copy() for k, v in self.obs.items()},
                self.state,
                deterministic=self.spec.deterministic,
                action_rng=self.rng,
            )
        self.obs, reward, terminated, truncated, info = self.env.step(action)
        core = self.env.unwrapped
        row = self.metrics.update(
            core.evaluator_snapshot(), self.obs, reward, info["transition"], core.collision_stage
        )
        row["terminated"], row["truncated"] = bool(terminated), bool(truncated)
        self.total_reward += reward
        self.done = (
            terminated
            or truncated
            or (self.replay_steps is not None and self.replay_index == len(self.replay_steps) - 1)
        )
        if self.done:
            self.result = self.metrics.finish(cancelled=truncated or not terminated)
        return self.frame()

    def export(self, path):
        # This is an inspection trace, NOT a verified S16 replay/checkpoint artifact.
        core = self.env.unwrapped
        payload = {
            "schema": "interactive-trace-v1",
            "scope": "debug_not_formal_evaluation",
            "implementation_id": implementation_id(),
            "spec": asdict(self.spec),
            "scenario": asdict(core.scenario),
            "scenario_hash": scenario_hash(core.scenario),
            "observation_config": core.sensor_config.model_dump(mode="json"),
            "reward_config": core.reward_config.model_dump(mode="json"),
            "steps": self.metrics.steps,
            "result": self.result,
            "status": "completed" if self.done else "partial",
        }
        if core.generated is not None:
            payload = {
                "schema": "g4-replay-v1",
                "source_revision": implementation_id(),
                "record": core.generated.payload(),
                "steps": self.metrics.steps,
                "result": self.result,
                "spec": asdict(self.spec),
                "scope": "interactive_debug",
            }
        write_record(path, payload)
        return str(path)

    def close(self):
        self.env.close()


class SessionController:
    """Serialize simulation work off the display thread; discard stale results."""

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="episode")
        self._session = None  # Access only inside this executor.
        self._request = 0
        self.future = None
        self.frame = None
        self.state = "empty"
        self.error = ""
        self.exported = None
        self.running = False
        self.spec = None
        self._operation = None

    def _submit(self, operation, *args):
        if self.future is not None:
            self.future.cancel()
        self._request += 1
        self._operation = operation
        request = self._request
        self.future = self.executor.submit(self._execute, request, operation, args)

    def _execute(self, request, operation, args):
        if request != self._request:
            return request, operation, None
        if operation == "generate":
            if self._session is not None:
                self._session.close()
                self._session = None
            self._session = EpisodeSession(args[0])
            result = self._session.frame()
        elif operation == "step":
            result = self._session.step(*args)
        elif operation == "export":
            result = self._session.export(*args)
        else:
            if self._session is not None:
                self._session.close()
                self._session = None
            result = None
        return request, operation, result

    def generate(self, spec):
        self.spec = spec
        self.running = False
        self.state, self.error, self.frame = "generating", "", None
        self._submit("generate", spec)

    def cancel(self):
        self.running = False
        self.state, self.frame = "empty", None
        self._submit("close")

    @property
    def busy(self):
        return self.future is not None

    def step(self, action=None):
        if (
            not self.busy
            and self.state != "error"
            and self.frame is not None
            and not self.frame["done"]
        ):
            self._submit("step", action)

    def pause(self):
        self.running = False
        if self.state != "error" and self.frame is not None and not self.frame["done"]:
            self.state = "paused"

    def start(self):
        if (
            self.state != "error"
            and self.frame is not None
            and not self.frame["done"]
            and self.spec.agent != "manual"
        ):
            self.running = True
            self.state = "running"

    def export(self, path):
        self.pause()
        if self.busy or self.frame is None:
            raise ValueError("Pause and wait for the current step before exporting")
        self._submit("export", path)

    def poll(self):
        if self.future is None or not self.future.done():
            return False
        future, self.future = self.future, None
        try:
            request, operation, result = future.result()
            if request != self._request:
                return False
            self.error = ""
            if operation == "export":
                self.exported = result
            elif operation != "close":
                self.frame = result
                if result["done"]:
                    self.running = False
                    self.state = "ended"
                else:
                    self.state = "running" if self.running else "paused"
        except Exception as error:  # noqa: BLE001 -- surface worker failures, no silent fallback
            self.running = False
            self.state = "paused" if self._operation == "export" else "error"
            self.error = f"{type(error).__name__}: {error}"
        return True

    def close(self):
        self.cancel()
        # Running generation is bounded by G2 budgets. Its result is invalidated;
        # cleanup runs serially afterwards, never closing an env during a step.
        self.executor.shutdown(wait=False, cancel_futures=False)
