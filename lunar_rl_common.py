"""Shared Lunar Lander dict-observation + PPO setup for the notebook and Optuna script."""

from __future__ import annotations

import copy
import csv
import os
import warnings
from collections.abc import Callable

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from gymnasium.wrappers.transform_observation import AddRenderObservation

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import FloatSchedule, LinearSchedule
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.vec_env import unwrap_vec_normalize

DEFAULT_ENV_ID = "LunarLander-v3"


def suggested_parallel_envs(
    reserve_cores: int = 0,
    min_envs: int = 8,
    max_envs: int = 192,
) -> int:
    """
    SubprocVecEnv worker count from os.cpu_count(): one process per env, mostly CPU-bound
    (Box2D + OpenCV). Keeps the previous default (~32) on typical workstations, scales up
    on large CPUs (capped at max_envs), and never exceeds n_cpu. Optional reserve_cores leaves
    headroom for the trainer / OS if you set it > 0.
    """
    n_cpu = int(os.cpu_count() or 1)
    desired = min(max_envs, max(32, n_cpu - reserve_cores))
    return max(min_envs, min(desired, n_cpu))


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_train_device(preference: str = "auto") -> str:
    """
    Where SB3 puts the policy/value networks and runs backward/optimizer steps.
    Vectorized envs (e.g. SubprocVecEnv) always execute in CPU worker processes; only the
    model's device is selected here.
    """
    p = (preference or "auto").strip().lower()
    if p == "cpu":
        return "cpu"
    if p == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if p == "auto":
        return get_device()
    raise ValueError(
        f"Unknown train device preference: {preference!r} (use 'auto', 'cpu', or 'cuda')"
    )


class VecNormalizeSaveCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_training_start(self) -> None:
        if isinstance(self.training_env, VecNormalize):
            d = os.path.dirname(self.save_path)
            if d:
                os.makedirs(d, exist_ok=True)
            self.training_env.save(self.save_path)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            if isinstance(self.training_env, VecNormalize):
                self.training_env.save(self.save_path)
        return True


class GrayscaleResizePixelsWrapper(gym.ObservationWrapper):
    """Dict obs: only processes obs[\"pixels\"] (RGB HWC uint8 -> grayscale CHW uint8 1x84x84)."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
                ),
                "pixels": spaces.Box(
                    low=0, high=255, shape=(1, 84, 84), dtype=np.uint8
                ),
            }
        )

    def observation(self, obs):
        obs = dict(obs)
        obs["state"] = np.asarray(obs["state"], dtype=np.float32)
        px = np.asarray(obs["pixels"], dtype=np.uint8)
        if px.ndim == 3:
            gray = cv2.cvtColor(px, cv2.COLOR_RGB2GRAY)
        else:
            gray = np.squeeze(px)
        gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        obs["pixels"] = np.expand_dims(gray, axis=0).astype(np.uint8)
        return obs


def make_lunar_dict_env(env_id: str | None = None):
    eid = env_id or DEFAULT_ENV_ID
    env = gym.make(eid, render_mode="rgb_array")
    env = AddRenderObservation(
        env, render_only=False, render_key="pixels", obs_key="state"
    )
    env = GrayscaleResizePixelsWrapper(env)
    env = Monitor(env)
    return env


def make_subproc_venv(n_envs, seed, env_id: str | None = None):
    def make_env(rank):
        def _init():
            e = make_lunar_dict_env(env_id)
            e.reset(seed=seed + rank)
            return e

        return _init

    return SubprocVecEnv([make_env(i) for i in range(n_envs)])


def make_train_vec_env(n_envs, seed, gamma, env_id: str | None = None):
    return VecNormalize(
        make_subproc_venv(n_envs, seed, env_id),
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=gamma,
        norm_obs_keys=["state"],
    )


def make_eval_vec_env_with_stats(stats_path, seed, env_id: str | None = None):
    def factory():
        return make_lunar_dict_env(env_id)

    venv = DummyVecEnv([factory])
    venv = VecNormalize.load(stats_path, venv)
    venv.training = False
    venv.norm_reward = False
    venv.seed(seed)
    venv.reset()
    return venv


def sync_vecnormalize_obs_rms(src: VecNormalize, dst: VecNormalize) -> None:
    """Copy running observation normalization stats from training VecNormalize to eval copy."""
    if not src.norm_obs or not dst.norm_obs:
        return
    dst.obs_rms = copy.deepcopy(src.obs_rms)


def make_eval_vec_env_synced(
    train_venv: VecNormalize, seed: int, env_id: str | None = None
) -> VecNormalize:
    """
    Single-env eval VecNormalize with the same norm settings as training, no reward norm,
    training=False (no stat updates). obs_rms matches train_venv at call time.
    """
    def factory():
        return make_lunar_dict_env(env_id)

    venv = DummyVecEnv([factory])
    ev = VecNormalize(
        venv,
        norm_obs=train_venv.norm_obs,
        norm_reward=False,
        clip_obs=train_venv.clip_obs,
        clip_reward=train_venv.clip_reward,
        gamma=train_venv.gamma,
        epsilon=train_venv.epsilon,
        norm_obs_keys=(
            list(train_venv.norm_obs_keys)
            if train_venv.norm_obs_keys is not None
            else None
        ),
    )
    ev.training = False
    sync_vecnormalize_obs_rms(train_venv, ev)
    ev.seed(seed)
    ev.reset()
    return ev


def make_ppo_lr_schedule(
    lr_start: float, lr_end_factor: float = 0.05, lr_floor: float = 1e-5
) -> FloatSchedule:
    """Linear LR decay: high at start, low at end. lr_end = max(lr_floor, lr_start * lr_end_factor)."""
    lr_end = max(lr_floor, float(lr_start) * lr_end_factor)
    return FloatSchedule(LinearSchedule(float(lr_start), lr_end, end_fraction=1.0))


def make_ppo_clip_range_schedule(
    clip_start: float = 0.2, clip_end: float = 0.05
) -> FloatSchedule:
    return FloatSchedule(
        LinearSchedule(float(clip_start), float(clip_end), end_fraction=1.0)
    )


def get_train_vec_normalize(env) -> VecNormalize | None:
    if isinstance(env, VecNormalize):
        return env
    return unwrap_vec_normalize(env)


class PeriodicEvalCallback(BaseCallback):
    """
    Periodic **held-out** evaluation during training:

    1. Take the current training ``VecNormalize`` stats (``get_train_vec_normalize``).
    2. Build a **single-env** eval stack via ``make_eval_vec_env_synced`` — same dict obs,
       same obs normalization as training, ``norm_reward=False``, ``training=False``.
    3. Run Stable-Baselines3 ``evaluate_policy`` for ``n_eval_episodes`` (default 10),
       **deterministic** actions, and record ``mean_reward`` and ``std_reward`` (std across
       those episode returns). Rows are appended to ``csv_path`` and ``eval_history``.

    This matches the notebook / Hub eval path (synced normalization, no reward clipping in eval).
    """

    def __init__(
        self,
        eval_freq: int,
        n_eval_episodes: int,
        seed: int,
        csv_path: str = "logs/periodic_eval.csv",
        env_id: str | None = None,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.seed = seed
        self.csv_path = csv_path
        self.env_id = env_id
        self.deterministic = deterministic
        self._last_eval_at = 0
        self.eval_history: list[dict] = []

    def _write_row(
        self,
        timesteps: int,
        mean_reward: float,
        std_reward: float,
        n_episodes: int,
    ) -> None:
        d = os.path.dirname(self.csv_path)
        if d:
            os.makedirs(d, exist_ok=True)
        write_header = not os.path.isfile(self.csv_path)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(
                    [
                        "timesteps",
                        "mean_reward",
                        "std_reward",
                        "n_episodes",
                        "n_eval_episodes",
                    ]
                )
            w.writerow(
                [
                    timesteps,
                    mean_reward,
                    std_reward,
                    n_episodes,
                    self.n_eval_episodes,
                ]
            )
        self.eval_history.append(
            {
                "timesteps": timesteps,
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "n_episodes": n_episodes,
            }
        )

    def _run_eval(self) -> None:
        train_vn = get_train_vec_normalize(self.training_env)
        if train_vn is None:
            warnings.warn("PeriodicEvalCallback: training env is not VecNormalize; skip.")
            return

        eval_env = make_eval_vec_env_synced(train_vn, self.seed, self.env_id)
        try:
            mean_r, std_r = evaluate_policy(
                self.model,
                eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
            )
            self._write_row(
                self.num_timesteps,
                float(mean_r),
                float(std_r),
                self.n_eval_episodes,
            )
        finally:
            eval_env.close()

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_at >= self.eval_freq:
            self._last_eval_at = self.num_timesteps
            self._run_eval()
        return True


class LiveRewardPlotCallback(BaseCallback):
    """
    Live training plot: rolling **training** episode stats (from env ``info[\"episode\"]``),
    not the same draw as ``evaluate_policy``:

    - **Mean reward**: rolling mean of the last ``window`` completed training episode returns.
    - **Std (red axis)**: for each index ``i``, sample std of training returns in
      ``rews[i-window:i]`` (last ``window`` episodes ending at ``i``).
    - **Score (mean − std)**: ``mean[i] - std[i]`` on that rolling window — a training-curve
      analogue of the leaderboard-style ``mean - std``, not the periodic eval numbers.

    Optional: overlay periodic eval **mean** and **mean − std** from
    ``PeriodicEvalCallback.eval_history`` (no ± band fill).

    Saves PNG; optional ``display_fn`` can refresh Jupyter output with ``display_id``.
    """

    DISPLAY_ID_DEFAULT = "rl_lunarlander_live_reward"

    def __init__(
        self,
        window: int = 50,
        plot_freq: int = 5000,
        periodic_eval_cb: PeriodicEvalCallback | None = None,
        eval_color: str = "#7B1FA2",
        eval_conservative_color: str = "#311B92",
        solved_reference_y: float | None = 350.0,
        save_path: str | None = None,
        display_fn: Callable[..., None] | None = None,
        display_id: str | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.window = int(window)
        self.plot_freq = int(plot_freq)
        self.periodic_eval_cb = periodic_eval_cb
        self.eval_color = eval_color
        self.eval_conservative_color = eval_conservative_color
        self.solved_reference_y = solved_reference_y
        self.save_path = save_path
        self.display_fn = display_fn
        self.display_id = display_id or self.DISPLAY_ID_DEFAULT
        self.episode_rewards: list[float] = []
        self.episode_timesteps: list[int] = []
        self._last_plot_at = 0
        self._plot_shown = False

    def _on_training_start(self) -> None:
        self._last_plot_at = int(self.num_timesteps)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                self.episode_rewards.append(float(ep["r"]))
                self.episode_timesteps.append(int(self.num_timesteps))

        if len(self.episode_rewards) >= self.window:
            if self.num_timesteps - self._last_plot_at >= self.plot_freq:
                self._last_plot_at = int(self.num_timesteps)
                self._update_plot()
        return True

    def _update_plot(self) -> None:
        rews = np.array(self.episode_rewards, dtype=np.float64)
        ts = np.array(self.episode_timesteps, dtype=np.int64)

        mean = np.convolve(rews, np.ones(self.window) / self.window, mode="valid")
        std = np.array(
            [
                float(rews[max(0, i - self.window) : i].std())
                for i in range(self.window, len(rews) + 1)
            ]
        )
        score = mean - std
        ts_valid = ts[self.window - 1 :]

        fig, ax1 = plt.subplots(figsize=(12, 5))

        ax1.plot(
            ts_valid, mean, color="#2196F3", linewidth=2, label="Mean reward"
        )
        ax1.plot(
            ts_valid,
            score,
            color="#FF9800",
            linewidth=2,
            label="Score (mean − std)",
        )

        pecb = self.periodic_eval_cb
        hist = getattr(pecb, "eval_history", None) if pecb is not None else None
        if hist:
            ex: list[int] = []
            em: list[float] = []
            es: list[float] = []
            for h in hist:
                m = float(h["mean_reward"])
                if np.isfinite(m):
                    ex.append(int(h["timesteps"]))
                    em.append(m)
                    s = float(h["std_reward"])
                    es.append(s if np.isfinite(s) else 0.0)
            if ex:
                em_a = np.array(em, dtype=np.float64)
                es_a = np.array(es, dtype=np.float64)
                ax1.plot(
                    ex,
                    em_a,
                    color=self.eval_color,
                    linewidth=2,
                    marker="o",
                    markersize=4,
                    label="Periodic eval (mean)",
                )
                ax1.plot(
                    ex,
                    em_a - es_a,
                    color=self.eval_conservative_color,
                    linewidth=2,
                    marker="s",
                    markersize=4,
                    label="Periodic eval (mean − std)",
                )

        if self.solved_reference_y is not None:
            ax1.axhline(
                float(self.solved_reference_y),
                color="green",
                linestyle="--",
                alpha=0.7,
                label=f"Solved ({self.solved_reference_y:g})",
            )

        ax1.set_xlabel("Timesteps")
        ax1.set_ylabel("Reward")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(
            ts_valid,
            std,
            color="#E53935",
            linewidth=1.5,
            alpha=0.7,
            linestyle=":",
            label="Std",
        )
        ax2.set_ylabel("Std", color="#E53935")
        ax2.tick_params(axis="y", labelcolor="#E53935")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

        latest_mean = float(mean[-1])
        latest_std = float(std[-1])
        latest_score = float(score[-1])
        ax1.set_title(
            f"Training Progress ({len(self.episode_rewards)} episodes) | "
            f"Mean: {latest_mean:.1f}  Std: {latest_std:.1f}  Score: {latest_score:.1f}"
        )
        plt.tight_layout()

        if self.display_fn is not None:
            fn_kw: dict = {"display_id": self.display_id, "update": self._plot_shown}
            if self.save_path:
                fn_kw["filename"] = os.path.basename(self.save_path)
            self.display_fn(fig, **fn_kw)
            self._plot_shown = True
        elif self.save_path:
            d = os.path.dirname(self.save_path)
            if d:
                os.makedirs(d, exist_ok=True)
            fig.savefig(self.save_path, format="png", bbox_inches="tight", dpi=100)
            try:
                sz = os.path.getsize(self.save_path)
            except OSError:
                sz = -1
            print(f"[nb plot] {self.save_path} ({sz} bytes)")

        plt.close(fig)


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 384):
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = int(self.cnn(torch.zeros(1, 1, 84, 84)).shape[1])
        self.cnn_lin = nn.Sequential(
            nn.Linear(n_flatten, 256),
            nn.ReLU(),
        )
        self.state_mlp = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

    def forward(self, observations):
        x = self.cnn(observations["pixels"])
        x = self.cnn_lin(x)
        s = self.state_mlp(observations["state"])
        return torch.cat([x, s], dim=1)


policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor,
    normalize_images=True,
    activation_fn=torch.nn.ReLU,
    net_arch=dict(pi=[512, 256], vf=[512, 256]),
)
