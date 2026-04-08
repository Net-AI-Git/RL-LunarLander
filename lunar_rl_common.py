"""Shared Lunar Lander dict-observation + PPO setup for the notebook and Optuna script."""

from __future__ import annotations

import copy
import csv
import os
import warnings

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from gymnasium.wrappers.transform_observation import AddRenderObservation

from stable_baselines3.common.callbacks import BaseCallback
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
    Run eval for a fixed step budget on a synced eval env (eval-only VecNormalize, no reward norm).
    Appends rows to CSV and stores records in eval_history for plotting.
    """

    def __init__(
        self,
        eval_freq: int,
        n_eval_steps: int,
        seed: int,
        csv_path: str = "logs/periodic_eval.csv",
        env_id: str | None = None,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_steps = n_eval_steps
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
                        "n_eval_steps",
                    ]
                )
            w.writerow(
                [
                    timesteps,
                    mean_reward,
                    std_reward,
                    n_episodes,
                    self.n_eval_steps,
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
            obs = eval_env.reset()
            ep_returns: list[float] = []
            ep_return = 0.0
            steps = 0
            while steps < self.n_eval_steps:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, rewards, dones, _ = eval_env.step(action)
                ep_return += float(rewards[0])
                steps += 1
                if dones[0]:
                    ep_returns.append(ep_return)
                    ep_return = 0.0
            if ep_return != 0.0 and not dones[0]:
                # unfinished episode at horizon — optional: could append; plan focuses on completed
                pass

            n_ep = len(ep_returns)
            if n_ep >= 1:
                mean_r = float(np.mean(ep_returns))
                std_r = float(np.std(ep_returns)) if n_ep > 1 else 0.0
            else:
                mean_r = float("nan")
                std_r = float("nan")
                warnings.warn(
                    "PeriodicEvalCallback: no completed episode in eval window; "
                    "increase n_eval_steps or check env."
                )

            self._write_row(self.num_timesteps, mean_r, std_r, n_ep)
        finally:
            eval_env.close()

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_at >= self.eval_freq:
            self._last_eval_at = self.num_timesteps
            self._run_eval()
        return True


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
