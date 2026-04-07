"""Shared Lunar Lander dict-observation + PPO setup for the notebook and Optuna script."""

from __future__ import annotations

import os

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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

DEFAULT_ENV_ID = "LunarLander-v3"


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


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
