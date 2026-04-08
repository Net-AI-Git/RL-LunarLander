#!/usr/bin/env python3
"""Short end-to-end smoke test: PPO + schedules + VecNormalize save + periodic eval row + evaluate_policy."""

from __future__ import annotations

import os
import shutil
import tempfile

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from lunar_rl_common import (
    PeriodicEvalCallback,
    VecNormalizeSaveCallback,
    make_eval_vec_env_with_stats,
    make_ppo_clip_range_schedule,
    make_ppo_lr_schedule,
    make_train_vec_env,
    policy_kwargs,
)


def main() -> int:
    # MlpPolicy short smoke: CPU avoids SB3 GPU warning for non-CNN policies.
    device = "cpu"
    seed = 42
    env_id = "LunarLander-v3"
    n_envs = 2
    gamma = 0.99
    td = tempfile.mkdtemp(prefix="lunar_smoke_")
    vec_path = os.path.join(td, "vecnormalize.pkl")
    csv_path = os.path.join(td, "periodic_eval.csv")
    try:
        train_env = make_train_vec_env(n_envs, seed, gamma, env_id=env_id)
        model = PPO(
            "MlpPolicy",
            train_env,
            seed=seed,
            device=device,
            verbose=0,
            policy_kwargs=policy_kwargs,
            learning_rate=make_ppo_lr_schedule(3e-4, lr_end_factor=0.05, lr_floor=1e-5),
            clip_range=make_ppo_clip_range_schedule(),
            target_kl=0.015,
            n_steps=128,
            batch_size=64,
            n_epochs=4,
            gamma=gamma,
            gae_lambda=0.95,
            ent_coef=0.0,
        )
        save_cb = VecNormalizeSaveCallback(save_freq=500, save_path=vec_path)
        # Trigger at least one eval during short run (eval_freq small)
        eval_cb = PeriodicEvalCallback(
            eval_freq=2048,
            n_eval_episodes=2,
            seed=seed,
            csv_path=csv_path,
            env_id=env_id,
        )
        model.learn(total_timesteps=6144, callback=[save_cb, eval_cb])
        train_env.save(vec_path)
        train_env.close()

        assert os.path.isfile(vec_path), "vecnormalize.pkl missing"
        assert os.path.isfile(csv_path), "periodic_eval.csv missing"
        assert eval_cb.eval_history, "periodic eval should have run at least once"

        eval_env = make_eval_vec_env_with_stats(vec_path, seed, env_id)
        mean_r, std_r = evaluate_policy(
            model, eval_env, n_eval_episodes=2, deterministic=True
        )
        eval_env.close()
        print(f"smoke OK: evaluate_policy mean={mean_r:.2f} std={std_r:.2f}")
        del model
        return 0
    finally:
        shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
