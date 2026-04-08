#!/usr/bin/env python3
"""Run Optuna hyperparameter search; writes best_hyperparams.json (or --output). Run separately from the notebook."""

from __future__ import annotations

import argparse
import json
import os
import tempfile

import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from lunar_rl_common import (
    make_eval_vec_env_with_stats,
    make_ppo_clip_range_schedule,
    make_ppo_lr_schedule,
    make_train_vec_env,
    policy_kwargs,
    resolve_train_device,
    suggested_parallel_envs,
)


def main() -> None:
    p = argparse.ArgumentParser(description="PPO + MultiInput Optuna tuning for Lunar Lander")
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--timesteps-per-trial", type=int, default=200_000)
    p.add_argument("--n-eval-episodes", type=int, default=10)
    p.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="SubprocVecEnv size; default: auto from CPU (see lunar_rl_common.suggested_parallel_envs)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="best_hyperparams.json")
    p.add_argument("--env-id", type=str, default="LunarLander-v3")
    p.add_argument("--study-name", type=str, default="ppo-lunarlander")
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="PPO device: auto=cuda if available; match notebook train_device",
    )
    args = p.parse_args()
    if args.n_envs is None:
        args.n_envs = suggested_parallel_envs()

    device = resolve_train_device(args.device)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        lr_start = trial.suggest_float("learning_rate", 3e-5, 3e-4, log=True)
        n_steps = trial.suggest_categorical("n_steps", [512, 1024])
        batch_size = trial.suggest_categorical("batch_size", [128, 256])
        n_epochs = trial.suggest_int("n_epochs", 3, 6)
        trial_gamma = trial.suggest_float("gamma", 0.98, 0.999)
        gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
        ent_coef = trial.suggest_float("ent_coef", 1e-3, 0.03, log=True)
        target_kl = trial.suggest_float("target_kl", 0.005, 0.03, log=True)

        if batch_size > n_steps * args.n_envs:
            raise optuna.TrialPruned()

        trial_env = make_train_vec_env(
            args.n_envs, args.seed, trial_gamma, env_id=args.env_id
        )

        model = PPO(
            "MultiInputPolicy",
            trial_env,
            seed=args.seed,
            device=device,
            verbose=0,
            policy_kwargs=policy_kwargs,
            learning_rate=make_ppo_lr_schedule(lr_start, lr_end_factor=0.05, lr_floor=1e-5),
            clip_range=make_ppo_clip_range_schedule(),
            target_kl=target_kl,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=trial_gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
        )

        model.learn(total_timesteps=args.timesteps_per_trial)

        trial_vec_path = os.path.join(
            tempfile.gettempdir(), f"optuna_vecnormalize_trial_{trial.number}.pkl"
        )
        trial_env.save(trial_vec_path)
        trial_env.close()

        eval_venv = make_eval_vec_env_with_stats(
            trial_vec_path, args.seed, env_id=args.env_id
        )
        mean_reward, std_reward = evaluate_policy(
            model,
            eval_venv,
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
        )
        eval_venv.close()
        try:
            os.remove(trial_vec_path)
        except OSError:
            pass
        del model

        score = mean_reward - std_reward
        trial.set_user_attr("mean_reward", mean_reward)
        trial.set_user_attr("std_reward", std_reward)
        return score

    print(
        f"Starting Optuna study: {args.n_trials} trials, "
        f"{args.timesteps_per_trial:,} timesteps each, {args.n_envs} envs"
    )
    study = optuna.create_study(direction="maximize", study_name=args.study_name)
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    best_data = {
        "params": study.best_trial.params,
        "score": study.best_trial.value,
        "mean_reward": study.best_trial.user_attrs["mean_reward"],
        "std_reward": study.best_trial.user_attrs["std_reward"],
        "trial_number": study.best_trial.number,
        "run": {
            "seed": args.seed,
            "n_envs": args.n_envs,
            "timesteps_per_trial": args.timesteps_per_trial,
            "n_eval_episodes": args.n_eval_episodes,
            "env_id": args.env_id,
        },
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(best_data, f, indent=2)

    print(f"\nBest trial #{study.best_trial.number}:")
    print(f"  Score (mean - std): {study.best_trial.value:.2f}")
    print(f"  Mean reward:        {study.best_trial.user_attrs['mean_reward']:.2f}")
    print(f"  Std reward:         {study.best_trial.user_attrs['std_reward']:.2f}")
    print("  Params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")
    print(f"\nSaved to {args.output!r}")


if __name__ == "__main__":
    main()
