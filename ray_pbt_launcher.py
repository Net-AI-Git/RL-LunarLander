#!/usr/bin/env python3
"""
Population Based Training launcher for Lunar Lander PPO.

Run from repo root (with Ray installed: pip install 'ray[tune]'):
  python ray_pbt_launcher.py

Rollout geometry (n_envs, n_steps, batch_size, n_epochs), gamma, gae_lambda,
policy_kwargs, VecNormalize settings, and train_device are fixed in ``ray_pbt_config.json``.
PBT only mutates the hyperparameters listed in ``HYPERPARAM_MUTATIONS``.
"""

from __future__ import annotations

import os

from ray import tune
from ray.tune import RunConfig, Tuner, TuneConfig
from ray.tune.schedulers import PopulationBasedTraining

from ray_pbt_train import get_default_pbt_config, train_lunarlander_pbt

# SubprocVecEnv uses one CPU per env by default; reserve headroom for the learner.
_DEFAULT_CPUS = int(os.environ.get("RAY_PBT_CPUS", "18"))


def _hyperparam_mutations() -> dict:
    """
    Only these keys are perturbed by PBT. Do not add n_envs, n_steps, batch_size, n_epochs,
    gamma, gae_lambda, policy/architecture, VecNormalize, or device — those stay in static JSON.
    """
    return {
        "learning_rate": tune.loguniform(1e-4, 3e-4),
        "lr_end": [5e-5, 7.5e-5, 1e-4],
        "ent_coef": tune.loguniform(0.005, 0.02),
        "ent_coef_end": [0.001, 0.002, 0.003],
        "schedule_flat_until": [0.20, 0.25, 0.33],
        "vf_coef": [0.4, 0.5, 0.6, 0.7],
        "target_kl": [0.02, 0.03, 0.04, 0.05],
        "clip_range": [0.15, 0.2, 0.25],
        "max_grad_norm": [0.3, 0.5, 0.7],
    }


def _initial_param_space(static: dict) -> dict:
    """
    Explicit baseline from ``static['base']`` for every mutated key (single choice each).
    PBT applies ``hyperparam_mutations`` later. ``seed`` is free so 8 trials are not identical.
    """
    b = static["base"]
    return {
        "learning_rate": tune.choice([float(b["learning_rate"])]),
        "lr_end": tune.choice([float(b["lr_end"])]),
        "ent_coef": tune.choice([float(b["ent_coef"])]),
        "ent_coef_end": tune.choice([float(b["ent_coef_end"])]),
        "schedule_flat_until": tune.choice([float(b["schedule_flat_until"])]),
        "vf_coef": tune.choice([float(b["vf_coef"])]),
        "target_kl": tune.choice([float(b["target_kl"])]),
        "clip_range": tune.choice([float(b["clip_range"])]),
        "max_grad_norm": tune.choice([float(b["max_grad_norm"])]),
        "seed": tune.randint(0, 2**31 - 1),
    }


def main() -> None:
    static = get_default_pbt_config()
    pbt_cfg = static["pbt"]

    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        metric=pbt_cfg["metric"],
        mode=pbt_cfg["mode"],
        perturbation_interval=int(pbt_cfg["perturbation_interval"]),
        burn_in_period=int(pbt_cfg["burn_in_period"]),
        hyperparam_mutations=_hyperparam_mutations(),
        quantile_fraction=float(pbt_cfg["quantile_fraction"]),
        resample_probability=float(pbt_cfg["resample_probability"]),
        log_config=True,
    )

    param_space = _initial_param_space(static)
    num_samples = int(pbt_cfg["num_samples"])

    trainable = tune.with_resources(
        train_lunarlander_pbt,
        resources={"cpu": _DEFAULT_CPUS},
    )

    tuner = Tuner(
        trainable,
        param_space=param_space,
        tune_config=TuneConfig(
            mode=pbt_cfg["mode"],
            metric=pbt_cfg["metric"],
            scheduler=pbt,
            num_samples=num_samples,
        ),
        run_config=RunConfig(name="lunarlander_pbt"),
    )

    tuner.fit()


if __name__ == "__main__":
    main()
