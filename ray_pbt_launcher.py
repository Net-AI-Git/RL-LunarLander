#!/usr/bin/env python3
"""
Population Based Training launcher for Lunar Lander PPO.

Run from repo root (with Ray installed: pip install 'ray[tune]'):
  python ray_pbt_launcher.py

Environment (optional):

- ``RAY_RESULTS_DIR`` — root for Tune storage (default: ``./ray_results`` under cwd).
- ``RAY_PBT_EXPERIMENT_NAME`` — run name (default: ``lunarlander_pbt``).
- ``RAY_PBT_CHECKPOINTS_TO_KEEP`` — max Tune checkpoints to retain per trial by score (default: ``2``;
  raise if you have plenty of disk).
- ``RAY_PBT_CPUS`` — logical CPUs reserved **per trial** (default: ``base.n_envs``).
- ``RAY_PBT_GPU_FRACTION`` — when ``base.train_device`` is ``cuda``, GPU per trial (default ``1.0``).
  On a single GPU (e.g. RTX 3070 8GB), keep ``1.0`` so trials do not share VRAM.

- **Continuation from a saved PBT export** (e.g. ``best_pbt_checkpoints_top5/rank01_...``): set
  ``pbt.seed_checkpoint_dir`` in ``ray_pbt_config.json`` to a directory containing
  ``model.zip``, ``vecnormalize.pkl``, and ``trainer_state.json``. Hyperparameters still come from
  the JSON ``base`` + ``initial_param_space`` (same starting values as a fresh run); only the policy
  weights and VecNormalize stats are loaded from disk. Override or disable with
  ``RAY_PBT_SEED_CHECKPOINT`` (absolute or repo-relative path; empty string disables). To train past
  the checkpoint's timestep count, raise ``base.total_timesteps`` (e.g. add more steps for a second
  training phase).

If Ray fails to start with ``Timed out waiting for ... gcs_server_port`` (GCS): stop stale Ray
(``ray stop --force``), remove old sessions under ``/tmp/ray/session_*`` when nothing is running,
ensure ``/tmp`` is writable and has space; optionally ``export RAY_TMPDIR=/path/to/fast/local/dir``.
On instances with **small root disks**, point ``RAY_RESULTS_DIR`` at a larger mount so ``ray_results/``
does not fill the volume.

Report cadence (see ``ray_pbt_config.json`` ``pbt`` + ``base``):

- Each ``tune.report`` is one **training_iteration** after ``report_interval_timesteps`` env steps
  (see JSON; wider intervals mean fewer checkpoints on disk) and ``periodic_eval_episodes`` eval passes.
- ``perturbation_interval=2`` means PBT considers exploit/explore every **2** reports (400K steps
  between perturbation checks).
- ``burn_in_period=4`` is **4** reports before PBT mutates (~800K steps), reducing checkpoint churn.

SB3 checkpoints are saved only inside ``train_lunarlander_pbt`` via
``tune.report(..., checkpoint=...)`` — not via ``CheckpointConfig.checkpoint_frequency``.

Rollout geometry (n_envs, n_steps, batch_size, n_epochs), gamma, gae_lambda,
policy_kwargs, VecNormalize settings, and train_device are fixed in ``ray_pbt_config.json``.
PBT only mutates the hyperparameters listed in ``_hyperparam_mutations``.
"""

from __future__ import annotations

import os
import warnings

import ray
from ray import tune
from ray.tune import CheckpointConfig, RunConfig, Tuner, TuneConfig
from ray.tune.schedulers import PopulationBasedTraining

from ray_pbt_train import get_default_pbt_config, train_lunarlander_pbt
from ray_tune_visualization import (
    TuneVisualizationCallback,
    print_and_save_run_summary,
    refresh_tune_visualizations,
)

_RESULTS_ROOT = os.environ.get("RAY_RESULTS_DIR")
_DEFAULT_STORAGE = os.path.join(os.getcwd(), "ray_results")


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


def build_run_config(
    *,
    experiment_name: str | None = None,
    callbacks: list | None = None,
    metric: str = "eval_score",
    mode: str = "max",
    top_k: int = 8,
) -> RunConfig:
    """
    Tune/AIR storage + checkpoint retention. Function-API trials still rely on manual
    ``tune.report(..., checkpoint=...)``; ``checkpoint_frequency`` stays 0.

    When ``callbacks`` is ``None`` (default), attaches ``TuneVisualizationCallback`` so
    ``visualizations/`` updates after each trial and at experiment end. Pass ``[]`` to disable.
    """
    name = experiment_name or os.environ.get("RAY_PBT_EXPERIMENT_NAME", "lunarlander_pbt")
    storage = _RESULTS_ROOT if _RESULTS_ROOT else _DEFAULT_STORAGE
    keep = int(os.environ.get("RAY_PBT_CHECKPOINTS_TO_KEEP", "2"))
    cb = callbacks
    if cb is None:
        cb = [
            TuneVisualizationCallback(metric=metric, mode=mode, top_k=top_k),
        ]
    return RunConfig(
        name=name,
        storage_path=storage,
        checkpoint_config=CheckpointConfig(
            num_to_keep=keep,
            checkpoint_score_attribute="eval_score",
            checkpoint_score_order="max",
            checkpoint_frequency=0,
        ),
        callbacks=cb,
    )


def initial_param_space(static: dict) -> dict:
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


def trainable_resources(static: dict) -> dict[str, float]:
    """
    CPUs follow ``base.n_envs`` (one SubprocVecEnv worker per env unless overridden).

    When ``base.train_device`` is ``cuda``, request ``gpu`` so Tune does not place multiple trials
    on the same GPU by default (important for 8GB cards).
    """
    n_envs = int(static["base"]["n_envs"])
    cpus = int(os.environ.get("RAY_PBT_CPUS", str(n_envs)))
    out: dict[str, float] = {"cpu": float(cpus)}
    dev = str(static["base"].get("train_device", "cpu")).strip().lower()
    if dev == "cuda":
        out["gpu"] = float(os.environ.get("RAY_PBT_GPU_FRACTION", "1.0"))
    return out


def main() -> None:
    # Initialize before Tune so ``tuner.fit`` skips auto-init; dashboard off reduces startup work.
    if not ray.is_initialized():
        ray.init(include_dashboard=False)

    static = get_default_pbt_config()
    pbt_cfg = static["pbt"]

    # metric/mode only on TuneConfig — PBT also accepts them on the scheduler, but then
    # tune.run would receive duplicate metric/mode and raise ValueError (Ray 2.5+).
    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=int(pbt_cfg["perturbation_interval"]),
        burn_in_period=int(pbt_cfg["burn_in_period"]),
        hyperparam_mutations=_hyperparam_mutations(),
        quantile_fraction=float(pbt_cfg["quantile_fraction"]),
        resample_probability=float(pbt_cfg["resample_probability"]),
        log_config=True,
    )

    param_space = initial_param_space(static)
    num_samples = int(pbt_cfg["num_samples"])

    trainable = tune.with_resources(
        train_lunarlander_pbt,
        resources=trainable_resources(static),
    )

    tuner = Tuner(
        trainable,
        param_space=param_space,
        tune_config=TuneConfig(
            metric=pbt_cfg["metric"],
            mode=pbt_cfg["mode"],
            scheduler=pbt,
            num_samples=num_samples,
        ),
        run_config=build_run_config(
            metric=pbt_cfg["metric"],
            mode=pbt_cfg["mode"],
        ),
    )

    result = tuner.fit()
    try:
        refresh_tune_visualizations(
            result.experiment_path,
            metric=pbt_cfg["metric"],
            mode=pbt_cfg["mode"],
        )
        print_and_save_run_summary(
            result.experiment_path,
            metric=pbt_cfg["metric"],
            mode=pbt_cfg["mode"],
        )
    except Exception as e:
        warnings.warn(f"Post-fit visualization refresh failed: {e}", UserWarning)


if __name__ == "__main__":
    main()
