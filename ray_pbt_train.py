#!/usr/bin/env python3
"""
Ray PBT entrypoint helpers: build PPO + VecNormalize + callbacks, eval (mean/std/score),
unified checkpoints (model.zip, vecnormalize.pkl, trainer_state.json), and Tune Function API
trainable ``train_lunarlander_pbt`` (chunked learn + ``train.report`` checkpoints).

Orchestration with ray.tune stays out of the notebook; import and call from a driver script.
"""

from __future__ import annotations

import copy
import json
import os
import warnings
import shutil
import tempfile
import types
from collections.abc import Mapping, MutableMapping
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

from lunar_rl_common import (
    DEFAULT_ENV_ID,
    EntropyCoefScheduleCallback,
    PeriodicEvalCallback,
    VecNormalizeSaveCallback,
    make_ent_coef_schedule_late_linear,
    make_ppo_lr_schedule_late_linear,
    make_subproc_venv,
    make_train_vec_env,
    make_eval_vec_env_synced,
    policy_kwargs as default_policy_kwargs,
    resolve_train_device,
)

CHECKPOINT_MODEL = "model.zip"
CHECKPOINT_VECNORMALIZE = "vecnormalize.pkl"
CHECKPOINT_TRAINER_STATE = "trainer_state.json"
PERIODIC_EVAL_CSV = "periodic_eval.csv"

_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "ray_pbt_config.json")

# Logged each tune.report so progress.csv contains HP evolution (config is stripped by Ray CSV).
_REPORT_METRIC_HP_KEYS = (
    "learning_rate",
    "lr_end",
    "ent_coef",
    "ent_coef_end",
    "schedule_flat_until",
    "vf_coef",
    "target_kl",
    "clip_range",
    "max_grad_norm",
)


def hp_metrics_from_merged(merged: Mapping[str, Any]) -> dict[str, float]:
    """Flatten current trial hyperparameters from ``merged[\"base\"]`` for Tune metrics."""
    b = merged["base"]
    out: dict[str, float] = {}
    for k in _REPORT_METRIC_HP_KEYS:
        if k not in b:
            continue
        v = b[k]
        if v is None:
            continue
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _activation_from_name(name: str) -> type[torch.nn.Module]:
    n = (name or "tanh").strip().lower()
    if n == "tanh":
        return torch.nn.Tanh
    if n == "relu":
        return torch.nn.ReLU
    raise ValueError(f"Unsupported activation_fn: {name!r}")


def policy_kwargs_from_fixed(fixed: Mapping[str, Any] | None) -> dict[str, Any]:
    """V2-style policy kwargs from ``fixed_policy_kwargs`` (e.g. ray_pbt_config.json)."""
    if not fixed:
        return dict(default_policy_kwargs)
    pi = fixed["pi"]
    vf = fixed["vf"]
    act = _activation_from_name(str(fixed.get("activation_fn", "tanh")))
    ortho = bool(fixed.get("ortho_init", True))
    return dict(
        net_arch=dict(pi=list(pi), vf=list(vf)),
        activation_fn=act,
        ortho_init=ortho,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={},
    )


def _to_json_serializable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(x) for x in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


def _report_interval_timesteps(config: Mapping[str, Any]) -> int:
    pbt = config.get("pbt") or {}
    return int(pbt.get("report_interval_timesteps", 200_000))


def patch_ppo_global_progress(model: PPO, global_total_timesteps: int) -> None:
    """
    SB3's LR schedule uses ``progress_remaining`` from each ``learn()`` call's local target.
    For chunked training against a **global** budget, use
    ``1 - num_timesteps / global_total_timesteps``.
    """
    g = max(1, int(global_total_timesteps))

    def _upd(self: PPO, num_timesteps: int, total_timesteps: int) -> None:
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(g)

    model._global_total_timesteps = g  # type: ignore[attr-defined]
    model._update_current_progress_remaining = types.MethodType(_upd, model)  # type: ignore[method-assign]


class EntropyCoefGlobalScheduleCallback(EntropyCoefScheduleCallback):
    """
    Late-linear entropy schedule with progress = ``num_timesteps / global_total`` (chunked
    ``learn`` + resume). ``_on_training_start`` uses the current timestep so resume is consistent.
    """

    def __init__(
        self,
        schedule_fn: Any,
        global_total_timesteps: int,
        verbose: int = 0,
    ):
        super().__init__(schedule_fn, verbose)
        self.global_total_timesteps = max(1, int(global_total_timesteps))

    def _progress_remaining(self) -> float:
        p = 1.0 - float(self.model.num_timesteps) / float(self.global_total_timesteps)
        return max(0.0, min(1.0, p))

    def _on_training_start(self) -> None:
        pr = self._progress_remaining()
        self.model.ent_coef = float(self.schedule_fn(pr))

    def _on_rollout_end(self) -> None:
        pr = self._progress_remaining()
        self.model.ent_coef = float(self.schedule_fn(pr))


def apply_schedules_from_base(
    model: PPO,
    ent_cb: EntropyCoefGlobalScheduleCallback,
    base: Mapping[str, Any],
    *,
    global_total_timesteps: int,
) -> None:
    """
    Rebuild LR and entropy schedules from the current trial ``base`` (after PBT mutation or
    resume). Updates ``model.learning_rate`` and ``ent_cb.schedule_fn``.
    """
    flat_until = float(base["schedule_flat_until"])
    lr_sched = make_ppo_lr_schedule_late_linear(
        lr_start=float(base["learning_rate"]),
        lr_end=float(base["lr_end"]),
        flat_until_progress=flat_until,
    )
    ent_sched = make_ent_coef_schedule_late_linear(
        ent_start=float(base["ent_coef"]),
        ent_end=float(base["ent_coef_end"]),
        flat_until_progress=flat_until,
    )
    model.learning_rate = lr_sched
    ent_cb.schedule_fn = ent_sched
    ent_cb.global_total_timesteps = max(1, int(global_total_timesteps))
    patch_ppo_global_progress(model, global_total_timesteps)


def merge_pbt_config(
    static_cfg: MutableMapping[str, Any],
    tune_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Deep-copy ``static_cfg`` and overlay Tune/PBT hyperparameters onto ``base``.

    ``tune_cfg`` may be flat (e.g. ``learning_rate``, ``lr_end``) or include a nested ``base`` dict.
    """
    out: dict[str, Any] = copy.deepcopy(dict(static_cfg))
    base = out.setdefault("base", {})

    known_base = {
        "train_device",
        "n_envs",
        "total_timesteps",
        "n_steps",
        "batch_size",
        "n_epochs",
        "gamma",
        "gae_lambda",
        "clip_range",
        "clip_range_vf",
        "normalize_advantage",
        "vf_coef",
        "max_grad_norm",
        "target_kl",
        "use_sde",
        "learning_rate",
        "lr_end",
        "ent_coef",
        "ent_coef_end",
        "schedule_flat_until",
        "periodic_eval_episodes",
        "vecnormalize_clip_obs",
        "vecnormalize_clip_reward",
    }

    nested = tune_cfg.get("base")
    if isinstance(nested, Mapping):
        for k, v in nested.items():
            if k in known_base:
                base[k] = v

    for k, v in tune_cfg.items():
        if k in ("base", "pbt", "fixed_policy_kwargs"):
            continue
        if k in known_base:
            base[k] = v

    return out


def build_model_and_env(
    config: Mapping[str, Any],
    *,
    seed: int,
    checkpoint_dir: str,
    env_id: str | None = None,
) -> tuple[PPO, VecNormalize, list[BaseCallback]]:
    """
    Load baseline from ``config["base"]``, build train env, PPO(MlpPolicy), and core callbacks.

    ``checkpoint_dir`` is used for ``vecnormalize.pkl``, ``periodic_eval.csv``, and must exist
    or be creatable by the caller before ``learn()`` if needed.
    """
    b = config["base"]
    env_id = env_id or DEFAULT_ENV_ID
    device = resolve_train_device(str(b.get("train_device", "cpu")))
    policy_kwargs = policy_kwargs_from_fixed(config.get("fixed_policy_kwargs"))

    train_env = make_train_vec_env(
        int(b["n_envs"]),
        seed,
        float(b["gamma"]),
        env_id=env_id,
        clip_obs=float(b["vecnormalize_clip_obs"]),
        clip_reward=float(b["vecnormalize_clip_reward"]),
    )

    flat_until = float(b["schedule_flat_until"])
    lr_sched = make_ppo_lr_schedule_late_linear(
        lr_start=float(b["learning_rate"]),
        lr_end=float(b["lr_end"]),
        flat_until_progress=flat_until,
    )
    ent_sched = make_ent_coef_schedule_late_linear(
        ent_start=float(b["ent_coef"]),
        ent_end=float(b["ent_coef_end"]),
        flat_until_progress=flat_until,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        seed=seed,
        device=device,
        verbose=0,
        policy_kwargs=policy_kwargs,
        learning_rate=lr_sched,
        clip_range=float(b["clip_range"]),
        clip_range_vf=b.get("clip_range_vf"),
        normalize_advantage=bool(b["normalize_advantage"]),
        vf_coef=float(b["vf_coef"]),
        max_grad_norm=float(b["max_grad_norm"]),
        use_sde=bool(b["use_sde"]),
        target_kl=float(b["target_kl"]),
        n_steps=int(b["n_steps"]),
        batch_size=int(b["batch_size"]),
        n_epochs=int(b["n_epochs"]),
        gamma=float(b["gamma"]),
        gae_lambda=float(b["gae_lambda"]),
        ent_coef=float(b["ent_coef"]),
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    vec_path = os.path.join(checkpoint_dir, CHECKPOINT_VECNORMALIZE)
    csv_path = os.path.join(checkpoint_dir, PERIODIC_EVAL_CSV)
    interval = _report_interval_timesteps(config)

    callbacks: list[BaseCallback] = [
        EntropyCoefScheduleCallback(ent_sched),
        VecNormalizeSaveCallback(save_freq=interval, save_path=vec_path),
        PeriodicEvalCallback(
            eval_freq=interval,
            n_eval_episodes=int(b["periodic_eval_episodes"]),
            seed=seed,
            csv_path=csv_path,
            env_id=env_id,
        ),
    ]
    return model, train_env, callbacks


def build_model_and_env_chunked(
    config: Mapping[str, Any],
    *,
    seed: int,
    env_id: str | None = None,
) -> tuple[PPO, VecNormalize, EntropyCoefGlobalScheduleCallback]:
    """
    PPO + VecNormalize for chunked Tune training: one global-progress entropy callback; no
    periodic eval (eval runs explicitly each chunk in ``train_lunarlander_pbt``).
    """
    b = config["base"]
    env_id = env_id or DEFAULT_ENV_ID
    device = resolve_train_device(str(b.get("train_device", "cpu")))
    policy_kwargs = policy_kwargs_from_fixed(config.get("fixed_policy_kwargs"))
    total_ts = int(b["total_timesteps"])

    train_env = make_train_vec_env(
        int(b["n_envs"]),
        seed,
        float(b["gamma"]),
        env_id=env_id,
        clip_obs=float(b["vecnormalize_clip_obs"]),
        clip_reward=float(b["vecnormalize_clip_reward"]),
    )

    flat_until = float(b["schedule_flat_until"])
    lr_sched = make_ppo_lr_schedule_late_linear(
        lr_start=float(b["learning_rate"]),
        lr_end=float(b["lr_end"]),
        flat_until_progress=flat_until,
    )
    ent_sched = make_ent_coef_schedule_late_linear(
        ent_start=float(b["ent_coef"]),
        ent_end=float(b["ent_coef_end"]),
        flat_until_progress=flat_until,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        seed=seed,
        device=device,
        verbose=0,
        policy_kwargs=policy_kwargs,
        learning_rate=lr_sched,
        clip_range=float(b["clip_range"]),
        clip_range_vf=b.get("clip_range_vf"),
        normalize_advantage=bool(b["normalize_advantage"]),
        vf_coef=float(b["vf_coef"]),
        max_grad_norm=float(b["max_grad_norm"]),
        use_sde=bool(b["use_sde"]),
        target_kl=float(b["target_kl"]),
        n_steps=int(b["n_steps"]),
        batch_size=int(b["batch_size"]),
        n_epochs=int(b["n_epochs"]),
        gamma=float(b["gamma"]),
        gae_lambda=float(b["gae_lambda"]),
        ent_coef=float(b["ent_coef"]),
    )

    ent_cb = EntropyCoefGlobalScheduleCallback(
        ent_sched,
        global_total_timesteps=total_ts,
    )
    apply_schedules_from_base(model, ent_cb, b, global_total_timesteps=total_ts)
    return model, train_env, ent_cb


def evaluate_current_model(
    model: PPO,
    train_venv: VecNormalize,
    seed: int,
    n_eval_episodes: int,
    env_id: str | None = None,
) -> tuple[float, float, float]:
    """
    Synced eval env, deterministic policy; returns
    ``(eval_mean_reward, eval_std_reward, eval_score)`` with ``eval_score = mean - std``.
    """
    eval_env = make_eval_vec_env_synced(train_venv, seed, env_id=env_id)
    try:
        mean_r, std_r = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
        )
        mean_f = float(mean_r)
        std_f = float(std_r)
        return mean_f, std_f, mean_f - std_f
    finally:
        eval_env.close()


def save_trial_checkpoint(
    checkpoint_dir: str,
    model: PPO,
    train_venv: VecNormalize,
    *,
    seed: int,
    current_config: Mapping[str, Any],
    best_eval_score_so_far: float | None,
) -> None:
    """Write ``model.zip``, ``vecnormalize.pkl``, and ``trainer_state.json`` under ``checkpoint_dir``."""
    if not isinstance(train_venv, VecNormalize):
        raise TypeError("train_venv must be VecNormalize")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, CHECKPOINT_MODEL)
    vec_path = os.path.join(checkpoint_dir, CHECKPOINT_VECNORMALIZE)
    state_path = os.path.join(checkpoint_dir, CHECKPOINT_TRAINER_STATE)

    model.save(model_path)
    train_venv.save(vec_path)

    payload = {
        "timesteps_done": int(model.num_timesteps),
        "seed": int(seed),
        "current_config": _to_json_serializable(dict(current_config)),
        "best_eval_score_so_far": best_eval_score_so_far,
    }
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_trial_checkpoint(
    checkpoint_dir: str,
    config: Mapping[str, Any],
    *,
    env_id: str | None = None,
    override_seed: int | None = None,
) -> tuple[PPO, VecNormalize, dict[str, Any]]:
    """
    Load ``model.zip``, ``vecnormalize.pkl``, restore ``timesteps_done`` from ``trainer_state.json``.

    Rebuilds a training ``SubprocVecEnv`` + ``VecNormalize`` compatible with the saved stats,
    then ``PPO.load``. Returns ``(model, train_env, state_dict)``.

    ``override_seed``: if set, use for vectorized env creation instead of ``trainer_state.json``
    (e.g. Ray Tune ``seed`` per trial while sharing the same policy weights).
    """
    state_path = os.path.join(checkpoint_dir, CHECKPOINT_TRAINER_STATE)
    vec_path = os.path.join(checkpoint_dir, CHECKPOINT_VECNORMALIZE)
    model_path = os.path.join(checkpoint_dir, CHECKPOINT_MODEL)

    if not os.path.isfile(state_path):
        raise FileNotFoundError(state_path)

    with open(state_path, encoding="utf-8") as f:
        state: dict[str, Any] = json.load(f)

    b = config["base"]
    env_id = env_id or DEFAULT_ENV_ID
    seed = int(override_seed) if override_seed is not None else int(state["seed"])
    n_envs = int(b["n_envs"])

    raw = make_subproc_venv(n_envs, seed, env_id)
    train_env = VecNormalize.load(vec_path, raw)
    device = resolve_train_device(str(b.get("train_device", "cpu")))

    model = PPO.load(model_path, env=train_env, device=device)

    ts = int(state.get("timesteps_done", model.num_timesteps))
    model.num_timesteps = ts

    return model, train_env, state


def load_ray_pbt_config(path: str) -> dict[str, Any]:
    """Load a JSON config file (e.g. ``ray_pbt_config.json``)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_default_pbt_config() -> dict[str, Any]:
    """Load bundled ``ray_pbt_config.json`` next to this module."""
    return load_ray_pbt_config(_DEFAULT_CONFIG_PATH)


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(_DEFAULT_CONFIG_PATH))


def resolve_seed_checkpoint_dir(static_cfg: Mapping[str, Any]) -> str | None:
    """
    Optional directory with ``model.zip``, ``vecnormalize.pkl``, ``trainer_state.json`` to start
    every PBT trial from the same pretrained policy (weights + VecNormalize stats).

    Resolution order:

    1. ``RAY_PBT_SEED_CHECKPOINT`` — absolute path, or relative to repo root. Set to empty string
       to disable and train from scratch even if ``pbt.seed_checkpoint_dir`` is set in JSON.
    2. ``static_cfg["pbt"]["seed_checkpoint_dir"]`` — same path rules; omit or null to disable.
    """
    env_val = os.environ.get("RAY_PBT_SEED_CHECKPOINT")
    if env_val is not None:
        raw = env_val.strip()
        if raw == "":
            return None
        resolved = _resolve_checkpoint_dir_path(raw)
        if resolved is None:
            warnings.warn(
                f"RAY_PBT_SEED_CHECKPOINT is set but directory not found: {raw!r}. "
                f"Tried repo root {_repo_root()!r} and cwd {os.getcwd()!r}. Training from scratch.",
                UserWarning,
                stacklevel=2,
            )
        return resolved

    pbt = static_cfg.get("pbt") or {}
    raw = pbt.get("seed_checkpoint_dir")
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    resolved = _resolve_checkpoint_dir_path(s)
    if resolved is None:
        warnings.warn(
            f"pbt.seed_checkpoint_dir is set but not found: {s!r} "
            f"(tried repo root {_repo_root()!r} and cwd {os.getcwd()!r}). "
            "Training from scratch.",
            UserWarning,
            stacklevel=2,
        )
    return resolved


def _resolve_checkpoint_dir_path(raw: str) -> str | None:
    p = os.path.expanduser(raw)
    if os.path.isabs(p):
        return p if os.path.isdir(p) else None
    rel_repo = os.path.join(_repo_root(), p)
    if os.path.isdir(rel_repo):
        return rel_repo
    rel_cwd = os.path.join(os.getcwd(), p)
    if os.path.isdir(rel_cwd):
        return rel_cwd
    return None


def _env_id_arg(config: Mapping[str, Any]) -> str | None:
    e = config.get("env_id")
    return e if isinstance(e, str) else None


def train_lunarlander_pbt(config: dict[str, Any]) -> None:
    """
    Ray Tune **Function API** trainable: chunked PPO, eval, Tune checkpoints.

    Uses ``ray.tune.get_checkpoint()`` to restore and ``ray.tune.report(..., checkpoint=...)`` to save
    (not ``ray.train`` — deprecated inside Tune function trainables, Ray 2.5+).
    Schedules are rebuilt from ``merged[\"base\"]`` each chunk (and on resume after PBT).

    **Starting from an exported best checkpoint** (no Tune checkpoint yet): if
    ``resolve_seed_checkpoint_dir`` finds ``pbt.seed_checkpoint_dir`` or ``RAY_PBT_SEED_CHECKPOINT``,
    loads ``model.zip`` / ``vecnormalize.pkl`` from that folder while keeping hyperparameters from
    ``merge_pbt_config(static, config)`` (current ``ray_pbt_config.json`` + Tune), not from the file's
    embedded ``current_config``.

    Requires: ``pip install 'ray[tune]'``.
    """
    try:
        from ray import tune
        from ray.tune import Checkpoint
    except ImportError as e:
        raise ImportError(
            "train_lunarlander_pbt requires Ray Tune; install with: pip install 'ray[tune]'"
        ) from e

    static = get_default_pbt_config()
    checkpoint = tune.get_checkpoint()
    env_id = _env_id_arg(config)
    seed_ckpt_dir = resolve_seed_checkpoint_dir(static)

    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            state_path = os.path.join(checkpoint_dir, CHECKPOINT_TRAINER_STATE)
            with open(state_path, encoding="utf-8") as f:
                prev_state = json.load(f)
            merged = merge_pbt_config(prev_state["current_config"], config)
            model, train_env, _state = load_trial_checkpoint(
                checkpoint_dir, merged, env_id=env_id
            )
            timesteps_done = int(model.num_timesteps)
            seed = int(prev_state["seed"])
            best_raw = prev_state.get("best_eval_score_so_far")
            best_eval = float(best_raw) if best_raw is not None else None
            ent_cb = EntropyCoefGlobalScheduleCallback(
                make_ent_coef_schedule_late_linear(),
                global_total_timesteps=int(merged["base"]["total_timesteps"]),
            )
            apply_schedules_from_base(
                model,
                ent_cb,
                merged["base"],
                global_total_timesteps=int(merged["base"]["total_timesteps"]),
            )
    elif seed_ckpt_dir:
        # Fresh Tune trial: load policy + VecNormalize from disk, but hyperparameters and
        # schedules come from ``merge_pbt_config(static, config)`` (current JSON + Tune), not from
        # the checkpoint's saved ``current_config`` — so PBT starts from the same HP baseline.
        merged = merge_pbt_config(static, config)
        state_path = os.path.join(seed_ckpt_dir, CHECKPOINT_TRAINER_STATE)
        with open(state_path, encoding="utf-8") as f:
            prev_state = json.load(f)
        trial_seed = int(config.get("seed", prev_state["seed"]))
        model, train_env, _state = load_trial_checkpoint(
            seed_ckpt_dir,
            merged,
            env_id=env_id,
            override_seed=trial_seed,
        )
        timesteps_done = int(model.num_timesteps)
        seed = trial_seed
        best_raw = prev_state.get("best_eval_score_so_far")
        best_eval = float(best_raw) if best_raw is not None else None
        ent_cb = EntropyCoefGlobalScheduleCallback(
            make_ent_coef_schedule_late_linear(),
            global_total_timesteps=int(merged["base"]["total_timesteps"]),
        )
        apply_schedules_from_base(
            model,
            ent_cb,
            merged["base"],
            global_total_timesteps=int(merged["base"]["total_timesteps"]),
        )
    else:
        merged = merge_pbt_config(static, config)
        seed = int(config.get("seed", 42))
        model, train_env, ent_cb = build_model_and_env_chunked(
            merged, seed=seed, env_id=env_id
        )
        timesteps_done = int(model.num_timesteps)
        best_eval = None

    base = merged["base"]
    total_timesteps = int(base["total_timesteps"])
    report_interval = _report_interval_timesteps(merged)
    n_eval_episodes = int(base["periodic_eval_episodes"])

    reported = False

    while timesteps_done < total_timesteps:
        apply_schedules_from_base(
            model,
            ent_cb,
            base,
            global_total_timesteps=total_timesteps,
        )

        chunk = min(report_interval, total_timesteps - timesteps_done)
        if chunk <= 0:
            break

        reset_ts = timesteps_done == 0
        model.learn(
            total_timesteps=chunk,
            callback=ent_cb,
            reset_num_timesteps=reset_ts,
            progress_bar=False,
        )
        timesteps_done = int(model.num_timesteps)

        mean_r, std_r, eval_score = evaluate_current_model(
            model,
            train_env,
            seed,
            n_eval_episodes,
            env_id=env_id,
        )
        if best_eval is None:
            best_eval = eval_score
        else:
            best_eval = max(best_eval, eval_score)

        ckpt_dir = tempfile.mkdtemp(prefix="lunarlander_tune_ckpt_")
        try:
            save_trial_checkpoint(
                ckpt_dir,
                model,
                train_env,
                seed=seed,
                current_config=merged,
                best_eval_score_so_far=best_eval,
            )
            # Tune increments ``training_iteration`` each report (PBT time_attr).
            report_metrics: dict[str, Any] = {
                "eval_mean_reward": mean_r,
                "eval_std_reward": std_r,
                "eval_score": eval_score,
                "timesteps_done": timesteps_done,
            }
            report_metrics.update(hp_metrics_from_merged(merged))
            tune.report(
                report_metrics,
                checkpoint=Checkpoint.from_directory(ckpt_dir),
            )
            reported = True
        finally:
            shutil.rmtree(ckpt_dir, ignore_errors=True)

    # If ``total_timesteps`` is not above the loaded checkpoint's timesteps, the loop runs zero
    # times — Ray Tune still requires at least one ``tune.report`` containing ``eval_score``.
    if not reported:
        warnings.warn(
            f"No training steps ran: timesteps_done={timesteps_done} >= total_timesteps={total_timesteps}. "
            "Raise base.total_timesteps above the checkpoint to continue training.",
            UserWarning,
            stacklevel=1,
        )
        apply_schedules_from_base(
            model,
            ent_cb,
            base,
            global_total_timesteps=total_timesteps,
        )
        mean_r, std_r, eval_score = evaluate_current_model(
            model,
            train_env,
            seed,
            n_eval_episodes,
            env_id=env_id,
        )
        if best_eval is None:
            best_eval = eval_score
        else:
            best_eval = max(best_eval, eval_score)
        ckpt_dir = tempfile.mkdtemp(prefix="lunarlander_tune_ckpt_")
        try:
            save_trial_checkpoint(
                ckpt_dir,
                model,
                train_env,
                seed=seed,
                current_config=merged,
                best_eval_score_so_far=best_eval,
            )
            report_metrics = {
                "eval_mean_reward": mean_r,
                "eval_std_reward": std_r,
                "eval_score": eval_score,
                "timesteps_done": timesteps_done,
            }
            report_metrics.update(hp_metrics_from_merged(merged))
            tune.report(
                report_metrics,
                checkpoint=Checkpoint.from_directory(ckpt_dir),
            )
        finally:
            shutil.rmtree(ckpt_dir, ignore_errors=True)

    train_env.close()
