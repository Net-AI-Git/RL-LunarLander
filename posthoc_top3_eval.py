#!/usr/bin/env python3
"""
Post-hoc evaluation: load the **final** checkpoint of the top-3 Ray Tune trials (by last-row
``eval_score``), then run 10 seeds × 10 episodes on ``posthoc_eval_seeds`` from config
(``ray_pbt_config.json`` / ``trainer_state.json``), deterministic policy.

Usage::

    python posthoc_top3_eval.py --experiment-path /path/to/ray_results/exp_name/driver_artifacts

Or pass explicit checkpoint directories (each must contain ``model.zip``, ``vecnormalize.pkl``,
``trainer_state.json``)::

    python posthoc_top3_eval.py --checkpoint-dirs ckpt_a ckpt_b ckpt_c

Requires: ``pip install 'ray[tune]'`` for ``--experiment-path``; core eval only needs SB3 + gymnasium.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from lunar_rl_common import posthoc_eval_seeds_from_base
from ray_pbt_train import CHECKPOINT_TRAINER_STATE, evaluate_current_model, load_trial_checkpoint


def _latest_checkpoint_subdir(trial_logdir: Path) -> Path:
    cps = [
        p
        for p in trial_logdir.iterdir()
        if p.is_dir() and p.name.startswith("checkpoint_")
    ]
    if not cps:
        raise FileNotFoundError(f"No checkpoint_* directory under {trial_logdir}")

    def sort_key(p: Path) -> int:
        tail = p.name.split("_")[-1]
        return int(tail) if tail.isdigit() else 0

    return max(cps, key=sort_key)


def _top3_final_checkpoints_from_experiment(experiment_path: str) -> list[Path]:
    try:
        from ray.tune import ExperimentAnalysis
    except ImportError as e:
        raise ImportError(
            "Install Ray Tune for --experiment-path: pip install 'ray[tune]'"
        ) from e

    analysis = ExperimentAnalysis(experiment_path)
    df = analysis.results_df
    if df is None or len(df) == 0:
        raise RuntimeError(f"No rows in ExperimentAnalysis results for {experiment_path!r}")

    if "eval_score" not in df.columns:
        raise RuntimeError(f"Column 'eval_score' missing from results; have: {list(df.columns)}")
    if "trial_id" not in df.columns:
        raise RuntimeError(f"Column 'trial_id' missing from results; have: {list(df.columns)}")

    idx = df.groupby("trial_id")["timesteps_done"].idxmax()
    final_rows = df.loc[idx].reset_index(drop=True)
    top3 = final_rows.nlargest(3, "eval_score")

    out: list[Path] = []
    for _, row in top3.iterrows():
        logdir = None
        for col in ("logdir", "trial_logdir"):
            v = row.get(col)
            if v is not None and str(v) not in ("nan", "None", ""):
                logdir = v
                break
        if logdir is None:
            raise RuntimeError("Row missing 'logdir' / 'trial_logdir'; cannot locate checkpoints.")
        ld = Path(str(logdir)).resolve()
        if not ld.is_dir():
            raise FileNotFoundError(f"Trial logdir not found: {ld}")
        ckpt = _latest_checkpoint_subdir(ld)
        st = ckpt / CHECKPOINT_TRAINER_STATE
        if not st.is_file():
            raise FileNotFoundError(f"Missing {CHECKPOINT_TRAINER_STATE} under {ckpt}")
        out.append(ckpt)
    return out


def _eval_one_checkpoint(checkpoint_dir: Path) -> dict[str, Any]:
    checkpoint_dir = checkpoint_dir.resolve()
    state_path = checkpoint_dir / CHECKPOINT_TRAINER_STATE
    with open(state_path, encoding="utf-8") as f:
        state: dict[str, Any] = json.load(f)
    merged = state["current_config"]
    base = merged["base"]
    seeds = posthoc_eval_seeds_from_base(base)
    n_ep = 10

    model, train_env, _ = load_trial_checkpoint(str(checkpoint_dir), merged, env_id=None)
    try:
        mean_r, std_r, score, diag = evaluate_current_model(
            model,
            train_env,
            base,
            env_id=None,
            return_diagnostics=True,
            eval_seeds_override=seeds,
            n_eval_episodes_per_seed=n_ep,
        )
    finally:
        train_env.close()

    return {
        "checkpoint_dir": str(checkpoint_dir),
        "trial_seed_saved": state.get("seed"),
        "posthoc_eval_seeds": list(seeds),
        "n_episodes_per_seed": n_ep,
        "total_episodes": len(seeds) * n_ep,
        "eval_mean_reward": mean_r,
        "eval_std_reward": std_r,
        "eval_score": score,
        **diag,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Post-hoc top-3 trial evaluation (10×10 hold-out seeds).")
    p.add_argument(
        "--experiment-path",
        type=str,
        default=None,
        help="Ray Tune experiment directory (ExperimentAnalysis root).",
    )
    p.add_argument(
        "--checkpoint-dirs",
        nargs="*",
        default=None,
        help="Explicit checkpoint directories (skip ExperimentAnalysis).",
    )
    args = p.parse_args()

    if args.checkpoint_dirs:
        ckpts = [Path(d).resolve() for d in args.checkpoint_dirs]
    elif args.experiment_path:
        ckpts = _top3_final_checkpoints_from_experiment(os.path.expanduser(args.experiment_path))
    else:
        p.error("Provide --experiment-path or --checkpoint-dirs")

    results = []
    for c in ckpts:
        results.append(_eval_one_checkpoint(c))

    winner = max(results, key=lambda r: r["eval_score"])
    out = {"results": results, "winner_by_posthoc_eval_score": winner}
    print(json.dumps(out, indent=2))
    print(
        "\nWinner by post-hoc eval_score:",
        winner.get("checkpoint_dir"),
        "eval_score",
        winner.get("eval_score"),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
