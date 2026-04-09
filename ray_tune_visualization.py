#!/usr/bin/env python3
"""
Automatic Ray Tune / PBT result visualization: progress plots, comparisons, CSV/JSON summary.

Called from ``TuneVisualizationCallback`` after each trial and at experiment end, and
optionally once more from the launcher after ``Tuner.fit()``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import warnings
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ray.air.constants import TRAINING_ITERATION
from ray.tune import Callback
from ray.tune.analysis import ExperimentAnalysis

logger = logging.getLogger(__name__)

DEFAULT_METRIC = "eval_score"
DEFAULT_MODE = "max"
DEFAULT_TOP_K = 8
OUTPUT_SUBDIR = "visualizations"

# Hyperparameters used for PBT-style perturbation lines and evolution plots
HP_TRACK_KEYS = (
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

HP_EVOLUTION_PLOT_KEYS = ("learning_rate", "ent_coef", "vf_coef", "target_kl")


def _trial_df_by_id(trial_dfs: dict[Any, pd.DataFrame], trial_id: Any) -> Optional[pd.DataFrame]:
    tid = str(trial_id)
    for k, v in trial_dfs.items():
        if str(k) == tid:
            return v
    return None


def _safe_trial_filename_id(trial_id: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", str(trial_id))[:200]


def _ensure_eval_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "eval_score" not in df.columns:
        if "eval_mean_reward" in df.columns and "eval_std_reward" in df.columns:
            df["eval_score"] = df["eval_mean_reward"] - df["eval_std_reward"]
    return df


def _x_series(df: pd.DataFrame) -> tuple[np.ndarray, str]:
    if "timesteps_done" in df.columns and df["timesteps_done"].notna().any():
        return df["timesteps_done"].to_numpy(dtype=float), "timesteps_done"
    if TRAINING_ITERATION in df.columns and df[TRAINING_ITERATION].notna().any():
        return df[TRAINING_ITERATION].to_numpy(dtype=float), TRAINING_ITERATION
    idx = np.arange(len(df), dtype=float)
    return idx, "index"


def _perturbation_x_indices(df: pd.DataFrame) -> list[float]:
    """X positions where any tracked HP changes (PBT mutation)."""
    if len(df) < 2:
        return []
    present = [k for k in HP_TRACK_KEYS if k in df.columns]
    if not present:
        return []
    xs, _ = _x_series(df)
    marks: list[float] = []
    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        row = df.iloc[i]
        changed = False
        for k in present:
            if k not in prev.index or k not in row.index:
                continue
            if pd.isna(prev[k]) or pd.isna(row[k]):
                continue
            if float(prev[k]) != float(row[k]):
                changed = True
                break
        if changed:
            marks.append(float(xs[i]))
    return marks


def _plot_trial_progress(
    df: pd.DataFrame,
    out_png: Path,
    out_html: Optional[Path],
    perturbation_xs: list[float],
    x_label: str,
) -> None:
    df = _ensure_eval_columns(df)
    xs, xl = _x_series(df)
    if xl != x_label:
        xl = x_label

    fig, ax = plt.subplots(figsize=(10, 5))
    if "eval_mean_reward" in df.columns:
        ax.plot(xs, df["eval_mean_reward"], label="eval_mean_reward", color="#1976D2")
    if "eval_std_reward" in df.columns:
        ax.plot(xs, df["eval_std_reward"], label="eval_std_reward", color="#E53935", linestyle="--")
    if "eval_score" in df.columns:
        ax.plot(xs, df["eval_score"], label="eval_score", color="#2E7D32", linewidth=2)
    for xv in perturbation_xs:
        ax.axvline(xv, color="gray", linestyle=":", alpha=0.7, linewidth=1)
    ax.set_xlabel(xl)
    ax.set_ylabel("reward / score")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)

    if out_html is not None:
        try:
            import plotly.graph_objects as go  # type: ignore[import-untyped]

            fig_p = go.Figure()
            if "eval_mean_reward" in df.columns:
                fig_p.add_trace(
                    go.Scatter(x=xs, y=df["eval_mean_reward"], name="eval_mean_reward")
                )
            if "eval_std_reward" in df.columns:
                fig_p.add_trace(
                    go.Scatter(x=xs, y=df["eval_std_reward"], name="eval_std_reward")
                )
            if "eval_score" in df.columns:
                fig_p.add_trace(go.Scatter(x=xs, y=df["eval_score"], name="eval_score"))
            for xv in perturbation_xs:
                fig_p.add_vline(x=xv, line_dash="dot", line_color="gray", opacity=0.6)
            fig_p.update_layout(title="Trial progress", xaxis_title=xl)
            fig_p.write_html(str(out_html))
        except ImportError:
            pass


def _plot_hp_evolution_single(
    df: pd.DataFrame,
    key: str,
    out_png: Path,
    perturbation_xs: list[float],
    x_label: str,
) -> None:
    if key not in df.columns or df[key].notna().sum() == 0:
        return
    xs, _ = _x_series(df)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, df[key], color="#6A1B9A", linewidth=1.5)
    for xv in perturbation_xs:
        ax.axvline(xv, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel(x_label)
    ax.set_ylabel(key)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_all_trials_comparison(
    trial_frames: dict[str, pd.DataFrame],
    out_path: Path,
    metric: str,
    best_trial_id: Optional[str],
    *,
    sort_legend_by_metric: bool,
    title_suffix: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    trial_max_scores: list[tuple[str, float]] = []
    xl = "x"
    for tid, df in trial_frames.items():
        try:
            df = _ensure_eval_columns(df)
            if metric not in df.columns:
                continue
            xs, xl = _x_series(df)
            m = float(df[metric].max())
            trial_max_scores.append((str(tid), m))
        except Exception as e:
            warnings.warn(f"trial {tid}: skip comparison line: {e}", UserWarning)
            continue
        tid_s = str(tid)
        bt = str(best_trial_id) if best_trial_id is not None else None
        lw = 2.8 if bt is not None and tid_s == bt else 1.2
        alpha = 1.0 if bt is not None and tid_s == bt else 0.75
        if bt is not None and tid_s == bt:
            ax.plot(xs, df[metric], label=tid_s, linewidth=lw, alpha=alpha, color="#F57C00")
        else:
            ax.plot(xs, df[metric], label=tid_s, linewidth=lw, alpha=alpha)
    ax.set_xlabel(xl)
    ax.set_ylabel(metric)
    ax.set_title(f"All trials — {metric}{title_suffix}")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if sort_legend_by_metric and trial_max_scores:
        score_by_tid = {tid: sc for tid, sc in trial_max_scores}
        pairs = sorted(
            zip(labels, handles),
            key=lambda lh: score_by_tid.get(lh[0], float("-inf")),
            reverse=True,
        )
        if pairs:
            labels, handles = zip(*pairs)
            ax.legend(handles, labels, loc="best", fontsize=8)
    else:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def refresh_tune_visualizations(
    experiment_path: str | os.PathLike,
    *,
    metric: str = DEFAULT_METRIC,
    mode: str = DEFAULT_MODE,
    top_k: int = DEFAULT_TOP_K,
    output_subdir: str = OUTPUT_SUBDIR,
) -> None:
    """
    Read Tune experiment results, write plots and summaries under
    ``<experiment_path>/<output_subdir>/``.
    """
    exp_path = Path(experiment_path).resolve()
    out_dir = exp_path / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        analysis = ExperimentAnalysis(
            str(exp_path),
            default_metric=metric,
            default_mode=mode,
        )
    except Exception as e:
        warnings.warn(
            f"refresh_tune_visualizations: could not open experiment at {exp_path}: {e}",
            UserWarning,
        )
        return

    trial_dfs = analysis.trial_dataframes
    if not trial_dfs:
        warnings.warn(f"No trial dataframes at {exp_path}", UserWarning)
        return

    best_tid: Optional[str] = None
    try:
        best_trial = analysis.get_best_trial(metric=metric, mode=mode, scope="all")
        if best_trial is not None:
            best_tid = str(best_trial.trial_id)
    except Exception as e:
        warnings.warn(f"Could not get best trial: {e}", UserWarning)

    # Per-trial artifacts
    summary_rows: list[dict[str, Any]] = []
    comparison_frames: dict[str, pd.DataFrame] = {}

    for trial_id, df in trial_dfs.items():
        if df is None or len(df) == 0:
            warnings.warn(f"Trial {trial_id}: empty dataframe, skipping.", UserWarning)
            continue
        try:
            df = _ensure_eval_columns(df)
            xs, x_label = _x_series(df)
            perts = _perturbation_x_indices(df)
            safe_id = _safe_trial_filename_id(trial_id)

            png = out_dir / f"trial_{safe_id}_progress.png"
            html = out_dir / f"trial_{safe_id}_progress.html"
            _plot_trial_progress(df, png, html, perts, x_label)

            for hk in HP_EVOLUTION_PLOT_KEYS:
                hp_png = out_dir / f"trial_{safe_id}_hp_{hk}.png"
                _plot_hp_evolution_single(df, hk, hp_png, perts, x_label)

            comparison_frames[str(trial_id)] = df

            max_score = float(df[metric].max()) if metric in df.columns else float("nan")
            max_mean = (
                float(df["eval_mean_reward"].max())
                if "eval_mean_reward" in df.columns
                else float("nan")
            )
            last_ts = float(df["timesteps_done"].iloc[-1]) if "timesteps_done" in df.columns else None
            summary_rows.append(
                {
                    "trial_id": trial_id,
                    "max_eval_score": max_score,
                    "max_eval_mean_reward": max_mean,
                    "last_timesteps_done": last_ts,
                    "rows": len(df),
                }
            )
        except Exception as e:
            warnings.warn(f"Trial {trial_id}: visualization failed: {e}", UserWarning)

    # Comparisons
    if comparison_frames:
        comp_path = out_dir / "all_trials_comparison.png"
        _plot_all_trials_comparison(
            comparison_frames,
            comp_path,
            metric,
            best_tid,
            sort_legend_by_metric=False,
        )
        sorted_path = out_dir / "all_trials_comparison_sorted.png"
        _plot_all_trials_comparison(
            comparison_frames,
            sorted_path,
            metric,
            best_tid,
            sort_legend_by_metric=True,
            title_suffix=" (legend sorted by max metric)",
        )

        # Top-K by max metric
        ranked = sorted(
            comparison_frames.items(),
            key=lambda kv: float(kv[1][metric].max()) if metric in kv[1].columns else float("-inf"),
            reverse=True,
        )[:top_k]
        top_dict = {k: v for k, v in ranked}
        top_path = out_dir / "top_trials_comparison.png"
        best_in_top = ranked[0][0] if ranked else None
        _plot_all_trials_comparison(
            top_dict,
            top_path,
            metric,
            best_in_top,
            sort_legend_by_metric=True,
            title_suffix=f" (top {len(top_dict)} by max {metric})",
        )

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("max_eval_score", ascending=False, na_position="last")
    summary_df.to_csv(out_dir / "trial_summary.csv", index=False)

    # best_trial_summary.json
    best_payload: dict[str, Any] = {
        "metric": metric,
        "mode": mode,
        "best_trial_id": best_tid,
        "experiment_path": str(exp_path),
    }
    try:
        bt_obj = analysis.get_best_trial(metric=metric, mode=mode, scope="all")
        if bt_obj is not None:
            bdf = _trial_df_by_id(trial_dfs, bt_obj.trial_id)
            if bdf is not None and len(bdf) > 0:
                bdf = _ensure_eval_columns(bdf)
                best_payload["best_eval_score"] = (
                    float(bdf[metric].max()) if metric in bdf.columns else None
                )
                best_payload["best_eval_mean_reward"] = (
                    float(bdf["eval_mean_reward"].max())
                    if "eval_mean_reward" in bdf.columns
                    else None
                )
                last = bdf.iloc[-1]
                best_payload["final_row_metrics"] = {
                    k: _json_scalar(last[k]) for k in last.index
                }
        cfg = analysis.get_best_config(metric=metric, mode=mode, scope="last")
        if cfg is not None:
            best_payload["best_config"] = cfg
        if bt_obj is not None:
            ckpt = analysis.get_best_checkpoint(bt_obj, metric=metric, mode=mode)
            if ckpt is not None:
                best_payload["best_checkpoint_path"] = getattr(ckpt, "path", str(ckpt))
    except Exception as e:
        warnings.warn(f"best trial summary incomplete: {e}", UserWarning)

    with open(out_dir / "best_trial_summary.json", "w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2, default=str)


def _json_scalar(v: Any) -> Any:
    if isinstance(v, (np.floating, float, np.integer, int, str, bool)) or v is None:
        return v
    if isinstance(v, np.ndarray):
        return v.tolist()
    return str(v)


def print_and_save_run_summary(
    experiment_path: str | os.PathLike,
    *,
    metric: str = DEFAULT_METRIC,
    mode: str = DEFAULT_MODE,
    output_subdir: str = OUTPUT_SUBDIR,
) -> None:
    """Print and append human-readable summary to ``run_summary.txt`` under visualizations."""
    exp_path = Path(experiment_path).resolve()
    out_dir = exp_path / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "run_summary.txt"
    lines: list[str] = []
    try:
        analysis = ExperimentAnalysis(
            str(exp_path),
            default_metric=metric,
            default_mode=mode,
        )
        bt = analysis.get_best_trial(metric=metric, mode=mode, scope="all")
        if bt is None:
            lines.append("No completed best trial found.")
        else:
            lines.append(f"Best trial ID: {bt.trial_id}")
            try:
                cfg = analysis.get_best_config(metric=metric, mode=mode, scope="last")
                lines.append(f"Best config (last scope): {json.dumps(cfg, indent=2, default=str)}")
            except Exception as e:
                lines.append(f"Config: (unavailable: {e})")
            try:
                ckpt = analysis.get_best_checkpoint(trial=bt, metric=metric)
                cp = getattr(ckpt, "path", str(ckpt)) if ckpt else None
                lines.append(f"Best checkpoint path: {cp}")
            except Exception as e:
                lines.append(f"Best checkpoint: (unavailable: {e})")
            df = _trial_df_by_id(analysis.trial_dataframes, bt.trial_id)
            if df is not None and len(df) > 0:
                df = _ensure_eval_columns(df)
                if metric in df.columns:
                    lines.append(f"Best max {metric}: {float(df[metric].max())}")
                if "eval_mean_reward" in df.columns:
                    lines.append(f"Best max eval_mean_reward: {float(df['eval_mean_reward'].max())}")
    except Exception as e:
        lines.append(f"Summary failed: {e}")
        warnings.warn(str(e), UserWarning)

    text = "\n".join(lines) + "\n"
    print(text)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(text)


def _experiment_path_from_trial(trial: Any) -> Optional[str]:
    try:
        lp = getattr(trial, "local_path", None) or getattr(trial, "logdir", None)
        if lp:
            return str(Path(lp).resolve().parent)
    except Exception:
        pass
    try:
        st = getattr(trial, "storage", None)
        if st is not None:
            p = getattr(st, "experiment_driver_staging_path", None) or getattr(
                st, "experiment_fs_path", None
            )
            if p:
                return str(p)
    except Exception:
        pass
    return None


class TuneVisualizationCallback(Callback):
    """Refresh visualization folder after each trial completes and when the experiment ends."""

    def __init__(
        self,
        *,
        metric: str = DEFAULT_METRIC,
        mode: str = DEFAULT_MODE,
        top_k: int = DEFAULT_TOP_K,
        output_subdir: str = OUTPUT_SUBDIR,
    ) -> None:
        super().__init__()
        self.metric = metric
        self.mode = mode
        self.top_k = top_k
        self.output_subdir = output_subdir

    def _refresh(self, trial: Any, *, with_summary: bool) -> None:
        exp_path = _experiment_path_from_trial(trial)
        if not exp_path:
            warnings.warn("TuneVisualizationCallback: could not resolve experiment path.", UserWarning)
            return
        try:
            refresh_tune_visualizations(
                exp_path,
                metric=self.metric,
                mode=self.mode,
                top_k=self.top_k,
                output_subdir=self.output_subdir,
            )
            if with_summary:
                print_and_save_run_summary(
                    exp_path,
                    metric=self.metric,
                    mode=self.mode,
                    output_subdir=self.output_subdir,
                )
        except Exception as e:
            warnings.warn(f"TuneVisualizationCallback: refresh failed: {e}", UserWarning)

    def on_trial_complete(
        self, iteration: int, trials: list, trial, **info
    ) -> None:
        self._refresh(trial, with_summary=False)

    def on_experiment_end(self, trials: list, **info) -> None:
        if trials:
            self._refresh(trials[0], with_summary=True)
