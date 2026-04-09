#!/usr/bin/env python3
"""Small smoke tests before a full PBT run (imports, RunConfig, viz refresh, Tuner construct)."""

from __future__ import annotations

import tempfile
import traceback
import warnings


def main() -> int:
    failed = 0

    try:
        import ray_pbt_launcher  # noqa: F401
        import ray_tune_visualization  # noqa: F401
        from ray_pbt_train import get_default_pbt_config, hp_metrics_from_merged
    except Exception:
        traceback.print_exc()
        return 1

    try:
        static = get_default_pbt_config()
        merged = {"base": static["base"]}
        hp = hp_metrics_from_merged(merged)
        assert isinstance(hp, dict)
        assert "learning_rate" in hp
    except Exception:
        failed += 1
        traceback.print_exc()

    try:
        from ray_pbt_launcher import build_run_config

        rc = build_run_config(metric="eval_score", mode="max")
        assert rc.callbacks is not None and len(rc.callbacks) >= 1
        rc_empty = build_run_config(callbacks=[])
        assert rc_empty.callbacks == []
    except Exception:
        failed += 1
        traceback.print_exc()

    try:
        from ray_tune_visualization import refresh_tune_visualizations

        with tempfile.TemporaryDirectory() as d:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                refresh_tune_visualizations(d)
    except Exception:
        failed += 1
        traceback.print_exc()

    try:
        from ray.tune.schedulers import PopulationBasedTraining
        from ray_pbt_launcher import _hyperparam_mutations

        from ray_tune_visualization import TuneVisualizationCallback

        TuneVisualizationCallback(metric="eval_score", mode="max")
        pbt = PopulationBasedTraining(
            time_attr="training_iteration",
            perturbation_interval=2,
            burn_in_period=1,
            hyperparam_mutations=_hyperparam_mutations(),
        )
        assert pbt is not None
    except Exception:
        failed += 1
        traceback.print_exc()

    try:
        from ray import tune
        from ray.tune import Tuner, TuneConfig
        from ray.tune.schedulers import PopulationBasedTraining as PBT

        from ray_pbt_launcher import build_run_config, initial_param_space, _hyperparam_mutations
        from ray_pbt_train import get_default_pbt_config, train_lunarlander_pbt

        static = get_default_pbt_config()
        pbt_cfg = static["pbt"]
        pbt = PBT(
            time_attr="training_iteration",
            perturbation_interval=int(pbt_cfg["perturbation_interval"]),
            burn_in_period=int(pbt_cfg["burn_in_period"]),
            hyperparam_mutations=_hyperparam_mutations(),
            quantile_fraction=float(pbt_cfg["quantile_fraction"]),
            resample_probability=float(pbt_cfg["resample_probability"]),
            log_config=True,
        )
        trainable = tune.with_resources(train_lunarlander_pbt, resources={"cpu": 1})
        _ = Tuner(
            trainable,
            param_space=initial_param_space(static),
            tune_config=TuneConfig(
                metric=pbt_cfg["metric"],
                mode=pbt_cfg["mode"],
                scheduler=pbt,
                num_samples=1,
            ),
            run_config=build_run_config(
                experiment_name="_smoke_do_not_run_fit",
                metric=pbt_cfg["metric"],
                mode=pbt_cfg["mode"],
                callbacks=[],
            ),
        )
    except Exception:
        failed += 1
        traceback.print_exc()

    if failed == 0:
        print("smoke_ray_tune: all checks passed")
    else:
        print(f"smoke_ray_tune: {failed} check(s) failed", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
