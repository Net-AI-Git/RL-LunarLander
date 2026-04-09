#!/usr/bin/env python3
"""
Replay a PBT hyperparameter **schedule** from a policy file on a single trial.

Requires the original PBT run to use ``log_config=True`` (see ``ray_pbt_launcher.py``).
Ray writes policy files under the experiment directory, e.g.::

  ray_results/lunarlander_pbt/pbt_policy_<trial_id>.txt

Usage::

  python ray_pbt_replay.py --policy-file /path/to/pbt_policy_....txt

Optional: ``RAY_RESULTS_DIR``, ``RAY_PBT_CHECKPOINTS_TO_KEEP``, ``RAY_PBT_CPUS`` — same as the main launcher.
Experiment name: ``--experiment-name`` or env ``RAY_PBT_REPLAY_NAME`` (default ``lunarlander_pbt_replay``).
"""

from __future__ import annotations

import argparse
import os

from ray import tune
from ray.tune import Tuner, TuneConfig
from ray.tune.schedulers import PopulationBasedTrainingReplay

from ray_pbt_launcher import _DEFAULT_CPUS, build_run_config, initial_param_space
from ray_pbt_train import get_default_pbt_config, train_lunarlander_pbt


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay PBT config schedule on one trial.")
    parser.add_argument(
        "--policy-file",
        required=True,
        help="Path to pbt_policy_*.txt from a previous PBT run (log_config=True).",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Tune run name (default: lunarlander_pbt_replay or RAY_PBT_REPLAY_NAME).",
    )
    args = parser.parse_args()

    policy_path = os.path.abspath(os.path.expanduser(args.policy_file))
    if not os.path.isfile(policy_path):
        raise FileNotFoundError(f"policy file not found: {policy_path}")

    static = get_default_pbt_config()
    pbt_cfg = static["pbt"]

    replay = PopulationBasedTrainingReplay(policy_path)

    trainable = tune.with_resources(
        train_lunarlander_pbt,
        resources={"cpu": _DEFAULT_CPUS},
    )

    exp_name = args.experiment_name or os.environ.get(
        "RAY_PBT_REPLAY_NAME", "lunarlander_pbt_replay"
    )

    tuner = Tuner(
        trainable,
        param_space=initial_param_space(static),
        tune_config=TuneConfig(
            mode=pbt_cfg["mode"],
            metric=pbt_cfg["metric"],
            scheduler=replay,
            num_samples=1,
        ),
        run_config=build_run_config(experiment_name=exp_name),
    )

    tuner.fit()


if __name__ == "__main__":
    main()
