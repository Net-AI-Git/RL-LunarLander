#!/usr/bin/env python3
"""Apply Lunar Lander notebook refactor (restore plan)."""
import json
import re
from pathlib import Path

p = Path("/workspace/unit1 - Lunar Lander agent_v3.ipynb")
nb = json.loads(p.read_text())
cells = nb["cells"]


def cell_text(c):
    return "".join(c.get("source", []))


def find_code(pred):
    for i, c in enumerate(cells):
        if c["cell_type"] == "code" and pred(cell_text(c)):
            return i
    return None


def find_md(pred):
    for i, c in enumerate(cells):
        if c["cell_type"] == "markdown" and pred(cell_text(c)):
            return i
    return None


# --- 1) Setup: replace install markdown, remove install code cells ---
idx_install_md = find_md(lambda s: "## Install dependencies" in s and "virtual screen" in s)
if idx_install_md is None:
    idx_install_md = find_md(lambda s: "Install dependencies" in s)

if idx_install_md is not None:
    cells[idx_install_md]["source"] = [
        "## Install dependencies and virtual screen (outside this notebook)\n",
        "\n",
        "All **pip** packages are listed in [`requirements.txt`](requirements.txt). "
        "Run the install commands in a terminal or in Colab **before** running the rest of this notebook — **the cells below do not run any installs**.\n",
        "\n",
        "Keep next to the notebook: `requirements.txt`, `lunar_rl_common.py`, and optionally `run_optuna_tuning.py` / `best_hyperparams.json`.\n",
        "\n",
        "**System (Linux / Colab)** — tools for Box2D, rendering, and Xvfb:\n",
        "\n",
        "```bash\n",
        "sudo apt-get update\n",
        "sudo apt-get install -y swig cmake python3-opengl ffmpeg xvfb\n",
        "```\n",
        "\n",
        "**Python**:\n",
        "\n",
        "```bash\n",
        "pip install -r requirements.txt\n",
        "```\n",
        "\n",
        "On **Google Colab**: upload the files via the file browser, then run the two command blocks above "
        "(for example in a one-off code cell you delete after installing).\n",
    ]

# Remove code cells: !apt, !sudo, %pip, !pip (iterate backwards)
to_remove = []
for i, c in enumerate(cells):
    if c["cell_type"] != "code":
        continue
    t = cell_text(c)
    if any(
        x in t
        for x in ("!apt install", "!sudo apt-get", "%pip install", "!pip install", "!pip3 install")
    ):
        to_remove.append(i)
for i in reversed(to_remove):
    del cells[i]

# Update markdown after install section (virtual display + restart)
idx_vd = find_md(
    lambda s: "During the notebook" in s
    or ("replay" in s.lower() and "video" in s.lower() and "## Install" not in s)
)
if idx_vd is not None:
    cells[idx_vd]["source"] = [
        "Replay videos need a **virtual display** (Xvfb). Make sure you installed the **xvfb** system package from the section above.\n",
        "\n",
        "If your environment recommends a restart after installs, run the next restart cell, then the virtual-display cell.\n",
    ]

idx_rs = find_md(lambda s: "sometimes it's required to restart" in s or "restart the notebook runtime" in s)
if idx_rs is not None:
    cells[idx_rs]["source"] = [
        "Sometimes, for newly installed libraries to load correctly, you need to **restart the notebook runtime**. "
        "The next cell forces a **deliberate process crash** — reconnect, then run again starting from the virtual-display cell below.\n",
    ]

# --- 2) Consolidated imports (first code cell after "## Import the packages" markdown) ---
idx_imp_md = find_md(lambda s: "## Import the packages" in s)
imp_code_idx = None
if idx_imp_md is not None:
    for j in range(idx_imp_md + 1, min(idx_imp_md + 5, len(cells))):
        if cells[j]["cell_type"] == "code":
            imp_code_idx = j
            break

IMPORTS_SOURCE = """# Reproducibility and all imports for the rest of the notebook
import copy
import glob
import inspect
import json
import os
import random
import subprocess
import sys
import tempfile

import gymnasium as gym
import gymnasium
import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import torch
import torch.nn as nn
from IPython.display import HTML, display

from huggingface_hub import ModelCard, notebook_login
from huggingface_sb3 import load_from_hub, package_to_hub

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
"""

if imp_code_idx is not None:
    cells[imp_code_idx]["source"] = [line + "\n" for line in IMPORTS_SOURCE.splitlines()]

# --- 3) Gym demo: remove top-level duplicate import gymnasium as gym ---
idx_gym_demo = find_code(lambda s: "LunarLander-v3" in s and "env.reset(seed=SEED)" in s and "for _ in range(5)" in s)
if idx_gym_demo is not None:
    t = cell_text(cells[idx_gym_demo])
    lines = [ln for ln in t.splitlines() if ln.strip() != "import gymnasium as gym"]
    cells[idx_gym_demo]["source"] = [l + "\n" for l in lines]

# --- 4) Hyperparameters + JSON ---
HP_SOURCE = """# ============================================================
# Hyperparameters — defaults; PPO params overridden if JSON exists
# ============================================================
from pathlib import Path

HYPERPARAMS_JSON = Path("best_hyperparams.json")

# --- Environment ---
env_id = "LunarLander-v3"
n_envs = 32

# --- Training ---
total_timesteps = 16_000_000
model_name = "ppo-LunarLander-v3"
checkpoint_dir = "./checkpoints"
vecnormalize_path = os.path.join(checkpoint_dir, "vecnormalize.pkl")
save_freq = 500_000

# PPO defaults (used if no JSON; overwritten from best_hyperparams.json)
learning_rate = 3e-4
n_steps = 1024
batch_size = 64
n_epochs = 4
gamma = 0.99
gae_lambda = 0.95
ent_coef = 0.01

# --- Progress video ---
video_eval_freq = 200_000

# --- Live reward plot ---
reward_plot_window = 50
reward_plot_freq = 5000

if HYPERPARAMS_JSON.is_file():
    with open(HYPERPARAMS_JSON, "r", encoding="utf-8") as f:
        best_data = json.load(f)
    best = best_data["params"]
    learning_rate = best["learning_rate"]
    n_steps = best["n_steps"]
    batch_size = best["batch_size"]
    n_epochs = best["n_epochs"]
    gamma = best["gamma"]
    gae_lambda = best["gae_lambda"]
    ent_coef = best["ent_coef"]
    print(f"Loaded PPO hyperparameters from {HYPERPARAMS_JSON}:")
    print(f"  trial_number={best_data.get('trial_number')!r} score={best_data.get('score')!r}")
    print(
        f"  learning_rate={learning_rate} n_steps={n_steps} batch_size={batch_size} "
        f"n_epochs={n_epochs} gamma={gamma} gae_lambda={gae_lambda} ent_coef={ent_coef}"
    )
else:
    print(f"No {HYPERPARAMS_JSON} found — using default PPO hyperparameters above.")
"""

idx_hp = find_code(lambda s: "# Hyperparameters" in s and "edit this cell" in s)
if idx_hp is None:
    idx_hp = find_code(lambda s: "optuna_n_trials" in s or ("learning_rate = 3e-4" in s and "n_envs" in s))

if idx_hp is not None:
    cells[idx_hp]["source"] = [line + "\n" for line in HP_SOURCE.splitlines()]

# --- 5) Replace giant env cell with lunar_rl_common import ---
idx_giant = find_code(
    lambda s: "class VecNormalizeSaveCallback" in s and "class GrayscaleResizePixelsWrapper" in s
)
COMMON_IMPORT = """from lunar_rl_common import (
    VecNormalizeSaveCallback,
    get_device,
    make_eval_vec_env_with_stats,
    make_lunar_dict_env,
    make_subproc_venv,
    make_train_vec_env,
    policy_kwargs,
)

device = get_device()
"""

if idx_giant is not None:
    cells[idx_giant]["source"] = [line + "\n" for line in COMMON_IMPORT.splitlines()]

# --- 6) Validation cell ---
idx_val = find_code(
    lambda s: "check_env" in s and "make_lunar_dict_env" in s and "_val_env" in s
)
if idx_val is not None:
    cells[idx_val]["source"] = [
        "_val_env = make_lunar_dict_env(env_id)\n",
        "check_env(_val_env, warn=True, skip_render_check=True)\n",
        "obs, _ = _val_env.reset(seed=SEED)\n",
        'assert set(obs.keys()) == {"state", "pixels"}\n',
        'assert obs["state"].shape == (8,) and obs["state"].dtype == np.float32\n',
        'assert obs["pixels"].shape == (1, 84, 84) and obs["pixels"].dtype == np.uint8\n',
        'print("Validation OK:", {k: (obs[k].shape, obs[k].dtype) for k in obs})\n',
        "_val_env.close()\n",
    ]

# --- 7) Smoke test ---
idx_smoke = find_code(lambda s: "Smoke test" in s and "smoke_n" in s)
if idx_smoke is not None:
    t = cell_text(cells[idx_smoke])
    t = t.replace("return make_lunar_dict_env()", "return make_lunar_dict_env(env_id)")
    cells[idx_smoke]["source"] = [l + "\n" for l in t.splitlines()]

# --- 8) make_train_vec_env line ---
idx_mtv = find_code(
    lambda s: s.strip().startswith("env = make_train_vec_env") and "n_envs=n_envs" in s
)
if idx_mtv is not None:
    cells[idx_mtv]["source"] = [
        "env = make_train_vec_env(n_envs=n_envs, seed=SEED, gamma=gamma, env_id=env_id)\n"
    ]

# --- 9) Remove Optuna / JSON save / JSON load code cells; replace Optuna markdown ---
idx_opt_md = find_md(lambda s: "Hyperparameter Tuning with Optuna" in s or "Optuna" in s and "best_hyperparams" in s)
if idx_opt_md is not None:
    cells[idx_opt_md]["source"] = [
        "## Optional: hyperparameter tuning with Optuna (outside the notebook)\n",
        "\n",
        "This notebook does **not** run Optuna. After you change the **architecture** or want a **one-off** search, run the script from the same directory (so `lunar_rl_common.py` is importable):\n",
        "\n",
        "```bash\n",
        "python run_optuna_tuning.py --n-trials 30 --timesteps-per-trial 200000 --n-envs 32 --seed 42\n",
        "```\n",
        "\n",
        "Results are saved to `best_hyperparams.json` (override with `--output`). "
        "The **Hyperparameters** cell above loads `params` from that file automatically when it exists.\n",
        "\n",
        "**Note:** Old tuning results from a different policy/observation setup are not comparable — re-run the study after architecture changes.\n",
    ]

to_remove_opt = []
for i, c in enumerate(cells):
    if c["cell_type"] != "code":
        continue
    t = cell_text(c)
    if "import optuna" in t and "def objective" in t:
        to_remove_opt.append(i)
    elif "study.optimize" in t and "optuna.create_study" in t:
        to_remove_opt.append(i)
    elif "best_params_path" in t and "json.load" in t and "learning_rate = best" in t:
        to_remove_opt.append(i)
for i in reversed(sorted(set(to_remove_opt))):
    del cells[i]

# --- 10) model_name duplicate cell ---
for i, c in enumerate(cells):
    if c["cell_type"] == "code" and cell_text(c).strip() == 'model_name = "ppo-LunarLander-v3"':
        cells[i]["source"] = ["# model_name is defined in the Hyperparameters cell\n"]
        break

# --- 11) Observation demo: DummyVecEnv ---
idx_demo = find_code(lambda s: "_demo = DummyVecEnv" in s)
if idx_demo is not None:
    t = cell_text(cells[idx_demo])
    t = t.replace(
        "_demo = DummyVecEnv([make_lunar_dict_env])",
        "_demo = DummyVecEnv([lambda: make_lunar_dict_env(env_id)])",
    )
    # strip duplicate imports if any at top
    lines = t.splitlines()
    out = []
    for ln in lines:
        if ln.startswith("import copy") or ln.startswith("import matplotlib.pyplot"):
            continue
        if ln.startswith("import numpy as np") and "import numpy" in IMPORTS_SOURCE:
            continue
        if ln.startswith("from stable_baselines3.common.vec_env import DummyVecEnv"):
            continue
        out.append(ln)
    cells[idx_demo]["source"] = [l + "\n" for l in out]

# --- 12) VideoProgressCallback ---
idx_vid = find_code(lambda s: "class VideoProgressCallback" in s)
if idx_vid is not None:
    t = cell_text(cells[idx_vid])
    lines = [ln for ln in t.splitlines() if not (ln.startswith("import ") or ln.startswith("from "))]
    t = "\n".join(lines)
    t = t.replace(
        "def __init__(self, eval_freq=200_000, vecnormalize_path=None, seed=42):",
        "def __init__(self, eval_freq=200_000, vecnormalize_path=None, seed=42, env_id=None):",
    )
    t = t.replace(
        "        self.seed = seed\n",
        "        self.seed = seed\n        self.env_id = env_id\n",
    )
    t = t.replace(
        "        eval_venv = make_eval_vec_env_with_stats(self.vecnormalize_path, self.seed)",
        "        eval_venv = make_eval_vec_env_with_stats(\n            self.vecnormalize_path, self.seed, self.env_id\n        )",
    )
    t = t.replace(
        "video_cb = VideoProgressCallback(\n    eval_freq=video_eval_freq,\n    vecnormalize_path=vecnormalize_path,\n    seed=SEED,\n)",
        "video_cb = VideoProgressCallback(\n    eval_freq=video_eval_freq,\n    vecnormalize_path=vecnormalize_path,\n    seed=SEED,\n    env_id=env_id,\n)",
    )
    cells[idx_vid]["source"] = [l + "\n" for l in t.splitlines()]

# --- 13) LiveRewardPlotCallback: strip imports ---
idx_live = find_code(lambda s: "class LiveRewardPlotCallback" in s)
if idx_live is not None:
    lines = [
        ln
        for ln in cell_text(cells[idx_live]).splitlines()
        if not (ln.startswith("import ") or ln.startswith("from "))
    ]
    cells[idx_live]["source"] = [l + "\n" for l in lines if l.strip() or l == ""]

# --- 14) First model.learn block (checkpoint) strip imports ---
idx_learn1 = None
for i, c in enumerate(cells):
    if c["cell_type"] != "code":
        continue
    t = cell_text(c)
    if (
        "model.learn" in t
        and "total_timesteps=total_timesteps" in t
        and "reset_num_timesteps" not in t
        and "checkpoint_callback" in t
    ):
        idx_learn1 = i
        break
if idx_learn1 is not None:
    lines = [
        ln
        for ln in cell_text(cells[idx_learn1]).splitlines()
        if ln not in ("import os", "from stable_baselines3.common.callbacks import CheckpointCallback")
    ]
    cells[idx_learn1]["source"] = [l + "\n" for l in lines]

# --- 15) Eval cell ---
idx_eval = find_code(
    lambda s: s.strip().startswith("eval_env = make_eval_vec_env_with_stats")
    and "mean_reward" in s
)
if idx_eval is not None:
    cells[idx_eval]["source"] = [
        "eval_env = make_eval_vec_env_with_stats(vecnormalize_path, SEED, env_id)\n",
        "mean_reward, std_reward = evaluate_policy(\n",
        "    model, eval_env, n_eval_episodes=20, deterministic=True\n",
        ")\n",
        "eval_env.close()\n",
        'print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")\n',
    ]

# --- 16) Resume training cell ---
idx_resume = find_code(lambda s: "reset_num_timesteps=False" in s and "glob.glob" in s)
if idx_resume is not None:
    t = cell_text(cells[idx_resume])
    t = t.replace("venv = make_subproc_venv(n_envs, SEED)", "venv = make_subproc_venv(n_envs, SEED, env_id)")
    lines = [
        ln
        for ln in t.splitlines()
        if ln
        not in (
            "import os, glob",
            "from stable_baselines3 import PPO",
            "from stable_baselines3.common.callbacks import CheckpointCallback",
        )
    ]
    # re-add glob usage - still need glob; it's in imports cell
    cells[idx_resume]["source"] = [l + "\n" for l in lines]

# Second VideoProgressCallback instantiation in resume cell
if idx_resume is not None:
    t = cell_text(cells[idx_resume])
    if "VideoProgressCallback(" in t and "env_id=env_id" not in t:
        t = t.replace(
            "video_cb = VideoProgressCallback(\n    eval_freq=video_eval_freq,\n    vecnormalize_path=vecnormalize_path,\n    seed=SEED,\n)",
            "video_cb = VideoProgressCallback(\n    eval_freq=video_eval_freq,\n    vecnormalize_path=vecnormalize_path,\n    seed=SEED,\n    env_id=env_id,\n)",
        )
        cells[idx_resume]["source"] = [l + "\n" for l in t.splitlines()]

# --- 17) package_to_hub cell ---
idx_hub = find_code(lambda s: "package_to_hub" in s and "_sig = inspect.signature" in s)
if idx_hub is not None:
    lines = [
        ln
        for ln in cell_text(cells[idx_hub]).splitlines()
        if ln not in ("import inspect", "from huggingface_sb3 import package_to_hub")
        and not ln.startswith("from huggingface_hub import ModelCard")
    ]
    t = "\n".join(lines)
    t = t.replace(
        "eval_env = make_eval_vec_env_with_stats(vecnormalize_path, SEED)",
        "eval_env = make_eval_vec_env_with_stats(vecnormalize_path, SEED, env_id)",
    )
    # Remove duplicate env_id assignment line for hub
    out_lines = []
    skip_todo_env = False
    for ln in t.splitlines():
        if "# TODO: Define the name of the environment" in ln:
            skip_todo_env = True
            continue
        if skip_todo_env and re.match(r'^\s*env_id\s*=\s*["\']', ln):
            skip_todo_env = False
            continue
        skip_todo_env = False
        out_lines.append(ln)
    cells[idx_hub]["source"] = [l + "\n" for l in "\n".join(out_lines).splitlines()]

p.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("Wrote", p)