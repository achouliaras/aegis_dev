import os
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

LABELS = {
          "AEGIS_101010": r"$\mathcal{L}^{FDM}+\mathcal{L}^{IDM}+\mathcal{L}^{DSCM}$",
          "AEGIS_050505": r"$0.5(\mathcal{L}^{FDM}+\mathcal{L}^{IDM}+\mathcal{L}^{DSCM})$",
          "AEGIS_100000": r"$\mathcal{L}^{FDM}$",
          "AEGIS_001000": r"$\mathcal{L}^{IDM}$",
          "AEGIS_000010": r"$\mathcal{L}^{DSCM}$",
          "AEGIS_101000": r"$\mathcal{L}^{FDM}+\mathcal{L}^{IDM}$",
          "AEGIS_001010": r"$\mathcal{L}^{IDM}+\mathcal{L}^{DSCM}$",
          "AEGIS_100010": r"$\mathcal{L}^{FDM}+\mathcal{L}^{DSCM}$",
          "AEGIS_050500": r"$0.5\mathcal{L}^{FDM}+0.5\mathcal{L}^{IDM}$",
          "AEGIS_000505": r"$0.5\mathcal{L}^{IDM}+0.5\mathcal{L}^{DSCM}$",
          "AEGIS_050005": r"$0.5\mathcal{L}^{FDM}+0.5\mathcal{L}^{DSCM}$",
          "AEGIS_101005": r"$\mathcal{L}^{FDM}+\mathcal{L}^{IDM}+0.5\mathcal{L}^{DSCM}$",
          "AEGIS_050510": r"$0.5\mathcal{L}^{FDM}+0.5\mathcal{L}^{IDM}+\mathcal{L}^{DSCM}$",
          "AEGIS_051010": r"$0.5\mathcal{L}^{FDM}+\mathcal{L}^{IDM}+\mathcal{L}^{DSCM}$",
          "AEGIS_100510": r"$\mathcal{L}^{FDM}+0.5\mathcal{L}^{IDM}+\mathcal{L}^{DSCM}$",
          "AEGIS_021010": r"$0.2\mathcal{L}^{FDM}+1.0\mathcal{L}^{IDM}+\mathcal{L}^{DSCM}$",
          "AEGIS_050710": r"$0.5\mathcal{L}^{FDM}+0.7\mathcal{L}^{IDM}+\mathcal{L}^{DSCM}$",
          "AEGIS": r"AEGIS($0.2\mathcal{L}^{FDM}+0.2\mathcal{L}^{IDM}+\mathcal{L}^{DSCM}$)",
          }

MODES = {"NoPreTrain": "NPT",
         "QuarterPreTrain": "1QPT",
         "HalfPreTrain": "2QPT",
         "ThreeQuarterPreTrain": "3QPT"}

# Custom colors and markers for each algo
COLORS = {"AEGIS": "blue",
          "AEGIS_101010": "cyan",
          "AEGIS_050505": "magenta",
          "AEGIS_100000": "orange",
          "AEGIS_001000": "green",
          "AEGIS_000010": "red",
          "AEGIS_101000": "lime",
          "AEGIS_001010": "teal",
          "AEGIS_100010": "navy",
          "AEGIS_050500": "purple",
          "AEGIS_000505": "brown",
          "AEGIS_050005": "pink",
          "AEGIS_051010": "darkorange",
          "AEGIS_100510": "darkgreen",
          "AEGIS_101005": "olive",
          "AEGIS_021010": "gold",
          "AEGIS_050510": "gray",
          "AEGIS_050710": "darkblue",
         }

MARKERS = {"AEGIS": "X",
           "AEGIS_101010": "o",
           "AEGIS_050505": "s",
           "AEGIS_100000": "<",
           "AEGIS_001000": ">",
           "AEGIS_000010": "^",
           "AEGIS_101000": "8",
           "AEGIS_001010": "p",
           "AEGIS_100010": "P",
           "AEGIS_050500": "v",
           "AEGIS_000505": "h",
           "AEGIS_050005": "*",
           "AEGIS_101005": "D",
           "AEGIS_050510": "P",
           "AEGIS_051010": "X",
           "AEGIS_100510": "s",
           "AEGIS_021010": "o",
           "AEGIS_050710": "v",
        }

def plot_algorithms_for_env(
    env: str,
    mode: str,
    algos: List[str],
    out_filename: str,
    base_path: str = "logs",
    n_seeds: int = 10,
    figsize: Tuple[int,int] = (10, 6),
    save_kwargs: Optional[dict] = None,
) -> Dict[str, pd.DataFrame]:
    """
    For a given env and mode, load logs/{env}/{algo}/{mode}/{seed}/rollout.csv
    for each algo and seed, average across seeds, plot mean ± std,
    save the figure to out_filename, and return a dict of averaged DataFrames.

    Parameters
    ----------
    env : str
        Environment name (kept outside in caller loops).
    mode : str
        Mode name (kept outside in caller loops).
    algos : list[str]
        List of algorithm folder names to compare (this loop is inside the function).
    out_filename : str
        Path to save the resulting figure (extension determines format, e.g. .png, .pdf).
    base_path : str
        Base logs folder (default "logs").
    n_seeds : int
        Number of seeds (folders 0 ... n_seeds-1).
    figsize : tuple
        Figure size in inches.
    save_kwargs : dict, optional
        Extra kwargs to pass to plt.savefig (e.g. dpi=300).
    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from algorithm name to a DataFrame with columns
        ['time/total_timesteps', 'mean', 'std'].
    """
    if save_kwargs is None:
        save_kwargs = {}

    fig, ax1 = plt.subplots(figsize=figsize)

    averaged: Dict[str, pd.DataFrame] = {}
    averaged_states: Dict[str, pd.DataFrame] = {}
    required_cols_reward = {"time/total_timesteps", "rollout/ep_info_rew_mean"}
    required_cols_states = {"time/total_timesteps", "rollout/ll_unique_states_per_step"}

    for algo in algos:
        seed_dfs_r, seed_dfs_s = [], []
        for seed in range(n_seeds):
            if algo == "DEIR":
                seed = seed + 10  # DEIR seeds start from 10
            csv_path = os.path.join(base_path, env, algo, mode, str(seed), "rollout.csv")
            if not os.path.exists(csv_path):
                print(f"⚠️ Missing file: {csv_path}")
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"❌ Error reading {csv_path}: {e}")
                continue
            if not required_cols_reward.issubset(df.columns) or not required_cols_states.issubset(df.columns):
                print(f"⚠️ Missing columns in {csv_path}")

            # Rewards
            if required_cols_reward.issubset(df.columns):
                df_r = df[["time/total_timesteps", "rollout/ep_info_rew_mean"]]
                seed_dfs_r.append(df_r)

            # Unique states
            if required_cols_states.issubset(df.columns):
                df_s = df[["time/total_timesteps", "rollout/ll_unique_states"]]
                seed_dfs_s.append(df_s)

        if not seed_dfs_r and not seed_dfs_s:
            print(f"ℹ️ No valid data for algo={algo}")
            continue

        # Merge on timesteps (inner join across seeds)
        merged_r = seed_dfs_r[0].rename(columns={"rollout/ep_info_rew_mean": f"seed0"})
        for i, df in enumerate(seed_dfs_r[1:], start=1):
            merged_r = pd.merge(
                merged_r,
                df.rename(columns={"rollout/ep_info_rew_mean": f"seed{i}"}),
                on="time/total_timesteps",
                how="inner"
            )
        merged_s = seed_dfs_s[0].rename(columns={"rollout/ll_unique_states": f"seed0"})
        for i, df in enumerate(seed_dfs_s[1:], start=1):
            merged_s = pd.merge(
                merged_s,
                df.rename(columns={"rollout/ll_unique_states": f"seed{i}"}),
                on="time/total_timesteps",
                how="inner"
            )

        # Compute mean/std across seeds
        rewards = merged_r.drop(columns=["time/total_timesteps"])
        merged_r["mean"] = rewards.mean(axis=1)
        merged_r["std"] = rewards.std(axis=1)

        states = merged_s.drop(columns=["time/total_timesteps"])
        merged_s["mean"] = states.mean(axis=1)
        merged_s["std"] = states.std(axis=1)

        averaged[algo] = merged_r[["time/total_timesteps", "mean", "std"]]
        averaged_states[algo] = merged_s[["time/total_timesteps", "mean", "std"]]

        # Plot with std band
        ax1.plot(merged_r["time/total_timesteps"], merged_r["mean"], label=LABELS[algo], color=COLORS.get(algo, None), marker=MARKERS.get(algo, None), markevery=0.1)
        ax1.fill_between(
            merged_r["time/total_timesteps"],
            merged_r["mean"] - merged_r["std"],
            merged_r["mean"] + merged_r["std"],
            color=COLORS.get(algo, None),
            alpha=0.2
        )
    if not averaged:
        print(f"ℹ️ No valid data found for env='{env}', mode='{mode}'. Nothing plotted.")
        plt.close(fig)
        return averaged

    ax1.xaxis.set_major_formatter(ScalarFormatter())
    ax1.ticklabel_format(style="sci", axis="x", scilimits=(0,0))
    ax1.xaxis.get_offset_text().set_fontsize(16)   # change scientific notation text size
    ax1.tick_params(axis="both", which="major", labelsize=16)  # bigger font for major ticks
    ax1.tick_params(axis="both", which="minor", labelsize=14)  # optional, minor ticks
    ax1.set_xlabel("Total Timesteps", fontsize=16)
    ax1.set_ylabel("Episode Reward Mean", fontsize=16)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True)
    ax1.legend(
        loc="upper left",
        fontsize=10,
    )

    # leave extra room on the right (adjust 0.8 if legend is too cramped)
    plt.tight_layout()

    # --- Add vertical line for pretraining/training split ---
    max_timestep = max(df["time/total_timesteps"].max() for df in averaged.values())

    if "ThreeQuarter" in mode:
        split_frac = 0.75
    elif "Quarter" in mode:
        split_frac = 0.25
    elif "Half" in mode:
        split_frac = 0.50
    else:
        split_frac = None

            
    if split_frac is not None:
        split_step = max_timestep * split_frac
        ax1.axvline(split_step, color="black", linestyle="--", label="Pretraining")

        # Annotate text
        ymin, ymax = ax1.get_ylim()
        ax1.text(split_step * 0.5, ymax * 1.1, "Pretraining",
                ha="center", va="top", color="black", fontsize=16, fontweight="bold")
        ax1.text(split_step + (max_timestep - split_step) * 0.5, ymax * 1.1, "Training",
                ha="center", va="top", color="black", fontsize=16, fontweight="bold")
    # -------------------------------------------------------
    try:
        # os.makedirs(os.path.dirname(f"figures/{mode}/"+out_filename), exist_ok=True)
        # plt.savefig(f"figures/{mode}/"+out_filename, **save_kwargs)
        # print(f"✅ Saved figure to 'figures/{mode}/"+out_filename)
        os.makedirs(os.path.dirname(f"figures/"+out_filename), exist_ok=True)
        plt.savefig(f"figures/"+out_filename, bbox_inches="tight", **save_kwargs)
        print(f"✅ Saved figure to 'figures/"+out_filename)
    except Exception as e:
        print(f"❌ Error saving figure '{out_filename}': {e}")
    finally:
        plt.close(fig)

    return averaged

if __name__ == "__main__":
    envs = ["MiniGrid-DoorKey-8x8-v0", 
            ] 
    modes = ["HalfPreTrain"]

    algos_to_compare = ["AEGIS_100000", "AEGIS_001000", "AEGIS_000010", "AEGIS_050500", "AEGIS_000505", "AEGIS_050005", "AEGIS_101000", "AEGIS_001010", "AEGIS_100010", "AEGIS_101010", "AEGIS_051010", "AEGIS"]
    algos_to_compare2 = ["AEGIS_050505", "AEGIS_101010", "AEGIS_101005", "AEGIS_050510", "AEGIS_051010", "AEGIS_100510", "AEGIS", "AEGIS_021010",  "AEGIS_050710"]
    for mode in modes:
        for env in envs:
            out_file = f"abl_loss_coef.png"
            data = plot_algorithms_for_env(env=env,
                                           mode=mode,
                                           algos=algos_to_compare,
                                           out_filename=out_file,
                                           base_path="logs",
                                           n_seeds=10,
                                           figsize=(8, 4),
                                           save_kwargs={"dpi": 200})
            
            out_file2 = f"abl_loss_coef2.png"
            data = plot_algorithms_for_env(env=env,
                                           mode=mode,
                                           algos=algos_to_compare2,
                                           out_filename=out_file2,
                                           base_path="logs",
                                           n_seeds=10,
                                           figsize=(8, 4),
                                           save_kwargs={"dpi": 200})
            # `data` is a dict {algo: DataFrame} you can use for further analysis if desired.
