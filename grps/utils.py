import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_densities(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6, 4), dpi=200)
    axes = axes.flat
    fig.suptitle("Population Densities")

    for i in range(4):
        df = data[data["seed"] == i]
        axes[i].plot(df["epoch"], df["R_density"], c="red", label="rock")
        axes[i].plot(df["epoch"], df["P_density"], c="blue", label="paper")
        axes[i].plot(df["epoch"], df["S_density"], c="green", label="scissors")
        if i == 2 or i == 3:
            axes[i].set_xlabel("Epoch")
        if i == 0 or i == 2:
            axes[i].set_ylabel("Count")
        axes[i].grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.05),
    )
    plt.tight_layout()
    plt.show()


def plot_average_invasions(data: pd.DataFrame) -> None:
    # Average invasion rate
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6, 4), dpi=200)
    axes = axes.flat
    fig.suptitle("Mean Invasion Probability")

    for i in range(4):
        df = data[data["seed"] == i]
        axes[i].plot(df["epoch"], df["R_invasion"], c="red", label="rock")
        axes[i].plot(df["epoch"], df["P_invasion"], c="blue", label="paper")
        axes[i].plot(df["epoch"], df["S_invasion"], c="green", label="scissors")
        axes[i].axhline(1.0, ls="--", c="gray", label="max invasion")
        if i == 2 or i == 3:
            axes[i].set_xlabel("Epoch")
        if i == 0 or i == 2:
            axes[i].set_ylabel("Invasion")
        axes[i].grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.05),
    )
    plt.tight_layout()
    plt.show()


def plot_average_age(data: pd.DataFrame) -> None:
    # Average age
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6, 4), dpi=200)
    axes = axes.flat
    fig.suptitle("Mean Population Age")

    for i in range(4):
        df = data[data["seed"] == i]
        axes[i].plot(df["epoch"], df["R_age"], c="red", label="rock")
        axes[i].plot(df["epoch"], df["P_age"], c="blue", label="paper")
        axes[i].plot(df["epoch"], df["S_age"], c="green", label="scissors")
        if i == 2 or i == 3:
            axes[i].set_xlabel("Epoch")
        if i == 0 or i == 2:
            axes[i].set_ylabel("Age")
        axes[i].grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.05),
    )
    plt.tight_layout()
    plt.show()


def plot_orbits(df: pd.DataFrame):
    _, axes = plt.subplots(
        2,
        2,
        figsize=(6, 4),
        dpi=200,
        sharex=True,
        sharey=True,
    )

    axes = axes.flat

    # vertici del simplesso
    R = np.array([0.0, 0.0])
    S = np.array([1.0, 0.0])
    P = np.array([0.5, np.sqrt(3) / 2])

    for ax, (_, run) in zip(axes, df.groupby("seed")):
        r = run["R_density"].to_numpy()
        p = run["P_density"].to_numpy()
        s = run["S_density"].to_numpy()

        # normalizzazione
        total = r + p + s
        r = r / total
        p = p / total
        s = s / total

        # coordinate barycentriche -> cartesiane
        x = s + 0.5 * p
        y = (np.sqrt(3) / 2) * p

        # triangolo
        ax.plot(
            [R[0], S[0], P[0], R[0]],
            [R[1], S[1], P[1], R[1]],
            c="black",
            lw=1,
        )

        # traiettoria
        ax.plot(x, y, lw=0.5, zorder=1)

        # inizio
        ax.scatter(x[0], y[0], marker="o", c="green", s=5, zorder=2)

        # fine
        ax.scatter(x[-1], y[-1], marker="x", c="red", s=5, zorder=2)

        ax.set_aspect("equal")
        ax.axis("off")

        # etichette vertici
        ax.text(-0.06, -0.03, "R")
        ax.text(1.02, -0.03, "S")
        ax.text(0.5, np.sqrt(3) / 2 + 0.03, "P")

    plt.tight_layout()
    plt.show()
