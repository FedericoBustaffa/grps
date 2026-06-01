import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from grps import RPSModel
from grps import evolution_policies as evo

if __name__ == "__main__":
    radius = 25
    policies = {
        "rock": evo.Genetic(sigma=0.01, radius=radius),
        "paper": evo.Stochastic(sigma=0.01),
        "scissors": evo.Stochastic(sigma=0.01),
    }
    model = RPSModel(
        dim=50,
        policies=policies,
        initial_invasions=[0.5, 0.5, 0.5],
        rng=42,
    )

    n_epochs = 100
    model.run_for(n_epochs)

    df = model.datacollector.get_model_vars_dataframe()
    print(df)

    # Population density
    plt.figure(dpi=150)
    plt.title("Population Density")
    plt.plot(df["R_density"], c="red", label="rock")
    plt.plot(df["P_density"], c="blue", label="paper")
    plt.plot(df["S_density"], c="green", label="scissors")
    plt.xlabel("Epoch")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("population_density.png")
    plt.show()

    # Average invasion rate
    plt.figure(dpi=150)
    plt.title("Average Invasion Rate")
    plt.plot(df["R_invasion"], c="red", label="rock")
    plt.plot(df["P_invasion"], c="blue", label="paper")
    plt.plot(df["S_invasion"], c="green", label="scissors")
    plt.axhline(1.0, ls="--", c="gray", label="max invasion")
    plt.xlabel("Epoch")
    plt.ylabel("Mean invasion probability")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("invasion_rate.png")
    plt.show()

    # Average age
    plt.figure(dpi=150)
    plt.title("Average Age")
    plt.plot(df["R_age"], c="red", label="rock")
    plt.plot(df["P_age"], c="blue", label="paper")
    plt.plot(df["S_age"], c="green", label="scissors")
    plt.xlabel("Epoch")
    plt.ylabel("Mean age (epochs)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("average_age.png")
    plt.show()

    # Final grid state
    color_map = {"rock": 0, "paper": 1, "scissors": 2}
    grid_arr = np.full((model.grid.width, model.grid.height), -1)
    for agent in model.agents:
        x, y = agent.cell.coordinate
        grid_arr[x, y] = color_map[agent.specie]

    plt.figure(dpi=150)
    plt.title("Final Grid State")
    cmap = ListedColormap(["red", "blue", "green"])
    im = plt.imshow(grid_arr.T, cmap=cmap, interpolation="nearest", origin="lower")
    cbar = plt.colorbar(im, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["rock", "paper", "scissors"])
    plt.tight_layout()
    plt.savefig("final_grid.png")
    plt.show()
