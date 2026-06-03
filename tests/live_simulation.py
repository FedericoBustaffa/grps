"""
live_simulation.py — RPS Solara dashboard  (heatmap + tutti i grafici su page 0)
Avvia con:  solara run live_simulation.py
"""

from functools import partial

import matplotlib.pyplot as plt
import mesa
import numpy as np
import solara
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.visualization import SolaraViz, make_plot_component
from mesa.visualization.utils import update_counter

from grps.evolution_policies import Genetic, Inheritance, Stochastic
from grps.rps_agent import RPSAgent

# ──────────────────────────────────────────────────────────────────────────────
# Reporters
# ──────────────────────────────────────────────────────────────────────────────


def _norm_density(specie, model):
    total = len(model.agents)
    if total == 0:
        return 0.0
    return len(model.agents.select(lambda a: a.specie == specie)) / total


def _invasion(specie, model):
    agents = model.agents.select(lambda a: a.specie == specie)
    return float(agents.agg("invasion", np.mean)) if len(agents) else 0.0


def _age(specie, model):
    agents = model.agents.select(lambda a: a.specie == specie)
    return float(agents.agg("age", np.mean)) if len(agents) else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

SPECIES = ("rock", "paper", "scissors")
POLICY_OPTIONS = ("inheritance", "stochastic", "genetic")

# Colori stabili: rock=rosso, paper=blu, scissors=verde
COLORS = {"rock": "#d64045", "paper": "#3a86ff", "scissors": "#38b000"}
CMAP = ListedColormap([COLORS["rock"], COLORS["paper"], COLORS["scissors"]])
SPECIE_IDX = {"rock": 0, "paper": 1, "scissors": 2}


def _build_policy(name: str, sigma: float, radius: int):
    if name == "inheritance":
        return Inheritance()
    if name == "stochastic":
        return Stochastic(sigma=sigma)
    return Genetic(sigma=sigma, radius=radius)


class RPSModel(mesa.Model):
    """Rock-Paper-Scissors — keyword-only constructor per SolaraViz."""

    def __init__(
        self,
        *,
        dim: int = 50,
        rock_invasion: float = 0.5,
        paper_invasion: float = 0.5,
        scissors_invasion: float = 0.5,
        rock_policy: str = "stochastic",
        paper_policy: str = "stochastic",
        scissors_policy: str = "stochastic",
        sigma: float = 0.01,
        radius: int = 5,
        rng=None,
    ) -> None:
        super().__init__(rng=rng)

        n = dim * dim
        self.dim = dim
        self.grid = OrthogonalMooreGrid((dim, dim), torus=True, random=self.random)
        self.epoch_length = n
        self.epoch = 0

        self.policies = {
            "rock": _build_policy(rock_policy, sigma, radius),
            "paper": _build_policy(paper_policy, sigma, radius),
            "scissors": _build_policy(scissors_policy, sigma, radius),
        }

        init_invasions = {
            "rock": rock_invasion,
            "paper": paper_invasion,
            "scissors": scissors_invasion,
        }
        cells = list(self.grid.all_cells)
        species_list = self.random.choices(SPECIES, k=n)
        invasion_list = [init_invasions[s] for s in species_list]

        RPSAgent.create_agents(
            model=self,
            n=n,
            cell=cells,
            specie=species_list,
            invasion=invasion_list,
        )

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "R_norm": partial(_norm_density, "rock"),
                "P_norm": partial(_norm_density, "paper"),
                "S_norm": partial(_norm_density, "scissors"),
                "R_invasion": partial(_invasion, "rock"),
                "P_invasion": partial(_invasion, "paper"),
                "S_invasion": partial(_invasion, "scissors"),
                "R_age": partial(_age, "rock"),
                "P_age": partial(_age, "paper"),
                "S_age": partial(_age, "scissors"),
            }
        )
        self.datacollector.collect(self)

    def step(self) -> None:
        for _ in range(self.epoch_length):
            self.random.choice(self.agents).hunt()
        self.agents.do("get_older")
        self.epoch += 1
        self.datacollector.collect(self)


# ──────────────────────────────────────────────────────────────────────────────
# Custom component — heatmap a quadrati pieni
# ──────────────────────────────────────────────────────────────────────────────


@solara.component
def GridHeatmap(model):
    """Visualizza la griglia come imshow (quadrati pieni, stile heatmap)."""
    update_counter.get()  # hook obbligatorio per aggiornamento reattivo

    dim = model.dim
    grid_arr = np.empty((dim, dim), dtype=np.int8)
    for agent in model.agents:
        x, y = agent.cell.coordinate
        grid_arr[x, y] = SPECIE_IDX[agent.specie]

    fig = Figure(figsize=(4.5, 4.5))
    ax = fig.subplots()

    ax.imshow(
        grid_arr.T,  # trasposta: x→colonne, y→righe, origine in basso
        cmap=CMAP,
        vmin=0,
        vmax=2,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )
    ax.set_title(f"Grid — epoch {model.epoch}", fontsize=10)
    ax.axis("off")

    # Legenda manuale
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=COLORS["rock"], label="Rock"),
        Patch(facecolor=COLORS["paper"], label="Paper"),
        Patch(facecolor=COLORS["scissors"], label="Scissors"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=8,
        framealpha=0.7,
    )

    solara.FigureMatplotlib(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Post-process per i plot — stile coerente
# ──────────────────────────────────────────────────────────────────────────────


def _pp_pop(ax):
    ax.set_title("Popolazione (normalizzata)", fontsize=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Frazione")
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.4)
    lines = ax.get_lines()
    colors = [COLORS["rock"], COLORS["paper"], COLORS["scissors"]]
    labels = ["Rock", "Paper", "Scissors"]
    for line, c, lbl in zip(lines, colors, labels):
        line.set_color(c)
        line.set_label(lbl)
    ax.legend(fontsize=8)


def _pp_invasion(ax):
    ax.set_title("Invasion rate medio", fontsize=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Invasion prob.")
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.4)
    lines = ax.get_lines()
    colors = [COLORS["rock"], COLORS["paper"], COLORS["scissors"]]
    labels = ["Rock", "Paper", "Scissors"]
    for line, c, lbl in zip(lines, colors, labels):
        line.set_color(c)
        line.set_label(lbl)
    ax.legend(fontsize=8)


def _pp_age(ax):
    ax.set_title("Età media (longevità / fitness)", fontsize=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Età (epoch)")
    ax.grid(True, linestyle="--", alpha=0.4)
    lines = ax.get_lines()
    colors = [COLORS["rock"], COLORS["paper"], COLORS["scissors"]]
    labels = ["Rock", "Paper", "Scissors"]
    for line, c, lbl in zip(lines, colors, labels):
        line.set_color(c)
        line.set_label(lbl)
    ax.legend(fontsize=8)


# ──────────────────────────────────────────────────────────────────────────────
# Componenti plot — tutti su page=0 (stessa pagina della heatmap)
# ──────────────────────────────────────────────────────────────────────────────

PopPlot = make_plot_component(["R_norm", "P_norm", "S_norm"], post_process=_pp_pop)
InvasionPlot = make_plot_component(
    ["R_invasion", "P_invasion", "S_invasion"], post_process=_pp_invasion
)
AgePlot = make_plot_component(["R_age", "P_age", "S_age"], post_process=_pp_age)


# ──────────────────────────────────────────────────────────────────────────────
# Parametri dashboard
# ──────────────────────────────────────────────────────────────────────────────

model_params = {
    "dim": {
        "type": "SliderInt",
        "value": 50,
        "label": "Grid size (dim × dim)",
        "min": 10,
        "max": 100,
        "step": 5,
    },
    "rock_invasion": {
        "type": "SliderFloat",
        "value": 0.5,
        "label": "Rock — initial invasion",
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
    },
    "paper_invasion": {
        "type": "SliderFloat",
        "value": 0.5,
        "label": "Paper — initial invasion",
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
    },
    "scissors_invasion": {
        "type": "SliderFloat",
        "value": 0.5,
        "label": "Scissors — initial invasion",
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
    },
    "rock_policy": {
        "type": "Select",
        "value": "stochastic",
        "label": "Rock — policy",
        "values": list(POLICY_OPTIONS),
    },
    "paper_policy": {
        "type": "Select",
        "value": "stochastic",
        "label": "Paper — policy",
        "values": list(POLICY_OPTIONS),
    },
    "scissors_policy": {
        "type": "Select",
        "value": "stochastic",
        "label": "Scissors — policy",
        "values": list(POLICY_OPTIONS),
    },
    "sigma": {
        "type": "SliderFloat",
        "value": 0.01,
        "label": "Mutation σ",
        "min": 0.0,
        "max": 0.2,
        "step": 0.005,
    },
    "radius": {
        "type": "SliderInt",
        "value": 5,
        "label": "Genetic partner radius",
        "min": 1,
        "max": 30,
        "step": 1,
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# SolaraViz — tutto su page=0, nessuna tab
# ──────────────────────────────────────────────────────────────────────────────

model = RPSModel()

page = SolaraViz(
    model,
    components=[GridHeatmap, PopPlot, InvasionPlot, AgePlot],
    model_params=model_params,
    name="Rock · Paper · Scissors",
)
