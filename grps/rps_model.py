from functools import partial
from typing import Mapping, Sequence

import mesa
import numpy as np
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.model import RNGLike, SeedLike
from tqdm import trange

from grps.evolution_policies import EvolutionPolicy
from grps.rps_agent import RPSAgent
from grps.utils import save_grid

# ---------------------------------------------------------------------------
# Data-collector reporter functions
# ---------------------------------------------------------------------------


def population_density(specie: str, model: mesa.Model) -> int:
    """Number of living agents of the given species."""
    return len(model.agents.select(lambda a: a.specie == specie))


def specie_invasion(specie: str, model: mesa.Model) -> float:
    """Mean invasion rate of the given species (0.0 if extinct)."""
    agents = model.agents.select(lambda a: a.specie == specie)
    if len(agents) == 0:
        return 0.0
    avg = agents.agg("invasion", np.mean)
    assert isinstance(avg, float)
    return float(avg)


def specie_age(specie: str, model: mesa.Model) -> float:
    """Mean age of the given species (0.0 if extinct)."""
    agents = model.agents.select(lambda a: a.specie == specie)
    if len(agents) == 0:
        return 0.0
    avg = agents.agg("age", np.mean)
    assert isinstance(avg, float)
    return float(avg)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class RPSModel(mesa.Model):
    """Spatial Rock-Paper-Scissors model on a toroidal Moore-neighbourhood grid.

    The grid is fully occupied (one agent per cell) at all times.  Predation
    coincides with reproduction: a successful hunt converts the prey cell
    rather than removing it, preserving the total population.

    Args:
        dim:      side length of the square grid (total agents = dim²).
        policies: mapping from species name to an :class:`EvolutionPolicy`
                  that determines the offspring's invasion rate.
        initial_invasions: initial values of invasion probabilities for each specie
        rng:      seed or RNG for reproducibility.
    """

    SPECIES = ("rock", "paper", "scissors")

    def __init__(
        self,
        dim: int,
        policies: Mapping[str, EvolutionPolicy],
        initial_invasions: Sequence[float],
        rng: "RNGLike | SeedLike | None" = None,
    ) -> None:
        super().__init__(rng=rng)

        n = dim * dim  # total number of cells / agents
        self.grid = OrthogonalMooreGrid((dim, dim), torus=True, random=self.random)
        self.epoch_length = n  # one epoch = N attempted hunts (as in the paper)
        self.policies = policies
        self.epoch = 0

        # ------------------------------------------------------------------
        # Populate the grid: one agent per cell, random species & invasion
        # ------------------------------------------------------------------
        cells = list(self.grid.all_cells)
        species_list = self.random.choices(self.SPECIES, k=n)
        invasion_list = [initial_invasions[self.SPECIES.index(s)] for s in species_list]

        RPSAgent.create_agents(
            model=self,
            n=n,
            cell=cells,
            specie=species_list,
            invasion=invasion_list,
        )

        # ------------------------------------------------------------------
        # Data collectors
        # ------------------------------------------------------------------
        model_reporters = {
            "R_density": partial(population_density, "rock"),
            "P_density": partial(population_density, "paper"),
            "S_density": partial(population_density, "scissors"),
            "R_invasion": partial(specie_invasion, "rock"),
            "P_invasion": partial(specie_invasion, "paper"),
            "S_invasion": partial(specie_invasion, "scissors"),
            "R_age": partial(specie_age, "rock"),
            "P_age": partial(specie_age, "paper"),
            "S_age": partial(specie_age, "scissors"),
            "epoch": lambda _: self.epoch,
        }

        self.datacollector = mesa.DataCollector(model_reporters=model_reporters)

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance the model by one epoch.

        Each epoch consists of `epoch_length` randomly-chosen hunt attempts,
        followed by aging all agents by one step.  Data are collected at the
        beginning of each epoch (i.e. before the hunts).
        """
        # Randomly activate agents (with replacement, as in the reference paper)
        for _ in range(self.epoch_length):
            ind = self.random.choice(self.agents)
            ind.hunt()

        # Every agent ages by 1 at the end of the epoch
        self.agents.do("get_older")

    def run_for(
        self,
        duration: float | int,
        filename: str = "grid",
        verbose: bool = False,
    ) -> None:
        """Run the model for `duration` epochs."""
        assert isinstance(duration, int)

        for epoch in trange(duration, ncols=80):
            self.datacollector.collect(self)

            if verbose:
                self._log_epoch()

            if epoch == 0:
                save_grid(self, f"images/{filename}_start.png")
            if epoch == duration // 2:
                save_grid(self, f"images/{filename}_mid.png")
            if epoch == duration - 1:
                save_grid(self, f"images/{filename}_end.png")

            self.epoch += 1
            self.step()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_epoch(self) -> None:
        r = population_density("rock", self)
        p = population_density("paper", self)
        s = population_density("scissors", self)
        ri = specie_invasion("rock", self)
        pi = specie_invasion("paper", self)
        si = specie_invasion("scissors", self)
        ra = specie_age("rock", self)
        pa = specie_age("paper", self)
        sa = specie_age("scissors", self)
        print(
            f"epoch {self.epoch:>4d} | "
            f"R: {r:>4d} (inv={ri:.3f}, age={ra:.1f})  "
            f"P: {p:>4d} (inv={pi:.3f}, age={pa:.1f})  "
            f"S: {s:>4d} (inv={si:.3f}, age={sa:.1f})"
        )
