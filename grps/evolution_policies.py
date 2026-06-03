from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from grps.rps_agent import RPSAgent


class EvolutionPolicy:
    """Base class for evolution policies.

    A policy is called after a successful predation event to compute the
    invasion rate of the newly-converted agent (the "offspring").

    Args:
        agent: the *predator* that just performed the hunt.

    Returns:
        The invasion probability [0, 1] assigned to the offspring.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, agent: "RPSAgent") -> float: ...


class Inheritance(EvolutionPolicy):
    """Offspring inherits the predator's invasion rate unchanged."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, agent: "RPSAgent") -> float:
        return agent.invasion


class Stochastic(EvolutionPolicy):
    """Offspring inherits the predator's invasion rate plus Gaussian noise.

    Args:
        sigma: standard deviation of the Gaussian mutation applied to
               the parent's invasion rate.
    """

    def __init__(self, sigma: float = 0.01) -> None:
        super().__init__()
        self.sigma = sigma

    def __call__(self, agent: "RPSAgent") -> float:
        offset = agent.model.random.gauss(0, self.sigma)
        return float(np.clip(agent.invasion + offset, 0.0, 1.0))


class Genetic(EvolutionPolicy):
    """Genetic recombination with fitness-weighted partner selection.

    After a successful predation the predator ("parent A") samples a mate
    ("parent B") from agents of the *same species* in its neighbourhood.
    The probability of selecting a candidate is proportional to

        P(candidate) ∝ fitness(candidate) / distance(predator, candidate)

    where ``fitness`` is the candidate's current age and ``distance`` is the
    Chebyshev distance on the torus grid.  The offspring's invasion rate is
    the average of the two parents' rates plus Gaussian mutation.

    If no same-species neighbour is found the predator self-reproduces with
    Stochastic mutation (fallback identical to :class:`Stochastic`).

    Args:
        sigma:       std-dev of the Gaussian mutation added to the recombined
                     invasion rate.
        radius:      how many Moore-neighbourhood "rings" to search for a
                     partner (default 3, so a 7x7 area).
        age_offset:  added to every candidate's age before computing fitness
                     so that very young agents still have non-zero weight
                     (default 1).
    """

    def __init__(
        self,
        sigma: float = 0.01,
        radius: int = 3,
        age_offset: int = 1,
    ) -> None:
        super().__init__()
        self.sigma = sigma
        self.radius = radius
        self.age_offset = age_offset

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chebyshev(ax: int, ay: int, bx: int, by: int, w: int, h: int) -> float:
        """Torus-aware Chebyshev distance."""
        dx = abs(ax - bx)
        dy = abs(ay - by)
        dx = min(dx, w - dx)
        dy = min(dy, h - dy)
        return float(max(dx, dy))

    def _partner_candidates(self, agent: "RPSAgent"):
        """Return all same-species agents within *radius* rings, with their distance."""
        from grps.rps_agent import (
            RPSAgent as _RPSAgent,
        )  # avoid circular at module level

        assert agent.cell is not None
        ax, ay = agent.cell.coordinate
        grid = agent.model.grid
        w, h = grid.width, grid.height

        candidates = []
        for cell in grid.all_cells:
            if cell is agent.cell:
                continue
            cx, cy = cell.coordinate
            dist = self._chebyshev(ax, ay, cx, cy, w, h)
            if dist > self.radius or dist == 0:
                continue
            for other in cell.agents:
                if isinstance(other, _RPSAgent) and other.specie == agent.specie:
                    candidates.append(other)

        return candidates

    # ------------------------------------------------------------------
    # policy call
    # ------------------------------------------------------------------

    def __call__(self, agent: "RPSAgent") -> float:
        candidates = self._partner_candidates(agent)

        if not candidates:
            # fallback: stochastic self-reproduction
            offset = agent.model.random.gauss(0, self.sigma)
            return float(np.clip(agent.invasion + offset, 0.0, 1.0))

        # Compute un-normalised weights: fitness / distance
        # fitness = age + age_offset  (longevity as proxy for fitness)
        weights = [(other.age + self.age_offset) for other in candidates]
        total = sum(weights)
        probs = [w / total for w in weights]

        # Sample a partner according to the computed distribution
        idx = agent.model.random.choices(range(len(candidates)), weights=probs, k=1)[0]
        partner = candidates[idx]

        # Recombine: weighted average of the two parents' invasion rates
        agent_age = agent.age + self.age_offset
        partner_age = partner.age + self.age_offset
        child_invasion = (
            agent.invasion * agent_age + partner.invasion * partner_age
        ) / (agent_age + partner_age)

        # Mutate
        mutation = agent.model.random.gauss(0, self.sigma)

        return float(np.clip(child_invasion + mutation, 0.0, 1.0))
