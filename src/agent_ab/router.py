"""
Routing strategies for A/B experiments.

ABRouter:                  Random allocation with configurable weights.
ChampionChallengerRouter:  Routes most traffic to the current champion,
                           a small slice to challenger(s).
"""

from __future__ import annotations
import random
from typing import Dict, List, Optional, Sequence, Tuple

from .experiment import Experiment


class ABRouter:
    """Route incoming requests to variants by weighted random sampling.

    Usage::

        router = ABRouter(experiment, weights={"v1": 0.5, "v2": 0.5})
        variant = router.choose()         # "v1" or "v2"
        router.record(variant, success=True)
    """

    def __init__(
        self,
        experiment: Experiment,
        weights: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.experiment = experiment
        self._rng = random.Random(seed)

        variant_names = list(experiment.variants.keys())
        if not variant_names:
            raise ValueError("Experiment has no variants.")

        if weights is None:
            # Uniform distribution
            self._weights = {name: 1.0 / len(variant_names) for name in variant_names}
        else:
            total = sum(weights.values())
            self._weights = {k: v / total for k, v in weights.items()}

    def choose(self) -> str:
        """Return a variant name according to the configured weights."""
        names = list(self._weights.keys())
        probs = [self._weights[n] for n in names]
        r = self._rng.random()
        cumulative = 0.0
        for name, prob in zip(names, probs):
            cumulative += prob
            if r <= cumulative:
                return name
        return names[-1]

    def record(self, variant_name: str, success: bool, score: Optional[float] = None) -> None:
        """Record a result for the chosen variant."""
        self.experiment.record(variant_name, success=success, score=score)

    def stats(self) -> Dict[str, Dict]:
        return {name: v.summary() for name, v in self.experiment.variants.items()}


class ChampionChallengerRouter:
    """Route most traffic to the current champion; a small slice to challengers.

    Usage::

        router = ChampionChallengerRouter(
            experiment,
            champion="v1",
            challenger_fraction=0.1,   # 10% to challengers
        )
        variant = router.choose()
        router.record(variant, success=True)
        # Champion auto-updates when a challenger achieves higher win rate
        router.maybe_promote_challenger()
    """

    def __init__(
        self,
        experiment: Experiment,
        champion: str,
        challenger_fraction: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        self.experiment = experiment
        self.champion = champion
        self.challenger_fraction = challenger_fraction
        self._rng = random.Random(seed)

        challengers = [n for n in experiment.variants if n != champion]
        self._challengers = challengers

    @property
    def challengers(self) -> List[str]:
        return [n for n in self.experiment.variants if n != self.champion]

    def choose(self) -> str:
        """Route traffic: (1 - fraction) to champion, fraction split across challengers."""
        challengers = self.challengers
        if not challengers:
            return self.champion
        r = self._rng.random()
        if r >= self.challenger_fraction:
            return self.champion
        # Pick a random challenger
        return self._rng.choice(challengers)

    def record(self, variant_name: str, success: bool, score: Optional[float] = None) -> None:
        self.experiment.record(variant_name, success=success, score=score)

    def maybe_promote_challenger(self, min_samples: int = 50) -> Optional[str]:
        """Promote a challenger to champion if it significantly beats the current one.

        Returns the new champion name if a promotion happened, else None.
        """
        champ_variant = self.experiment.variants.get(self.champion)
        if champ_variant is None:
            return None

        for name in self.challengers:
            challenger = self.experiment.variants[name]
            if challenger.n < min_samples or champ_variant.n < min_samples:
                continue
            if challenger.win_rate > champ_variant.win_rate:
                self.champion = name
                return name
        return None

    def status(self) -> Dict:
        champ = self.experiment.variants.get(self.champion)
        return {
            "champion": self.champion,
            "champion_win_rate": champ.win_rate if champ else None,
            "challenger_fraction": self.challenger_fraction,
            "challengers": {
                name: self.experiment.variants[name].summary()
                for name in self.challengers
            },
        }
