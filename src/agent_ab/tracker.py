"""
ExperimentTracker: manage multiple A/B experiments with optional persistence.
"""

from __future__ import annotations
import json
import os
from typing import Dict, List, Optional

from .experiment import Experiment, Variant, VariantResult
from .stats import proportion_z_test, is_significant, confidence_interval


class ExperimentTracker:
    """Central tracker for all A/B experiments.

    Usage::

        tracker = ExperimentTracker()
        exp = tracker.create("prompt-ab-test", variants=["v1", "v2"])

        # Record outcomes
        tracker.record("prompt-ab-test", "v1", success=True)
        tracker.record("prompt-ab-test", "v2", success=False)

        # Analyze
        report = tracker.report("prompt-ab-test")
        winner = tracker.winner("prompt-ab-test")
    """

    def __init__(self, persist_dir: Optional[str] = None) -> None:
        self._experiments: Dict[str, Experiment] = {}
        self._persist_dir = persist_dir
        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            self._load_all()

    def create(
        self,
        name: str,
        variants: Optional[List[str]] = None,
        description: str = "",
    ) -> Experiment:
        """Create and register a new experiment."""
        exp = Experiment(name=name, description=description)
        for vname in (variants or []):
            exp.add_variant(vname)
        self._experiments[name] = exp
        if self._persist_dir:
            self._save(exp)
        return exp

    def get(self, name: str) -> Optional[Experiment]:
        return self._experiments.get(name)

    def record(
        self,
        experiment_name: str,
        variant_name: str,
        success: bool,
        score: Optional[float] = None,
        latency_ms: Optional[float] = None,
    ) -> VariantResult:
        """Record a result for a specific experiment+variant."""
        exp = self._experiments.get(experiment_name)
        if exp is None:
            raise KeyError(f"Experiment {experiment_name!r} not found.")
        result = exp.record(variant_name, success=success, score=score, latency_ms=latency_ms)
        if self._persist_dir:
            self._save(exp)
        return result

    def report(self, experiment_name: str) -> Dict:
        """Return a detailed analysis report for an experiment."""
        exp = self._experiments.get(experiment_name)
        if exp is None:
            raise KeyError(f"Experiment {experiment_name!r} not found.")

        leaderboard = exp.leaderboard()
        variants = list(exp.variants.values())

        sig_results = {}
        if len(variants) >= 2:
            a, b = variants[0], variants[1]
            z, p = proportion_z_test(a.n, a.wins, b.n, b.wins)
            sig = is_significant(a.n, a.wins, b.n, b.wins)
            ci_a = confidence_interval(a.n, a.wins)
            ci_b = confidence_interval(b.n, b.wins)
            sig_results = {
                "z_score": z,
                "p_value": p,
                "significant": sig,
                f"{a.name}_ci_95": ci_a,
                f"{b.name}_ci_95": ci_b,
            }

        return {
            "experiment": experiment_name,
            "status": exp.status.value,
            "winner": exp.winner(),
            "leaderboard": leaderboard,
            "statistics": sig_results,
            "total_observations": sum(v.n for v in exp.variants.values()),
        }

    def winner(self, experiment_name: str) -> Optional[str]:
        exp = self._experiments.get(experiment_name)
        return exp.winner() if exp else None

    def list_experiments(self) -> List[str]:
        return list(self._experiments.keys())

    def _save(self, exp: Experiment) -> None:
        path = os.path.join(self._persist_dir, f"{exp.name}.json")  # type: ignore
        with open(path, "w") as fp:
            json.dump(exp.to_dict(), fp, indent=2)

    def _load_all(self) -> None:
        for fname in os.listdir(self._persist_dir):  # type: ignore
            if fname.endswith(".json"):
                with open(os.path.join(self._persist_dir, fname)) as fp:  # type: ignore
                    data = json.load(fp)
                exp = Experiment.from_dict(data)
                self._experiments[exp.name] = exp
