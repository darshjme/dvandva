"""
agent-ab: A/B testing framework for LLM agent prompts and strategies.

Provides experiment tracking, win rate computation, statistical significance
testing, and champion/challenger routing. Zero dependencies, pure Python.
"""

from .experiment import Experiment, Variant, VariantResult
from .tracker import ExperimentTracker
from .stats import (
    proportion_z_test,
    chi_square_test,
    confidence_interval,
    is_significant,
)
from .router import ABRouter, ChampionChallengerRouter

__all__ = [
    "Experiment",
    "Variant",
    "VariantResult",
    "ExperimentTracker",
    "proportion_z_test",
    "chi_square_test",
    "confidence_interval",
    "is_significant",
    "ABRouter",
    "ChampionChallengerRouter",
]

__version__ = "0.1.0"
