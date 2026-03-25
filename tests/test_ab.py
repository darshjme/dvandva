"""Tests for agent-ab library"""
import math
import pytest
from agent_ab.experiment import Experiment, Variant, VariantResult, ExperimentStatus
from agent_ab.stats import (
    proportion_z_test, chi_square_test, confidence_interval, is_significant,
)
from agent_ab.tracker import ExperimentTracker
from agent_ab.router import ABRouter, ChampionChallengerRouter
import tempfile
import os


# ── Variant & Experiment ─────────────────────────────────────────────────────

def test_variant_record_and_win_rate():
    v = Variant(name="v1")
    v.record(success=True)
    v.record(success=True)
    v.record(success=False)
    assert v.n == 3
    assert v.wins == 2
    assert math.isclose(v.win_rate, 2/3)


def test_variant_avg_score():
    v = Variant(name="v1")
    v.record(success=True, score=4.0)
    v.record(success=False, score=2.0)
    assert math.isclose(v.avg_score, 3.0)


def test_variant_win_rate_empty():
    v = Variant(name="empty")
    assert v.win_rate == 0.0
    assert v.avg_score is None


def test_experiment_add_variant_and_record():
    exp = Experiment(name="test-exp")
    exp.add_variant("a")
    exp.add_variant("b")
    exp.record("a", success=True)
    exp.record("b", success=False)
    assert exp.variants["a"].wins == 1
    assert exp.variants["b"].wins == 0


def test_experiment_record_unknown_variant():
    exp = Experiment(name="x")
    exp.add_variant("v1")
    with pytest.raises(KeyError):
        exp.record("nonexistent", success=True)


def test_experiment_winner():
    exp = Experiment(name="winner-test")
    exp.add_variant("slow")
    exp.add_variant("fast")
    for _ in range(3):
        exp.record("slow", success=True)
    for _ in range(9):
        exp.record("fast", success=True)
    exp.record("slow", success=False)
    assert exp.winner() == "fast"


def test_experiment_leaderboard_sorted():
    exp = Experiment(name="lb")
    exp.add_variant("bad")
    exp.add_variant("good")
    exp.record("bad", success=False)
    exp.record("good", success=True)
    lb = exp.leaderboard()
    assert lb[0]["name"] == "good"


def test_experiment_conclude():
    exp = Experiment(name="conclude-test")
    exp.conclude()
    assert exp.status == ExperimentStatus.CONCLUDED
    assert exp.concluded_at is not None


def test_experiment_roundtrip():
    exp = Experiment(name="rt", description="test")
    exp.add_variant("v1", config={"prompt": "hello"})
    exp.record("v1", success=True, score=5.0, latency_ms=120.0)
    exp.finish = exp.conclude
    d = exp.to_dict()
    restored = Experiment.from_dict(d)
    assert restored.id == exp.id
    assert restored.variants["v1"].n == 1
    assert restored.variants["v1"].results[0].score == 5.0


# ── Stats ────────────────────────────────────────────────────────────────────

def test_proportion_z_test_equal():
    # Equal proportions → z=0, p=1
    z, p = proportion_z_test(100, 50, 100, 50)
    assert math.isclose(z, 0.0, abs_tol=1e-9)
    assert p > 0.9


def test_proportion_z_test_different():
    # 90% vs 50% with 200 samples each → should be highly significant
    z, p = proportion_z_test(200, 180, 200, 100)
    assert abs(z) > 5
    assert p < 0.001


def test_proportion_z_test_zero_samples():
    z, p = proportion_z_test(0, 0, 100, 50)
    assert z == 0.0
    assert p == 1.0


def test_chi_square_equal():
    chi2, p = chi_square_test(100, 50, 100, 50)
    assert math.isclose(chi2, 0.0, abs_tol=1e-6)
    assert p > 0.9


def test_chi_square_different():
    chi2, p = chi_square_test(200, 180, 200, 100)
    assert chi2 > 25
    assert p < 0.001


def test_confidence_interval_known():
    # 50/100 = 50% → CI should contain 0.5
    lo, hi = confidence_interval(100, 50)
    assert lo < 0.5 < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_confidence_interval_zero_samples():
    lo, hi = confidence_interval(0, 0)
    assert lo == 0.0
    assert hi == 1.0


def test_is_significant_true():
    # Very different, large samples
    assert is_significant(200, 180, 200, 80, alpha=0.05, min_samples=30) is True


def test_is_significant_small_samples():
    # Not enough samples
    assert is_significant(10, 8, 10, 4, min_samples=30) is False


def test_is_significant_equal():
    # Equal rates → not significant
    assert is_significant(1000, 500, 1000, 500, alpha=0.05) is False


# ── ExperimentTracker ────────────────────────────────────────────────────────

def test_tracker_create_and_record():
    tracker = ExperimentTracker()
    tracker.create("exp1", variants=["a", "b"])
    tracker.record("exp1", "a", success=True)
    tracker.record("exp1", "b", success=False)
    exp = tracker.get("exp1")
    assert exp.variants["a"].wins == 1


def test_tracker_report():
    tracker = ExperimentTracker()
    tracker.create("report-test", variants=["v1", "v2"])
    for _ in range(10):
        tracker.record("report-test", "v1", success=True)
    for _ in range(10):
        tracker.record("report-test", "v2", success=False)
    report = tracker.report("report-test")
    assert report["winner"] == "v1"
    assert report["total_observations"] == 20


def test_tracker_list_experiments():
    tracker = ExperimentTracker()
    tracker.create("e1", variants=["a"])
    tracker.create("e2", variants=["b"])
    assert set(tracker.list_experiments()) >= {"e1", "e2"}


def test_tracker_unknown_experiment():
    tracker = ExperimentTracker()
    with pytest.raises(KeyError):
        tracker.record("nope", "v1", success=True)


def test_tracker_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        t1 = ExperimentTracker(persist_dir=tmpdir)
        t1.create("persistent", variants=["x", "y"])
        t1.record("persistent", "x", success=True)
        t1.record("persistent", "y", success=False)

        t2 = ExperimentTracker(persist_dir=tmpdir)
        exp = t2.get("persistent")
        assert exp is not None
        assert exp.variants["x"].wins == 1


# ── ABRouter ─────────────────────────────────────────────────────────────────

def test_ab_router_uniform_distribution():
    exp = Experiment(name="ab-uniform")
    exp.add_variant("a")
    exp.add_variant("b")
    router = ABRouter(exp, seed=42)
    counts = {"a": 0, "b": 0}
    for _ in range(1000):
        choice = router.choose()
        counts[choice] += 1
    # With uniform weights, both should be ~500 ± 50
    assert 400 < counts["a"] < 600
    assert 400 < counts["b"] < 600


def test_ab_router_weighted():
    exp = Experiment(name="ab-weighted")
    exp.add_variant("a")
    exp.add_variant("b")
    router = ABRouter(exp, weights={"a": 0.9, "b": 0.1}, seed=42)
    counts = {"a": 0, "b": 0}
    for _ in range(1000):
        counts[router.choose()] += 1
    assert counts["a"] > counts["b"]


def test_ab_router_record():
    exp = Experiment(name="ab-record")
    exp.add_variant("x")
    router = ABRouter(exp, seed=1)
    router.record("x", success=True)
    assert exp.variants["x"].wins == 1


def test_ab_router_no_variants_raises():
    exp = Experiment(name="empty")
    with pytest.raises(ValueError):
        ABRouter(exp)


# ── ChampionChallengerRouter ─────────────────────────────────────────────────

def test_cc_router_champion_gets_most_traffic():
    exp = Experiment(name="cc")
    exp.add_variant("champion")
    exp.add_variant("challenger")
    router = ChampionChallengerRouter(exp, champion="champion", challenger_fraction=0.1, seed=42)
    counts = {"champion": 0, "challenger": 0}
    for _ in range(1000):
        counts[router.choose()] += 1
    assert counts["champion"] > 850


def test_cc_router_maybe_promote():
    exp = Experiment(name="cc-promote")
    exp.add_variant("champ")
    exp.add_variant("better")
    # Give challenger a higher win rate with enough samples
    for _ in range(60):
        exp.record("champ", success=True)
    for _ in range(40):
        exp.record("champ", success=False)
    for _ in range(60):
        exp.record("better", success=True)
    for _ in range(10):
        exp.record("better", success=False)

    router = ChampionChallengerRouter(exp, champion="champ", seed=42)
    promoted = router.maybe_promote_challenger(min_samples=50)
    assert promoted == "better"
    assert router.champion == "better"


def test_cc_router_no_promotion_insufficient_samples():
    exp = Experiment(name="cc-no-promote")
    exp.add_variant("champ")
    exp.add_variant("newbie")
    exp.record("champ", success=True)
    exp.record("newbie", success=True)
    router = ChampionChallengerRouter(exp, champion="champ")
    result = router.maybe_promote_challenger(min_samples=50)
    assert result is None
    assert router.champion == "champ"
