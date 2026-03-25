# agent-ab

A/B testing framework for LLM agent prompts and strategies — pure Python, zero dependencies.

**Win rate tracking · Statistical significance · Champion/challenger routing · Persistence**

## Install

```bash
pip install agent-ab
```

## Quick Start

```python
from agent_ab import ExperimentTracker, ABRouter, ChampionChallengerRouter, is_significant

# Create an experiment
tracker = ExperimentTracker()
exp = tracker.create("prompt-experiment", variants=["v1-concise", "v2-detailed"])

# Record outcomes as your agent runs
tracker.record("prompt-experiment", "v1-concise", success=True, latency_ms=320)
tracker.record("prompt-experiment", "v2-detailed", success=False, latency_ms=890)

# Get a report with statistical significance
report = tracker.report("prompt-experiment")
print(report["winner"])      # "v1-concise"
print(report["statistics"])  # {z_score, p_value, significant, ci_95...}

# Route traffic with champion/challenger (90/10 split)
router = ChampionChallengerRouter(exp, champion="v1-concise", challenger_fraction=0.1)
variant = router.choose()           # "v1-concise" ~90% of the time
router.record(variant, success=True)
router.maybe_promote_challenger()   # Auto-promote if challenger beats champion
```

## Features

| Module | What it provides |
|--------|-----------------|
| `experiment` | `Experiment`, `Variant`, `VariantResult` — data model |
| `stats` | `proportion_z_test`, `chi_square_test`, `confidence_interval`, `is_significant` |
| `tracker` | `ExperimentTracker` — manage experiments with optional JSON persistence |
| `router` | `ABRouter` (weighted random), `ChampionChallengerRouter` (auto-promotion) |

## Zero Dependencies

Only the Python standard library (`math`, `random`, `json`, `dataclasses`, `enum`).
