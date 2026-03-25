"""
Experiment and Variant data models.
"""

from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ExperimentStatus(str, Enum):
    RUNNING = "running"
    PAUSED = "paused"
    CONCLUDED = "concluded"


@dataclass
class VariantResult:
    """A single observation (win/loss/score) for a variant."""
    variant_name: str
    success: bool
    score: Optional[float] = None          # Numeric score (e.g. 1-5 rating)
    latency_ms: Optional[float] = None     # Response latency
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "variant_name": self.variant_name,
            "success": self.success,
            "score": self.score,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VariantResult":
        return cls(
            id=d["id"],
            variant_name=d["variant_name"],
            success=d["success"],
            score=d.get("score"),
            latency_ms=d.get("latency_ms"),
            metadata=d.get("metadata", {}),
            timestamp=d.get("timestamp", time.time()),
        )


@dataclass
class Variant:
    """An experimental variant (e.g., a prompt or strategy)."""

    name: str
    config: Any = None                    # Prompt string, strategy dict, etc.
    description: str = ""
    results: List[VariantResult] = field(default_factory=list)

    def record(self, success: bool, score: Optional[float] = None,
               latency_ms: Optional[float] = None, metadata: Optional[Dict] = None) -> VariantResult:
        """Record one observation for this variant."""
        result = VariantResult(
            variant_name=self.name,
            success=success,
            score=score,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )
        self.results.append(result)
        return result

    @property
    def n(self) -> int:
        """Total observations."""
        return len(self.results)

    @property
    def wins(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def win_rate(self) -> float:
        """Win rate in [0, 1]. Returns 0.0 if no observations."""
        if self.n == 0:
            return 0.0
        return self.wins / self.n

    @property
    def avg_score(self) -> Optional[float]:
        scores = [r.score for r in self.results if r.score is not None]
        return sum(scores) / len(scores) if scores else None

    @property
    def avg_latency_ms(self) -> Optional[float]:
        lats = [r.latency_ms for r in self.results if r.latency_ms is not None]
        return sum(lats) / len(lats) if lats else None

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n": self.n,
            "wins": self.wins,
            "win_rate": self.win_rate,
            "avg_score": self.avg_score,
            "avg_latency_ms": self.avg_latency_ms,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "config": self.config,
            "description": self.description,
            "results": [r.to_dict() for r in self.results],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Variant":
        v = cls(
            name=d["name"],
            config=d.get("config"),
            description=d.get("description", ""),
        )
        v.results = [VariantResult.from_dict(r) for r in d.get("results", [])]
        return v


@dataclass
class Experiment:
    """An A/B experiment comparing multiple variants."""

    name: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    variants: Dict[str, Variant] = field(default_factory=dict)
    status: ExperimentStatus = ExperimentStatus.RUNNING
    created_at: float = field(default_factory=time.time)
    concluded_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_variant(self, name: str, config: Any = None, description: str = "") -> Variant:
        """Register a new variant."""
        v = Variant(name=name, config=config, description=description)
        self.variants[name] = v
        return v

    def record(self, variant_name: str, success: bool, score: Optional[float] = None,
               latency_ms: Optional[float] = None, metadata: Optional[Dict] = None) -> VariantResult:
        """Record a result for a named variant."""
        if variant_name not in self.variants:
            raise KeyError(f"Unknown variant: {variant_name!r}. Add it first with add_variant().")
        return self.variants[variant_name].record(
            success=success, score=score, latency_ms=latency_ms, metadata=metadata
        )

    def conclude(self) -> None:
        self.status = ExperimentStatus.CONCLUDED
        self.concluded_at = time.time()

    def winner(self) -> Optional[str]:
        """Return the variant with the highest win rate (None if no data)."""
        best_name = None
        best_rate = -1.0
        for name, v in self.variants.items():
            if v.n > 0 and v.win_rate > best_rate:
                best_rate = v.win_rate
                best_name = name
        return best_name

    def leaderboard(self) -> List[Dict[str, Any]]:
        """Return variants sorted by win rate descending."""
        summaries = [v.summary() for v in self.variants.values()]
        return sorted(summaries, key=lambda x: x["win_rate"], reverse=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "created_at": self.created_at,
            "concluded_at": self.concluded_at,
            "metadata": self.metadata,
            "variants": {k: v.to_dict() for k, v in self.variants.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Experiment":
        exp = cls(
            id=d["id"],
            name=d["name"],
            description=d.get("description", ""),
            status=ExperimentStatus(d.get("status", "running")),
            created_at=d.get("created_at", time.time()),
            concluded_at=d.get("concluded_at"),
            metadata=d.get("metadata", {}),
        )
        for name, vdata in d.get("variants", {}).items():
            exp.variants[name] = Variant.from_dict(vdata)
        return exp
