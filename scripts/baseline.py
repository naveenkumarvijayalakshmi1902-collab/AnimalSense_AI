from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IndividualBaseline:
    behaviours: list[str]
    ema_alpha: float = 0.02
    start_time: float | None = None
    observations: int = 0
    distribution: dict[str, float] = field(default_factory=dict)
    frozen: bool = False

    def __post_init__(self) -> None:
        self.distribution = {b: 1.0 / len(self.behaviours) for b in self.behaviours}

    def update(self, current: dict[str, float], timestamp: float, warmup_seconds: float, min_observations: int) -> None:
        if self.start_time is None:
            self.start_time = timestamp
        self.observations += 1
        if self.frozen:
            return
        a = self.ema_alpha
        for b in self.behaviours:
            self.distribution[b] = (1 - a) * self.distribution.get(b, 0.0) + a * float(current.get(b, 0.0))
        if timestamp - self.start_time >= warmup_seconds and self.observations >= min_observations:
            self.frozen = True

    def ready(self) -> bool:
        return self.frozen
