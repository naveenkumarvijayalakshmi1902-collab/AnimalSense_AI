from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DeviationEngine:
    deviation_threshold: float = 45.0
    warning_threshold: float = 60.0
    critical_threshold: float = 80.0
    persistence_windows: int = 5
    consecutive_high: dict[str, int] = field(default_factory=dict)

    def score(self, current: dict[str, float], baseline: dict[str, float]):
        per = {b: abs(current.get(b, 0.0) - baseline.get(b, 0.0)) * 100 for b in baseline}
        l1 = sum(abs(current.get(b, 0.0) - baseline.get(b, 0.0)) for b in baseline)
        return min(100.0, l1 / 2.0 * 100.0), per

    def status(self, animal_id: str, score: float, ready: bool):
        if not ready:
            self.consecutive_high[animal_id] = 0
            return "INSUFFICIENT HISTORY", False
        if score >= self.warning_threshold:
            self.consecutive_high[animal_id] = self.consecutive_high.get(animal_id, 0) + 1
        else:
            self.consecutive_high[animal_id] = 0
        n = self.consecutive_high[animal_id]
        if score >= self.critical_threshold and n >= self.persistence_windows:
            return "EARLY WARNING", True
        if score >= self.warning_threshold and n >= self.persistence_windows:
            return "SIGNIFICANT DEVIATION", True
        if score >= self.deviation_threshold:
            return "BEHAVIOUR CHANGING", False
        return "NORMAL", False
