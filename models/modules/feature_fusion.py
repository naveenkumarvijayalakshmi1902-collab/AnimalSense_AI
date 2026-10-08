from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FusionWeights:
    movement: float = 0.30
    behaviour_distribution: float = 0.40
    geometry: float = 0.15
    persistence: float = 0.15

    def normalized(self) -> "FusionWeights":
        total = self.movement + self.behaviour_distribution + self.geometry + self.persistence
        if total <= 0:
            return FusionWeights(0.25, 0.25, 0.25, 0.25)
        return FusionWeights(*(v / total for v in (self.movement, self.behaviour_distribution, self.geometry, self.persistence)))


def weighted_deviation(movement: float, behaviour: float, geometry: float, persistence: float, weights: FusionWeights | None = None) -> float:
    w = (weights or FusionWeights()).normalized()
    v = np.clip(np.asarray([movement, behaviour, geometry, persistence], dtype=float), 0.0, 1.0)
    c = np.asarray([w.movement, w.behaviour_distribution, w.geometry, w.persistence])
    return float(np.dot(v, c) * 100.0)
