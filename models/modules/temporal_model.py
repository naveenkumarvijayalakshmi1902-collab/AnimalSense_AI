from __future__ import annotations

import numpy as np


class TemporalWindow:
    def __init__(self, max_len: int = 60) -> None:
        self.max_len = max(5, int(max_len))
        self.values: list[np.ndarray] = []

    def append(self, vector: np.ndarray) -> None:
        self.values.append(np.asarray(vector, dtype=float))
        if len(self.values) > self.max_len:
            self.values.pop(0)

    def mean(self) -> np.ndarray:
        return np.mean(np.stack(self.values), axis=0) if self.values else np.zeros(1)
