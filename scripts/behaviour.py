from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from math import hypot


@dataclass
class TrackObservation:
    frame: int
    timestamp: float
    cx: float
    cy: float
    width: float
    height: float
    confidence: float
    movement: float


@dataclass
class BehaviourEngine:
    config: dict
    observation_history: deque = field(default_factory=lambda: deque(maxlen=300))
    behaviour_history: deque = field(default_factory=lambda: deque(maxlen=300))

    def _inside(self, x: float, y: float, zone: list[float]) -> bool:
        x1, y1, x2, y2 = zone
        return x1 <= x <= x2 and y1 <= y <= y2

    def infer(self, frame: int, timestamp: float, bbox: tuple[float, float, float, float], confidence: float):
        x1, y1, x2, y2 = bbox
        width = max(x2 - x1, 1.0)
        height = max(y2 - y1, 1.0)
        fw = float(self.config.get("frame_width", 1.0))
        fh = float(self.config.get("frame_height", 1.0))
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        nx, ny = cx / fw, cy / fh
        movement = 0.0
        if self.observation_history:
            prev = self.observation_history[-1]
            movement = hypot(cx - prev.cx, cy - prev.cy) / max(hypot(fw, fh), 1.0)
        self.observation_history.append(TrackObservation(frame, timestamp, cx, cy, width, height, confidence, movement))

        walking = float(self.config.get("walking_motion_threshold", 0.012))
        lying = float(self.config.get("lying_aspect_threshold", 0.95))
        rumination = float(self.config.get("rumination_motion_max", 0.006))
        if self._inside(nx, ny, self.config.get("drinking_zone", [0, 0, 0.35, 0.35])) and movement < walking:
            label, bconf = "drinking", min(0.95, 0.60 + confidence * 0.35)
        elif self._inside(nx, ny, self.config.get("feeding_zone", [0, 0.55, 1, 1])) and movement < walking:
            label, bconf = "feeding", min(0.95, 0.58 + confidence * 0.35)
        elif movement >= walking:
            label, bconf = "walking", min(0.98, 0.65 + min(movement / max(walking, 1e-6), 3.0) * 0.08)
        elif width / height >= lying:
            label, bconf = "lying", min(0.90, 0.62 + confidence * 0.25)
        elif movement <= rumination:
            label, bconf = "ruminating", min(0.85, 0.55 + confidence * 0.25)
        else:
            label, bconf = "standing", min(0.90, 0.60 + confidence * 0.25)
        self.behaviour_history.append((timestamp, label, float(bconf)))
        return label, float(bconf), {"movement": movement, "aspect_ratio": width / height, "cx": nx, "cy": ny}

    def distribution(self, now: float, window_seconds: float, behaviours: list[str]) -> dict[str, float]:
        recent = [x for x in self.behaviour_history if now - x[0] <= window_seconds]
        c = Counter(x[1] for x in recent)
        total = max(len(recent), 1)
        return {b: c.get(b, 0) / total for b in behaviours}
