from __future__ import annotations


def explain_reasons(per_behaviour: dict[str, float], current: dict[str, float], baseline: dict[str, float], limit: int = 3):
    ranked = sorted(per_behaviour.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    reasons = []
    for behaviour, _ in ranked:
        c = current.get(behaviour, 0.0) * 100
        b = baseline.get(behaviour, 0.0) * 100
        direction = "increased" if c > b else "decreased"
        reasons.append(f"{behaviour.capitalize()} activity {direction} ({abs(c-b):.1f} percentage points)")
    return reasons


def alert_record(timestamp: float, animal_id: str, status: str, score: float, reasons: list[str]) -> dict:
    return {
        "timestamp_seconds": round(timestamp, 3),
        "animal_id": animal_id,
        "status": status,
        "deviation_score": round(score, 3),
        "reasons": reasons,
    }
