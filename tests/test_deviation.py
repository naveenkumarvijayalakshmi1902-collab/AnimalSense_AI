from scripts.deviation import DeviationEngine

def test_zero_deviation():
    e = DeviationEngine()
    score, _ = e.score({"standing": 0.5, "walking": 0.5}, {"standing": 0.5, "walking": 0.5})
    assert score == 0.0
