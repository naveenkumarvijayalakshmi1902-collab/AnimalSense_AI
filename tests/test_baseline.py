from scripts.baseline import IndividualBaseline

def test_baseline_ready():
    b = IndividualBaseline(["standing", "walking"], ema_alpha=1.0)
    for i in range(35): b.update({"standing": 1.0, "walking": 0.0}, float(i), 30, 30)
    assert b.ready()
    assert b.distribution["standing"] > 0.9
