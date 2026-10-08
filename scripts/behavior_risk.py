import csv
from collections import defaultdict
from pathlib import Path


LOG_FILE = Path(
    r"D:\AnimalSense_AI\outputs\stable_cow_behavior_log.csv"
)

BASELINE_FILE = Path(
    r"D:\AnimalSense_AI\outputs\adaptive_baseline.csv"
)

OUTPUT_FILE = Path(
    r"D:\AnimalSense_AI\outputs\behavior_risk_report.csv"
)


BEHAVIORS = [
    "stand",
    "lying_down",
    "foraging",
    "drinking_water",
    "rumination"
]


# --------------------------------------------------
# Load adaptive baselines
# --------------------------------------------------

baselines = {}

with BASELINE_FILE.open(
    "r",
    newline="",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        cow_id = row["cow_id"]

        baselines[cow_id] = {
            "status": row["status"],
            "stand": float(row["stand_pct"]),
            "lying_down": float(row["lying_down_pct"]),
            "foraging": float(row["foraging_pct"]),
            "drinking_water": float(row["drinking_water_pct"]),
            "rumination": float(row["rumination_pct"])
        }


# --------------------------------------------------
# Load current observations
# --------------------------------------------------

observations = defaultdict(list)

with LOG_FILE.open(
    "r",
    newline="",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        observations[row["cow_id"]].append(
            row["behavior"]
        )


# --------------------------------------------------
# Calculate behavioral deviation
# --------------------------------------------------

reports = []


for cow_id, behaviors in observations.items():

    if cow_id not in baselines:
        continue

    baseline = baselines[cow_id]

    total = len(behaviors)

    # Not enough information
    if (
        total < 20
        or baseline["status"] != "BASELINE_READY"
    ):

        reports.append({
            "cow_id": cow_id,
            "observations": total,
            "risk_score": 0,
            "risk_level": "INSUFFICIENT_DATA",
            "main_reason": "Not enough observations for reliable baseline"
        })

        continue


    # Current behavior distribution
    counts = defaultdict(int)

    for behavior in behaviors:
        counts[behavior] += 1


    current = {
        behavior:
        (counts[behavior] / total) * 100
        for behavior in BEHAVIORS
    }


    # --------------------------------------------------
    # Calculate deviation
    # --------------------------------------------------

    deviations = {}

    for behavior in BEHAVIORS:

        deviations[behavior] = abs(
            current[behavior]
            -
            baseline[behavior]
        )


    # Average behavioral deviation
    average_deviation = (
        sum(deviations.values())
        /
        len(BEHAVIORS)
    )


    # Convert deviation to 0-100 risk score
    risk_score = min(
        100,
        average_deviation * 2
    )


    # Risk category
    if risk_score >= 60:
        risk_level = "HIGH"

    elif risk_score >= 30:
        risk_level = "MEDIUM"

    else:
        risk_level = "LOW"


    # Find largest deviation
    largest_behavior = max(
        deviations,
        key=deviations.get
    )

    largest_change = deviations[
        largest_behavior
    ]


    reason = (
        f"{largest_behavior} deviation "
        f"{largest_change:.1f}% from baseline"
    )


    reports.append({
        "cow_id": cow_id,
        "observations": total,
        "risk_score": round(risk_score, 2),
        "risk_level": risk_level,
        "main_reason": reason
    })


# --------------------------------------------------
# Save report
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


with OUTPUT_FILE.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    fieldnames = [
        "cow_id",
        "observations",
        "risk_score",
        "risk_level",
        "main_reason"
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(reports)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print()
print("==========================================")
print("ANIMAL SENSE BEHAVIOR RISK ANALYSIS")
print("==========================================")

for report in reports:

    print()
    print(
        f"Cow #{report['cow_id']}"
    )

    print(
        f"Observations: "
        f"{report['observations']}"
    )

    print(
        f"Risk Score: "
        f"{report['risk_score']}/100"
    )

    print(
        f"Risk Level: "
        f"{report['risk_level']}"
    )

    print(
        f"Reason: "
        f"{report['main_reason']}"
    )


print()
print(
    f"Report saved to: {OUTPUT_FILE}"
)