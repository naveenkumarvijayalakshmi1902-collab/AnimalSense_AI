import csv
from collections import defaultdict, Counter
from pathlib import Path


INPUT = Path(
    r"D:\AnimalSense_AI\outputs\stable_cow_behavior_log.csv"
)

OUTPUT = Path(
    r"D:\AnimalSense_AI\outputs\rolling_anomaly_report.csv"
)

BEHAVIORS = [
    "stand",
    "lying_down",
    "foraging",
    "drinking_water",
    "rumination"
]

# 25 frames = 1 second because the video is 25 FPS.
WINDOW_SIZE = 25

# Require at least 2 windows before comparing
# current behavior with previous behavior.
MIN_HISTORY_WINDOWS = 2

# Maximum number of historical windows retained
HISTORY_WINDOWS = 10


# --------------------------------------------------
# Load observations
# --------------------------------------------------

cow_frames = defaultdict(list)

with INPUT.open(
    "r",
    newline="",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        cow_frames[row["cow_id"]].append({
            "frame": int(row["frame"]),
            "behavior": row["behavior"]
        })


# --------------------------------------------------
# Helper
# --------------------------------------------------

def behavior_distribution(items):

    counts = Counter(
        item["behavior"]
        for item in items
    )

    total = len(items)

    if total == 0:
        return {
            behavior: 0.0
            for behavior in BEHAVIORS
        }

    return {
        behavior:
        (counts[behavior] / total) * 100
        for behavior in BEHAVIORS
    }


# --------------------------------------------------
# Analyze rolling windows
# --------------------------------------------------

reports = []


for cow_id, observations in cow_frames.items():

    observations.sort(
        key=lambda x: x["frame"]
    )

    # Create sequential windows
    windows = []

    for start in range(
        0,
        len(observations),
        WINDOW_SIZE
    ):

        window = observations[
            start:start + WINDOW_SIZE
        ]

        if len(window) < WINDOW_SIZE:
            continue

        windows.append(window)


    if len(windows) <= MIN_HISTORY_WINDOWS:
        reports.append({
            "cow_id": cow_id,
            "window": "final",
            "observations": len(observations),
            "risk_score": 0,
            "risk_level": "INSUFFICIENT_DATA",
            "main_reason": "Not enough temporal windows"
        })

        continue


    history = []


    for window_index, current_window in enumerate(windows):

        current_distribution = behavior_distribution(
            current_window
        )


        # First windows establish the cow's normal pattern.
        if len(history) < MIN_HISTORY_WINDOWS:

            history.append(
                current_distribution
            )

            continue


        # --------------------------------------------------
        # Calculate adaptive historical baseline
        # --------------------------------------------------

        baseline = {
            behavior: 0.0
            for behavior in BEHAVIORS
        }


        for previous in history:

            for behavior in BEHAVIORS:

                baseline[behavior] += (
                    previous[behavior]
                )


        for behavior in BEHAVIORS:

            baseline[behavior] /= len(history)


        # --------------------------------------------------
        # Calculate deviations
        # --------------------------------------------------

        deviations = {}

        for behavior in BEHAVIORS:

            deviations[behavior] = abs(
                current_distribution[behavior]
                -
                baseline[behavior]
            )


        average_deviation = (
            sum(deviations.values())
            /
            len(BEHAVIORS)
        )


        # Convert behavioral deviation to 0-100.
        risk_score = min(
            100,
            average_deviation * 2
        )


        if risk_score >= 60:
            risk_level = "HIGH"

        elif risk_score >= 30:
            risk_level = "MEDIUM"

        else:
            risk_level = "LOW"


        # Largest behavioral change
        largest_behavior = max(
            deviations,
            key=deviations.get
        )

        largest_change = deviations[
            largest_behavior
        ]


        direction = (
            "increased"
            if current_distribution[largest_behavior]
            >
            baseline[largest_behavior]
            else
            "decreased"
        )


        reason = (
            f"{largest_behavior} {direction} "
            f"by {largest_change:.1f}% "
            f"relative to adaptive baseline"
        )


        reports.append({
            "cow_id": cow_id,
            "window": window_index,
            "observations": len(current_window),
            "risk_score": round(
                risk_score,
                2
            ),
            "risk_level": risk_level,
            "main_reason": reason
        })


        # Add current window to adaptive history
        history.append(
            current_distribution
        )


        # Keep only recent history
        if len(history) > HISTORY_WINDOWS:

            history.pop(0)


# --------------------------------------------------
# Save report
# --------------------------------------------------

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

with OUTPUT.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    fieldnames = [
        "cow_id",
        "window",
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
# Display alerts
# --------------------------------------------------

print()
print("==========================================")
print("ROLLING BEHAVIOR ANOMALY ANALYSIS")
print("==========================================")

if not reports:

    print("No temporal windows available.")

else:

    for report in reports:

        print()
        print(
            f"Cow #{report['cow_id']} "
            f"| Window {report['window']}"
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
    f"Report saved to: {OUTPUT}"
)