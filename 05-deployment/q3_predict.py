from __future__ import annotations

import pickle
import sys
from pathlib import Path

HW_DIR = (
    Path(__file__).resolve().parent.parent
    / "teorico"
    / "cohorts"
    / "2026"
    / "homework"
    / "05-deployment"
)
sys.path.insert(0, str(HW_DIR))

from model import normalize_record  # noqa: E402

LEAD = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41,
}


def main() -> None:
    with (HW_DIR / "pipeline.bin").open("rb") as model_file:
        pipeline = pickle.load(model_file)

    record = normalize_record(LEAD)
    probability = float(pipeline.predict_proba([record])[0, 1])
    print(f"conversion_probability = {probability:.6f}")
    print(f"rounded (3 decimals)   = {probability:.3f}")
    print(f"conversion             = {probability >= 0.5}")


if __name__ == "__main__":
    main()
