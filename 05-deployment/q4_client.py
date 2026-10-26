from __future__ import annotations

import json
from urllib.request import Request, urlopen


LEAD = {
    "lead_source": "organic_search",
    "industry": "technology",
    "employment_status": "employed",
    "location": "europe",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0,
    "interaction_count": 7,
    "lead_score": 0.74,
}


def main() -> None:
    request = Request(
        "http://localhost:9696/predict",
        data=json.dumps(LEAD).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        result = json.load(response)
    print(result)
    print(f"conversion_probability (3 decimals) = {result['conversion_probability']:.3f}")


if __name__ == "__main__":
    main()
