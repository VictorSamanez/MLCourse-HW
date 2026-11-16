"""Small client that verifies the running API."""

from __future__ import annotations

import json
import sys
from urllib.request import Request, urlopen

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9696/predict"

CLIENT = {
    "age": 41,
    "job": "technician",
    "marital": "married",
    "education": "secondary",
    "default": "no",
    "balance": 1200,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown",
}


def main() -> None:
    request = Request(
        URL,
        data=json.dumps(CLIENT).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        result = json.load(response)

    assert "subscription_probability" in result
    assert 0.0 <= result["subscription_probability"] <= 1.0
    assert isinstance(result["subscription"], bool)
    print(result)


if __name__ == "__main__":
    main()
