"""Download and verify the UCI Bank Marketing dataset (bank-full.csv).

Source: https://archive.ics.uci.edu/dataset/222/bank+marketing
License: CC BY 4.0 (Moro, Rita & Cortez, 2014).
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
EXPECTED_SHA256 = "d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d"
DATA_DIR = Path(__file__).resolve().parent
OUTPUT = DATA_DIR / "bank-full.csv"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        digest = sha256(OUTPUT.read_bytes())
        if digest == EXPECTED_SHA256:
            print(f"Dataset already present and verified: {OUTPUT}")
            return
        print("Existing file has a different checksum; re-downloading.")

    print(f"Downloading {URL}")
    with urllib.request.urlopen(URL) as response:
        outer = zipfile.ZipFile(io.BytesIO(response.read()))

    with outer.open("bank.zip") as inner_file:
        inner = zipfile.ZipFile(io.BytesIO(inner_file.read()))

    content = inner.read("bank-full.csv")
    digest = sha256(content)
    if digest != EXPECTED_SHA256:
        raise SystemExit(
            f"Checksum mismatch: got {digest}, expected {EXPECTED_SHA256}"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(content)
    print(f"Saved {OUTPUT} ({len(content):,} bytes)")
    print(f"sha256 = {digest}")


if __name__ == "__main__":
    main()
