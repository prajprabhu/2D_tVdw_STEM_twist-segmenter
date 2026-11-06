"""
Download the sample HAADF-STEM image used by the apps and example notebooks.

    python examples/download_sample_data.py

Fetches Moire.npz (a 512x512 bilayer moire HAADF-STEM image, ~0.5 MB) from
the Gomb-Net repository (https://github.com/AustinHouston/Gomb-Net) into
examples/data/moire.npz. The file is not redistributed with this repo: it
belongs to the Gomb-Net authors -- see examples/data/README.md for the
citation.

Standard library only, so it runs on Windows, macOS and Linux before any
other dependency is installed.
"""

import os
import sys
import urllib.request

URL = "https://raw.githubusercontent.com/AustinHouston/Gomb-Net/main/Moire.npz"
EXPECTED_BYTES = 524_894
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "moire.npz")


def main() -> int:
    if os.path.isfile(DEST) and os.path.getsize(DEST) == EXPECTED_BYTES:
        print(f"Already downloaded: {DEST}")
        return 0

    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    tmp = DEST + ".part"
    print(f"Downloading {URL}\n        -> {DEST}")
    try:
        urllib.request.urlretrieve(URL, tmp)
    except OSError as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1

    size = os.path.getsize(tmp)
    if size != EXPECTED_BYTES:
        os.remove(tmp)
        print(f"Unexpected file size ({size} bytes, expected {EXPECTED_BYTES}); "
              "the upstream file may have changed. Nothing was saved.", file=sys.stderr)
        return 1

    os.replace(tmp, DEST)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
