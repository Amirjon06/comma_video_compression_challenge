"""Quantize a checkpoint into archive.zip and report where the bytes went."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import torch

from .codec import encode_model, write_archive
from .config import REPO_ROOT

ORIGINAL_BYTES = 37_545_489


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weight-bits", type=int, default=6)
    ap.add_argument("--embed-bits", type=int, default=6)
    args = ap.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu")
    write_archive(
        state, args.out, weight_bits=args.weight_bits, embed_bits=args.embed_bits
    )

    size = args.out.stat().st_size
    rate = size / ORIGINAL_BYTES
    print(f"archive: {size:,} bytes")
    print(f"rate: {rate:.6f}   rate term: {25 * rate:.4f}")

    with zipfile.ZipFile(args.out) as zf:
        payload = zf.read("p")
    print(f"payload: {len(payload):,} bytes, zip overhead {size - len(payload)} bytes")


if __name__ == "__main__":
    main()
