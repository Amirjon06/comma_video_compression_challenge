"""Turn a trained model into the archive, and back.

Weights are quantized to a small number of bits with one scale per output
channel, then arithmetic coded. Tensors are grouped so the frequency tables
cost a few hundred bytes instead of a few kilobytes: convolution weights,
biases and norms, and the frame latents each get one table, because their
value distributions are nothing alike.
"""

from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path

import numpy as np
import torch

from . import rangecoder as rc

MAGIC = b"HNM1"
DEFAULT_WEIGHT_BITS = 6
DEFAULT_EMBED_BITS = 6


def _group_of(name: str) -> str:
    if name.startswith("embed"):
        return "embed"
    if name.endswith(".bias"):
        return "bias"
    return "weight"


def _quantize(values: np.ndarray, bits: int, per_channel: bool):
    """Symmetric uniform quantization. Returns codes in [0, 2**bits) and scales."""
    levels = (1 << bits) - 1
    half = levels // 2
    flat = values.reshape(values.shape[0], -1) if per_channel and values.ndim > 1 else values.reshape(1, -1)
    peak = np.abs(flat).max(axis=1)
    peak = np.maximum(peak, 1e-8).astype(np.float32)
    scales = (peak / half).astype(np.float16).astype(np.float32)
    codes = np.rint(flat / scales[:, None]).clip(-half, half - 1) + half
    return codes.astype(np.int64).reshape(-1), scales.astype(np.float16)


def _dequantize(codes: np.ndarray, scales: np.ndarray, shape, bits: int, per_channel: bool):
    levels = (1 << bits) - 1
    half = levels // 2
    rows = shape[0] if per_channel and len(shape) > 1 else 1
    grid = codes.reshape(rows, -1).astype(np.float32) - half
    return (grid * scales.astype(np.float32)[:, None]).reshape(shape)


def encode_model(
    state: dict[str, torch.Tensor],
    weight_bits: int = DEFAULT_WEIGHT_BITS,
    embed_bits: int = DEFAULT_EMBED_BITS,
) -> bytes:
    entries, scale_parts = [], []
    buckets: dict[str, list[np.ndarray]] = {"weight": [], "bias": [], "embed": []}

    for name, tensor in state.items():
        values = tensor.detach().cpu().float().numpy()
        group = _group_of(name)
        bits = embed_bits if group == "embed" else weight_bits
        per_channel = values.ndim > 1
        codes, scales = _quantize(values, bits, per_channel)
        buckets[group].append(codes)
        scale_parts.append(scales.tobytes())
        entries.append(
            {
                "name": name,
                "shape": list(values.shape),
                "group": group,
                "bits": bits,
                "per_channel": bool(per_channel),
                "count": int(codes.size),
                "scales": int(scales.size),
            }
        )

    payload = bytearray()
    tables: dict[str, list[int]] = {}
    streams: dict[str, bytes] = {}
    for group, chunks in buckets.items():
        if not chunks:
            continue
        symbols = np.concatenate(chunks)
        alphabet = 1 << max(e["bits"] for e in entries if e["group"] == group)
        counts = rc.build_table(symbols, alphabet)
        tables[group] = counts.tolist()
        streams[group] = rc.encode(symbols, counts)

    header = json.dumps({"entries": entries, "tables": tables}, separators=(",", ":")).encode()
    payload += MAGIC
    payload += struct.pack("<I", len(header))
    payload += header
    payload += b"".join(scale_parts)
    for group in ("weight", "bias", "embed"):
        blob = streams.get(group, b"")
        payload += struct.pack("<I", len(blob))
        payload += blob
    return bytes(payload)


def decode_model(blob: bytes) -> dict[str, torch.Tensor]:
    if blob[:4] != MAGIC:
        raise ValueError("not an HNM1 payload")
    header_len = struct.unpack_from("<I", blob, 4)[0]
    header = json.loads(blob[8 : 8 + header_len])
    entries = header["entries"]
    tables = {k: np.asarray(v, dtype=np.int64) for k, v in header["tables"].items()}

    cursor = 8 + header_len
    scales: dict[str, np.ndarray] = {}
    for entry in entries:
        size = entry["scales"] * 2
        scales[entry["name"]] = np.frombuffer(blob[cursor : cursor + size], dtype=np.float16)
        cursor += size

    decoded: dict[str, np.ndarray] = {}
    for group in ("weight", "bias", "embed"):
        length = struct.unpack_from("<I", blob, cursor)[0]
        cursor += 4
        stream = blob[cursor : cursor + length]
        cursor += length
        total = sum(e["count"] for e in entries if e["group"] == group)
        if total == 0:
            continue
        decoded[group] = rc.decode(stream, tables[group], total)

    offsets = {group: 0 for group in decoded}
    state: dict[str, torch.Tensor] = {}
    for entry in entries:
        group = entry["group"]
        start = offsets[group]
        end = start + entry["count"]
        offsets[group] = end
        values = _dequantize(
            decoded[group][start:end],
            scales[entry["name"]],
            tuple(entry["shape"]),
            entry["bits"],
            entry["per_channel"],
        )
        state[entry["name"]] = torch.from_numpy(values.astype(np.float32))
    return state


def write_archive(state: dict[str, torch.Tensor], out_path: Path, **kwargs) -> Path:
    blob = encode_model(state, **kwargs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("p", blob)
    return out_path


def read_archive(path: Path) -> dict[str, torch.Tensor]:
    with zipfile.ZipFile(path) as zf:
        return decode_model(zf.read("p"))
