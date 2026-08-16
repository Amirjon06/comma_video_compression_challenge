"""Arithmetic coder with static per-stream frequency tables.

Classic Witten-Neal-Cleary integer arithmetic coding with underflow counting.
Picked over a range coder because the carry handling is unambiguous and easy
to test exhaustively, and decode speed is irrelevant here: the archive is
under 200 KB and inflation has a 30 minute budget.
"""

from __future__ import annotations

import numpy as np

CODE_BITS = 32
TOP = (1 << CODE_BITS) - 1
QUARTER = 1 << (CODE_BITS - 2)
HALF = 2 * QUARTER
THREE_QUARTER = 3 * QUARTER
MAX_TOTAL = QUARTER - 1


class _BitWriter:
    def __init__(self):
        self.bits = bytearray()

    def put(self, bit: int, pending: int) -> int:
        self.bits.append(bit)
        for _ in range(pending):
            self.bits.append(bit ^ 1)
        return 0

    def bytes(self) -> bytes:
        pad = (-len(self.bits)) % 8
        self.bits.extend([0] * pad)
        packed = np.packbits(np.frombuffer(bytes(self.bits), dtype=np.uint8))
        return packed.tobytes()


class _BitReader:
    def __init__(self, data: bytes):
        self.bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        self.pos = 0

    def get(self) -> int:
        if self.pos >= len(self.bits):
            return 0
        bit = int(self.bits[self.pos])
        self.pos += 1
        return bit


def build_table(symbols: np.ndarray, alphabet: int) -> np.ndarray:
    """Frequency counts, floored at 1 so every symbol stays decodable."""
    counts = np.bincount(symbols.astype(np.int64), minlength=alphabet)[:alphabet]
    counts = np.maximum(counts, 1).astype(np.int64)
    total = int(counts.sum())
    if total > MAX_TOTAL:
        scale = MAX_TOTAL / total
        counts = np.maximum((counts * scale).astype(np.int64), 1)
    return counts


def encode(symbols: np.ndarray, counts: np.ndarray) -> bytes:
    cum = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    total = int(cum[-1])
    low, high, pending = 0, TOP, 0
    writer = _BitWriter()

    for sym in symbols.astype(np.int64):
        span = high - low + 1
        high = low + (span * int(cum[sym + 1])) // total - 1
        low = low + (span * int(cum[sym])) // total

        while True:
            if high < HALF:
                pending = writer.put(0, pending)
            elif low >= HALF:
                pending = writer.put(1, pending)
                low -= HALF
                high -= HALF
            elif low >= QUARTER and high < THREE_QUARTER:
                pending += 1
                low -= QUARTER
                high -= QUARTER
            else:
                break
            low = (low << 1) & TOP
            high = ((high << 1) | 1) & TOP

    pending += 1
    if low < QUARTER:
        writer.put(0, pending)
    else:
        writer.put(1, pending)
    return writer.bytes()


def decode(data: bytes, counts: np.ndarray, count: int) -> np.ndarray:
    cum = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    total = int(cum[-1])
    reader = _BitReader(data)
    low, high = 0, TOP
    value = 0
    for _ in range(CODE_BITS):
        value = (value << 1) | reader.get()

    out = np.empty(count, dtype=np.int64)
    for i in range(count):
        span = high - low + 1
        target = ((value - low + 1) * total - 1) // span
        sym = int(np.searchsorted(cum, target, side="right") - 1)
        out[i] = sym

        high = low + (span * int(cum[sym + 1])) // total - 1
        low = low + (span * int(cum[sym])) // total

        while True:
            if high < HALF:
                pass
            elif low >= HALF:
                low -= HALF
                high -= HALF
                value -= HALF
            elif low >= QUARTER and high < THREE_QUARTER:
                low -= QUARTER
                high -= QUARTER
                value -= QUARTER
            else:
                break
            low = (low << 1) & TOP
            high = ((high << 1) | 1) & TOP
            value = ((value << 1) | reader.get()) & TOP

    return out


def pack_table(counts: np.ndarray) -> bytes:
    return np.asarray(counts, dtype=np.uint32).tobytes()


def unpack_table(blob: bytes, alphabet: int) -> np.ndarray:
    return np.frombuffer(blob[: 4 * alphabet], dtype=np.uint32).astype(np.int64)
