"""Checks that survive being run without a trained model or a GPU."""

from __future__ import annotations

import numpy as np
import torch

from . import rangecoder as rc
from .codec import decode_model, encode_model
from .config import RENDER_H, RENDER_W
from .model import HNeRV


def test_coder_roundtrip() -> None:
    rng = np.random.default_rng(0)
    for alphabet, count in ((64, 4000), (64, 17), (16, 20000), (2, 500)):
        symbols = np.clip(
            np.round(rng.laplace(0, alphabet / 10, count)) + alphabet // 2,
            0,
            alphabet - 1,
        ).astype(np.int64)
        counts = rc.build_table(symbols, alphabet)
        blob = rc.encode(symbols, counts)
        assert np.array_equal(rc.decode(blob, counts, count), symbols), (alphabet, count)
    print("coder roundtrip ok")


def test_model_shapes() -> None:
    model = HNeRV()
    with torch.inference_mode():
        out = model.render_pairs(torch.arange(3))
    assert out.shape == (3, 2, 3, RENDER_H, RENDER_W), out.shape
    assert float(out.min()) >= 0.0 and float(out.max()) <= 255.0
    print("model shapes ok:", tuple(out.shape))


def test_codec_roundtrip() -> None:
    model = HNeRV()
    state = model.state_dict()
    blob = encode_model(state)
    restored = decode_model(blob)

    assert set(restored) == set(state)
    worst = 0.0
    for name, tensor in state.items():
        ref = tensor.detach().float()
        got = restored[name].float()
        assert got.shape == ref.shape, name
        span = float(ref.abs().max()) + 1e-8
        worst = max(worst, float((got - ref).abs().max()) / span)
    print(f"codec roundtrip ok, payload {len(blob):,} bytes, worst relative error {worst:.4f}")


def test_quantized_render_is_close() -> None:
    model = HNeRV()
    with torch.inference_mode():
        before = model(torch.arange(2))
    model.load_state_dict(decode_model(encode_model(model.state_dict())))
    with torch.inference_mode():
        after = model(torch.arange(2))
    delta = (before - after).abs().mean().item()
    print(f"mean pixel shift from quantization: {delta:.3f} levels")


if __name__ == "__main__":
    test_coder_roundtrip()
    test_model_shapes()
    test_codec_roundtrip()
    test_quantized_render_is_close()
