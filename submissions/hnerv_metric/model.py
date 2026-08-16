"""HNeRV-style decoder: a per-frame latent goes in, a frame comes out.

The archive is the quantized weights plus the quantized latents, so parameter
count is a direct bit cost. Blocks are depthwise-separable to keep the count
near 120k while still reaching 512x384 from a 4x3x4 latent.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import NUM_FRAMES, RENDER_H, RENDER_W

EMBED_C, EMBED_H, EMBED_W = 4, 3, 4
DEFAULT_WIDTHS = (96, 96, 80, 64, 48, 32, 24)


def scaled_widths(mult: float, base=DEFAULT_WIDTHS) -> tuple[int, ...]:
    return tuple(max(8, int(round(w * mult))) for w in base)


class UpBlock(nn.Module):
    """Depthwise 3x3, pointwise expand to 4x channels, then pixel shuffle 2x."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch * 4, 1)
        self.shuffle = nn.PixelShuffle(2)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.shuffle(self.pw(self.act(self.dw(x)))))


class Decoder(nn.Module):
    def __init__(self, widths=DEFAULT_WIDTHS, embed_c: int = EMBED_C):
        super().__init__()
        self.stem = nn.Conv2d(embed_c, widths[0], 1)
        blocks = []
        for i in range(len(widths) - 1):
            blocks.append(UpBlock(widths[i], widths[i + 1]))
        blocks.append(UpBlock(widths[-1], widths[-1]))
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Conv2d(widths[-1], 3, 3, padding=1)

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        x = self.stem(embed)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        assert x.shape[-2:] == (RENDER_H, RENDER_W), x.shape
        # Model works in [-1, 1] and the judges want [0, 255].
        return (torch.tanh(x) * 0.5 + 0.5) * 255.0


class HNeRV(nn.Module):
    def __init__(
        self,
        num_frames: int = NUM_FRAMES,
        widths=DEFAULT_WIDTHS,
        embed_c: int = EMBED_C,
    ):
        super().__init__()
        self.embed = nn.Parameter(
            torch.randn(num_frames, embed_c, EMBED_H, EMBED_W) * 0.1
        )
        self.decoder = Decoder(widths, embed_c)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.embed[indices])

    def render_pairs(self, pair_indices: torch.Tensor) -> torch.Tensor:
        """(B,) pair ids -> (B, 2, 3, H, W), the shape the judges want."""
        flat = torch.stack([2 * pair_indices, 2 * pair_indices + 1], dim=1).reshape(-1)
        out = self(flat)
        return out.reshape(pair_indices.shape[0], 2, 3, RENDER_H, RENDER_W)

    def param_report(self) -> dict:
        decoder = sum(p.numel() for p in self.decoder.parameters())
        embed = self.embed.numel()
        total = (decoder + embed) * 6 // 8
        return {
            "decoder_params": decoder,
            "embed_params": embed,
            "decoder_bytes_at_6bit": decoder * 6 // 8,
            "embed_bytes_at_6bit": embed * 6 // 8,
            "archive_floor_bytes": total,
            "rate_term": round(25 * total / 37_545_489, 4),
        }


if __name__ == "__main__":
    model = HNeRV()
    report = model.param_report()
    for key, value in report.items():
        print(f"{key}: {value:,}")
    total = report["decoder_bytes_at_6bit"] + report["embed_bytes_at_6bit"]
    print(f"rough archive floor: {total:,} bytes -> rate term {25 * total / 37_545_489:.4f}")
    out = model.render_pairs(torch.arange(2))
    print("render_pairs:", tuple(out.shape), float(out.min()), float(out.max()))
