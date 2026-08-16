"""How much image fidelity does PoseNet's first output dimension need?

Training plateaus at dim0 error ~21.5 regardless of model size or schedule,
while the untouched frames score ~3e-5. This degrades the real frames in
controlled ways and reports what each degradation costs, which turns "our
reconstruction isn't good enough" into a number we can aim at.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F

from . import frames as frames_mod
from . import judges
from .config import POSE_DIMS
from .model import HNeRV, scaled_widths


def gaussian_kernel(sigma: float, device) -> torch.Tensor:
    radius = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-(x**2) / (2 * sigma**2))
    return k / k.sum()


def blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    b, t, c, h, w = x.shape
    flat = x.reshape(b * t, c, h, w)
    k = gaussian_kernel(sigma, x.device)
    pad = (len(k) - 1) // 2
    flat = F.conv2d(F.pad(flat, (pad, pad, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    flat = F.conv2d(F.pad(flat, (0, 0, pad, pad), mode="reflect"), k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    return flat.reshape(b, t, c, h, w)


def resample(x: torch.Tensor, factor: float) -> torch.Tensor:
    b, t, c, h, w = x.shape
    flat = x.reshape(b * t, c, h, w)
    small = F.interpolate(flat, scale_factor=1 / factor, mode="bilinear", antialias=True)
    back = F.interpolate(small, size=(h, w), mode="bilinear")
    return back.reshape(b, t, c, h, w)


def report(name, pose_out, pose_ref):
    got = pose_out["pose"][..., :POSE_DIMS]
    per_dim = (got - pose_ref).pow(2).mean(0)
    total = per_dim.mean().item()
    contribution = math.sqrt(10 * total)
    print(
        f"{name:<34} dim0 {per_dim[0].item():10.4f}   pose {total:10.5f}   "
        f"score term {contribution:7.3f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pairs", type=int, default=24)
    ap.add_argument("--checkpoint", default=".hnerv_cache/night.pt")
    ap.add_argument("--embed-c", type=int, default=16)
    ap.add_argument("--width-mult", type=float, default=2.0)
    args = ap.parse_args()

    device = torch.device(args.device)
    small = frames_mod.load_small()
    _, pose_t = judges.load_targets()
    net = judges.load_distortion_net(device)

    pairs = np.arange(args.pairs)
    ids = np.stack([2 * pairs, 2 * pairs + 1], 1).reshape(-1)
    real = torch.from_numpy(np.asarray(small[ids]).astype(np.float32)).to(device)
    real = real.reshape(len(pairs), 2, 3, real.shape[-2], real.shape[-1])
    pose_ref = torch.from_numpy(np.asarray(pose_t[pairs])).to(device).float()

    print("degradation applied to the ORIGINAL frames")
    print("-" * 78)
    with torch.no_grad():
        for sigma in (0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
            out, _ = judges.judge_outputs(net, blur(real, sigma))
            report(f"gaussian blur sigma {sigma}", out, pose_ref)

        print()
        for factor in (1.5, 2.0, 3.0, 4.0):
            out, _ = judges.judge_outputs(net, resample(real, factor))
            report(f"downsample {factor}x and back", out, pose_ref)

        print()
        for sigma in (2.0, 5.0, 10.0, 20.0):
            noisy = (real + sigma * torch.randn_like(real)).clamp(0, 255)
            out, _ = judges.judge_outputs(net, noisy)
            report(f"gaussian noise {sigma} levels", out, pose_ref)

        print()
        for step in (4, 8, 16, 32):
            quant = (real / step).round() * step
            out, _ = judges.judge_outputs(net, quant.clamp(0, 255))
            report(f"quantize to {step} levels", out, pose_ref)

        print()
        out, _ = judges.judge_outputs(net, real.flip(1))
        report("frame order swapped", out, pose_ref)

        still = real[:, :1].expand(-1, 2, -1, -1, -1).contiguous()
        out, _ = judges.judge_outputs(net, still)
        report("second frame = first frame", out, pose_ref)

        print()
        print("our model for comparison")
        print("-" * 78)
        model = HNeRV(
            widths=scaled_widths(args.width_mult), embed_c=args.embed_c
        ).to(device)
        try:
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
            model.eval()
            recon = model.render_pairs(torch.from_numpy(pairs).to(device))
            out, _ = judges.judge_outputs(net, recon)
            report("trained model", out, pose_ref)
            err = (recon - real).pow(2).mean().item()
            print(f"model pixel MSE {err:.2f} (RMSE {math.sqrt(err):.2f} levels)")
        except Exception as exc:  # checkpoint shape mismatch is not fatal here
            print(f"skipped model: {exc}")


if __name__ == "__main__":
    main()
