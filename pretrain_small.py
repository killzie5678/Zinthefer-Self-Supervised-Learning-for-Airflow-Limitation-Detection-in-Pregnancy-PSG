# small self-supervised pretraining run (masked reconstruction) on the CPU
#
#   python pretrain_small.py                              (2000 train / 500 val windows)
#   python pretrain_small.py --train-windows 5000 --epochs 20
#
# This is a check that the pipeline works end to end, not the real pretraining.
#
# Masked reconstruction: random 1 s chunks of each 30 s window (--mask-ratio of
# them) are hidden in all six channels, and the model learns to fill them in
# from the surrounding signal. The loss only counts hidden samples of channels
# that are present, so a missing belt is never scored.
#
# Baseline: normalized signals are centered near 0, so "predict 0" is the
# trivial answer. If val loss goes clearly below that, the model is learning
# real breathing structure. Saves the model, a loss plot and example
# reconstructions in "pretrain runs/<time>/".

import argparse
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from channels import respiratory_channels
from pretrain_data import WindowDataset

RATE = 200
CHUNK = RATE  # mask in 1 s chunks

INK = "#1f2933"
MUTED = "#6b7280"
GRID = "#e5e7eb"
ACCENT = "#2563eb"   # the model's guess; the real signal is INK
HIDDEN = "#e5e7eb"   # shaded chunks the model could not see


def load(split, n, seed):
    """n random windows of one split, loaded into memory."""
    ds = WindowDataset(split)
    idx = np.random.default_rng(seed).choice(len(ds), min(n, len(ds)), replace=False)
    xs, ps = zip(*(ds[i] for i in idx))
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ps))


def random_mask(batch, length, ratio, generator):
    """(batch, length) bool: True where the signal is hidden, in whole 1 s chunks."""
    chunks = torch.rand(batch, length // CHUNK, generator=generator) < ratio
    return chunks.repeat_interleave(CHUNK, dim=1)


class Encoder(nn.Module):
    """Six channels at 200 Hz -> 64 features at 12.5 Hz (one per 80 ms)."""
    def __init__(self, width=64):
        super().__init__()
        layers, c_in = [], 7  # 6 signals + the mask, so it knows what is hidden
        for c_out in (32, 48, 64, width):
            layers += [nn.Conv1d(c_in, c_out, 7, stride=2, padding=3), nn.BatchNorm1d(c_out), nn.GELU()]
            c_in = c_out
        for d in (1, 2, 4, 8, 16):
            layers += [nn.Conv1d(c_in, c_out, 3, padding=d, dilation=d), nn.BatchNorm1d(c_out), nn.GELU()]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        layers, c_in = [], width
        for c_out in (64, 48, 32):
            layers += [nn.ConvTranspose1d(c_in, c_out, 4, stride=2, padding=1), nn.BatchNorm1d(c_out), nn.GELU()]
            c_in = c_out
        layers += [nn.ConvTranspose1d(c_in, 6, 4, stride=2, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class MaskedAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder, self.decoder = Encoder(), Decoder()

    def forward(self, x, mask):
        visible = x * ~mask[:, None, :]
        return self.decoder(self.encoder(torch.cat([visible, mask[:, None, :].float()], dim=1)))


def masked_loss(pred, x, mask, present):
    """Mean squared error over hidden samples of present channels only."""
    weight = mask[:, None, :] & present[:, :, None]
    return ((pred - x) ** 2 * weight).sum() / weight.sum().clamp(min=1)


def evaluate(model, x, p, ratio, batch_size):
    """Val loss for the model and for always predicting 0, on a fixed mask."""
    g = torch.Generator().manual_seed(123)  # same hidden chunks every epoch
    model.eval()
    total = zero = n = 0.0
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb, pb = x[i:i + batch_size], p[i:i + batch_size]
            m = random_mask(len(xb), xb.shape[2], ratio, g)
            total += masked_loss(model(xb, m), xb, m, pb).item() * len(xb)
            zero += masked_loss(torch.zeros_like(xb), xb, m, pb).item() * len(xb)
            n += len(xb)
    return total / n, zero / n


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def plot_losses(history, baseline, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = np.arange(1, len(history) + 1)
    ax.plot(epochs, [h[0] for h in history], color=MUTED, linewidth=2)
    ax.plot(epochs, [h[1] for h in history], color=INK, linewidth=2)
    ax.axhline(baseline, color=MUTED, linewidth=1, linestyle="--")
    ax.text(epochs[-1], [h[0] for h in history][-1], "  train", color=MUTED, fontsize=9, va="center")
    ax.text(epochs[-1], [h[1] for h in history][-1], "  val", color=INK, fontsize=9, va="center")
    ax.text(1, baseline, "predict-0 baseline", color=MUTED, fontsize=8, va="bottom")
    ax.set_xlabel("epoch", fontsize=9, color=MUTED)
    ax.set_ylabel("masked MSE", fontsize=9, color=MUTED)
    ax.set_ylim(0, baseline * 1.1)
    ax.set_title("Masked reconstruction loss", loc="left", fontsize=10, color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_example(model, x, p, ratio, out):
    """One val window: real signal, hidden chunks shaded, model's fill-in."""
    g = torch.Generator().manual_seed(7)
    m = random_mask(1, x.shape[2], ratio, g)
    model.eval()
    with torch.no_grad():
        pred = model(x[:1], m)[0].numpy()
    real, hidden = x[0].numpy(), m[0].numpy()
    t = np.arange(real.shape[1]) / RATE
    fig, axes = plt.subplots(6, 1, figsize=(12, 9), sharex=True)
    for c, ax in enumerate(axes):
        edges = np.flatnonzero(np.diff(np.concatenate([[0], hidden.astype(int), [0]])))
        for a, b in zip(edges[::2], edges[1::2]):
            ax.axvspan(a / RATE, b / RATE, color=HIDDEN, linewidth=0, zorder=0)
        ax.plot(t, real[c], color=INK, linewidth=0.8)
        guess = np.where(hidden, pred[c], np.nan)
        ax.plot(t, guess, color=ACCENT, linewidth=1.2)
        ax.set_ylabel(respiratory_channels[c] + ("" if p[0, c] else "\n(missing)"),
                      rotation=0, ha="right", va="center", fontsize=9, color=INK)
        style(ax)
    axes[-1].set_xlabel("seconds", fontsize=9, color=MUTED)
    fig.suptitle("Val window: real signal (dark), hidden 1 s chunks (shaded), model's fill-in (blue)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Small masked-reconstruction pretraining run")
    ap.add_argument("--train-windows", type=int, default=2000)
    ap.add_argument("--val-windows", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mask-ratio", type=float, default=0.4)
    ap.add_argument("--out", default="pretrain runs")
    args = ap.parse_args()
    torch.manual_seed(0)

    started = time.time()
    x_train, p_train = load("train", args.train_windows, seed=0)
    x_val, p_val = load("val", args.val_windows, seed=1)
    print(f"loaded {len(x_train)} train / {len(x_val)} val windows in {time.time() - started:.0f} s")

    model = MaskedAutoencoder()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    g = torch.Generator().manual_seed(0)

    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        order = torch.randperm(len(x_train), generator=g)
        total = 0.0
        for i in range(0, len(order), args.batch_size):
            b = order[i:i + args.batch_size]
            xb, pb = x_train[b], p_train[b]
            m = random_mask(len(xb), xb.shape[2], args.mask_ratio, g)
            loss = masked_loss(model(xb, m), xb, m, pb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        val, baseline = evaluate(model, x_val, p_val, args.mask_ratio, args.batch_size)
        history.append((total / len(order), val))
        print(f"epoch {epoch:2d}: train {total / len(order):.3f}  val {val:.3f}  "
              f"(predict-0 {baseline:.3f}, val is {100 * val / baseline:.0f}% of it)  "
              f"{time.time() - started:.0f} s")

    out = Path(args.out) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    torch.save({"model": model.state_dict(), "args": vars(args), "history": history}, out / "model.pt")
    plot_losses(history, baseline, out / "loss.png")
    plot_example(model, x_val, p_val, args.mask_ratio, out / "example.png")
    print(f"saved model, loss.png and example.png in {out}")


if __name__ == "__main__":
    main()
