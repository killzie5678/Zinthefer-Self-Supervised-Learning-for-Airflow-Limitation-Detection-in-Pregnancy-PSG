# plot processed recordings from the pretraining dataset: the whole night with
# the detected gaps shaded, plus a 60 s zoom, for each of the six channels
#
#   python plot_night.py 4304 4318 4341
#   python plot_night.py 4304 --at 120          (zoom starts 120 min into the night)
#
# Reads "<id>.npy", manifest.csv and gaps.csv from the dataset folder (--data)
# and saves one PNG per recording in "pretrain plots" (change with --out).

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from channels import respiratory_channels

INK = "#1f2933"       # one line color for every channel - the panel label says which is which
MUTED = "#6b7280"
GRID = "#e5e7eb"
GAP = "#d1d5db"       # shaded gap bands, labelled "gap" so they don't rely on color
ZOOM = "#9ca3af"
UNITS = ["cmH2O", "cmH2O", "cmH2O", "V/s", "V", "V"]
BIN_S = 10            # overview draws the min-max range of each 10 s stretch


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def plot_recording(recording_id, data_dir, out_dir, at_min, seconds):
    rows = list(csv.DictReader(open(data_dir / "manifest.csv", encoding="utf-8")))
    info = next(r for r in rows if r["recording_id"] == recording_id)
    gaps = [g for g in csv.DictReader(open(data_dir / "gaps.csv", encoding="utf-8"))
            if g["recording_id"] == recording_id]
    x = np.load(data_dir / f"{recording_id}.npy", mmap_mode="r")
    rate = int(info["rate"])
    hours = x.shape[1] / rate / 3600
    start = datetime.fromisoformat(info["start"])

    zoom_s = x.shape[1] / rate / 2 if at_min is None else at_min * 60
    zoom_s = max(0, min(zoom_s, x.shape[1] / rate - seconds))

    fig, axes = plt.subplots(6, 2, figsize=(15, 9), sharex="col",
                             gridspec_kw={"width_ratios": [3, 1]})
    for c, name in enumerate(respiratory_channels):
        night, zoom = axes[c]
        signal = np.asarray(x[c])

        # overview: min-max envelope per 10 s, y range set by the bulk of the
        # night so a few large artifacts don't flatten everything else
        n_bins = len(signal) // (BIN_S * rate)
        bins = signal[:n_bins * BIN_S * rate].reshape(n_bins, -1)
        lo, hi = bins.min(axis=1), bins.max(axis=1)
        t = (np.arange(n_bins) + 0.5) * BIN_S / 3600
        night.fill_between(t, lo, hi, color=INK, linewidth=0)
        ylim = np.percentile(lo, 1), np.percentile(hi, 99)
        pad = (ylim[1] - ylim[0]) * 0.1 or 1e-6
        night.set_ylim(ylim[0] - pad, ylim[1] + pad)
        for g in gaps:
            if g["channel"] == name:
                a = float(g["start_s"]) / 3600
                b = a + float(g["duration_s"]) / 3600
                night.axvspan(a, b, color=GAP, linewidth=0, zorder=0)
                night.text((a + b) / 2, 0.92, f"{g.get('kind') or 'gap'} {float(g['duration_s']) / 60:.0f} min",
                           transform=night.get_xaxis_transform(), ha="center", va="top",
                           fontsize=7, color=MUTED)
        night.axvspan(zoom_s / 3600, (zoom_s + seconds) / 3600, color=ZOOM, alpha=0.4, linewidth=0)
        night.set_ylabel(f"{name}\n({UNITS[c]})", rotation=0, ha="right", va="center",
                         fontsize=9, color=INK)
        style(night)

        # zoom: the actual 200 Hz samples
        a, b = int(zoom_s * rate), int((zoom_s + seconds) * rate)
        zoom.plot(np.arange(b - a) / rate, signal[a:b], color=INK, linewidth=0.7)
        style(zoom)

    axes[-1, 0].set_xlim(0, hours)
    ticks = np.arange(0, hours + 1e-9, 1)
    axes[-1, 0].set_xticks(ticks, [f"{start + timedelta(hours=h):%H:%M}" for h in ticks])
    axes[-1, 0].set_xlabel("clock time (whole night, min-max per 10 s)", fontsize=9, color=MUTED)
    axes[-1, 1].set_xlim(0, seconds)
    axes[-1, 1].set_xlabel(f"seconds from {start + timedelta(seconds=zoom_s):%H:%M:%S} "
                           f"(gray band on the left)", fontsize=9, color=MUTED)

    title = (f"{recording_id}  -  {start:%Y-%m-%d %H:%M}, {hours:.1f} h, "
             f"usable {float(info['usable_h']):.1f} h, gaps {info['gap_min']} min")
    if info["missing_min"]:
        title += f", {info['missing_min']} min missing at start/end of the recording"
    if info["selected"] != "yes":
        title += f"  [not selected: {info['selection_note']}]"
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{recording_id}_night.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description="Plot processed recordings with their gaps")
    ap.add_argument("ids", nargs="+", help="recording ids, e.g. 4304 5415_f9151")
    ap.add_argument("--data", default="pretrain dataset", help="dataset folder")
    ap.add_argument("--out", default="pretrain plots", help="folder for the PNGs")
    ap.add_argument("--at", type=float, default=None,
                    help="minutes into the night where the zoom starts (default: the middle)")
    ap.add_argument("--seconds", type=float, default=60, help="zoom length in seconds (default 60)")
    args = ap.parse_args()
    for recording_id in args.ids:
        print("saved", plot_recording(recording_id, Path(args.data), Path(args.out), args.at, args.seconds))


if __name__ == "__main__":
    main()
