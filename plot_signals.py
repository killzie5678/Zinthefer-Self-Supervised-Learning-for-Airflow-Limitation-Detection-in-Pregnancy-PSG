# plot the six respiratory channels of one or more EDF files, stacked on one time axis
#
#   python plot_signals.py "Batch 2 edf/5001.edf" "Batch 2 edf/5040.edf"
#   python plot_signals.py "Batch 2 edf/5040.edf" --at 120 --seconds 30
#
# --at       minutes into the recording where the window starts (default: middle of the night)
# --seconds  how long a window to plot (default 60)
# Saves one PNG per file in "Batch 2 plots" (change with --out).

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from channels import EdfReader, find_channel, respiratory_channels

INK = "#1f2933"       # one line color for every channel - the panel label says which is which
MUTED = "#6b7280"
GRID = "#e5e7eb"


def plot_file(path, at_min, seconds, out_dir):
    reader = EdfReader(path)
    labels = reader.getSignalLabels()
    duration = reader.getFileDuration()
    start_s = duration / 2 if at_min is None else at_min * 60
    start_s = max(0, min(start_s, duration - seconds))
    clock0 = datetime.strptime(reader.start, "%d.%m.%y%H.%M.%S") + timedelta(seconds=start_s)

    found = [(name, find_channel(labels, name)) for name in respiratory_channels]
    found = [(name, i) for name, i in found if i is not None]
    missing = [name for name in respiratory_channels if name not in dict(found)]

    fig, axes = plt.subplots(len(found), 1, figsize=(12, 1.6 * len(found)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (name, i) in zip(axes, found):
        rate = reader.getSampleFrequency(i)
        a, b = int(start_s * rate), int((start_s + seconds) * rate)
        signal = reader.readSignal(i)[a:b]
        t = np.arange(len(signal)) / rate
        ax.plot(t, signal, color=INK, linewidth=0.8)
        unit = reader.getPhysicalDimension(i)
        shown = labels[i] if "".join(labels[i].split()).lower() != "".join(name.split()).lower() else ""
        ax.set_ylabel(f"{name}\n({unit})" if unit else name, rotation=0, ha="right", va="center",
                      fontsize=9, color=INK)
        if shown:  # e.g. Thorax stored as "Chest" in this file
            ax.text(1.0, 1.0, f'labelled "{shown}" in file', transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8, color=MUTED)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    axes[-1].set_xlabel(f"seconds from {clock0:%H:%M:%S}", fontsize=9, color=MUTED)
    axes[-1].set_xlim(0, seconds)

    title = f"{Path(path).name}  -  {seconds:g} s from {clock0:%Y-%m-%d %H:%M:%S} " \
            f"({start_s / 3600:.1f} h into a {duration / 3600:.1f} h recording)"
    if missing:
        title += f"\nnot in this file: {', '.join(missing)}"
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{Path(path).stem}_{clock0:%H%M%S}_{seconds:g}s.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    reader.close()
    return out


def main():
    ap = argparse.ArgumentParser(description="Plot the respiratory channels of EDF files")
    ap.add_argument("edf", nargs="+", help="EDF file(s) to plot")
    ap.add_argument("--at", type=float, default=None,
                    help="minutes into the recording to start (default: the middle)")
    ap.add_argument("--seconds", type=float, default=60, help="window length in seconds (default 60)")
    ap.add_argument("--out", default="Batch 2 plots", help='folder for the PNGs (default "Batch 2 plots")')
    args = ap.parse_args()
    for path in args.edf:
        print("saved", plot_file(path, args.at, args.seconds, Path(args.out)))


if __name__ == "__main__":
    main()
