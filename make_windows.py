# split the pretraining dataset into 30 s windows and assign patients to train / val
#
#   python make_windows.py
#   python make_windows.py --min-usable-hours 3 --val-percent 20
#
# Reads manifest.csv and gaps.csv (from build_pretrain_dataset.py) and writes
# windows.csv next to them: one row per 30 s window, saying which recording it
# comes from, where it starts, whether it is train or val, which channels (if
# any) have a gap in it, and whether it is flagged. The signal data stays in the .npy files;
# a loader reads each window straight out of them:
#
#   x = np.load(f"pretrain dataset/{recording_id}.npy", mmap_mode="r")
#   window = np.asarray(x[:, start_sample:start_sample + 6000])   # (6, 6000)
#
# Rules:
#   - only the selected recording per patient, and only nights with at least
#     --min-usable-hours of usable signal
#   - every window is KEPT; "missing" lists the channels with a gap in it
#   - a window is FLAGGED (flagged = yes) if Flow, Nasal Pressure or Snore has
#     a gap, since those are what airflow limitation and snoring are judged
#     from. Whatever uses the windows decides whether to skip flagged ones.
#   - each patient is train or val by a fixed hash of their ID, so nobody
#     switches sides when new exports are added, and no patient is in both

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np

from channels import respiratory_channels

REQUIRED = {"Flow", "Nasal Pressure", "Snore"}
FIELDS = ["recording_id", "patient_id", "split", "window", "start_sample", "missing", "flagged"]


def split_for(patient_id, val_percent):
    """'val' for a fixed ~val_percent of patients, 'train' for the rest."""
    bucket = int(hashlib.sha256(patient_id.encode()).hexdigest(), 16) % 100
    return "val" if bucket < val_percent else "train"


def main():
    ap = argparse.ArgumentParser(description="Make the 30 s window list for pretraining")
    ap.add_argument("--data", default="pretrain dataset", help="dataset folder")
    ap.add_argument("--window-seconds", type=int, default=30)
    ap.add_argument("--min-usable-hours", type=float, default=2.0,
                    help="leave out nights with less usable signal than this (default 2)")
    ap.add_argument("--val-percent", type=int, default=15,
                    help="percent of patients held out for validation (default 15)")
    args = ap.parse_args()
    data = Path(args.data)

    manifest = list(csv.DictReader(open(data / "manifest.csv", encoding="utf-8")))
    gaps = list(csv.DictReader(open(data / "gaps.csv", encoding="utf-8")))
    gaps_by_source = {}
    for g in gaps:
        gaps_by_source.setdefault(g["source"], []).append(g)

    selected = [r for r in manifest if r["selected"] == "yes"]
    used = [r for r in selected if float(r["usable_h"]) >= args.min_usable_hours]
    print(f"{len(selected)} selected recordings; {len(selected) - len(used)} left out "
          f"for < {args.min_usable_hours:g} usable h; using {len(used)}")

    rows = []
    stats = {"train": [0, 0, set()], "val": [0, 0, set()]}  # windows kept, with any gaps, patients
    n_flagged = 0
    for rec in sorted(used, key=lambda r: r["recording_id"]):
        rate = int(rec["rate"])
        n_seconds = int(rec["n_samples"]) // rate
        n_windows = n_seconds // args.window_seconds
        split = split_for(rec["patient_id"] or rec["recording_id"], args.val_percent)

        # per channel, per second of the night: is it inside a gap?
        bad = {name: np.zeros(n_seconds + 1, bool) for name in respiratory_channels}
        for g in gaps_by_source.get(rec["source"], []):
            start = int(float(g["start_s"]))
            end = int(np.ceil(float(g["start_s"]) + float(g["duration_s"])))
            bad[g["channel"]][start:end] = True
        # per channel, per window: does the window touch a gap anywhere?
        usable_len = n_windows * args.window_seconds
        bad_windows = {name: mask[:usable_len].reshape(n_windows, -1).any(axis=1)
                       for name, mask in bad.items()}

        for w in range(n_windows):
            missing = [name for name in respiratory_channels if bad_windows[name][w]]
            flagged = "yes" if REQUIRED & set(missing) else "no"
            n_flagged += flagged == "yes"

            rows.append({
                "recording_id": rec["recording_id"], "patient_id": rec["patient_id"],
                "split": split, "window": w, "start_sample": w * args.window_seconds * rate,
                "missing": ";".join(missing),
                "flagged": flagged
            })
            stats[split][0] += 1
            stats[split][1] += bool(missing)
            stats[split][2].add(rec["patient_id"])

    out = data / "windows.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{n_flagged} windows flagged (gap in Flow / Nasal Pressure / Snore)")
    for split, (kept, any_gap, patients) in stats.items():
        print(f"{split:5s}: {len(patients):3d} patients, {kept:7d} windows "
              f"({kept * args.window_seconds / 3600:.0f} h), {any_gap} with any gap")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
