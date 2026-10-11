# summarize what is missing from each recording in the pretraining dataset
#
#   python missing_summary.py
#
# Reads manifest.csv and gaps.csv (written by build_pretrain_dataset.py) and
# writes "missing summary.csv" next to them: one row per recording with the
# minutes missing at the start/end of the night, the gap minutes per channel,
# and a plain-English description. Re-run it after each dataset build.

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

from channels import respiratory_channels

# channels that drop out together, described as one problem
GROUPS = [
    ("nasal cannula", ["Flow", "Nasal Pressure"]),
    ("belts", ["RIP Flow", "Thorax", "Abdomen"]),
    ("snore", ["Snore"]),
]


def merged(intervals):
    """Union of (start, end) intervals, as a sorted list of non-overlapping ones."""
    out = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def describe(row, gaps):
    """Plain-English list of what is missing, biggest problems first."""
    parts = []
    if row["missing_min"]:
        parts.append(f"{float(row['missing_min']):.0f} min of the recording missing at start/end of EDF")
    start = datetime.fromisoformat(row["start"]) if row["start"] else None
    found = []
    for name, channels in GROUPS:
        spans = merged((float(g["start_s"]), float(g["start_s"]) + float(g["duration_s"]))
                       for g in gaps if g["channel"] in channels)
        if not spans:
            continue
        total = sum(b - a for a, b in spans) / 60
        a, b = max(spans, key=lambda s: s[1] - s[0])
        when = f" from {start + timedelta(seconds=a):%H:%M}" if start else ""
        kinds = sorted({g["kind"] for g in gaps if g["channel"] in channels})
        text = f"{name} ({' + '.join(kinds)}): {total:.0f} min"
        if len(spans) > 1:
            text += f" in {len(spans)} gaps, longest {(b - a) / 60:.0f} min{when}"
        else:
            text += when
        found.append((total, text))
    parts += [text for _, text in sorted(found, reverse=True)]
    return "; ".join(parts) or "nothing missing"


def main():
    ap = argparse.ArgumentParser(description="Summarize what is missing from each recording")
    ap.add_argument("--data", default="pretrain dataset", help="dataset folder")
    args = ap.parse_args()
    data = Path(args.data)

    manifest = list(csv.DictReader(open(data / "manifest.csv", encoding="utf-8")))
    gaps = list(csv.DictReader(open(data / "gaps.csv", encoding="utf-8")))
    by_recording = {}
    for g in gaps:
        by_recording.setdefault(g["source"], []).append(g)

    fields = (["recording_id", "patient_id", "selected", "status", "duration_h", "usable_h",
               "usable_pct", "missing_start_end_min"]
              + [f"{name} gap min" for name in respiratory_channels] + ["what is missing"])
    rows = []
    for row in sorted(manifest, key=lambda r: r["recording_id"]):
        out = {"recording_id": row["recording_id"], "patient_id": row["patient_id"],
               "selected": row["selected"], "status": row["status"]}
        if row["status"] != "processed":
            out["what is missing"] = f"not processed: {row['reason']}"
            rows.append(out)
            continue
        mine = by_recording.get(row["source"], [])
        out.update({
            "duration_h": row["duration_h"], "usable_h": row["usable_h"],
            "usable_pct": f"{100 * float(row['usable_h']) / float(row['duration_h']):.0f}",
            "missing_start_end_min": row["missing_min"],
            "what is missing": describe(row, mine),
        })
        for name, minutes in zip(respiratory_channels, row["flat_min"].split(";")):
            out[f"{name} gap min"] = minutes
        rows.append(out)

    out_path = data / "missing summary.csv"
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:  # -sig so Excel reads it right
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path} ({len(rows)} recordings)")


if __name__ == "__main__":
    main()
