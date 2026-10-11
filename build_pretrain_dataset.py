"""
build_pretrain_dataset.py - build the unlabeled (pretraining) dataset from the
exported EDFs, a few recordings at a time.

For every EDF in the input folders this writes one .npy file holding the six
respiratory channels as a (6, n_samples) float32 array, all resampled to the
same rate (200 Hz by default) and kept in physical units (cmH2O, V, V/s).
Channel order is the order in channels.respiratory_channels.

A manifest.csv next to the .npy files has one row per EDF: whether it was
processed or skipped (and why), its duration, the original sampling rates,
per-channel median / IQR so windows can be normalized at load time, and how
much of it is gaps.

GAPS come in two kinds, both tracked:
    - dropouts inside the EDF: a channel stuck at one value for --min-gap-seconds
      or longer (e.g. a belt not yet connected). Each one is a row in gaps.csv;
      the manifest has the per-channel flat minutes, the minutes where ANY
      channel is flat (gap_min), and usable_h = duration minus gap_min.
    - coverage gaps: the EDF starts late / ends early compared with the
      recording itself. These come from the missing-data CSV that compares EDFs
      with their recording folders (missing_min). has_gaps is "yes" if either
      kind is present.

ONE RECORDING PER PATIENT: every EDF is still processed and nothing is deleted,
but the "selected" column marks exactly one recording per patient - the one
with the most usable hours (ties go to the plain-named file, e.g. 5415.edf over
5415_f9151.edf). Use only selected == "yes" rows for training. The selection
is redone on every run, so it can change when a better export arrives.

It is INCREMENTAL: run it again as more exports arrive and it only processes
EDFs that are new or have changed since the last run. EDFs modified in the last
few minutes are left alone, since they may still be exporting.

Usage (PowerShell, from the folder containing this script):
    python build_pretrain_dataset.py --limit 3          (try a few first)
    python build_pretrain_dataset.py                    (everything new)
    python build_pretrain_dataset.py "Batch 2 edf" --out "pretrain dataset"

Loading one recording for training:
    rows = [r for r in csv.DictReader(open("pretrain dataset/manifest.csv")) if r["selected"] == "yes"]
    x = np.load("pretrain dataset/5017.npy", mmap_mode="r")   # (6, n) - instant
    window = np.asarray(x[:, start:start + 30 * 200])

This script only READS the EDFs; it never modifies them.
"""

import argparse
import csv
import os
import re
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from channels import EdfReader, find_channel, respiratory_channels
from edf_qc import EDFError, read_edf_header

# bump when the output changes, so rows made by an older version get redone
VERSION = "4"

# Snore is legitimately near-silent whenever the patient isn't snoring, so it
# only counts as a gap when it is completely flat, never for being quiet
LOW_SIGNAL_EXEMPT = {"Snore"}

MANIFEST_FIELDS = [
    "recording_id", "patient_id", "selected", "selection_note", "has_gaps",
    "usable_h", "gap_min", "longest_gap_min", "missing_min", "flat_min",
    "source", "source_size", "source_mtime", "status", "reason", "start",
    "duration_h", "n_samples", "rate", "orig_rates", "medians", "iqrs", "version",
]
GAP_FIELDS = ["recording_id", "source", "channel", "kind", "start_s", "duration_s"]


def runs(mask, min_length):
    """(start, length) of every stretch of True in mask at least min_length long."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return [(start, end - start)
            for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
            if end - start >= min_length]


def find_gaps(signal, rate, low_signal, min_flat_s, min_low_s, low_ratio):
    """(kind, start_s, duration_s) of every gap in one channel, checked in ~1 s
    blocks at the original rate (before resampling, so constants stay exact):
      flat       - stuck at one value for at least min_flat_s
      low signal - amplitude below low_ratio x the channel's typical breathing
                   amplitude (90th percentile of 10 s std) for at least min_low_s,
                   e.g. a nasal cannula that has come off. The long minimum keeps
                   real apneas (also near-zero flow, but short) from counting."""
    block = int(rate)
    block_s = block / rate
    blocks = signal[:len(signal) // block * block].reshape(-1, block)
    flat = blocks.max(axis=1) == blocks.min(axis=1)
    bad = np.zeros(len(blocks), bool)
    for start, length in runs(flat, min_flat_s / block_s):
        bad[start:start + length] = True
    if low_signal:
        n10 = len(blocks) // 10
        std10 = blocks[:n10 * 10].reshape(n10, -1).std(axis=1)
        low = np.repeat(std10 < low_ratio * np.percentile(std10, 90), 10)
        for start, length in runs(low, min_low_s / block_s):
            bad[start:start + length] = True
    # a spike or two inside a long dropout shouldn't split it into many gaps:
    # fill any stretch of signal shorter than 60 s that has a gap on both sides
    for start, length in runs(~bad, 1):
        if start > 0 and start + length < len(bad) and length * block_s < 60:
            bad[start:start + length] = True
    return [("flat" if flat[start:start + length].all() else "low signal",
             start * block_s, length * block_s)
            for start, length in runs(bad, 1)]


def resample(signal, rate, target_rate, n_out):
    """Resample to target_rate and return exactly n_out samples.

    Rates like 203.7 Hz would need huge exact up/down factors (2000/2037), so
    this uses a close small-integer ratio for the anti-aliased resampling and
    then interpolates onto the exact output grid to absorb the remaining tiny
    difference. That keeps every channel time-aligned over a whole night."""
    if rate != target_rate:
        ratio = Fraction(target_rate / rate).limit_denominator(50)
        signal = resample_poly(signal, ratio.numerator, ratio.denominator)
    if len(signal) != n_out:
        old_t = np.arange(len(signal)) / len(signal)
        new_t = np.arange(n_out) / n_out
        signal = np.interp(new_t, old_t, signal)
    return signal.astype(np.float32)


def extract(path, target_rate, gap_args):
    """Read the six respiratory channels from one EDF. Returns (array, info, gaps)
    where gaps is a list of (channel, kind, start_s, duration_s)."""
    reader = EdfReader(path)
    try:
        labels = reader.getSignalLabels()
        duration = reader.getFileDuration()
        n_out = int(round(duration * target_rate))
        rows, orig_rates, gaps = [], [], []
        for name in respiratory_channels:
            index = find_channel(labels, name)
            if index is None:
                raise ValueError(f"no {name} channel")
            rate = reader.getSampleFrequency(index)
            orig_rates.append(rate)
            signal = reader.readSignal(index)
            gaps += [(name, kind, start, length) for kind, start, length
                     in find_gaps(signal, rate, name not in LOW_SIGNAL_EXEMPT, *gap_args)]
            rows.append(resample(signal, rate, target_rate, n_out))
    finally:
        reader.close()

    data = np.stack(rows)
    q25, median, q75 = np.percentile(data, [25, 50, 75], axis=1)
    # seconds of the night where at least one channel is flat
    any_flat = np.zeros(int(np.ceil(duration)), bool)
    for _, _, start, length in gaps:
        any_flat[int(start):int(np.ceil(start + length))] = True
    flat_min = [sum(length for ch, _, _, length in gaps if ch == name) / 60 for name in respiratory_channels]
    info = {
        "duration_h": f"{duration / 3600:.3f}",
        "n_samples": data.shape[1],
        "orig_rates": ";".join(f"{r:g}" for r in orig_rates),
        "medians": ";".join(f"{m:.6g}" for m in median),
        "iqrs": ";".join(f"{q:.6g}" for q in q75 - q25),
        "flat_min": ";".join(f"{m:.1f}" for m in flat_min),
        "gap_min": f"{any_flat.sum() / 60:.1f}",
        "longest_gap_min": f"{max((length for _, _, _, length in gaps), default=0) / 60:.1f}",
        "usable_h": f"{(duration - any_flat.sum()) / 3600:.3f}",
    }
    return data, info, gaps


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, fields, rows):
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def save(out_dir, manifest, gaps):
    write_csv(out_dir / "manifest.csv", MANIFEST_FIELDS, [manifest[s] for s in sorted(manifest)])
    write_csv(out_dir / "gaps.csv", GAP_FIELDS,
              sorted(gaps, key=lambda g: (g["source"], g["channel"], float(g["start_s"]))))


def select_one_per_patient(manifest):
    """Mark the processed recording with the most usable hours for each patient."""
    by_patient = {}
    for row in manifest.values():
        row["selected"], row["selection_note"] = "", ""
        if row.get("version") != VERSION:
            row["selection_note"] = "made by an older version of this script - not selectable until re-run"
        elif row["status"] == "processed":
            by_patient.setdefault(row["patient_id"] or row["recording_id"], []).append(row)
    for patient, rows in by_patient.items():
        # most usable hours first; on a tie prefer the plain name (5415 over 5415_f9151)
        rows.sort(key=lambda r: (-float(r["usable_h"]), len(r["recording_id"]), r["recording_id"]))
        best, others = rows[0], rows[1:]
        best["selected"] = "yes"
        if others:
            best["selection_note"] = "chosen over " + ", ".join(r["recording_id"] for r in others)
        for r in others:
            r["selected"] = "no"
            r["selection_note"] = f"patient {patient} uses {best['recording_id']}"
    return len(by_patient)


def main():
    ap = argparse.ArgumentParser(description="Build the unlabeled pretraining dataset from EDFs.")
    ap.add_argument("edf_dirs", nargs="*", default=["Batch 1 edf", "Batch 2 edf"],
                    help='EDF folders to read (default: "Batch 1 edf" "Batch 2 edf")')
    ap.add_argument("--out", default="pretrain dataset", help="Output folder")
    ap.add_argument("--limit", type=int, help="Process at most this many new EDFs this run")
    ap.add_argument("--target-rate", type=int, default=200, help="Output sampling rate in Hz")
    ap.add_argument("--min-hours", type=float, default=1.0,
                    help="Skip recordings shorter than this (default 1)")
    ap.add_argument("--settle-minutes", type=float, default=10.0,
                    help="Leave EDFs modified more recently than this alone (default 10)")
    ap.add_argument("--min-gap-seconds", type=float, default=10.0,
                    help="A channel flat for at least this long counts as a gap (default 10)")
    ap.add_argument("--min-low-seconds", type=float, default=120.0,
                    help="A channel this quiet for at least this long counts as a gap (default 120)")
    ap.add_argument("--low-ratio", type=float, default=0.05,
                    help="'Quiet' = amplitude below this fraction of the channel's typical "
                         "breathing amplitude (default 0.05)")
    ap.add_argument("--missing-csv", default="Batch 2 short recordings - missing data.csv",
                    help="CSV of minutes missing at the start/end of each EDF (matched on edf_file)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    manifest = {row["source"]: row for row in read_csv(out_dir / "manifest.csv")}
    gaps = read_csv(out_dir / "gaps.csv")
    missing = {row["edf_file"]: row["total_missing_min"] for row in read_csv(Path(args.missing_csv))}

    files = sorted(p for d in args.edf_dirs for p in Path(d).glob("*.edf"))
    counts = {"processed": 0, "skipped": 0, "unchanged": 0, "pending": 0}
    done_this_run = 0

    for path in files:
        if args.limit is not None and done_this_run >= args.limit:
            break
        source = path.as_posix()
        stat = path.stat()
        old = manifest.get(source)
        if (old and old["source_size"] == str(stat.st_size)
                and old["source_mtime"] == str(int(stat.st_mtime))
                and old.get("version") == VERSION
                and (old["status"] != "processed" or (out_dir / f"{old['recording_id']}.npy").exists())):
            counts["unchanged"] += 1
            continue
        if time.time() - stat.st_mtime < args.settle_minutes * 60:
            print(f"pending  {source}: modified in the last {args.settle_minutes:g} min, may still be exporting")
            counts["pending"] += 1
            continue

        recording_id = path.stem
        patient = re.match(r"\d{4}", recording_id)
        row = {
            "recording_id": recording_id, "patient_id": patient.group() if patient else "",
            "source": source, "source_size": stat.st_size, "source_mtime": int(stat.st_mtime),
            "rate": args.target_rate, "version": VERSION,
            "missing_min": missing.get(path.name, ""),
        }
        done_this_run += 1
        gaps = [g for g in gaps if g["source"] != source]  # replaced below if it has any

        reason = ""
        try:
            header = read_edf_header(path)
            row["start"] = header["start"].isoformat(sep=" ") if header["start"] else ""
            if header["expected_size"] is not None and header["size"] < header["expected_size"]:
                reason = "truncated (export likely interrupted)"
            elif header["n_records"] * header["rec_dur"] / 3600 < args.min_hours:
                hours = header["n_records"] * header["rec_dur"] / 3600
                reason = f"too short ({hours:.2f} h < {args.min_hours:g} h)"
        except EDFError as e:
            reason = f"unreadable header: {e}"

        # two exports of one patient (e.g. 5415.edf, 5415_f9151.edf) have distinct stems,
        # but two folders could both hold e.g. 4304.edf - don't let one overwrite the other
        clash = [r for s, r in manifest.items() if r["recording_id"] == recording_id and s != source]
        if not reason and clash:
            reason = f"recording_id clashes with {clash[0]['source']}"

        if not reason:
            started = time.time()
            try:
                data, info, found = extract(str(path), args.target_rate,
                                            (args.min_gap_seconds, args.min_low_seconds, args.low_ratio))
                tmp = out_dir / f"{recording_id}.tmp.npy"
                np.save(tmp, data)
                os.replace(tmp, out_dir / f"{recording_id}.npy")
                row.update(info, status="processed")
                gaps += [{"recording_id": recording_id, "source": source, "channel": channel,
                          "kind": kind, "start_s": f"{start:.0f}", "duration_s": f"{length:.0f}"}
                         for channel, kind, start, length in found]
                has_gaps = float(info["gap_min"]) > 0 or float(row["missing_min"] or 0) > 0
                row["has_gaps"] = "yes" if has_gaps else "no"
                note = f", gaps {info['gap_min']} min" if found else ""
                if row["missing_min"]:
                    note += f", {row['missing_min']} min missing at start/end"
                print(f"ok       {source}: {info['duration_h']} h{note} "
                      f"({time.time() - started:.0f} s)")
            except (ValueError, OSError) as e:
                reason = f"extraction failed: {e}"

        if reason:
            row.update(status="skipped", reason=reason)
            print(f"skipped  {source}: {reason}")
        counts[row["status"]] += 1
        manifest[source] = {field: row.get(field, "") for field in MANIFEST_FIELDS}
        save(out_dir, manifest, gaps)  # after every file, so a crash loses nothing

    n_patients = select_one_per_patient(manifest)
    save(out_dir, manifest, gaps)
    selected = [r for r in manifest.values() if r["selected"] == "yes"]
    print(f"\nprocessed {counts['processed']}, skipped {counts['skipped']}, "
          f"unchanged {counts['unchanged']}, pending {counts['pending']}")
    print(f"{n_patients} patients selected ({sum(r['has_gaps'] == 'yes' for r in selected)} with gaps, "
          f"{sum(float(r['usable_h']) for r in selected):.0f} usable hours) -> {out_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
