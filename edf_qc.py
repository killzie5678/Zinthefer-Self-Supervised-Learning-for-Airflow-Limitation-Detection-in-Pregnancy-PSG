"""
edf_qc.py - batch quality check for exported EDF/EDF+ files.

Checks every EDF in a folder for:
    - corrupted / truncated files (header unreadable, or file is shorter than
      the header says it should be - the classic sign of an interrupted export)
    - missing or extra channels, compared to the rest of the batch (or to a
      known-good reference EDF you point it at)
    - inconsistent sampling rates for the same channel across files
    - unexpectedly short recordings
    - (optional, --signals) channels that are completely flat or all-NaN

Usage (PowerShell, from the folder containing this script):
    python edf_qc.py "Batch 2 edf"
    python edf_qc.py "Batch 2 edf" --reference "5017.edf"
    python edf_qc.py "Batch 2 edf" --signals          (slower: reads the data too)
    python edf_qc.py "Batch 2 edf" --csv qc_report.csv

HOW "EXPECTED" CHANNELS AND RATES ARE DECIDED:
    By default the script assumes the MOST COMMON channel set (and, per
    channel, the most common sampling rate) across the folder is correct, and
    flags files that differ. That's a sensible default when most exports are
    fine, but it can't tell you whether the batch as a whole is wrong. If you
    have one export you've personally verified against Noxturnal (e.g. 5017),
    pass it with --reference and the whole folder is checked against that
    instead.

    Channel names are compared exactly (after trimming spaces). If Noxturnal
    labels the same signal differently in different exports, that will show up
    as one channel "missing" and another "extra" - which is what you want to
    know before analysis.

The header checks need no extra packages. --signals needs pyedflib + numpy.
This script only READS files; it never modifies anything.
"""

import argparse
import csv
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ANNOTATION_LABEL = "EDF Annotations"  # EDF+ events channel, not a physiological signal

# (field name, bytes per signal) in the order they appear in the EDF header
SIGNAL_FIELDS = [
    ("label", 16), ("transducer", 80), ("physdim", 8), ("physmin", 8),
    ("physmax", 8), ("digmin", 8), ("digmax", 8), ("prefilter", 80),
    ("nsamp", 8), ("reserved", 32),
]


class EDFError(Exception):
    pass


def _ascii(b):
    return b.decode("ascii", "replace").strip()


def read_edf_header(path):
    """
    Parse an EDF header by hand so that corrupted files can be diagnosed
    rather than just crashing a library. Returns a dict, or raises EDFError.
    """
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        fixed = f.read(256)
        if len(fixed) < 256:
            raise EDFError("file is shorter than the 256-byte EDF header")
        if _ascii(fixed[0:8]) != "0":
            raise EDFError("not an EDF file (version field is not '0')")
        try:
            header_bytes = int(_ascii(fixed[184:192]))
            n_records = int(_ascii(fixed[236:244]))
            rec_dur = float(_ascii(fixed[244:252]))
            ns = int(_ascii(fixed[252:256]))
        except ValueError:
            raise EDFError("header numbers are unreadable (corrupted header)")
        if ns <= 0 or ns > 512:
            raise EDFError(f"implausible signal count in header ({ns})")
        if header_bytes != 256 + ns * 256:
            raise EDFError(f"header size ({header_bytes}) doesn't match {ns} signals")
        rest = f.read(ns * 256)
        if len(rest) < ns * 256:
            raise EDFError("file ends inside the header (truncated)")

    # Per-signal fields are stored field-by-field: all labels, then all transducers, ...
    pos, sig = 0, {}
    for name, width in SIGNAL_FIELDS:
        sig[name] = [_ascii(rest[pos + i * width: pos + (i + 1) * width]) for i in range(ns)]
        pos += ns * width
    try:
        nsamp = [int(x) for x in sig["nsamp"]]
    except ValueError:
        raise EDFError("samples-per-record field is unreadable (corrupted header)")
    if rec_dur <= 0:
        raise EDFError("record duration in header is zero or negative")

    bytes_per_record = sum(nsamp) * 2  # EDF stores 16-bit samples
    data_bytes = size - header_bytes
    if n_records >= 0:
        expected_size = header_bytes + n_records * bytes_per_record
        actual_records = data_bytes // bytes_per_record if bytes_per_record else 0
    else:  # -1 means "unknown" in the spec; infer from file size
        actual_records = data_bytes // bytes_per_record if bytes_per_record else 0
        expected_size = None

    start = None
    try:
        d, mo, yy = (int(x) for x in _ascii(fixed[168:176]).split("."))
        h, mi, s = (int(x) for x in _ascii(fixed[176:184]).split("."))
        start = datetime(1900 + yy if yy >= 85 else 2000 + yy, mo, d, h, mi, s)
    except (ValueError, TypeError):
        pass

    channels = []
    for i in range(ns):
        channels.append({
            "label": sig["label"][i],
            "rate": nsamp[i] / rec_dur,
            "nsamp": nsamp[i],
        })

    return {
        "size": size, "n_records": n_records, "rec_dur": rec_dur,
        "expected_size": expected_size, "actual_records": actual_records,
        "start": start, "channels": channels,
        "format": "EDF+" if _ascii(fixed[192:236]).startswith("EDF+") else "EDF",
    }


def check_signals(path, channels):
    """
    Optional, slower: read the actual samples and flag dead channels.
    Returns (list_of_problem_strings, note_or_None).
    """
    try:
        import numpy as np
        import pyedflib
    except ImportError:
        return [], "signal checks skipped (pyedflib/numpy not installed)"

    problems = []
    try:
        reader = pyedflib.EdfReader(str(path))
    except Exception as e:  # pyedflib raises assorted exceptions on bad files
        return [f"pyedflib could not open the file ({e})"], None
    try:
        for i, ch in enumerate(channels):
            if ch["label"] == ANNOTATION_LABEL:
                continue
            try:
                x = reader.readSignal(i)
            except Exception as e:
                problems.append(f"{ch['label']}: could not read samples ({e})")
                continue
            if x.size == 0:
                problems.append(f"{ch['label']}: no samples")
            elif np.all(np.isnan(x)):
                problems.append(f"{ch['label']}: all NaN")
            elif np.nanmax(x) == np.nanmin(x):
                problems.append(f"{ch['label']}: completely flat (constant value {x[0]:.4g})")
            elif x.size > 1 and np.mean(np.diff(x) == 0) >= 0.98:
                problems.append(f"{ch['label']}: >=98% of consecutive samples identical (mostly flat)")
    finally:
        reader.close()
    return problems, None


def main():
    ap = argparse.ArgumentParser(description="Batch QC for exported EDF/EDF+ files.")
    ap.add_argument("edf_dir", help='Folder containing exported EDFs, e.g. "Batch 2 edf"')
    ap.add_argument("--reference", help="A known-good EDF to define expected channels/rates "
                                        "(otherwise the most common ones in the folder are used)")
    ap.add_argument("--min-hours", type=float, default=3.0,
                    help="Warn if a recording is shorter than this many hours (default 3)")
    ap.add_argument("--signals", action="store_true",
                    help="Also read the samples to find flat/dead channels (slower; needs pyedflib)")
    ap.add_argument("--csv", help="Write one row per file to this CSV")
    args = ap.parse_args()

    edf_dir = Path(args.edf_dir)
    if not edf_dir.is_dir():
        print(f"ERROR: folder not found: {edf_dir}")
        sys.exit(1)

    files = sorted(p for p in edf_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".edf")
    if not files:
        print(f"No .edf files found under {edf_dir}")
        sys.exit(0)

    # ---- Pass 1: parse every header ----
    parsed = {}   # path -> header dict
    results = {}  # path -> {"fail": [...], "warn": [...]}
    for p in files:
        results[p] = {"fail": [], "warn": []}
        try:
            parsed[p] = read_edf_header(p)
        except EDFError as e:
            results[p]["fail"].append(f"unreadable: {e}")

    # ---- Decide what "expected" means ----
    if args.reference:
        try:
            ref = read_edf_header(Path(args.reference))
        except (EDFError, OSError) as e:
            print(f"ERROR: could not read reference file: {e}")
            sys.exit(1)
        ref_chs = [c for c in ref["channels"] if c["label"] != ANNOTATION_LABEL]
        expected_set = {c["label"] for c in ref_chs}
        expected_rate = {c["label"]: c["rate"] for c in ref_chs}
        basis = f"reference file {args.reference}"
    else:
        sets = Counter(
            frozenset(c["label"] for c in h["channels"] if c["label"] != ANNOTATION_LABEL)
            for h in parsed.values()
        )
        expected_set = set(sets.most_common(1)[0][0]) if sets else set()
        expected_rate = {}
        for label in expected_set:
            rates = Counter(round(c["rate"], 6) for h in parsed.values()
                            for c in h["channels"] if c["label"] == label)
            expected_rate[label] = rates.most_common(1)[0][0]
        basis = "the most common channels/rates in this folder"

    # ---- Pass 2: judge every file ----
    signal_note_shown = False
    for p, h in parsed.items():
        r = results[p]

        # Integrity
        if h["n_records"] == 0 or h["actual_records"] == 0:
            r["fail"].append("contains no data records")
        elif h["expected_size"] is not None and h["size"] < h["expected_size"]:
            missing_mb = (h["expected_size"] - h["size"]) / (1024 * 1024)
            pct = 100 * h["actual_records"] / h["n_records"]
            r["fail"].append(f"TRUNCATED: file is {missing_mb:.1f} MB smaller than its header says "
                             f"(~{pct:.0f}% of the recording present) - export was likely interrupted")
        elif h["expected_size"] is not None and h["size"] > h["expected_size"]:
            r["warn"].append("file is larger than its header says (trailing extra bytes)")

        # Channels
        have = {c["label"] for c in h["channels"] if c["label"] != ANNOTATION_LABEL}
        missing, extra = sorted(expected_set - have), sorted(have - expected_set)
        if missing:
            r["warn"].append("missing channel(s): " + ", ".join(missing))
        if extra:
            r["warn"].append("unexpected extra channel(s): " + ", ".join(extra))
        dupes = [l for l, n in Counter(c["label"] for c in h["channels"]).items() if n > 1]
        if dupes:
            r["warn"].append("duplicate channel label(s): " + ", ".join(dupes))

        # Sampling rates
        for c in h["channels"]:
            exp = expected_rate.get(c["label"])
            if exp is not None and abs(c["rate"] - exp) > 1e-6:
                r["warn"].append(f"{c['label']}: sampling rate {c['rate']:g} Hz (expected {exp:g} Hz)")

        # Duration
        # Use the records actually present on disk if the header count is unknown (-1)
        # or the file is truncated; otherwise trust the header.
        truncated = h["expected_size"] is not None and h["size"] < h["expected_size"]
        n_eff = h["actual_records"] if (h["n_records"] < 0 or truncated) else h["n_records"]
        dur_h = n_eff * h["rec_dur"] / 3600
        h["duration_h"] = dur_h
        if 0 < dur_h < args.min_hours:
            r["warn"].append(f"short recording: {dur_h:.1f} h (threshold {args.min_hours:g} h)")

        # Optional: signal content
        if args.signals and not r["fail"]:
            probs, note = check_signals(p, h["channels"])
            r["warn"].extend(probs)
            if note and not signal_note_shown:
                print(f"NOTE: {note}\n")
                signal_note_shown = True

    # ---- Report ----
    def status(p):
        r = results[p]
        return "FAIL" if r["fail"] else ("WARN" if r["warn"] else "PASS")

    counts = Counter(status(p) for p in files)
    print("=" * 70)
    print(f"EDF QC: {len(files)} file(s) checked in {edf_dir}")
    print(f"  PASS {counts['PASS']}   WARN {counts['WARN']}   FAIL {counts['FAIL']}")
    if parsed:
        print(f"  'Expected' = {basis}")
        print(f"  Expected channels ({len(expected_set)}): " + ", ".join(sorted(expected_set)))
    print("=" * 70)

    for p in files:
        s = status(p)
        if s == "PASS":
            continue
        print(f"\n[{s}] {p.relative_to(edf_dir)}")
        for msg in results[p]["fail"]:
            print(f"    FAIL: {msg}")
        for msg in results[p]["warn"]:
            print(f"    warn: {msg}")

    if counts["PASS"]:
        print(f"\nPASS ({counts['PASS']}): " + ", ".join(
            str(p.relative_to(edf_dir)) for p in files if status(p) == "PASS"))

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["file", "status", "format", "size_mb", "duration_h", "n_channels",
                        "start", "problems"])
            for p in files:
                h = parsed.get(p)
                issues = results[p]["fail"] + results[p]["warn"]
                w.writerow([
                    str(p.relative_to(edf_dir)), status(p),
                    h["format"] if h else "",
                    f"{h['size'] / (1024 * 1024):.1f}" if h else f"{p.stat().st_size / (1024 * 1024):.1f}",
                    f"{h['duration_h']:.2f}" if h else "",
                    len([c for c in h["channels"] if c["label"] != ANNOTATION_LABEL]) if h else "",
                    h["start"].isoformat(sep=" ") if h and h["start"] else "",
                    " | ".join(issues),
                ])
        print(f"\nWrote {args.csv}")

    if counts["FAIL"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
