"""
export_watch.py - live progress bar for a running export_batch.py.

    python export_watch.py "Batch 2 to export" "Batch 2 edf"

Run it in a second window while export_batch.py is running. It only READS
the export log, the queue folder and the staging folder; it never changes
anything, so it's safe to start, stop (Ctrl+C) and restart at any time.
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

from export_queue import is_link

BAR_WIDTH = 40
TYPICAL_EDF_MB = 520   # a full 8-hour recording; only used for the current-file bar


def bar(fraction, width=BAR_WIDTH):
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {fraction * 100:5.1f}%"


def snapshot(queue_dir, log_path, staging, since):
    rows = []
    if log_path.exists():
        with open(log_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    last_status = {r["recording"]: r["status"] for r in rows}
    this_run = [r for r in rows if datetime.fromisoformat(r["time"]) >= since]

    # still to do = queue links not already logged as SHORT/FAILED (those are skipped)
    remaining = 0
    for p in queue_dir.iterdir():
        try:
            linked = is_link(p)
        except OSError:      # export_batch.py removed this link just now (it was exported)
            continue
        if linked and last_status.get(p.name) not in ("SHORT", "FAILED"):
            remaining += 1
    # a recording that fails during this run stays in the queue but is now logged, so it isn't counted above

    in_progress = []   # MB written so far, one entry per export running right now
    if staging.is_dir():
        for p in staging.rglob("export*.edf"):
            if "annotations" not in p.name.lower():
                try:
                    in_progress.append(p.stat().st_size / 1024 / 1024)
                except OSError:  # finished and moved into place just now
                    pass
    return this_run, remaining, sorted(in_progress, reverse=True)


def main():
    ap = argparse.ArgumentParser(description="Progress bar for export_batch.py")
    ap.add_argument("queue_dir")
    ap.add_argument("edf_dir")
    ap.add_argument("--since", default=None,
                    help='Count exports logged after this time, e.g. "2026-10-04 21:07" '
                         "(default: start of the current export run, read from the log)")
    ap.add_argument("--every", type=float, default=5, help="Seconds between refreshes (default 5)")
    args = ap.parse_args()

    queue_dir, edf_dir = Path(args.queue_dir).resolve(), Path(args.edf_dir).resolve()
    log_path = edf_dir.parent / f"{edf_dir.name} export log.csv"
    staging = edf_dir.parent / f"{edf_dir.name} staging"
    if not queue_dir.is_dir():
        sys.exit(f"ERROR: queue folder not found: {queue_dir}")

    if args.since:
        since = datetime.fromisoformat(args.since)
    else:
        # the run started when its staging folder was created (export_batch.py makes it at start)
        since = datetime.fromtimestamp(staging.stat().st_ctime) if staging.is_dir() else datetime.now()

    while True:
        this_run, remaining, in_progress = snapshot(queue_dir, log_path, staging, since)
        done = len(this_run)
        total = done + remaining
        counts = {}
        for r in this_run:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        elapsed = (datetime.now() - since).total_seconds()
        if done:
            eta_h = elapsed / done * remaining / 3600
            eta = f"~{eta_h:.1f} h left (about {datetime.fromtimestamp(time.time() + eta_h * 3600):%a %H:%M})"
        else:
            eta = "estimating after the first recording..."

        lines = [
            f"Batch export progress   (started {since:%a %H:%M}, refreshed {datetime.now():%H:%M:%S})",
            "",
            f"Overall  {bar(done / total if total else 1)}  {done} of {total} recordings",
            f"         {eta}",
            "         " + ("   ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "no results yet"),
            "",
        ]
        if in_progress:
            for n, mb in enumerate(in_progress, 1):
                lines.append(f"{'Running' if n == 1 else '':8} {bar(mb / TYPICAL_EDF_MB)}  {mb:.0f} MB written")
        else:
            lines.append("Running  (reading recordings / between files)")
        if this_run:
            last = this_run[-1]
            lines += ["", f"Last:    {last['status']}  {last['recording']}  -> {Path(last['edf']).name if last['edf'] else '-'}"]
            if last["note"]:
                lines.append(f"         {last['note'][:110]}")
        if remaining == 0:
            lines += ["", "Queue is empty - all done."]

        sys.stdout.write("\x1b[2J\x1b[H" + "\n".join(lines) + "\n")
        sys.stdout.flush()
        if remaining == 0:
            return
        time.sleep(args.every)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped watching. (The export itself keeps running.)")
