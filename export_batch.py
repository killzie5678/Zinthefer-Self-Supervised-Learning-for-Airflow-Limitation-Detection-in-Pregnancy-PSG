"""
export_batch.py - export every recording in the "to export" queue to EDF+
using Noxturnal's command-line tool (NoxCmd.exe), no clicking.

    python export_batch.py "Batch 2 to export" "Batch 2 edf"
    python export_batch.py "Batch 2 to export" "Batch 2 edf" --limit 2
    python export_batch.py "Batch 2 to export" "Batch 2 edf" --dry-run
    python export_batch.py "Batch 2 to export" "Batch 2 edf" --workers 6

Each export only uses one CPU core, so by default 4 run at the same time
(--workers). Watch progress in a second window with export_watch.py.

Run it from PowerShell or cmd, NOT Git Bash (Git Bash rewrites NoxCmd's
/nox-style switches into file paths and nothing gets exported).

For each recording link in the queue folder (see export_queue.py):
    1. NoxCmd /details   -> the recording's real start time and length
    2. NoxCmd /export    -> an EDF+ in a staging folder (NOT inside the EDF
                            folder, so a half-written file never counts as done)
    3. checks: EDF+ header, a patient ID in the header, and that the EDF
       starts when the recording starts and is as long as the recording.
       (NoxCmd sometimes exports a shortened file - e.g. 5001 came out
       38 minutes short - so this check matters.)
    4. OK    -> moved to "<EDF folder>/<patient ID>.edf", the same naming as
                the GUI exports. If that name is taken (a second night for
                the same person), it becomes "<ID>_<folder code>.edf",
                e.g. "5006_c3e7.edf". Nothing is ever overwritten.
                The recording's link is then removed from the queue, so
                re-running the script picks up where it left off.
       SHORT -> kept in "<EDF folder> short exports" for reference; the link
                stays in the queue so you can export it through the GUI.
       FAILED-> nothing kept; the link stays in the queue.
    Every result is appended to "<EDF folder> export log.csv". Recordings
    already logged as SHORT or FAILED are skipped on later runs. Pass --retry
    to try the FAILED ones again (SHORT ones come out short every time, so
    they're never retried - export those in the GUI).

NoxCmd opens each recording the same way the Noxturnal GUI does, which
updates some files inside the recording folder (DATA.ndb, derived signals).
That is no different from exporting it by hand.

SAFETY: the only things this script deletes are its own temporary files in
the staging folder, empty annotation files NoxCmd creates alongside each
export, and queue LINKS (os.rmdir on a junction - never the recording).
"""

import argparse
import csv
import json
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

from export_progress import code_from_name
from export_queue import is_link, remove_link

NOXCMD = Path(r"C:\Program Files\Nox Medical\Noxturnal US\NoxCmd.exe")
EXPORT_TIMEOUT_S = 30 * 60
DETAILS_TIMEOUT_S = 5 * 60
START_TOLERANCE_S = 2     # EDF start vs recording start
LENGTH_TOLERANCE_S = 20   # EDF may end up to 2 data records before the recording does
LOG_FIELDS = ["time", "recording", "status", "edf", "patient_id",
              "recording_start", "recording_length", "edf_start", "edf_length", "note"]


def run_noxcmd(args, timeout):
    """Run NoxCmd, return (ok, output). ok means exit 0 and every task reported Success."""
    try:
        r = subprocess.run([str(NOXCMD), "/nox", *args], capture_output=True,
                           text=True, errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"NoxCmd timed out after {timeout} s"
    out = (r.stdout + r.stderr).strip()
    # Parallel NoxCmd processes share one log file, so log4net (Noxturnal's
    # logging library) prints errors about it. Those don't affect the task.
    # So judge success only by NoxCmd's own "Task: ..." lines: every task must be
    # followed by a "Success" line.
    lines = [l.strip() for l in out.splitlines() if l.strip() and not l.startswith("log4net:")]
    tasks = [i for i, l in enumerate(lines) if l.startswith("Task:")]
    ok = (r.returncode == 0 and bool(tasks)
          and all(i + 1 < len(lines) and lines[i + 1] == "Success" for i in tasks))
    return ok, "\n".join(lines)


def parse_nox_time(s):
    """'2015-06-25T21:30:01.05' -> datetime (fractional seconds of any length)."""
    s = re.sub(r"(\.\d{6})\d+", r"\1", s)
    return datetime.fromisoformat(s)


def recording_details(rec_path, staging):
    """(start, stop) of the recording from NoxCmd /details, or raise RuntimeError."""
    out_json = staging / "details.json"
    if out_json.exists():
        out_json.unlink()
    ok, out = run_noxcmd(["/recording", str(rec_path), "/details", "/output", str(staging)],
                         DETAILS_TIMEOUT_S)
    if not ok or not out_json.exists():
        raise RuntimeError("NoxCmd /details failed: " + last_line(out))
    text = out_json.read_text(encoding="utf-8-sig")
    out_json.unlink()
    info = json.loads(text)["Info"]
    return parse_nox_time(info["RecordingStart"]), parse_nox_time(info["RecordingStop"])


def read_edf_header(path):
    """The few EDF header fields we check, or raise RuntimeError."""
    with open(path, "rb") as f:
        h = f.read(256)
    if len(h) < 256:
        raise RuntimeError("EDF header is incomplete")
    txt = h.decode("latin-1")
    try:
        start = datetime.strptime(txt[168:184], "%d.%m.%y%H.%M.%S")
        n_records = int(txt[236:244])
        record_s = float(txt[244:252])
    except ValueError:
        raise RuntimeError("EDF header has unreadable start time / record count")
    patient = txt[8:88].strip()
    return {
        "reserved": txt[192:236].strip(),
        "patient_id": patient.split(" ")[0] if patient else "",
        "start": start,
        "length": timedelta(seconds=n_records * record_s),
    }


def last_line(text):
    lines = [l for l in text.splitlines() if l.strip()]
    return lines[-1][:200] if lines else "(no output)"


def fmt_td(td):
    s = int(round(td.total_seconds()))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def name_options(stem, code, start):
    """File names to try, in order: '<ID>.edf', then '<ID>_<folder code>.edf'
    (or '<ID>_<YYYYMMDD>.edf' when the folder name has no code, e.g.
    'Nox repeat 3rd trimester')."""
    return [f"{stem}.edf", f"{stem}_{code or start.strftime('%Y%m%d')}.edf"]


def free_name(folder, names):
    """The first of `names` not already in `folder`, or None if all are taken."""
    for name in names:
        if not (folder / name).exists():
            return folder / name
    return None


def export_one(name, rec_path, edf_dir, short_dir, staging, name_lock):
    """Export one recording. Returns a log row (dict).
    `staging` must be this worker's own folder; `name_lock` is held while
    picking a free file name and moving the file in, so parallel workers
    can't both claim the same name."""
    row = {"recording": name, "status": "FAILED", "edf": "", "patient_id": "",
           "recording_start": "", "recording_length": "", "edf_start": "", "edf_length": "", "note": ""}

    for p in staging.glob("export*"):      # leftovers from an interrupted run
        p.unlink()

    try:
        rec_start, rec_stop = recording_details(rec_path, staging)
    except (RuntimeError, OSError, ValueError, KeyError) as e:
        row["note"] = str(e)
        return row
    rec_len = rec_stop - rec_start
    row["recording_start"], row["recording_length"] = rec_start.isoformat(sep=" "), fmt_td(rec_len)

    ok, out = run_noxcmd(["/recording", str(rec_path), "/export", str(staging / "export.edf")],
                         EXPORT_TIMEOUT_S)
    produced = sorted(p for p in staging.glob("export*.edf"))
    signals = [p for p in produced if "annotations" not in p.name.lower()]
    annotations = [p for p in produced if "annotations" in p.name.lower()]
    if not ok or len(signals) != 1:
        row["note"] = (f"NoxCmd produced {len(signals)} signal EDFs; " if ok else "") + last_line(out)
        for p in produced:
            p.unlink()
        return row
    edf = signals[0]

    notes = []
    try:
        hdr = read_edf_header(edf)
    except (RuntimeError, OSError) as e:
        edf.unlink()
        row["note"] = str(e)
        return row
    row["patient_id"] = hdr["patient_id"]
    row["edf_start"], row["edf_length"] = hdr["start"].isoformat(sep=" "), fmt_td(hdr["length"])

    if not hdr["reserved"].startswith("EDF+"):
        notes.append(f"not EDF+ (header says '{hdr['reserved']}')")
    if not re.fullmatch(r"[A-Za-z0-9-]+", hdr["patient_id"]) or hdr["patient_id"] == "X":
        notes.append(f"no usable patient ID in header ('{hdr['patient_id']}')")
    # EDF start has whole-second precision; the recording's has fractions
    start_off = abs((hdr["start"] - rec_start.replace(microsecond=0)).total_seconds())
    missing = (rec_len - hdr["length"]).total_seconds()
    short = start_off > START_TOLERANCE_S or missing > LENGTH_TOLERANCE_S
    if short:
        notes.append(f"export starts {start_off:.0f} s after the recording and is "
                     f"{missing / 60:.1f} min shorter than it - export this one in the GUI")

    with name_lock:
        finish(name, row, edf, annotations, hdr, notes, short, edf_dir, short_dir)
    return row


def finish(name, row, edf, annotations, hdr, notes, short, edf_dir, short_dir):
    """Name the checked export and move it into place (call with name_lock held)."""
    # Annotation EDFs that NoxCmd writes are normally empty. An empty one is
    # deleted; a non-empty one is kept next to short exports, never in the EDF
    # folder (it has the same start time and would confuse export_progress.py).
    for a in annotations:
        try:
            empty = int(a.read_bytes()[236:244]) <= 0
        except (OSError, ValueError):
            empty = False
        if empty:
            a.unlink()
        else:
            short_dir.mkdir(exist_ok=True)
            dest = free_name(short_dir, name_options(f"{hdr['patient_id'] or name}_annotations",
                                                     code_from_name(name), hdr["start"]))
            if dest:
                shutil.move(str(a), dest)
                notes.append(f"annotations kept in {dest}")
            else:
                a.unlink()

    code = code_from_name(name)
    stem = hdr["patient_id"] if hdr["patient_id"] and not any("patient ID" in n for n in notes) else (code or "unknown")
    names = name_options(stem, code, hdr["start"])
    if short or any(n.startswith("not EDF+") for n in notes):
        short_dir.mkdir(exist_ok=True)
        dest = free_name(short_dir, names)
        row["status"] = "SHORT" if short else "FAILED"
    else:
        dest = free_name(edf_dir, names)
        row["status"] = "OK" if not notes else "CHECK"
    if dest is None:
        edf.unlink()
        row["status"] = "FAILED"
        notes.append(f"{' and '.join(names)} already exist - nothing overwritten")
    else:
        shutil.move(str(edf), dest)
        row["edf"] = str(dest)
    row["note"] = "; ".join(notes)


def read_log(log_path):
    if not log_path.exists():
        return {}
    with open(log_path, newline="", encoding="utf-8") as f:
        return {r["recording"]: r["status"] for r in csv.DictReader(f)}   # last entry wins


def append_log(log_path, row):
    new = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({"time": datetime.now().isoformat(sep=" ", timespec="seconds"), **row})


def main():
    ap = argparse.ArgumentParser(description="Batch-export queued Noxturnal recordings to EDF+.")
    ap.add_argument("queue_dir", help='Folder of links to recordings to export, e.g. "Batch 2 to export"')
    ap.add_argument("edf_dir", help='Folder for the EDFs, e.g. "Batch 2 edf"')
    ap.add_argument("--limit", type=int, default=None, help="Export at most this many recordings this run")
    ap.add_argument("--retry", action="store_true", help="Also retry recordings logged as FAILED")
    ap.add_argument("--workers", type=int, default=4,
                    help="Exports to run at the same time (default 4). Each NoxCmd uses one CPU core.")
    ap.add_argument("--dry-run", action="store_true", help="List what would be exported; export nothing")
    args = ap.parse_args()

    queue_dir, edf_dir = Path(args.queue_dir).resolve(), Path(args.edf_dir).resolve()
    if not NOXCMD.exists():
        sys.exit(f"ERROR: NoxCmd.exe not found at {NOXCMD}")
    for label, p in (("Queue folder", queue_dir), ("EDF folder", edf_dir)):
        if not p.is_dir():
            sys.exit(f"ERROR: {label} not found: {p}")
    short_dir = edf_dir.parent / f"{edf_dir.name} short exports"
    staging = edf_dir.parent / f"{edf_dir.name} staging"
    log_path = edf_dir.parent / f"{edf_dir.name} export log.csv"

    logged = read_log(log_path)
    todo, skipped = [], []
    for p in sorted(queue_dir.iterdir()):
        if not is_link(p) or not p.exists() or not (p / "DATA.ndb").exists():
            continue          # export_queue.py reports anything odd in the queue folder
        status = logged.get(p.name)
        if status == "SHORT" or (status == "FAILED" and not args.retry):
            skipped.append(p.name)
            continue
        todo.append(p)
    if args.limit is not None:
        todo = todo[:args.limit]

    print(f"{len(todo)} recording(s) to export -> {edf_dir}")
    if skipped:
        print(f"  ({len(skipped)} skipped: logged as SHORT/FAILED before; --retry to include the FAILED ones)")
    if args.dry_run:
        for p in todo:
            print(f"  {p.name}")
        print("DRY RUN - nothing exported.")
        return
    if not todo:
        return
    workers = max(1, min(args.workers, len(todo)))
    staging.mkdir(exist_ok=True)
    free_staging = queue.Queue()           # one private staging folder per worker
    for w in range(workers):
        d = staging / f"worker{w + 1}"
        d.mkdir(exist_ok=True)
        free_staging.put(d)
    name_lock = threading.Lock()

    def job(link):
        d = free_staging.get()
        try:
            t = time.time()
            # export from the real folder, not through the link
            row = export_one(link.name, link.resolve(), edf_dir, short_dir, d, name_lock)
            return row, time.time() - t
        finally:
            free_staging.put(d)

    print(f"Running {workers} export(s) at a time.", flush=True)
    counts = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(job, link): link for link in todo}
        try:
            for i, fut in enumerate(as_completed(futures), 1):
                link = futures[fut]
                try:
                    row, took = fut.result()
                except Exception as e:     # a bug or I/O error in one export shouldn't stop the batch
                    row, took = {"recording": link.name, "status": "FAILED", "edf": "",
                                 "note": f"{type(e).__name__}: {e}"}, 0.0
                append_log(log_path, row)
                counts[row["status"]] = counts.get(row["status"], 0) + 1
                if row["status"] in ("OK", "CHECK"):
                    try:
                        remove_link(link)
                    except OSError as e:
                        row["note"] += f"; could not remove queue link ({e})"
                eta = (time.time() - t0) / i * (len(todo) - i)
                print(f"[{i}/{len(todo)}] {link.name} ... {row['status']} "
                      f"{Path(row['edf']).name if row['edf'] else ''} "
                      f"({took / 60:.1f} min, ~{eta / 3600:.1f} h left)"
                      + (f"\n      {row['note']}" if row["note"] else ""), flush=True)
        except KeyboardInterrupt:
            # Ctrl+C also stops the running NoxCmd processes; their recordings
            # stay in the queue and are redone on the next run.
            for f in futures:
                f.cancel()
            raise

    for d in staging.iterdir():            # only removes empty folders
        try:
            d.rmdir()
        except OSError:
            pass
    try:
        staging.rmdir()
    except OSError:
        pass
    print("=" * 70)
    print("DONE  " + "   ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    print(f"log: {log_path}")
    if counts.get("SHORT") or counts.get("FAILED") or counts.get("CHECK"):
        print("SHORT/FAILED recordings are still in the queue - export those in the GUI.")
        print("CHECK = exported, but read the note in the log.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Run the script again to continue where it left off.")
