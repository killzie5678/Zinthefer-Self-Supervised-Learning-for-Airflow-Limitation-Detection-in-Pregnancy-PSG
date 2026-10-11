"""
export_progress.py - track manual EDF export progress for a batch.

Compares the recording folders you still need to export (e.g. "Batch 2 flat",
one subfolder per recording, each containing DATA.ndb) against the EDF files
you've already exported (e.g. "Batch 2 edf") and tells you:

    - which recordings are DONE (have a matching EDF)
    - which recordings are STILL TO DO
    - which EDF files don't match any recording (misnamed? wrong batch?)
    - which EDFs look suspiciously small or unreadable (possible bad export)

Usage (PowerShell, from the folder containing this script):
    python export_progress.py "Batch 2 flat" "Batch 2 edf"
    python export_progress.py "Batch 2 flat" "Batch 2 edf" --csv progress.csv
    python export_progress.py "Batch 2 flat" "Batch 2 edf" --debug

HOW MATCHING WORKS (read this once - it decides how much you can trust "done"):
    Most recording folders are named "<start time> - <code>", for example
    "20151009T220001 - c3e7". They usually do NOT contain the patient ID, so
    EDFs are matched to folders in three steps, in this order:

   -1. EXPORT LOG. EDFs made by export_batch.py are listed in
       "<EDF folder> export log.csv" with the exact folder they came from,
       so those are matched with no guessing. The steps below are only
       needed for EDFs exported by hand in the GUI.
    0. CODE IN FILENAME. If an EDF's filename contains a folder's code
       (e.g. "5006_c3e7.edf" contains "c3e7"), that's an exact match. This is
       the only fully reliable way to pair two recordings that share a start
       time, so use it for those (the report lists which ones).
    1. START TIME. The folder name's start time vs the start time stored in
       each EDF header (within --time-tolerance seconds).
         - one folder + one EDF at that time -> matched.
         - N folders + N EDFs all at the same time -> all N counted as done,
           but WHICH EDF belongs to WHICH folder is not determined. These are
           listed under PAIRING UNCERTAIN.
         - any other mix (e.g. 2 folders but only 1 EDF) -> NOT counted as
           done. The tracker can't know which folder you exported, so both
           stay on the to-do list and are shown under COULD NOT MATCH.
    2. ID IN THE NAME. A standalone 4-digit number shared by the folder name
       and the EDF filename, used only when it points to exactly one
       candidate on each side.

    One limit to know about: if you export the SAME recording twice under
    different names, the extra EDF can make a recording look done when it
    isn't (the report warns when two EDFs in a shared-time group have
    identical size).

    This script only READS file names and the first 256 bytes of each EDF.
    It never modifies, moves, or deletes anything.
"""

import argparse
import csv
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Folder names look like "5017-20151105T220001 - eadf" or "20150625T213001 - 4161"
TIMESTAMP_RE = re.compile(r"(\d{8})T(\d{6})")
# A standalone 4-digit number: not part of a longer run of digits (so the
# 8-digit date in a timestamp is not mistaken for an ID).
ID_RE = re.compile(r"(?<!\d)\d{4}(?!\d)")
# The trailing " - code" at the end of a folder name
CODE_RE = re.compile(r"\s-\s(\S{4,})\s*$")


def timestamp_from_name(name):
    m = TIMESTAMP_RE.search(name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def ids_from_name(name):
    return set(ID_RE.findall(name))


def code_from_name(name):
    m = CODE_RE.search(name)
    return m.group(1).lower() if m else None


def code_in_text(code, text):
    """True if `code` appears in `text` as its own token (not inside a longer alphanumeric run)."""
    return re.search(r"(?<![0-9a-z])" + re.escape(code) + r"(?![0-9a-z])", text.lower()) is not None


def read_edf_start(path):
    """
    Returns (start_datetime_or_None, problem_or_None) from the first 256
    bytes of an EDF. Never raises: an unreadable file just gets a problem
    string, which the report surfaces.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(256)
    except OSError as e:
        return None, f"cannot open ({e})"

    if len(head) < 256:
        return None, "file shorter than an EDF header (incomplete export?)"
    if head[0:8].decode("ascii", "replace").strip() != "0":
        return None, "does not look like an EDF file"

    try:
        date_s = head[168:176].decode("ascii").strip()  # dd.mm.yy
        time_s = head[176:184].decode("ascii").strip()  # hh.mm.ss
        d, mo, yy = (int(x) for x in date_s.split("."))
        h, mi, s = (int(x) for x in time_s.split("."))
        year = 1900 + yy if yy >= 85 else 2000 + yy  # EDF spec's 2-digit year rule
        return datetime(year, mo, d, h, mi, s), None
    except (ValueError, UnicodeDecodeError):
        return None, "start date/time in header could not be read"


def find_recordings(recordings_dir):
    """One entry per immediate subfolder of recordings_dir."""
    recs = []
    for sub in sorted(p for p in recordings_dir.iterdir() if p.is_dir()):
        recs.append({
            "name": sub.name,
            "has_ndb": (sub / "DATA.ndb").exists(),
            "ts": timestamp_from_name(sub.name),
            "ids": ids_from_name(sub.name),
            "code": code_from_name(sub.name),
            "edf": None,          # a single matched EDF, or...
            "group_edfs": None,   # ...a shared list of EDFs when pairing within a group is undetermined
            "method": None,
        })
    return recs


def find_edfs(edf_dir):
    edfs = []
    for p in sorted(edf_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() == ".edf":
            start, problem = read_edf_start(p)
            nbytes = p.stat().st_size
            edfs.append({
                "path": p,
                "rel": p.relative_to(edf_dir),
                "stem": p.stem,
                "bytes": nbytes,
                "size_mb": nbytes / (1024 * 1024),
                "start": start,
                "problem": problem,
                "ids": ids_from_name(p.stem),
                "rec": None,
            })
    return edfs


def edfs_of(rec):
    """All EDFs credited to a recording (1 for a normal match, several for an undetermined group)."""
    if rec["group_edfs"]:
        return rec["group_edfs"]
    return [rec["edf"]] if rec["edf"] else []


def link(rec, edf, method):
    rec["edf"], rec["method"] = edf, method
    edf["rec"] = rec


def cluster_by_time(items, key, gap_s):
    """Group items whose timestamps are within gap_s seconds of their neighbour."""
    timed = sorted((i for i in items if key(i) is not None), key=key)
    clusters = []
    for it in timed:
        if clusters and (key(it) - key(clusters[-1][-1])).total_seconds() <= gap_s:
            clusters[-1].append(it)
        else:
            clusters.append([it])
    return clusters


def export_log_path(edf_dir):
    """Where export_batch.py logs its exports, e.g. 'Batch 2 edf export log.csv'."""
    edf_dir = Path(edf_dir).resolve()
    return edf_dir.parent / f"{edf_dir.name} export log.csv"


def read_export_log(edf_dir):
    """{recording folder name: EDF path} for every successful export_batch.py export."""
    log_path = export_log_path(edf_dir)
    if not log_path.exists():
        return {}
    exported = {}
    with open(log_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["status"] in ("OK", "CHECK") and row["edf"]:
                exported[row["recording"]] = Path(row["edf"]).resolve()
    return exported


def match_all(recs, edfs, tolerance_s, edf_dir=None):
    """
    Links recordings to EDFs (see the module docstring for the rules).
    Returns a list of (recording, candidate_edfs, reason) for recordings that
    could not be matched confidently.
    """
    ambiguous = {}  # id(rec) -> (rec, cands, reason)

    # Pass -1: export_batch.py's log says exactly which folder each EDF came
    # from - no guessing needed (only if that EDF is still where it was saved).
    if edf_dir is not None:
        by_path = {e["path"].resolve(): e for e in edfs}
        for rec_name, edf_path in read_export_log(edf_dir).items():
            rec = next((r for r in recs if r["name"] == rec_name), None)
            e = by_path.get(edf_path)
            if rec is not None and e is not None and e["rec"] is None and not edfs_of(rec):
                link(rec, e, "export log")

    # Pass 0: the folder's code appears in the EDF filename -> exact match
    for rec in recs:
        code = rec["code"]
        if not code or sum(1 for r in recs if r["code"] == code) != 1:
            continue
        cands = [e for e in edfs if e["rec"] is None and code_in_text(code, e["stem"])]
        if len(cands) == 1:
            link(rec, cands[0], "code in filename")

    # Pass 1: start time, judged cluster by cluster so shared start times are handled safely
    tol = timedelta(seconds=tolerance_s)
    free = [r for r in recs if not edfs_of(r) and r["ts"] is not None]
    for cluster in cluster_by_time(free, lambda r: r["ts"], 2 * tolerance_s):
        lo, hi = cluster[0]["ts"] - tol, cluster[-1]["ts"] + tol
        cands = [e for e in edfs if e["rec"] is None and e["start"] is not None and lo <= e["start"] <= hi]
        k, m = len(cluster), len(cands)
        if m == 0:
            continue
        if k == 1 and m == 1:
            link(cluster[0], cands[0], "start time")
        elif k >= 2 and m == k:
            for r in cluster:
                r["group_edfs"] = cands
                r["method"] = f"start time; {k} recordings share it, pairing undetermined"
            for e in cands:
                e["rec"] = cluster[0]
        else:
            for r in cluster:
                ambiguous[id(r)] = (r, cands,
                                    f"{k} recording(s) share this start time but {m} EDF(s) were found there")

    # Pass 2: shared 4-digit ID, only when unambiguous in BOTH directions
    for rec in recs:
        if edfs_of(rec) or not rec["ids"]:
            continue
        cands = [e for e in edfs if e["rec"] is None and e["ids"] & rec["ids"]]
        if len(cands) == 1:
            others = [r for r in recs
                      if r is not rec and not edfs_of(r) and r["ids"] & cands[0]["ids"]]
            if not others:
                link(rec, cands[0], "ID in name")
                continue
        if cands and id(rec) not in ambiguous:
            ambiguous[id(rec)] = (rec, cands, "shares a 4-digit ID with more than one candidate")

    # Drop candidates that a later pass has since matched to a different recording, and drop
    # recordings left with no candidates at all (they're simply not exported yet).
    result = []
    for rec, cands, why in ambiguous.values():
        if edfs_of(rec):
            continue
        live = [c for c in cands if c["rec"] is None]
        if live:
            result.append((rec, live, why))
    return result


def main():
    ap = argparse.ArgumentParser(description="Track manual EDF export progress for a batch.")
    ap.add_argument("recordings_dir", help='Folder of recordings to export, e.g. "Batch 2 flat"')
    ap.add_argument("edf_dir", help='Folder where exported EDFs are saved, e.g. "Batch 2 edf"')
    ap.add_argument("--csv", help="Also write the full per-recording status to this CSV file")
    ap.add_argument("--min-mb", type=float, default=1.0,
                    help="Flag EDFs smaller than this many MB as suspicious (default 1). "
                         "Raise it once you know what a normal export size looks like.")
    ap.add_argument("--time-tolerance", type=int, default=2,
                    help="Seconds of difference allowed when matching start times (default 2)")
    ap.add_argument("--debug", action="store_true",
                    help="Also show what was read from each UNMATCHED folder name and EDF "
                         "(timestamp, IDs, code), to see why they didn't pair up")
    args = ap.parse_args()

    rec_dir, edf_dir = Path(args.recordings_dir), Path(args.edf_dir)
    for label, p in (("Recordings folder", rec_dir), ("EDF folder", edf_dir)):
        if not p.is_dir():
            print(f"ERROR: {label} not found: {p}")
            sys.exit(1)

    recs = find_recordings(rec_dir)
    edfs = find_edfs(edf_dir)
    if not recs:
        print(f"No recording subfolders found in {rec_dir}.")
        sys.exit(0)

    ambiguous = match_all(recs, edfs, args.time_tolerance, edf_dir)

    done = [r for r in recs if edfs_of(r)]
    todo = [r for r in recs if not edfs_of(r)]
    orphans = [e for e in edfs if e["rec"] is None]
    suspicious = [e for e in edfs if e["problem"] or e["size_mb"] < args.min_mb]

    print("=" * 70)
    print(f"EXPORT PROGRESS: {len(done)} of {len(recs)} recordings exported "
          f"({100 * len(done) / len(recs):.0f}%), {len(todo)} to go")
    print(f"  recordings folder: {rec_dir}")
    print(f"  EDF folder:        {edf_dir}  ({len(edfs)} EDF file(s) found)")
    print("=" * 70)

    if done:
        print(f"\nDONE ({len(done)}):  recording folder  ->  EDF   [how it matched]")
        for r in done:
            if r["group_edfs"]:
                names = " or ".join(str(e["rel"]) for e in r["group_edfs"])
                print(f"  {r['name']}  ->  {names}   [{r['method']}]")
            else:
                print(f"  {r['name']}  ->  {r['edf']['rel']}   [{r['method']}]")

    # Groups of recordings that share a start time and whose EDF pairing is undetermined
    groups, seen = [], set()
    for r in done:
        if r["group_edfs"] and id(r["group_edfs"]) not in seen:
            seen.add(id(r["group_edfs"]))
            groups.append(([x for x in done if x["group_edfs"] is r["group_edfs"]], r["group_edfs"]))
    if groups:
        print(f"\nPAIRING UNCERTAIN ({len(groups)} group(s)): each group's recordings share a start "
              "time, so all count as exported but which EDF belongs to which folder is unknown:")
        for members, es in groups:
            print("  recordings: " + ", ".join(m["name"] for m in members))
            print("  EDFs:       " + ", ".join(str(e["rel"]) for e in es))
            if len({e["bytes"] for e in es}) < len(es):
                print("  (!) two of these EDFs have identical size - possibly the SAME recording "
                      "exported twice, in which case one of these recordings is NOT really exported")

    if todo:
        ambiguous_names = {rec["name"] for rec, _, _ in ambiguous}
        print(f"\nSTILL TO EXPORT ({len(todo)}):")
        for r in todo:
            note = "" if r["has_ndb"] else "   (!) no DATA.ndb in this folder"
            if r["name"] in ambiguous_names:
                note += "   (?) an EDF might already exist - see 'could not match' below"
            print(f"  {r['name']}{note}")

    if ambiguous:
        print(f"\nCOULD NOT MATCH CONFIDENTLY ({len(ambiguous)}) - check by hand:")
        for rec, cands, why in ambiguous:
            print(f"  {rec['name']}  could be: " + ", ".join(str(c['rel']) for c in cands))
            print(f"      why: {why}")
        print("  Tip: to settle these for good, rename the EDF to include the folder's code "
              "(the text after ' - '), e.g. 5006_c3e7.edf, then run this again.")

    # Recordings that share a start time with another recording and aren't both done yet
    shared = [c for c in cluster_by_time(recs, lambda r: r["ts"], 2 * args.time_tolerance)
              if len(c) >= 2 and any(not edfs_of(r) for r in c)]
    if shared:
        print(f"\nSHARE A START TIME WITH ANOTHER RECORDING ({len(shared)} group(s) not fully exported):")
        print("  Start time can't tell these apart. When you export one, include its code (the text")
        print("  after ' - ') in the EDF filename, e.g. 5006_c3e7.edf, so it can be matched exactly.")
        for c in shared[:30]:
            print("  " + "  |  ".join(r["name"] + (" (done)" if edfs_of(r) else "") for r in c))
        if len(shared) > 30:
            print(f"  ... and {len(shared) - 30} more group(s)")

    if orphans:
        print(f"\nEDF FILES MATCHING NO RECORDING ({len(orphans)}) - misnamed, from another batch, or a duplicate?")
        for e in orphans:
            print(f"  {e['rel']}")

    if suspicious:
        print(f"\nSUSPICIOUS EDF FILES ({len(suspicious)}) - re-export or inspect these:")
        for e in suspicious:
            why = e["problem"] or (f"only {e['size_mb']:.2f} MB, below the {args.min_mb:g} MB threshold "
                                   f"- compare with your other exports (change with --min-mb)")
            print(f"  {e['rel']}: {why}")
        print("  (For a thorough integrity check, run edf_qc.py on the EDF folder.)")

    if not edfs:
        print("\nNo EDF files found yet in the EDF folder - nothing exported, or wrong folder?")

    if args.debug:
        print("\n" + "-" * 70)
        print("DEBUG: what the matcher saw (unmatched items only)")
        print("-" * 70)
        print(f"\nUnmatched RECORDING FOLDERS ({len(todo)}): name | start time found in name | 4-digit IDs | code")
        for r in todo[:40]:
            ts = r["ts"].isoformat(sep=" ") if r["ts"] else "none"
            print(f"  {r['name']} | {ts} | {', '.join(sorted(r['ids'])) or 'none'} | {r['code'] or 'none'}")
        if len(todo) > 40:
            print(f"  ... and {len(todo) - 40} more")
        print(f"\nUnmatched EDF FILES ({len(orphans)}): name | start time in EDF header | 4-digit IDs found in filename")
        for e in orphans:
            st = e["start"].isoformat(sep=" ") if e["start"] else f"unreadable ({e['problem']})"
            print(f"  {e['rel']} | {st} | {', '.join(sorted(e['ids'])) or 'none'}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["recording_folder", "status", "edf_file", "matched_by", "edf_size_mb"])
            for r in recs:
                es = edfs_of(r)
                if es:
                    w.writerow([r["name"], "done", " | ".join(str(e["rel"]) for e in es), r["method"],
                                " | ".join(f"{e['size_mb']:.1f}" for e in es)])
                else:
                    w.writerow([r["name"], "to do", "", "", ""])
            for e in orphans:
                w.writerow(["", "edf with no recording", str(e["rel"]), "", f"{e['size_mb']:.1f}"])
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()