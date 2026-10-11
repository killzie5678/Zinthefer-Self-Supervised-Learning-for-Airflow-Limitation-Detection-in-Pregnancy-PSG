"""
export_queue.py - keep a "to export" folder that only contains recordings you
have NOT exported yet, WITHOUT copying any recording data.

    python export_queue.py "Batch 2 flat" "Batch 2 edf" "Batch 2 to export"

How it works:
    - "Batch 2 flat" is the master set. This script NEVER modifies, moves,
      renames, or deletes anything in it.
    - "Batch 2 to export" is filled with LINKS (Windows "junctions") that point
      at the recording folders in "Batch 2 flat". They look and open like normal
      folders in File Explorer and Noxturnal, but take no disk space.
    - Each time you run the script it re-checks "Batch 2 edf" and:
        * adds a link for every recording that has no valid EDF yet
        * removes the link for every recording that now has one
      Removing a link only removes the shortcut. The real recording stays in
      "Batch 2 flat".
    - A recording whose EDF looks bad (unreadable header, or smaller than
      --min-mb) is NOT counted as exported, so it stays in the queue to be redone.
    - Recordings the tracker can't match confidently (see export_progress.py)
      also stay in the queue, because the safe assumption is "not done yet".

Run it BETWEEN recordings, after you've closed the recording you just exported
(Noxturnal may still be using the folder while it's open). Run it again any
time to refresh.

Options:
    --dry-run   show what would change, change nothing
    --limit N   only add up to N new links this run (use --limit 3 the first
                time, to confirm Noxturnal opens recordings through the links)

Needs export_progress.py in the same folder (it reuses its matching logic).

SAFETY: the only things this script ever deletes are links inside the
"to export" folder. It uses os.rmdir/os.unlink (which cannot delete a folder's
contents) and never recursive deletion. Anything inside the "to export"
folder that is a real folder or file, not a link, is left alone and reported.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from export_progress import edfs_of, find_edfs, find_recordings, match_all


def is_link(path):
    """True for symlinks and Windows junctions (any reparse point)."""
    if os.path.islink(path):
        return True
    attrs = getattr(os.lstat(path), "st_file_attributes", 0)  # Windows only
    return bool(attrs & 0x400)  # FILE_ATTRIBUTE_REPARSE_POINT


IS_WINDOWS = os.name == "nt"


def _junction_via_winapi(target, link_path):
    """Create the junction inside Python itself - no separate process, so it's fast."""
    import _winapi  # part of CPython on Windows
    _winapi.CreateJunction(str(target), str(link_path))


def _junction_via_mklink(target, link_path):
    """Fallback: ask Windows' own mklink command (slower: one process per link)."""
    try:
        r = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link_path), str(target)],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        raise OSError("mklink took longer than 30 seconds")
    if r.returncode != 0:
        raise OSError((r.stdout + r.stderr).strip() or "mklink failed")


def make_link(link_path, target):
    """Create a directory link at link_path pointing to target."""
    target = Path(target).resolve()
    link_path = os.path.abspath(link_path)
    if IS_WINDOWS:
        # Junctions need no admin rights, unlike symlinks. Local drives only.
        try:
            _junction_via_winapi(target, link_path)
        except (ImportError, AttributeError):
            _junction_via_mklink(target, link_path)
    else:
        os.symlink(target, link_path, target_is_directory=True)


def remove_link(link_path):
    """Remove ONLY the link, never what it points to."""
    if IS_WINDOWS:
        os.rmdir(link_path)   # removes a junction/dir-symlink itself, fails if it's a real non-empty folder
    else:
        os.unlink(link_path)


def main():
    ap = argparse.ArgumentParser(description="Maintain a folder of links to not-yet-exported recordings.")
    ap.add_argument("recordings_dir", help='Master folder, e.g. "Batch 2 flat" (never modified)')
    ap.add_argument("edf_dir", help='Folder of exported EDFs, e.g. "Batch 2 edf"')
    ap.add_argument("queue_dir", help='Folder to fill with links to unexported recordings, e.g. "Batch 2 to export"')
    ap.add_argument("--min-mb", type=float, default=1.0,
                    help="EDFs smaller than this don't count as exported (default 1)")
    ap.add_argument("--time-tolerance", type=int, default=2,
                    help="Seconds of start-time difference allowed when matching (default 2)")
    ap.add_argument("--dry-run", action="store_true", help="Show what would change; change nothing")
    ap.add_argument("--limit", type=int, default=None, help="Add at most this many new links this run")
    args = ap.parse_args()

    rec_dir, edf_dir, queue_dir = Path(args.recordings_dir), Path(args.edf_dir), Path(args.queue_dir)
    for label, p in (("Recordings folder", rec_dir), ("EDF folder", edf_dir)):
        if not p.is_dir():
            print(f"ERROR: {label} not found: {p}")
            sys.exit(1)
    if queue_dir.resolve() in (rec_dir.resolve(), edf_dir.resolve()):
        print("ERROR: the queue folder must be different from the recordings and EDF folders.")
        sys.exit(1)

    recs = find_recordings(rec_dir)
    edfs = find_edfs(edf_dir)
    match_all(recs, edfs, args.time_tolerance, edf_dir)

    exported, requeued, no_ndb = set(), [], []
    for r in recs:
        es = edfs_of(r)   # one EDF normally; several when recordings share a start time
        if not es:
            continue
        bad = [e for e in es if e["problem"] or e["size_mb"] < args.min_mb]
        if bad:
            requeued.append((r["name"], ", ".join(str(e["rel"]) for e in bad)))   # looks bad: redo it
        else:
            exported.add(r["name"])

    wanted = []  # recordings that belong in the queue
    for r in recs:
        if r["name"] in exported:
            continue
        if not r["has_ndb"]:
            no_ndb.append(r["name"])
            continue
        wanted.append(r["name"])

    if not args.dry_run:
        queue_dir.mkdir(parents=True, exist_ok=True)
    existing = {}
    if queue_dir.is_dir():
        existing = {p.name: p for p in queue_dir.iterdir()}

    added, removed, kept, blocked, failed, repaired = [], [], 0, [], [], []

    # Remove links for recordings that are no longer wanted (exported, or gone from the master folder)
    for name, p in sorted(existing.items()):
        if name in wanted:
            continue
        if is_link(p):
            if not args.dry_run:
                try:
                    remove_link(p)
                except OSError as e:
                    failed.append((name, str(e)))
                    continue
            removed.append(name)
        else:
            blocked.append(name)  # a real folder/file: never touched

    # Add links for wanted recordings that have none (or have a broken one)
    print(f"Updating links for {len(wanted)} recording(s) still to export "
          "(progress is printed every 25)...", flush=True)
    new_budget = args.limit
    for name in wanted:
        p = existing.get(name)
        if p is not None and is_link(p):
            if p.exists():          # link works
                kept += 1
                continue
            if not args.dry_run:    # broken link: replace it
                try:
                    remove_link(p)
                except OSError as e:
                    failed.append((name, str(e)))
                    continue
        elif p is not None:
            # A real (non-link) folder with this name is in the way. If it is EMPTY it is almost
            # certainly a leftover from an interrupted run, and os.rmdir can only remove empty
            # folders, so replacing it is safe. Anything with contents is left alone.
            if p.is_dir() and not any(p.iterdir()):
                if not args.dry_run:
                    try:
                        os.rmdir(p)
                    except OSError as e:
                        failed.append((name, str(e)))
                        continue
                repaired.append(name)
            else:
                blocked.append(name)
                continue
        if new_budget is not None and new_budget <= 0:
            continue
        if not args.dry_run:
            try:
                make_link(queue_dir / name, rec_dir / name)
            except OSError as e:
                failed.append((name, str(e)))
                continue
        added.append(name)
        if not args.dry_run and len(added) % 25 == 0:
            print(f"  ...{len(added)} links created so far", flush=True)
        if new_budget is not None:
            new_budget -= 1

    tag = "  (DRY RUN - nothing changed)" if args.dry_run else ""
    print("=" * 70)
    print(f"EXPORT QUEUE{tag}")
    extra = f" (+{len(no_ndb)} with no DATA.ndb, not queued)" if no_ndb else ""
    print(f"  {len(exported)} of {len(recs)} recordings exported; {len(wanted)} still to do{extra}")
    print(f"  master (untouched): {rec_dir}")
    print(f"  queue:              {queue_dir}")
    print("=" * 70)
    print(f"  links added: {len(added)}   removed: {len(removed)}   already in queue: {kept}")
    if args.limit is not None and len(added) >= args.limit:
        print(f"  (--limit {args.limit} reached; run again without --limit to add the rest)")
    if removed:
        print("\nREMOVED from queue (now exported, or no longer in the master folder):")
        for n in removed:
            print(f"  {n}")
    if repaired:
        print(f"\nREPAIRED {len(repaired)} empty leftover folder(s) (from an interrupted run), replaced with links.")
    if requeued:
        print("\nKEPT IN QUEUE because their EDF looks bad (unreadable or too small) - redo these:")
        for n, rel in requeued:
            print(f"  {n}   (EDF: {rel})")
    if no_ndb:
        print("\nNOT QUEUED (no DATA.ndb in the folder, so there's nothing to open):")
        for n in no_ndb:
            print(f"  {n}")
    if blocked:
        print("\nLEFT ALONE (a real folder/file with this name is in the queue folder, not a link):")
        for n in blocked:
            print(f"  {n}")
    if failed:
        print("\nFAILED:")
        for n, why in failed:
            print(f"  {n}: {why}")
    print("\nNothing in the master folder was modified.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Nothing in the master folder was touched. Run the script again to pick up where it left off.")