"""
make_upload_tree.py - lay the exported EDFs out in the same folder structure
as the original download, ready to upload to Google Drive.

    python make_upload_tree.py "Batch 2" "Batch 2 flat" "Batch 2 edf" "Batch 2 tri1 EDF"

For every recording folder in the original download, e.g.
    Batch 2/Batch 2 tri1-...-1-001/Batch 2 tri1/5190-5200/5194/5194/20171109T220004 - c410/
its EDF is placed at
    Batch 2 tri1 EDF/5190-5200/5194/5194/20171109T220004 - c410/5194.edf
(the "Batch 2 tri1-...-1-00N" level is only Google Drive splitting the download
into several zips, so it is dropped). A recording that appears in two places in
the download gets its EDF in both places.

Files are HARD LINKS to the EDFs in "Batch 2 edf": they take no extra disk
space and upload like ordinary files. Don't edit them - a hard link is the same
file, so an edit would change the one in "Batch 2 edf" too.

Checks: the ID in each EDF's file name must appear in its folder path (below
the ID-range folder). Anything that fails is listed and NOT placed. The output
folder must not exist yet.
"""

import re
import os
import shutil
import sys
from pathlib import Path

from export_progress import edfs_of, find_edfs, find_recordings, match_all

PART_RE = re.compile(r"-\d{8}T\d{6}Z-\d+-\d{3}$")   # "...-20260927T191718Z-1-001"


def originals(download_dir):
    """{recording folder name: [path relative to 'Batch 2 tri1', ...]}"""
    found = {}
    for root, dirs, files in os.walk(download_dir):
        if any(f.lower() == "data.ndb" for f in files):
            rel = Path(root).relative_to(download_dir).parts
            if PART_RE.search(rel[0]):
                rel = rel[1:]          # drop the zip-part folder
            if len(rel) > 1:
                rel = rel[1:]          # drop "Batch 2 tri1" itself
            found.setdefault(Path(root).name, []).append(Path(*rel))
    return found


def patient_id(edf_path):
    """The ID the file is named by: '5041.edf' -> '5041', '5234_8c5dc.edf' -> '5234'."""
    return edf_path.stem.split("_")[0]


def id_in_path(pid, path):
    """True if `pid` names one of the folders below the ID-range folder - the
    range folder itself ('5002-5006, 5008-5012') lists several IDs, so it can't
    tell recordings apart."""
    below_range = Path(*path.parts[1:]) if len(path.parts) > 1 else path
    return re.search(r"(?<!\d)" + re.escape(pid) + r"(?!\d)", str(below_range)) is not None


def main():
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    download, flat, edf_dir, out = (Path(a).resolve() for a in sys.argv[1:])
    if out.exists():
        sys.exit(f"ERROR: {out} already exists - delete or rename it first")

    where = originals(download)
    recs = find_recordings(flat)
    edfs = find_edfs(edf_dir)
    match_all(recs, edfs, 2, edf_dir)

    placed, problems = [], []
    for rec in recs:
        cands = edfs_of(rec)
        paths = where.get(rec["name"], [])
        if not paths:
            problems.append((rec["name"], "original folder not found in the download"))
            continue
        # for recordings matched only as a group, keep the EDF whose patient ID is in the path
        fits = [e for e in cands if all(id_in_path(patient_id(e["path"]), p) for p in paths)]
        if len(fits) != 1:
            problems.append((rec["name"], f"{len(cands)} EDF candidate(s), {len(fits)} with a matching "
                                          f"patient ID: {[e['rel'].name for e in cands]} for {paths[0]}"))
            continue
        for p in paths:
            placed.append((fits[0]["path"], out / p / fits[0]["path"].name))

    for src, dest in placed:
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dest)
        except OSError:
            shutil.copy2(src, dest)    # e.g. different drive: fall back to a real copy

    short_list = edf_dir.parent / f"{edf_dir.name.replace(' edf', '')} short recordings - missing data.csv"
    if short_list.exists():
        shutil.copy2(short_list, out / short_list.name)

    print(f"Placed {len(placed)} EDF(s) for {len(recs) - len(problems)} of {len(recs)} recordings in {out}")
    if problems:
        print(f"\nNOT PLACED ({len(problems)}) - check these by hand:")
        for name, why in problems:
            print(f"  {name}: {why}")


if __name__ == "__main__":
    main()
