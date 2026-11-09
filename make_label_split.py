# assign the labeled patients to training / validation / testing (70 / 15 / 15)
#
#   python make_label_split.py
#   python make_label_split.py --seed 1 --force     (overwrite an existing split)
#
# AI-generated (Claude, via Claude Code); reviewed by the author before use.
#
# Reads only the FILENAMES of the .edf files in --folder; it never opens them.
# Participant IDs are the 4-digit numbers starting with 4 or 5 (confirmed by
# the PI). A filename can hold two of them, e.g. "5402_4286_1.edf" (the last
# part is the trimester); one recording is one person, so both are the same
# participant. Files sharing any ID are joined into one participant, written
# as all their IDs, e.g. "4286;5402".
#
# The split is by patient, not by breath, so no patient's breaths end up in
# both training and testing. Patients are sorted, then shuffled with a fixed
# --seed, so rerunning gives exactly the same split. Writes one row per
# patient: patient_id, split. This file holds patient IDs: keep it off GitHub.

import argparse
import csv
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Patient-level 70/15/15 split of the labeled patients")
    ap.add_argument("--folder", default="psg_data_cleaned/psg_data", help="folder with the labeled .edf files")
    ap.add_argument("--out", default="labeled patient split.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-percent", type=int, default=15)
    ap.add_argument("--test-percent", type=int, default=15)
    ap.add_argument("--force", action="store_true", help="overwrite --out if it already exists")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} already exists; the split is meant to stay fixed. Use --force to overwrite.")

    files = sorted(Path(args.folder).glob("*.edf"))
    groups, unmatched = [], []  # groups: one set of IDs per participant
    for f in files:
        ids = {part for part in f.stem.split("_") if len(part) == 4 and part.isdigit() and part[0] in "45"}
        if not ids:
            unmatched.append(f.name)
            continue
        for g in [g for g in groups if g & ids]:  # merge every group sharing an ID with this file
            groups.remove(g)
            ids |= g
        groups.append(ids)
    if unmatched:
        print(f"{len(unmatched)} files with no 4xxx/5xxx ID, left out (check these yourself): {unmatched}")
    patients = sorted(";".join(sorted(g)) for g in groups)
    random.Random(args.seed).shuffle(patients)

    n_val = round(len(patients) * args.val_percent / 100)
    n_test = round(len(patients) * args.test_percent / 100)
    n_train = len(patients) - n_val - n_test
    splits = ["training"] * n_train + ["validation"] * n_val + ["testing"] * n_test

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "split"])
        writer.writerows(sorted(zip(patients, splits)))

    print(f"{len(files)} .edf files, {len(patients)} patients: "
          f"{n_train} training, {n_val} validation, {n_test} testing (seed {args.seed})")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
