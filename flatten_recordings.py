"""
Finds every recording folder (identified by containing a DATA.ndb file)
buried anywhere inside a messy, nested folder tree, and copies each one
up into a single flat destination folder - the structure STAR (SSRE)
requires, where every subfolder is exactly one recording.

Usage (Windows, PowerShell):
    python flatten_recordings.py "C:\\...\\Batch 2" "C:\\...\\Batch 2 flat"

Arguments:
    1. The messy source folder (e.g. your extracted "Batch 2", with
       recordings buried under extra zip-name folders).
    2. The destination folder to create, where each recording will end
       up as a direct subfolder (e.g. "Batch 2 flat\\5388\\DATA.ndb").

This COPIES folders rather than moving them, so your original extracted
files are left untouched - safe to re-run, and nothing is deleted.
"""

import shutil
import sys
from pathlib import Path


def find_recording_folders(root_folder: Path):
    """
    Finds every folder anywhere under root_folder that directly contains
    a DATA.ndb file - i.e. an actual recording, not just an intermediate
    zip-name folder.
    """
    recording_folders = []
    for ndb_file in root_folder.rglob("DATA.ndb"):
        recording_folders.append(ndb_file.parent)
    return recording_folders


def main():
    if len(sys.argv) != 3:
        print("Usage: python flatten_recordings.py \"<source folder>\" \"<destination folder>\"")
        print(r'Example: python flatten_recordings.py "C:\Users\lilyj\Downloads\pregnant snoring research\Batch 2" "C:\Users\lilyj\Downloads\pregnant snoring research\Batch 2 flat"')
        sys.exit(1)

    source_folder = Path(sys.argv[1])
    dest_folder = Path(sys.argv[2])

    if not source_folder.exists():
        print(f"ERROR: Source folder not found: {source_folder}")
        sys.exit(1)

    dest_folder.mkdir(parents=True, exist_ok=True)

    print(f"Searching for recording folders (containing DATA.ndb) under:\n  {source_folder}\n")

    recording_folders = find_recording_folders(source_folder)

    if not recording_folders:
        print("No recording folders found. Nothing to do.")
        print("(Looked for any folder directly containing a file named DATA.ndb.)")
        sys.exit(0)

    print(f"Found {len(recording_folders)} recording folder(s).\n")

    copied = 0
    skipped = 0

    for rec_folder in recording_folders:
        # Use the recording folder's own name (e.g. "5388") as the new
        # flat folder name. If that name already exists at the
        # destination (e.g. two different zips both had a "5388" folder),
        # skip it and warn rather than silently overwriting.
        target = dest_folder / rec_folder.name

        if target.exists():
            print(f"  SKIPPING '{rec_folder}' -> '{target}' already exists (possible duplicate patient ID or naming collision). Check this manually.")
            skipped += 1
            continue

        print(f"  Copying: {rec_folder}  ->  {target}")
        shutil.copytree(rec_folder, target)
        copied += 1

    print(f"\nDone. Copied {copied} recording(s) into: {dest_folder}")
    if skipped:
        print(f"Skipped {skipped} recording(s) due to name collisions - review these manually.")
    print("\nYou can now point STAR's 'recordings folder' at this flat destination folder.")


if __name__ == "__main__":
    main()
