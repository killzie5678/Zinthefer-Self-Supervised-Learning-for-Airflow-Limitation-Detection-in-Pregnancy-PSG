"""
Recursively extracts every zip file found inside a folder, including zips
nested inside other zips (e.g. Batch 1/5236.zip -> contains another zip
inside it -> that gets extracted too), until no NEW zip files are found.

Usage (Windows, in Command Prompt or PowerShell):
    python unzip_recursive.py "C:\\Users\\lilyj\\Downloads\\pregnant snoring research\\batch 1"

What it does:
    1. Walks the target folder looking for any .zip file.
    2. Extracts each one into a folder next to it, named after the zip
       (e.g. 5236.zip -> 5236/).
    3. Marks that zip as "already handled" (see WHY THIS MATTERS below) so
       it is never extracted a second time, even if it's still sitting on
       disk or somehow reappears inside its own extracted contents.
    4. Repeats until a full pass extracts zero NEW zips, so zips-within-zips
       get unpacked automatically without looping forever.

WHY THIS MATTERS (infinite loop fix):
    Some export tools package a zip that - once extracted - contains a
    copy of itself or a same-named archive one level down. If a script
    just keeps re-scanning for "any zip file" and re-extracting it without
    remembering what it already did, it can loop forever, going one
    folder deeper each pass (5236/5236/5236/5236/...). This version keeps
    a record of every zip's absolute path it has already extracted, and
    skips it permanently once handled - so each real zip is extracted
    exactly once, no matter how many times it's encountered.

This does NOT touch Noxturnal or your data itself - it only extracts
existing zip archives that are already on your computer.
"""

import zipfile
import sys
from pathlib import Path

# Set to True to delete each .zip after it's successfully extracted, to
# save disk space. Keeping them (False, the default) is safer while
# you're still verifying everything extracted correctly.
DELETE_ZIPS = False

# Safety valve: stop after this many passes even if something unexpected
# is still generating new zips, so a bug can never run forever unattended.
MAX_PASSES = 25


def extract_all_zips(root_folder: Path, already_handled: set) -> int:
    """
    Finds every .zip file under root_folder, extracts any that haven't
    been handled before, and records them so they're never re-extracted.
    Returns the number of NEW zips extracted in this pass.
    """
    zip_files = list(root_folder.rglob("*.zip"))
    count = 0

    for zip_path in zip_files:
        resolved = str(zip_path.resolve())

        if resolved in already_handled:
            continue  # Already extracted this one - skip it for good.

        # Extract into a folder named after the zip, sitting next to it.
        # e.g. .../batch 1/5236.zip -> .../batch 1/5236/
        extract_to = zip_path.parent / zip_path.stem

        try:
            print(f"Extracting: {zip_path.name}  ->  {extract_to}")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_to)
            count += 1
            already_handled.add(resolved)

            if DELETE_ZIPS:
                zip_path.unlink()
                print(f"  Deleted original zip: {zip_path.name}")

        except zipfile.BadZipFile:
            print(f"  WARNING: {zip_path.name} is not a valid zip file (corrupted or incomplete download?). Skipping permanently.")
            already_handled.add(resolved)  # Don't retry a broken zip forever either.
        except Exception as e:
            print(f"  ERROR extracting {zip_path.name}: {e}")
            already_handled.add(resolved)

    return count


def main():
    if len(sys.argv) != 2:
        print("Usage: python unzip_recursive.py \"<folder path>\"")
        print(r'Example: python unzip_recursive.py "C:\Users\lilyj\Downloads\pregnant snoring research\batch 1"')
        sys.exit(1)

    root_folder = Path(sys.argv[1])

    if not root_folder.exists():
        print(f"ERROR: Folder not found: {root_folder}")
        sys.exit(1)

    print(f"Starting recursive unzip in: {root_folder}\n")

    total_extracted = 0
    already_handled = set()
    pass_num = 1

    while pass_num <= MAX_PASSES:
        print(f"--- Pass {pass_num} ---")
        extracted_this_pass = extract_all_zips(root_folder, already_handled)
        total_extracted += extracted_this_pass

        if extracted_this_pass == 0:
            print("\nNo new zip files found. Done.")
            break

        pass_num += 1
        print()
    else:
        print(f"\nStopped after reaching the safety limit of {MAX_PASSES} passes.")
        print("This usually means a zip's contents keep producing what look like new zips")
        print("(e.g. very deeply nested archives, or a zip containing a copy of itself).")
        print("Check the folder structure manually before continuing.")

    print(f"\nTotal zip files extracted: {total_extracted}")
    print("You can now point Noxturnal's Browse dialog at this folder to see the recordings.")


if __name__ == "__main__":
    main()
