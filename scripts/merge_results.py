"""
Merge Results — Admin Tool
===========================
Combines all volunteer .xz (or .jsonl) result files into a single
clean final dataset.

Usage:
    python scripts/merge_results.py --results-dir ./results --output final_dataset.jsonl

Accepts:
    - *.xz   — compressed files submitted by volunteers via GitHub issues
    - *.jsonl — raw files (if you have any uncompressed)
"""

import json
import lzma
import argparse
from pathlib import Path


def iter_lines(path: Path):
    """Yield lines from either a .xz or .jsonl file."""
    if path.suffix == ".xz":
        with lzma.open(path, "rt", encoding="utf-8") as f:
            yield from f
    else:
        with open(path, encoding="utf-8") as f:
            yield from f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="./results",
                        help="Folder containing volunteer .xz (or .jsonl) files")
    parser.add_argument("--output",      default="final_dataset.jsonl",
                        help="Output path for merged dataset")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    files       = sorted(results_dir.glob("*.xz")) + sorted(results_dir.glob("*.jsonl"))

    if not files:
        print(f"❌  No .xz or .jsonl files found in {results_dir}")
        return

    print(f"📂  Found {len(files)} result files:\n")

    total, good, errors, duplicates = 0, 0, 0, 0
    seen_ids = set()

    with open(args.output, "w", encoding="utf-8") as out_f:
        for fpath in files:
            file_total, file_good, file_errors, file_dupes = 0, 0, 0, 0
            try:
                for line in iter_lines(fpath):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        cid    = record.get("chunk_id", "")
                        if cid in seen_ids:
                            file_dupes += 1
                            continue
                        seen_ids.add(cid)
                        file_total += 1
                        if record.get("parse_error"):
                            file_errors += 1
                        else:
                            file_good += 1
                            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    except json.JSONDecodeError:
                        file_errors += 1
            except Exception as e:
                print(f"  ⚠️  Could not read {fpath.name}: {e}")
                continue

            print(f"  {fpath.name:<45} {file_total:>5} records  "
                  f"({file_good} good, {file_errors} errors, {file_dupes} dupes skipped)")
            total      += file_total
            good       += file_good
            errors     += file_errors
            duplicates += file_dupes

    out_size_mb = Path(args.output).stat().st_size / 1_048_576

    print(f"""
{'='*60}
  ✅  Merge complete!
  Output         : {args.output}  ({out_size_mb:.1f} MB)
  Total records  : {total:,}
  Clean records  : {good:,}  (written to output)
  Parse errors   : {errors:,}  (excluded)
  Duplicates     : {duplicates:,}  (skipped)
{'='*60}
""")


if __name__ == "__main__":
    main()
