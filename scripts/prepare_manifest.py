import argparse
import csv
from pathlib import Path

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a"}


def scan_folder(input_dir, domain_id=0, relative_to=None):
    input_dir = Path(input_dir)
    relative_to = Path(relative_to) if relative_to else input_dir
    rows = []
    for path in input_dir.rglob("*"):
        if path.suffix.lower() not in AUDIO_EXTS:
            continue
        label = path.parent.name
        rel_path = path.relative_to(relative_to)
        rows.append(
            {
                "path": str(rel_path.as_posix()),
                "label": label,
                "speaker": "",
                "domain": str(domain_id),
            }
        )
    return rows


def read_csv(input_csv, domain_id=0):
    rows = []
    with Path(input_csv).open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row.setdefault("domain", str(domain_id))
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["folder", "csv"], default="folder")
    parser.add_argument("--input_dir")
    parser.add_argument("--input_csv")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--domain_id", type=int, default=0)
    parser.add_argument("--relative_to")
    args = parser.parse_args()

    if args.mode == "folder":
        if not args.input_dir:
            raise ValueError("--input_dir is required for folder mode")
        rows = scan_folder(args.input_dir, args.domain_id, args.relative_to)
    else:
        if not args.input_csv:
            raise ValueError("--input_csv is required for csv mode")
        rows = read_csv(args.input_csv, args.domain_id)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "speaker", "domain"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "path": row.get("path") or row.get("filepath") or row.get("audio"),
                    "label": row.get("label") or row.get("emotion"),
                    "speaker": row.get("speaker", ""),
                    "domain": row.get("domain", args.domain_id),
                }
            )

    print(f"Saved {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
