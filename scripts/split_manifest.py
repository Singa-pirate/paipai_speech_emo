import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def load_rows(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "speaker", "domain"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def split_by_speaker(rows, val_ratio, test_ratio, seed):
    groups = defaultdict(list)
    for row in rows:
        speaker = row.get("speaker") or row.get("path")
        groups[speaker].append(row)

    speakers = list(groups.keys())
    random.Random(seed).shuffle(speakers)

    total = len(speakers)
    test_count = int(total * test_ratio)
    val_count = int(total * val_ratio)

    test_speakers = set(speakers[:test_count])
    val_speakers = set(speakers[test_count : test_count + val_count])

    train_rows, val_rows, test_rows = [], [], []
    for speaker, items in groups.items():
        if speaker in test_speakers:
            test_rows.extend(items)
        elif speaker in val_speakers:
            val_rows.extend(items)
        else:
            train_rows.extend(items)

    return train_rows, val_rows, test_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_rows(args.input_csv)
    train_rows, val_rows, test_rows = split_by_speaker(
        rows, args.val_ratio, args.test_ratio, args.seed
    )

    output_dir = Path(args.output_dir)
    write_rows(output_dir / "train.csv", train_rows)
    write_rows(output_dir / "val.csv", val_rows)
    write_rows(output_dir / "test.csv", test_rows)

    print(
        f"Split complete: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}"
    )


if __name__ == "__main__":
    main()
