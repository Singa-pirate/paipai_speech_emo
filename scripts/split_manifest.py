"""數據清單拆分腳本

用於將數據清單CSV文件按照說話者（speaker）進行分拆
生成訓練集、驗證集和測試集三個獨立的CSV文件
確保同一說話者的音訊文件只出現在一個數據集中
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def load_rows(path):
    """從CSV文件加載所有行數據

    Args:
        path: CSV文件路徑

    Returns:
        包含所有行數據的字典列表
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def write_rows(path, rows):
    """將行數據寫入CSV文件

    Args:
        path: 輸出CSV文件路徑
        rows: 要寫入的行數據字典列表
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "speaker", "domain"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def split_by_speaker(rows, val_ratio, test_ratio, seed):
    """按照說話者拆分數據集

    將數據按照說話者分組，然後隨機分配說話者到訓練集、驗證集和測試集
    確保同一說話者的所有數據都只出現在一個數據集中
    如果沒有說話者信息，則按照文件路徑進行分組

    Args:
        rows: 原始數據行列表
        val_ratio: 驗證集比例
        test_ratio: 測試集比例
        seed: 隨機種子，用於重現拆分結果

    Returns:
        訓練集行列表、驗證集行列表、測試集行列表
    """
    # 按照說話者分組數據
    groups = defaultdict(list)
    for row in rows:
        speaker = row.get("speaker") or row.get("path")
        groups[speaker].append(row)

    # 打亂說話者順序
    speakers = list(groups.keys())
    random.Random(seed).shuffle(speakers)

    # 計算各數據集的說話者數量
    total = len(speakers)
    test_count = int(total * test_ratio)
    if test_ratio > 0 and total > 1:
        test_count = max(1, test_count)
    val_count = int(total * val_ratio)
    remaining = total - test_count
    if val_ratio > 0 and remaining > 1:
        val_count = max(1, val_count)
        val_count = min(val_count, remaining - 1)
    else:
        val_count = 0

    # 分配說話者到不同數據集
    test_speakers = set(speakers[:test_count])
    val_speakers = set(speakers[test_count : test_count + val_count])

    # 收集各數據集的行數據
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
    """主函數

    解析命令行參數，加載數據清單，拆分數據集，保存結果

    命令行參數：
        --input_csv: 輸入數據清單CSV文件路徑（必填）
        --output_dir: 輸出目錄路徑（必填）
        --val_ratio: 驗證集比例，默認為0.15
        --test_ratio: 測試集比例，默認為0.15
        --seed: 隨機種子，用於重現拆分結果，默認為42
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="輸入數據清單CSV文件路徑")
    parser.add_argument("--output_dir", required=True, help="輸出目錄路徑")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="驗證集比例")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="測試集比例")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子")
    args = parser.parse_args()

    # 加載原始數據
    rows = load_rows(args.input_csv)
    
    # 拆分數據集
    train_rows, val_rows, test_rows = split_by_speaker(
        rows, args.val_ratio, args.test_ratio, args.seed
    )

    # 保存拆分結果
    output_dir = Path(args.output_dir)
    write_rows(output_dir / "train.csv", train_rows)
    write_rows(output_dir / "val.csv", val_rows)
    write_rows(output_dir / "test.csv", test_rows)

    print(
        f"Split complete: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}"
    )


if __name__ == "__main__":
    main()
