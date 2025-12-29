"""數據清單準備腳本

用於將音訊文件目錄或CSV文件轉換為模型訓練所需的數據清單格式
支持兩種模式：文件夾掃描和CSV文件讀取
"""
import argparse
import csv
from pathlib import Path

# 支持的音訊文件擴展名
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a"}


def infer_speaker(path, relative_to, speaker_level):
    if speaker_level <= 0:
        return ""
    rel_parts = path.relative_to(relative_to).parts
    idx = len(rel_parts) - speaker_level - 1
    if idx < 0:
        return ""
    return rel_parts[idx]


def scan_folder(input_dir, domain_id=0, relative_to=None, speaker_level=2):
    """掃描文件夾中的音訊文件

    遞歸掃描文件夾，收集所有音訊文件的路徑和標籤信息
    標籤從文件所屬文件夾的名稱中獲取

    Args:
        input_dir: 要掃描的輸入文件夾
        domain_id: 領域ID，默認為0
        relative_to: 計算相對路徑的基準文件夾，默認為input_dir

    Returns:
        包含音訊文件信息的字典列表
    """
    input_dir = Path(input_dir)
    relative_to = Path(relative_to) if relative_to else input_dir
    rows = []
    for path in input_dir.rglob("*"):
        if path.suffix.lower() not in AUDIO_EXTS:
            continue
        label = path.parent.name
        rel_path = path.relative_to(relative_to)
        speaker = infer_speaker(path, relative_to, speaker_level)
        rows.append(
            {
                "path": str(rel_path.as_posix()),
                "label": label,
                "speaker": speaker,
                "domain": str(domain_id),
            }
        )
    return rows


def read_csv(input_csv, domain_id=0):
    """從CSV文件中讀取音訊文件信息

    讀取現有的CSV文件，並為每一行添加領域ID（如果不存在）

    Args:
        input_csv: 輸入CSV文件路徑
        domain_id: 領域ID，默認為0

    Returns:
        包含音訊文件信息的字典列表
    """
    rows = []
    with Path(input_csv).open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row.setdefault("domain", str(domain_id))
            rows.append(row)
    return rows


def main():
    """主函數

    解析命令行參數，根據指定的模式處理音訊文件信息，生成數據清單CSV

    命令行參數：
        --mode: 處理模式，可選值為"folder"或"csv"，默認為"folder"
        --input_dir: 輸入文件夾路徑（folder模式下必填）
        --input_csv: 輸入CSV文件路徑（csv模式下必填）
        --output_csv: 輸出數據清單CSV文件路徑（必填）
        --domain_id: 領域ID，默認為0
        --relative_to: 計算相對路徑的基準文件夾（folder模式下可用）
        --speaker_level: speaker欄位取檔案向上第幾層目錄（預設2）
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["folder", "csv"], default="folder", help="處理模式")
    parser.add_argument("--input_dir", help="輸入文件夾路徑")
    parser.add_argument("--input_csv", help="輸入CSV文件路徑")
    parser.add_argument("--output_csv", required=True, help="輸出數據清單CSV文件路徑")
    parser.add_argument("--domain_id", type=int, default=0, help="領域ID")
    parser.add_argument("--relative_to", help="計算相對路徑的基準文件夾")
    parser.add_argument("--speaker_level", type=int, default=2, help="speaker欄位取檔案向上第幾層目錄")
    args = parser.parse_args()

    # 根據模式處理音訊文件信息
    if args.mode == "folder":
        if not args.input_dir:
            raise ValueError("--input_dir is required for folder mode")
        rows = scan_folder(args.input_dir, args.domain_id, args.relative_to, args.speaker_level)
    else:
        if not args.input_csv:
            raise ValueError("--input_csv is required for csv mode")
        rows = read_csv(args.input_csv, args.domain_id)

    # 保存數據清單到CSV文件
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
