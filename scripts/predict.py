"""單個音訊文件預測腳本

用於對單個音訊文件進行情感分類預測
調用推論模塊的predict_file函數實現預測功能
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.inference.predict import predict_file


def main():
    """主函數

    解析命令行參數，調用預測函數，輸出預測結果

    命令行參數：
        --checkpoint: 模型檢查點路徑
        --audio: 待預測的音訊文件路徑
        --device: 運行設備，可選值為"cpu"或"cuda"，默認為"cpu"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="模型檢查點路徑")
    parser.add_argument("--audio", required=True, help="待預測的音訊文件路徑")
    parser.add_argument("--device", default="cpu", help="運行設備")
    args = parser.parse_args()

    # 調用預測函數
    result = predict_file(args.checkpoint, args.audio, device=args.device)
    
    # 輸出預測結果
    for label, score in result.items():
        print(f"{label}: {score:.4f}")


if __name__ == "__main__":
    main()
