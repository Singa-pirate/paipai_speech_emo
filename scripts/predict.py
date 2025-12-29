import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.inference.predict import predict_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    result = predict_file(args.checkpoint, args.audio, device=args.device)
    for label, score in result.items():
        print(f"{label}: {score:.4f}")


if __name__ == "__main__":
    main()
