import argparse
from pathlib import Path

import torch

from ship_detection.model import create_model, load_model
from ship_detection.predict import predict_image

def resolve_device(device_name):
    if device_name == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but CUDA is not available."
        )

    return torch.device(device_name)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify an 80x80 satellite image as ship or no-ship."
    )

    parser.add_argument(
        "image",
        type=Path,
        help="Path to an 80x80 RGB image."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a trained model checkpoint."
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use fo inference."
    )

    return parser.parse_args()

def main():
    args = parse_args()

    device = resolve_device(args.device)

    model = create_model()

    model, _ = load_model(
        model,
        args.checkpoint,
        device
    )

    result = predict_image(
        model,
        args.image,
        device
    )

    print(f"Prediction: {result['label']}")
    print(f"Class ID: {result['class_id']}")
    print(f"Confidence: {result['confidence']:.2%}")

if __name__ == "__main__":
    main()