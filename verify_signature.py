import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from utils import get_embeddings, make_transform

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAMES = ["simple", "complex", "resnet18", "mobilenet_v3_small", "efficientnet_b0", "densenet121"]


def choose(items):
    for i, item in enumerate(items, 1):
        print(f"{i}. {item}")
    while True:
        answer = input("Choose a number or name: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(items):
            return items[int(answer) - 1]
        for item in items:
            if answer == str(item):
                return item
        print("Invalid choice.")


def prepare_image(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGBA")
        background = Image.new("RGBA", image.size, "white")
        gray = np.array(Image.alpha_composite(background, image).convert("L"))
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary)


@torch.no_grad()
def compare_signatures(reference_path, query_path, model_dir=None, model_name="simple"):
    model_dir = Path(model_dir or BASE_DIR / "model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    saved = torch.load(model_dir / f"{model_name}_model.pt", map_location=device, weights_only=False)
    model = saved["model"].eval()
    threshold = json.loads((model_dir / f"{model_name}_threshold.json").read_text())["threshold"]
    rgb = model_name not in ["simple", "complex"]
    transform = make_transform(saved["config"]["image_size"], rgb)
    reference = prepare_image(reference_path)
    query = prepare_image(query_path)
    images = torch.stack([transform(image.convert("RGB" if rgb else "L")) for image in [reference, query]])
    vectors = get_embeddings(model, images.to(device), model_name)
    distance = (vectors[0] - vectors[1]).square().sum().item()
    result = {
        "model": model_name,
        "distance": distance,
        "threshold": threshold,
        "prediction": "Real" if distance <= threshold else "Fake",
    }
    return result, reference, query


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_NAMES)
    parser.add_argument("--model-dir", default=str(BASE_DIR / "model"))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--reference")
    group.add_argument("--reference-dir")
    parser.add_argument("--query")
    parser.add_argument("--output")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    name = args.model or choose([name for name in MODEL_NAMES
                                if (model_dir / f"{name}_model.pt").is_file()
                                and (model_dir / f"{name}_threshold.json").is_file()])
    reference = args.reference or BASE_DIR / "Signatures/CR7/cr7_riel.jpg"
    folder = Path(args.reference_dir) if args.reference_dir else BASE_DIR / "dang_riel"
    if args.reference_dir or (not args.reference and folder.is_dir()):
        paths = sorted(path for path in folder.iterdir()
                       if path.is_file() and path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"])
        reference = paths[0] if len(paths) == 1 else choose(paths)
    query = args.query or input("User signature image path: ").strip().strip("\"\'")
    result, reference, query = compare_signatures(reference, query, model_dir, name)
    print(json.dumps(result, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, image, title in zip(axes, [reference, query], ["Genuine reference", "User signature"]):
        axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(f"{name}: {result['prediction']} | Distance: {result['distance']:.4f} | Threshold: {result['threshold']:.4f}")
    plt.tight_layout()
    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
