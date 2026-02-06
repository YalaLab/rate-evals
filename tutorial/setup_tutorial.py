"""Generate synthetic data for the RATE-Evals tutorial.

Creates:
  - assets/CXR145_IM-0290-1001.png   Synthetic 1024x1024 grayscale chest X-ray
  - tutorial/dummy_labels.json        Labels for 100 dummy studies (3 binary findings)

Dependencies: Pillow (already a project dependency) + stdlib.

Usage:
    python tutorial/setup_tutorial.py
"""

import json
import os
import random

from PIL import Image, ImageDraw, ImageFilter

SEED = 42
NUM_SAMPLES = 100
POSITIVE_RATE = 0.20  # ~20 % positive rate per finding

QUESTIONS = [
    "Is there evidence of cardiomegaly?",
    "Is there a pleural effusion?",
    "Is there lung consolidation?",
]

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
LABELS_PATH = os.path.join(os.path.dirname(__file__), "dummy_labels.json")


def _create_synthetic_cxr(path: str, size: int = 1024) -> None:
    """Create a synthetic grayscale image that loosely resembles a chest X-ray."""
    random.seed(SEED)
    img = Image.new("L", (size, size), color=30)
    draw = ImageDraw.Draw(img)

    # Dark oval for the thoracic cavity
    draw.ellipse(
        [size // 6, size // 8, size - size // 6, size - size // 10],
        fill=50,
    )

    # Lighter region in the centre (mediastinum)
    cx, cy = size // 2, size // 2
    draw.ellipse(
        [cx - size // 8, cy - size // 4, cx + size // 8, cy + size // 4],
        fill=80,
    )

    # Random speckle to give texture
    for _ in range(2000):
        x = random.randint(0, size - 1)
        y = random.randint(0, size - 1)
        v = random.randint(20, 90)
        draw.point((x, y), fill=v)

    img = img.filter(ImageFilter.GaussianBlur(radius=3))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    print(f"Created synthetic image: {path}")


def _create_dummy_labels(path: str) -> None:
    """Create labels JSON in qa_results format for dummy_study_0000..0099."""
    random.seed(SEED)
    labels = {}

    for idx in range(NUM_SAMPLES):
        accession = f"dummy_study_{idx:04d}"
        findings = []
        for q in QUESTIONS:
            answer = "yes" if random.random() < POSITIVE_RATE else "no"
            findings.append({q: answer})

        labels[accession] = {"qa_results": {"findings": findings}}

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"Created labels for {NUM_SAMPLES} studies: {path}")


def main() -> None:
    image_path = os.path.join(ASSETS_DIR, "CXR145_IM-0290-1001.png")
    _create_synthetic_cxr(image_path)
    _create_dummy_labels(LABELS_PATH)
    print("\nSetup complete! You can now follow TUTORIAL.md.")


if __name__ == "__main__":
    main()
