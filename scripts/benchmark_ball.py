"""Mesure le taux de detection de la balle selon le modele et le seuil.

La balle est le point faible du pipeline. Plutot que de choisir un modele au
jugé, ce script mesure : pour chaque couple (poids, seuil), il compte sur
combien de frames une balle est trouvee et releve la confiance moyenne.

Les chiffres publies dans la section Results du README sortent de ce script.

    python scripts/benchmark_ball.py --frames 30
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tennisvision.detection import BallDetector  # noqa: E402
from tennisvision.io import read_frames  # noqa: E402

CONFIGURATIONS = [
    ("yolov8n.pt", 0.15),
    ("yolov8n.pt", 0.05),
    ("yolov8x.pt", 0.15),
    ("yolov8x.pt", 0.05),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="data/input/match.mp4")
    parser.add_argument("--frames", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    video = Path(args.video)
    if not video.is_absolute():
        video = PROJECT_ROOT / video
    if not video.exists():
        print(f"Video introuvable : {video}\nVoir docs/DATA.md")
        return 1

    frames = read_frames(video, limit=args.frames)
    print(f"Video : {video.name}  |  {len(frames)} frames\n")

    header = f"{'modele':<14}{'seuil':>7}{'detectee':>12}{'taux':>8}{'confiance':>12}{'duree':>9}"
    print(header)
    print("-" * len(header))

    for weights, confidence in CONFIGURATIONS:
        detector = BallDetector(weights, confidence)

        started = time.time()
        found = [detector.detect(frame) for frame in frames]
        elapsed = time.time() - started

        hits = [d for d in found if d is not None]
        mean_confidence = sum(d.confidence for d in hits) / len(hits) if hits else 0.0

        print(
            f"{weights:<14}{confidence:>7.2f}{len(hits):>8}/{len(frames):<3}"
            f"{len(hits) / len(frames):>7.0%}{mean_confidence:>12.2f}{elapsed:>8.1f}s"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
