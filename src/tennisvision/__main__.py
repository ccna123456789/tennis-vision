"""Interface en ligne de commande.

    python -m tennisvision calibrate
    python -m tennisvision analyze --max-frames 90
    python -m tennisvision analyze --no-video
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from .geometry import CourtHomography, CourtModel
from .io import read_frames
from .pipeline import AnalysisConfig, MatchAnalyser

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CORNER_PROMPTS = [
    "1/4  fond de court ELOIGNE, cote GAUCHE",
    "2/4  fond de court ELOIGNE, cote DROIT",
    "3/4  fond de court PROCHE, cote DROIT",
    "4/4  fond de court PROCHE, cote GAUCHE",
]


def load_config(path) -> dict:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        raise SystemExit(f"Configuration introuvable : {config_path}")
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# calibrate
# ----------------------------------------------------------------------
def pick_corners(frame):
    """Recueille 4 clics sur la premiere frame."""
    import cv2

    corners = []
    window = "Calibration - cliquez les 4 coins du court de double"

    def on_mouse(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            corners.append((float(x), float(y)))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        canvas = frame.copy()

        for index, (x, y) in enumerate(corners):
            cv2.circle(canvas, (int(x), int(y)), 7, (60, 60, 240), -1, cv2.LINE_AA)
            cv2.putText(canvas, str(index + 1), (int(x) + 12, int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 240), 2, cv2.LINE_AA)

        if len(corners) == 4:
            points = [(int(x), int(y)) for x, y in corners]
            for i in range(4):
                cv2.line(canvas, points[i], points[(i + 1) % 4], (90, 230, 120), 2, cv2.LINE_AA)
            message = "ENTREE pour valider  |  r pour recommencer  |  q pour annuler"
        else:
            message = CORNER_PROMPTS[len(corners)]

        cv2.putText(canvas, message, (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (60, 240, 240), 2, cv2.LINE_AA)
        cv2.imshow(window, canvas)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("r"):
            corners = []
        elif key == ord("q"):
            cv2.destroyAllWindows()
            return None
        elif key in (10, 13) and len(corners) == 4:
            cv2.destroyAllWindows()
            return corners


def command_calibrate(args) -> int:
    config = load_config(args.config)
    video = args.video or config["video"]
    output = args.output or config["calibration"]

    if args.corners:
        corners = [tuple(float(v) for v in raw.split(",")) for raw in args.corners]
    else:
        video_path = Path(video)
        if not video_path.is_absolute():
            video_path = PROJECT_ROOT / video_path
        frame = read_frames(video_path, limit=1)[0]
        print(f"Video : {video_path.name}  ({frame.shape[1]}x{frame.shape[0]})")
        corners = pick_corners(frame)
        if corners is None:
            print("Calibration annulee.")
            return 1

    homography = CourtHomography(corners)
    destination = Path(output)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    homography.save(destination)

    model = CourtModel()
    print(f"\nCalibration enregistree dans {destination}")
    print(f"Erreur de reprojection : {homography.reprojection_error():.4f} m")
    print(f"Court de reference     : {model.width} m x {model.length} m\n")
    for prompt, corner in zip(CORNER_PROMPTS, corners):
        position = homography.to_court(corner)
        print(f"  {prompt:<40} ({corner[0]:7.1f}, {corner[1]:7.1f}) px"
              f"  ->  ({position.across:5.2f}, {position.along:5.2f}) m")
    return 0


# ----------------------------------------------------------------------
# analyze
# ----------------------------------------------------------------------
def command_analyze(args) -> int:
    raw = load_config(args.config)

    config = AnalysisConfig(
        video=raw["video"],
        calibration=raw["calibration"],
        player_weights=raw["models"]["players"],
        ball_weights=raw["models"]["ball"],
        player_confidence=raw["models"]["player_confidence"],
        ball_confidence=raw["models"]["ball_confidence"],
        max_frames=args.max_frames if args.max_frames is not None else raw.get("max_frames"),
        sample_step=raw["analysis"]["sample_step"],
        max_ball_gap=raw["analysis"]["max_ball_gap"],
        cache_dir=raw["paths"]["cache"],
        output_video=raw["paths"]["output_video"],
        output_report=raw["paths"]["output_report"],
        use_cache=not args.no_cache,
    )

    analyser = MatchAnalyser(config, root=PROJECT_ROOT)
    result = analyser.run(render=not args.no_video)

    print("\n" + "=" * 52)
    print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))
    return 0


# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tennisvision", description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate = subparsers.add_parser("calibrate", help="Reperer les 4 coins du court")
    calibrate.add_argument("--video", help="Video a calibrer")
    calibrate.add_argument("--output", help="Fichier de calibration a ecrire")
    calibrate.add_argument("--corners", nargs=4, metavar="X,Y",
                           help="Coins fournis directement, sans interface graphique")
    calibrate.set_defaults(handler=command_calibrate)

    analyze = subparsers.add_parser("analyze", help="Analyser une sequence")
    analyze.add_argument("--max-frames", type=int, default=None)
    analyze.add_argument("--no-video", action="store_true", help="Statistiques seules")
    analyze.add_argument("--no-cache", action="store_true", help="Forcer la re-detection")
    analyze.set_defaults(handler=command_analyze)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
