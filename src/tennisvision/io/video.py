"""Lecture et ecriture de fichiers video."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

DEFAULT_FPS = 30.0


@dataclass(frozen=True)
class VideoInfo:
    """Metadonnees d'une video."""

    path: str
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


def probe(path) -> VideoInfo:
    """Lit les metadonnees sans charger les images."""
    capture = _open(path)
    info = VideoInfo(
        path=str(path),
        fps=capture.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS,
        frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    capture.release()
    return info


def read_frames(path, limit: int | None = None, start: int = 0):
    """Charge les frames en memoire.

    `limit` sert a iterer vite pendant la mise au point : sur processeur,
    l'inference coute de l'ordre d'une seconde par frame.
    """
    capture = _open(path)

    if start:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)

    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
        if limit is not None and len(frames) >= limit:
            break

    capture.release()

    if not frames:
        raise ValueError(f"Aucune frame lue depuis {path}")
    return frames


def write_frames(frames, path, fps: float = DEFAULT_FPS) -> None:
    """Ecrit une sequence de frames dans un fichier video."""
    if not frames:
        raise ValueError("Aucune frame a ecrire.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))

    for frame in frames:
        writer.write(frame)
    writer.release()


def _open(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise IOError(f"Impossible d'ouvrir la video : {path}")
    return capture
