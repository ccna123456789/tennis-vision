"""Cache disque des detections.

L'inference est de loin l'etape la plus couteuse du pipeline. La mettre en
cache decouple le travail sur les statistiques et le rendu du cout de la
detection : une fois les detections calculees, les executions suivantes sont
quasi instantanees.

La cle du cache inclut le nom des poids et le seuil de confiance : changer de
modele invalide automatiquement le cache, ce qui evite l'erreur classique de
comparer deux configurations en relisant les memes resultats.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path


class DetectionCache:
    """Stocke et relit des detections indexees par configuration."""

    def __init__(self, directory, enabled: bool = True):
        self.directory = Path(directory)
        self.enabled = enabled

    def key(self, video_path, weights: str, confidence: float, frame_count: int) -> str:
        raw = f"{Path(video_path).name}|{weights}|{confidence}|{frame_count}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def _path(self, name: str, key: str) -> Path:
        return self.directory / f"{name}-{key}.pkl"

    def load(self, name: str, key: str):
        if not self.enabled:
            return None
        path = self._path(name, key)
        if not path.exists():
            return None
        with path.open("rb") as handle:
            return pickle.load(handle)

    def store(self, name: str, key: str, payload) -> None:
        if not self.enabled:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        with self._path(name, key).open("wb") as handle:
            pickle.dump(payload, handle)
