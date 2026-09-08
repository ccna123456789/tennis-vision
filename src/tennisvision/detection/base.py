"""Enveloppe autour du detecteur YOLO.

Isoler Ultralytics derriere une seule classe presente deux avantages : le reste
du code manipule des `Detection` typees plutot que des tenseurs, et changer de
bibliotheque de detection ne toucherait que ce fichier.
"""

from __future__ import annotations

from ..types import BoundingBox, Detection


class YoloBackend:
    """Charge un modele YOLO et renvoie des detections typees.

    Parameters
    ----------
    weights:
        Nom ou chemin des poids. Les modeles officiels sont telecharges
        automatiquement au premier usage.
    confidence:
        Seuil de confiance minimal.
    """

    def __init__(self, weights: str, confidence: float = 0.25):
        from ultralytics import YOLO  # import differe : demarrage plus rapide

        self.weights = weights
        self.confidence = confidence
        self.model = YOLO(weights)

    @property
    def class_names(self) -> dict:
        return self.model.names

    def class_id(self, name: str) -> int | None:
        """Retrouve l'identifiant d'une classe par son nom."""
        for index, label in self.model.names.items():
            if label == name:
                return int(index)
        return None

    def detect(self, frame, class_filter: int | None = None) -> list[Detection]:
        """Detecte sans suivi. Utilise pour les objets sans identite stable."""
        result = self.model.predict(frame, conf=self.confidence, verbose=False)[0]
        return self._collect(result, class_filter, with_tracking=False)

    def track(self, frame, class_filter: int | None = None) -> list[Detection]:
        """Detecte en maintenant une identite d'une frame a l'autre."""
        result = self.model.track(frame, persist=True, verbose=False)[0]
        return self._collect(result, class_filter, with_tracking=True)

    @staticmethod
    def _collect(result, class_filter, with_tracking: bool) -> list[Detection]:
        detections = []

        for box in result.boxes:
            if class_filter is not None and int(box.cls.tolist()[0]) != class_filter:
                continue

            track_id = None
            if with_tracking:
                # Le tracker n'attribue un identifiant qu'une fois la piste
                # confirmee : avant cela, `box.id` vaut None.
                if box.id is None:
                    continue
                track_id = int(box.id.tolist()[0])

            detections.append(
                Detection(
                    box=BoundingBox.from_xyxy(box.xyxy.tolist()[0]),
                    confidence=float(box.conf.tolist()[0]),
                    track_id=track_id,
                )
            )

        return detections
