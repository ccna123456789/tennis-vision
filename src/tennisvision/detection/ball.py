"""Detection de la balle."""

from __future__ import annotations

from .base import YoloBackend

# Sur un modele COCO generique, la balle de tennis tombe dans la classe
# "sports ball", concue pour des ballons. La detection est donc peu fiable :
# c'est la limite principale du pipeline, documentee dans le README.
BALL_CLASS_NAME = "sports ball"


class BallDetector:
    """Retient au plus une balle par frame : la detection la plus sure.

    Le seuil de confiance est volontairement tres bas. Une balle en mouvement
    occupe quelques pixels et sort floue ; exiger une confiance elevee revient
    a ne rien detecter du tout. Les faux positifs qui en resultent sont ecartes
    plus loin par le filtre de vitesse physiquement plausible.
    """

    def __init__(self, weights: str = "yolov8x.pt", confidence: float = 0.05):
        self.backend = YoloBackend(weights, confidence)
        self.ball_class = self.backend.class_id(BALL_CLASS_NAME)
        self.is_specialised = len(self.backend.class_names) == 1

    def detect(self, frame):
        """Renvoie la meilleure detection de balle, ou None."""
        class_filter = None if self.is_specialised else self.ball_class
        detections = self.backend.detect(frame, class_filter=class_filter)

        if not detections:
            return None
        return max(detections, key=lambda d: d.confidence)
