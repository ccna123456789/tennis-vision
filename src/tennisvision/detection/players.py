"""Detection et identification des deux joueurs."""

from __future__ import annotations

from collections import Counter

from .base import YoloBackend

# Marge, en metres, ajoutee autour des lignes pour delimiter la zone de jeu.
# Elle est calibree sur deux observations : un joueur se tient jusqu'a environ
# 3 m derriere sa ligne de fond au service ou sur un retour profond, tandis que
# l'arbitre de chaise se trouve a plus de 3 m au-dela du couloir lateral.
PLAY_AREA_MARGIN = 3.0


class PlayerDetector:
    """Detecte les personnes, puis isole les deux joueurs."""

    def __init__(self, weights: str = "yolov8n.pt", confidence: float = 0.25):
        self.backend = YoloBackend(weights, confidence)
        self.person_class = self.backend.class_id("person")
        if self.person_class is None:
            raise ValueError(
                f"Le modele {weights} ne connait pas la classe 'person'. "
                "Utilisez un modele entraine sur COCO."
            )

    def detect(self, frame):
        """Toutes les personnes de la frame, avec identite de suivi."""
        return self.backend.track(frame, class_filter=self.person_class)

    @staticmethod
    def identify_players(frames_detections, homography, sample_size: int = 30):
        """Determine quelles pistes correspondent aux deux joueurs.

        Le critere est geometrique : on projette le point d'appui au sol de
        chaque personne et on compte sur combien de frames elle se trouve dans
        la zone de jeu. Les deux pistes les plus souvent presentes sont les
        joueurs.

        Comparer des distances en pixels, comme le ferait une approche naive,
        n'aurait pas de sens physique constant : un spectateur au premier rang
        est proche en pixels tout en etant hors du court.
        """
        presence = Counter()

        for detections in frames_detections[:sample_size]:
            for detection in detections:
                if detection.track_id is None:
                    continue
                position = homography.to_court(detection.box.ground_contact)
                if homography.in_play_area(position, margin=PLAY_AREA_MARGIN):
                    presence[detection.track_id] += 1

        return [track_id for track_id, _ in presence.most_common(2)]

    @staticmethod
    def keep_only(frames_detections, track_ids):
        """Filtre les detections pour ne garder que les pistes retenues."""
        wanted = set(track_ids)
        return [
            [d for d in detections if d.track_id in wanted]
            for detections in frames_detections
        ]
