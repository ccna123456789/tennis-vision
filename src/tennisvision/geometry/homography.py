"""Passage du plan image au plan du court.

Une camera qui filme une surface plane induit une transformation projective
entre le plan image et cette surface. Elle est decrite par une matrice 3x3,
definie a un facteur d'echelle pres : 8 degres de liberte, donc 4
correspondances de points suffisent, chacune apportant deux equations.

Ce module accepte 4 correspondances — resolution exacte — ou davantage, auquel
cas la matrice est estimee au sens des moindres carres. Pointer les
intersections des lignes de service en plus des coins reduit alors l'effet
d'une imprecision de clic.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ..types import CourtPosition
from .court import CourtModel


class CalibrationError(Exception):
    """Levee quand une homographie ne peut pas etre calculee ou appliquee."""


class CourtHomography:
    """Convertit des points image en positions metriques sur le court.

    Parameters
    ----------
    image_points:
        Points reperes dans l'image, en pixels.
    court_points:
        Leurs positions reelles en metres. Si omis, on suppose que
        `image_points` sont les 4 coins du court de double, dans l'ordre du
        modele.
    """

    def __init__(self, image_points, court_points=None, model: CourtModel | None = None):
        self.model = model or CourtModel()

        if court_points is None:
            court_points = self.model.corners

        image_array = np.asarray(image_points, dtype=np.float64)
        court_array = np.asarray(court_points, dtype=np.float64)

        # L'ordre de ces deux controles compte : avec trop peu de points image,
        # le nombre attendu de points court est celui du modele, si bien que le
        # controle de correspondance signalerait un ecart de comptage la ou le
        # vrai probleme est le minimum requis par une homographie.
        if len(image_array) < 4:
            raise CalibrationError(
                "Une homographie demande au moins 4 correspondances de points, "
                f"{len(image_array)} fournie(s)."
            )
        if image_array.shape != court_array.shape:
            raise CalibrationError(
                "Autant de points image que de points court sont requis "
                f"({len(image_array)} contre {len(court_array)})."
            )

        matrix, _ = cv2.findHomography(image_array, court_array, method=0)
        if matrix is None:
            raise CalibrationError(
                "Homographie non calculable. Verifiez que les points ne sont pas "
                "alignes trois par trois et qu'ils sont donnes dans le bon ordre."
            )

        self.matrix = matrix
        self.inverse = np.linalg.inv(matrix)
        self.image_points = image_array
        self.court_points = court_array

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------
    def to_court(self, image_point) -> CourtPosition:
        """Projette un point image (pixels) sur le court (metres)."""
        across, along = _project(self.matrix, image_point)
        return CourtPosition(across, along)

    def to_image(self, court_position) -> tuple[float, float]:
        """Projette une position du court (metres) dans l'image (pixels)."""
        if isinstance(court_position, CourtPosition):
            court_position = court_position.as_tuple()
        return _project(self.inverse, court_position)

    def in_play_area(self, court_position, margin: float = 0.0) -> bool:
        """Indique si une position tombe dans la zone de jeu."""
        if isinstance(court_position, CourtPosition):
            court_position = court_position.as_tuple()
        return self.model.contains(court_position, margin=margin)

    # ------------------------------------------------------------------
    # Controle qualite
    # ------------------------------------------------------------------
    def reprojection_error(self) -> float:
        """Erreur moyenne de reprojection, en metres.

        On reprojette les points de calibration et on mesure l'ecart aux
        positions attendues. Avec exactement 4 points la solution est exacte et
        l'erreur est nulle ; au-dela, cette valeur dit si les points pointes
        sont coherents entre eux.
        """
        errors = []
        for image_point, expected in zip(self.image_points, self.court_points):
            across, along = _project(self.matrix, image_point)
            errors.append(float(np.hypot(across - expected[0], along - expected[1])))
        return float(np.mean(errors)) if errors else 0.0

    # ------------------------------------------------------------------
    # Persistance
    # ------------------------------------------------------------------
    def save(self, path) -> None:
        """Enregistre les correspondances, pas la matrice.

        Sauver les points sources plutot que la matrice rend le fichier
        lisible, verifiable et rejouable si le calcul evolue.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "image_points": self.image_points.tolist(),
            "court_points": self.court_points.tolist(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "CourtHomography":
        path = Path(path)
        if not path.exists():
            raise CalibrationError(
                f"Calibration introuvable : {path}\n"
                "Lancez d'abord : python -m tennisvision calibrate"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(payload["image_points"], payload.get("court_points"))


def _project(matrix: np.ndarray, point) -> tuple[float, float]:
    """Applique une transformation projective a un point.

    Le point passe en coordonnees homogenes (x, y, 1), est multiplie par la
    matrice, puis ramene en coordonnees cartesiennes par division par la
    troisieme composante. C'est cette division qui encode la perspective : sans
    elle la transformation serait affine et ne pourrait pas rendre compte du
    retrecissement des objets eloignes.
    """
    vector = np.array([point[0], point[1], 1.0], dtype=np.float64)
    projected = matrix @ vector

    scale = projected[2]
    if abs(scale) < 1e-9:
        raise CalibrationError(
            "Point projete a l'infini : il n'appartient pas au plan du court."
        )

    return (float(projected[0] / scale), float(projected[1] / scale))
