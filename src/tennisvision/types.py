"""Types de base partages par tout le pipeline.

Le choix de structures typees plutot que de dictionnaires anonymes est
deliberatif : une boite englobante sait calculer elle-meme son point d'appui au
sol, ce qui evite de disperser cette regle geometrique dans chaque module qui en
a besoin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

Point = tuple[float, float]


@dataclass(frozen=True)
class BoundingBox:
    """Boite englobante en pixels, coin superieur gauche et inferieur droit."""

    left: float
    top: float
    right: float
    bottom: float

    @classmethod
    def from_xyxy(cls, values) -> "BoundingBox":
        left, top, right, bottom = values
        return cls(float(left), float(top), float(right), float(bottom))

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.left, self.top, self.right, self.bottom)

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def centre(self) -> Point:
        return ((self.left + self.right) / 2.0, (self.top + self.bottom) / 2.0)

    @property
    def ground_contact(self) -> Point:
        """Point ou l'objet touche le sol : milieu du bord inferieur.

        C'est le seul point de la boite dont on puisse raisonnablement dire
        qu'il appartient au plan du court. Toute projection par homographie doit
        partir de la, sous peine d'une erreur qui croit avec l'eloignement.
        """
        return ((self.left + self.right) / 2.0, self.bottom)


@dataclass(frozen=True)
class Detection:
    """Un objet detecte sur une frame."""

    box: BoundingBox
    confidence: float
    track_id: Optional[int] = None


@dataclass(frozen=True)
class CourtPosition:
    """Position sur le court, en metres, dans le repere du modele de court."""

    across: float  # axe lateral, 0 au couloir gauche
    along: float   # axe filet-fond, 0 au fond eloigne

    def as_tuple(self) -> Point:
        return (self.across, self.along)

    def distance_to(self, other: "CourtPosition") -> float:
        return ((self.across - other.across) ** 2 + (self.along - other.along) ** 2) ** 0.5
