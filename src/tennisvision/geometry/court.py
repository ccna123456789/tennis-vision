"""Modele geometrique d'un court de tennis.

Toutes les mesures sont celles du reglement ITF, exprimees en metres.

Le repere est defini ainsi :

    (0, 0) ------------------------- (10.97, 0)      fond de court eloigne
      |                                   |
      |            filet a 11.885         |
      |                                   |
    (0, 23.77) --------------------- (10.97, 23.77)  fond de court proche

L'axe `across` est lateral, l'axe `along` va du fond eloigne au fond proche.

Ce module ne connait ni image, ni pixel : il decrit le court reel. C'est
volontaire — le meme modele sert de cible a l'homographie et de source au
dessin de la vue du dessus, ce qui garantit que les deux restent coherents.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Dimensions reglementaires ITF, en metres ---------------------------------
DOUBLES_WIDTH = 10.97
SINGLES_WIDTH = 8.23
COURT_LENGTH = 23.77
SERVICE_BOX_DEPTH = 6.40      # du filet a la ligne de service
DOUBLES_ALLEY_WIDTH = 1.37    # largeur d'un couloir de double
NET_DISTANCE = COURT_LENGTH / 2.0

# Taille moyenne d'un joueur professionnel, servant d'echelle de reference.
REFERENCE_PLAYER_HEIGHT = 1.88


@dataclass(frozen=True)
class Segment:
    """Un trait du marquage, defini par ses deux extremites en metres."""

    start: tuple[float, float]
    end: tuple[float, float]
    name: str


class CourtModel:
    """Geometrie d'un court de tennis en coordonnees metriques."""

    @property
    def corners(self) -> list[tuple[float, float]]:
        """Les 4 coins du court de double, dans le sens horaire.

        Ordre : fond eloigne gauche, fond eloigne droite, fond proche droite,
        fond proche gauche. C'est la cible de l'homographie.
        """
        return [
            (0.0, 0.0),
            (DOUBLES_WIDTH, 0.0),
            (DOUBLES_WIDTH, COURT_LENGTH),
            (0.0, COURT_LENGTH),
        ]

    @property
    def width(self) -> float:
        return DOUBLES_WIDTH

    @property
    def length(self) -> float:
        return COURT_LENGTH

    def contains(self, position, margin: float = 0.0) -> bool:
        """Indique si une position metrique tombe dans le court, a `margin` pres.

        La marge sert a definir une *zone de jeu* plus large que les lignes :
        un joueur se tient couramment derriere sa ligne de fond, au service
        comme sur un retour profond.
        """
        across, along = position
        return (
            -margin <= across <= DOUBLES_WIDTH + margin
            and -margin <= along <= COURT_LENGTH + margin
        )

    def half(self, position) -> str:
        """Renvoie 'far' ou 'near' selon le cote du filet."""
        return "far" if position[1] < NET_DISTANCE else "near"

    def markings(self) -> list[Segment]:
        """Tous les traits du marquage, en metres.

        Cette liste est la source unique du dessin de la vue du dessus : aucune
        coordonnee de pixel n'est ecrite en dur nulle part.
        """
        alley = DOUBLES_ALLEY_WIDTH
        singles_left = alley
        singles_right = DOUBLES_WIDTH - alley
        service_far = NET_DISTANCE - SERVICE_BOX_DEPTH
        service_near = NET_DISTANCE + SERVICE_BOX_DEPTH
        centre = DOUBLES_WIDTH / 2.0

        return [
            # Perimetre du court de double
            Segment((0.0, 0.0), (DOUBLES_WIDTH, 0.0), "fond eloigne"),
            Segment((0.0, COURT_LENGTH), (DOUBLES_WIDTH, COURT_LENGTH), "fond proche"),
            Segment((0.0, 0.0), (0.0, COURT_LENGTH), "couloir gauche"),
            Segment((DOUBLES_WIDTH, 0.0), (DOUBLES_WIDTH, COURT_LENGTH), "couloir droit"),
            # Lignes de simple
            Segment((singles_left, 0.0), (singles_left, COURT_LENGTH), "simple gauche"),
            Segment((singles_right, 0.0), (singles_right, COURT_LENGTH), "simple droite"),
            # Lignes de service
            Segment((singles_left, service_far), (singles_right, service_far), "service eloigne"),
            Segment((singles_left, service_near), (singles_right, service_near), "service proche"),
            # Ligne mediane des carres de service
            Segment((centre, service_far), (centre, service_near), "mediane"),
            # Filet
            Segment((0.0, NET_DISTANCE), (DOUBLES_WIDTH, NET_DISTANCE), "filet"),
        ]
