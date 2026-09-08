"""Tests de l'identification des deux joueurs parmi les personnes detectees.

Le scenario reproduit une configuration reelle : un serveur dans son court, un
relanceur place derriere sa ligne de fond, et l'arbitre de chaise sur le cote.
"""

import pytest

from tennisvision.detection.players import PLAY_AREA_MARGIN, PlayerDetector
from tennisvision.geometry import CourtHomography, CourtModel
from tennisvision.types import BoundingBox, Detection

# Vue du dessus a 100 px/m : la projection se reduit a un changement d'echelle,
# ce qui isole la logique de selection de toute question de perspective.
TOP_DOWN = [(x * 100, y * 100) for x, y in CourtModel().corners]

SERVER_POSITION = (7.07, 0.59)     # dans son court, au service
RETURNER_POSITION = (3.23, 25.46)  # 1,7 m derriere sa ligne de fond
UMPIRE_POSITION = (14.61, -8.00)   # arbitre de chaise, hors zone de jeu


@pytest.fixture
def homography():
    return CourtHomography(TOP_DOWN)


def person_at(position, track_id):
    """Detection dont le point d'appui tombe sur la position metrique voulue."""
    x, y = position[0] * 100, position[1] * 100
    return Detection(
        box=BoundingBox(x - 20, y - 170, x + 20, y),
        confidence=0.9,
        track_id=track_id,
    )


SERVER = person_at(SERVER_POSITION, 2)
RETURNER = person_at(RETURNER_POSITION, 1)
UMPIRE = person_at(UMPIRE_POSITION, 3)


def test_both_players_are_identified(homography):
    frames = [[RETURNER, SERVER, UMPIRE] for _ in range(30)]
    assert set(PlayerDetector.identify_players(frames, homography)) == {1, 2}


def test_umpire_alone_yields_nothing(homography):
    frames = [[UMPIRE] for _ in range(30)]
    assert PlayerDetector.identify_players(frames, homography) == []


def test_the_returner_needs_the_margin():
    """Documente pourquoi la marge existe, et pourquoi elle vaut 3 m.

    A marge nulle, le relanceur — pourtant en position de jeu normale — serait
    ecarte. Avec la marge retenue il est conserve, tandis que l'arbitre de
    chaise reste exclu : les deux contraintes bornent la valeur.
    """
    model = CourtModel()

    assert model.contains(RETURNER_POSITION, margin=0.0) is False
    assert model.contains(RETURNER_POSITION, margin=PLAY_AREA_MARGIN) is True
    assert model.contains(UMPIRE_POSITION, margin=PLAY_AREA_MARGIN) is False


def test_fleeting_detections_rank_below_players(homography):
    """Une piste presente sur 2 frames ne doit pas primer sur un joueur."""
    frames = [[RETURNER, SERVER] for _ in range(30)]
    for frame in frames[:2]:
        frame.append(person_at((5.0, 12.0), 9))

    assert set(PlayerDetector.identify_players(frames, homography)) == {1, 2}


def test_detections_without_track_id_are_ignored(homography):
    """Le tracker ne fournit pas d'identifiant tant qu'une piste n'est pas sure."""
    unconfirmed = Detection(box=SERVER.box, confidence=0.9, track_id=None)
    frames = [[RETURNER, SERVER, unconfirmed] for _ in range(30)]

    assert set(PlayerDetector.identify_players(frames, homography)) == {1, 2}


def test_keep_only_filters_other_tracks():
    frames = [[RETURNER, SERVER, UMPIRE] for _ in range(5)]
    filtered = PlayerDetector.keep_only(frames, [1, 2])

    assert all({d.track_id for d in frame} == {1, 2} for frame in filtered)
