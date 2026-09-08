"""Tests de la geometrie du court et de l'homographie.

Ces tests verifient des proprietes mathematiques exactes : projeter les points
ayant servi au calcul doit redonner precisement les dimensions reglementaires.
Une erreur de signe, d'ordre des coins ou de division par la composante
homogene se voit immediatement.
"""

import pytest

from tennisvision.geometry import (
    COURT_LENGTH,
    DOUBLES_WIDTH,
    NET_DISTANCE,
    CalibrationError,
    CourtHomography,
    CourtModel,
)

# Quadrilatere en perspective : le fond eloigne apparait plus etroit que le
# premier plan, comme sur une prise de vue reelle depuis les gradins.
IMAGE_CORNERS = [
    (578.0, 300.0),
    (1345.0, 300.0),
    (1597.0, 851.0),
    (365.0, 851.0),
]


@pytest.fixture
def homography():
    return CourtHomography(IMAGE_CORNERS)


# --- Modele de court ---------------------------------------------------------
def test_court_dimensions_are_regulation():
    model = CourtModel()
    assert model.width == pytest.approx(10.97)
    assert model.length == pytest.approx(23.77)
    assert NET_DISTANCE == pytest.approx(11.885)


def test_markings_are_all_inside_the_court():
    model = CourtModel()
    for segment in model.markings():
        for point in (segment.start, segment.end):
            assert model.contains(point), f"{segment.name} sort du court"


def test_half_detection():
    model = CourtModel()
    assert model.half((5.0, 3.0)) == "far"
    assert model.half((5.0, 20.0)) == "near"


# --- Homographie -------------------------------------------------------------
def test_corners_project_onto_regulation_rectangle(homography):
    expected = CourtModel().corners
    for image_corner, target in zip(IMAGE_CORNERS, expected):
        position = homography.to_court(image_corner)
        assert position.across == pytest.approx(target[0], abs=1e-3)
        assert position.along == pytest.approx(target[1], abs=1e-3)


def test_round_trip_is_identity(homography):
    point = (900.0, 600.0)
    back = homography.to_image(homography.to_court(point))
    assert back[0] == pytest.approx(point[0], abs=1e-3)
    assert back[1] == pytest.approx(point[1], abs=1e-3)


def test_perspective_is_corrected(homography):
    """Deux largeurs tres differentes en pixels mesurent la meme chose en metres.

    C'est le coeur du projet : au fond du court 767 px couvrent la largeur,
    au premier plan il en faut 1232 pour la meme distance reelle.
    """
    far_px = IMAGE_CORNERS[1][0] - IMAGE_CORNERS[0][0]
    near_px = IMAGE_CORNERS[2][0] - IMAGE_CORNERS[3][0]
    assert near_px > far_px

    far = homography.to_court(IMAGE_CORNERS[1]).across - homography.to_court(IMAGE_CORNERS[0]).across
    near = homography.to_court(IMAGE_CORNERS[2]).across - homography.to_court(IMAGE_CORNERS[3]).across

    assert far == pytest.approx(near, abs=1e-3)
    assert far == pytest.approx(DOUBLES_WIDTH, abs=1e-3)


def test_exact_solution_has_no_reprojection_error(homography):
    assert homography.reprojection_error() == pytest.approx(0.0, abs=1e-6)


def test_extra_correspondences_are_accepted():
    """Au-dela de 4 points, la matrice est estimee au sens des moindres carres."""
    model = CourtModel()
    exact = CourtHomography(IMAGE_CORNERS)

    # On ajoute deux points supplementaires parfaitement coherents.
    extra_court = [(DOUBLES_WIDTH / 2, NET_DISTANCE), (1.37, COURT_LENGTH / 2)]
    extra_image = [exact.to_image(p) for p in extra_court]

    refined = CourtHomography(
        list(IMAGE_CORNERS) + extra_image,
        model.corners + extra_court,
    )
    assert refined.reprojection_error() < 1e-3


def test_play_area_margin(homography):
    assert homography.in_play_area((5.0, 12.0)) is True
    assert homography.in_play_area((3.2, 25.5), margin=0.0) is False
    assert homography.in_play_area((3.2, 25.5), margin=3.0) is True
    assert homography.in_play_area((14.6, -8.0), margin=3.0) is False


def test_too_few_points_is_rejected():
    with pytest.raises(CalibrationError, match="4 correspondances"):
        CourtHomography(IMAGE_CORNERS[:3])


def test_mismatched_point_counts_is_rejected():
    with pytest.raises(CalibrationError, match="Autant de points"):
        CourtHomography(IMAGE_CORNERS, CourtModel().corners[:3])


def test_save_and_load(tmp_path, homography):
    path = tmp_path / "calibration.json"
    homography.save(path)

    reloaded = CourtHomography.load(path)
    point = (1000.0, 700.0)
    assert reloaded.to_court(point).as_tuple() == pytest.approx(
        homography.to_court(point).as_tuple(), abs=1e-9
    )


def test_load_missing_file_is_actionable(tmp_path):
    with pytest.raises(CalibrationError, match="calibrate"):
        CourtHomography.load(tmp_path / "absent.json")
