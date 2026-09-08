"""Tests des types de base."""

import pytest

from tennisvision.types import BoundingBox, CourtPosition, Detection


def test_ground_contact_is_bottom_centre():
    box = BoundingBox(100.0, 200.0, 140.0, 360.0)
    assert box.ground_contact == (120.0, 360.0)


def test_centre_differs_from_ground_contact():
    """Le centre est a hauteur de torse : il n'appartient pas au plan du sol."""
    box = BoundingBox(100.0, 200.0, 140.0, 360.0)
    assert box.centre[1] < box.ground_contact[1]


def test_box_dimensions():
    box = BoundingBox.from_xyxy([10, 20, 50, 200])
    assert box.width == 40
    assert box.height == 180


def test_court_position_distance():
    a = CourtPosition(0.0, 0.0)
    b = CourtPosition(3.0, 4.0)
    assert a.distance_to(b) == pytest.approx(5.0)


def test_detection_defaults_to_no_track_id():
    detection = Detection(box=BoundingBox(0, 0, 1, 1), confidence=0.5)
    assert detection.track_id is None
