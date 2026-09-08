"""Tests des grandeurs cinematiques et de la detection des coups.

La strategie : construire des deplacements dont la reponse se calcule a la main,
puis verifier l'egalite exacte plutot qu'un ordre de grandeur.
"""

import pytest

from tennisvision.analytics import detect_shots, summarise_motion
from tennisvision.tracking import Trajectory
from tennisvision.types import CourtPosition

FPS = 30.0


def test_distance_matches_hand_computation():
    """0,2 m par frame sur 30 intervalles : 6,00 m exactement."""
    positions = [CourtPosition(0.0 + i * 0.2, 5.0) for i in range(31)]
    summary = summarise_motion(Trajectory(positions, FPS), "test", step=5)

    assert summary.distance_m == pytest.approx(6.0, abs=1e-6)


def test_speed_matches_hand_computation():
    """0,2 m par frame a 30 fps = 6 m/s = 21,6 km/h."""
    positions = [CourtPosition(0.0 + i * 0.2, 5.0) for i in range(31)]
    summary = summarise_motion(Trajectory(positions, FPS), "test", step=5)

    assert summary.mean_speed_ms == pytest.approx(6.0, abs=1e-6)
    assert summary.mean_speed_kmh == pytest.approx(21.6, abs=1e-4)


def test_final_window_is_counted():
    """23 frames, pas de 5 : les 2 dernieres frames doivent compter."""
    positions = [CourtPosition(0.0 + i * 0.2, 5.0) for i in range(23)]
    summary = summarise_motion(Trajectory(positions, FPS), "test", step=5)

    assert summary.distance_m == pytest.approx(22 * 0.2, abs=1e-6)


def test_implausible_segments_are_rejected_and_counted():
    """Un saut de 20 m en 5 frames est un artefact de suivi, pas un deplacement."""
    positions = [CourtPosition(0.0, 5.0)] * 5 + [CourtPosition(20.0, 5.0)]
    summary = summarise_motion(Trajectory(positions, FPS), "test", step=5)

    assert summary.distance_m == 0.0
    assert summary.rejected_segments == 1


def test_gaps_do_not_break_the_computation():
    positions = [CourtPosition(0.0, 5.0)] + [None] * 10 + [CourtPosition(2.0, 5.0)]
    summary = summarise_motion(Trajectory(positions, FPS), "test", step=5)

    assert summary.distance_m == pytest.approx(2.0, abs=1e-6)


def test_empty_summary_is_safe():
    summary = summarise_motion(Trajectory([], FPS), "test")
    assert summary.distance_m == 0.0
    assert summary.mean_speed_kmh == 0.0
    assert summary.peak_speed_kmh == 0.0


# --- Coups -------------------------------------------------------------------
def build_rally(*legs):
    """Construit une trajectoire faite de trajets alternes le long du court."""
    positions = []
    for start, stop in legs:
        count = int(abs(stop - start) / 0.5) + 1
        direction = 1 if stop > start else -1
        for i in range(count):
            positions.append(CourtPosition(5.0, start + direction * i * 0.5))
    return Trajectory(positions, FPS)


def test_two_direction_changes_are_two_shots():
    trajectory = build_rally((2.0, 20.0), (20.0, 2.0), (2.0, 20.0))
    assert len(detect_shots(trajectory)) == 2


def test_a_straight_run_has_no_shot():
    assert detect_shots(build_rally((2.0, 20.0))) == []


def test_shots_record_their_direction():
    shots = detect_shots(build_rally((2.0, 20.0), (20.0, 2.0)))
    assert len(shots) == 1
    assert shots[0].direction == "vers le fond eloigne"


def test_too_few_observations_yields_no_shot():
    assert detect_shots(Trajectory([None, None, None], FPS)) == []
