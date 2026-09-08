"""Tests de la serie temporelle de positions."""

import pytest

from tennisvision.tracking import Trajectory
from tennisvision.types import CourtPosition

FPS = 30.0


def straight_line(count, step_m=0.2):
    """Trajectoire rectiligne : une position tous les `step_m` metres."""
    return [CourtPosition(1.0 + i * step_m, 5.0) for i in range(count)]


def test_coverage():
    positions = [CourtPosition(0, 0), None, CourtPosition(1, 1), None]
    assert Trajectory(positions, FPS).coverage == pytest.approx(0.5)


def test_empty_trajectory_has_zero_coverage():
    assert Trajectory([], FPS).coverage == 0.0


def test_invalid_fps_is_rejected():
    with pytest.raises(ValueError, match="fps"):
        Trajectory([], 0)


# --- Comblement des absences -------------------------------------------------
def test_short_gaps_are_interpolated():
    positions = [CourtPosition(0.0, 0.0), None, None, CourtPosition(3.0, 3.0)]
    filled = Trajectory(positions, FPS).fill_gaps(max_gap=4)

    assert filled[1].across == pytest.approx(1.0)
    assert filled[2].across == pytest.approx(2.0)
    assert filled.coverage == 1.0


def test_long_gaps_are_left_open():
    """Une longue absence n'est pas comblee : ce serait inventer une trajectoire."""
    positions = [CourtPosition(0.0, 0.0)] + [None] * 20 + [CourtPosition(20.0, 0.0)]
    filled = Trajectory(positions, FPS).fill_gaps(max_gap=6)

    assert filled[10] is None
    assert filled.coverage == pytest.approx(2 / 22)


def test_leading_absences_are_not_invented():
    """Avant la premiere detection, il n'y a rien sur quoi s'appuyer."""
    positions = [None, None, CourtPosition(5.0, 5.0), CourtPosition(6.0, 5.0)]
    filled = Trajectory(positions, FPS).fill_gaps(max_gap=6)

    assert filled[0] is None
    assert filled[1] is None


# --- Echantillonnage ---------------------------------------------------------
def test_sampling_respects_the_step():
    trajectory = Trajectory(straight_line(21), FPS)
    assert trajectory.sample_indices(step=5) == [0, 5, 10, 15, 20]


def test_last_observation_is_always_sampled():
    """Sans cette regle, tout deplacement de la fenetre finale serait perdu.

    Avec 23 frames et un pas de 5, la grille s'arrete a 20 ; les frames 21 et 22
    doivent malgre tout etre prises en compte.
    """
    trajectory = Trajectory(straight_line(23), FPS)
    samples = trajectory.sample_indices(step=5)

    assert samples[-1] == 22
    assert samples == [0, 5, 10, 15, 20, 22]


def test_sampling_with_a_single_observation():
    trajectory = Trajectory([CourtPosition(0, 0)] + [None] * 5, FPS)
    assert trajectory.sample_indices(step=5) == [0]


def test_duration_between_frames():
    trajectory = Trajectory(straight_line(31), FPS)
    assert trajectory.duration_between(0, 30) == pytest.approx(1.0)
