"""Serie temporelle de positions sur le court.

Regrouper les positions d'un objet dans un seul objet `Trajectory` evite de
faire circuler des listes paralleles de positions, d'indices et de vitesses.
Toutes les grandeurs cinematiques se calculent a partir de la, avec une
definition unique du pas d'echantillonnage et du filtrage.
"""

from __future__ import annotations

from typing import Iterator, Optional

from ..types import CourtPosition


class Trajectory:
    """Positions metriques d'un objet, frame par frame.

    Une case vaut `None` quand l'objet n'a pas ete detecte sur cette frame.

    Parameters
    ----------
    positions:
        Une entree par frame, dans l'ordre chronologique.
    fps:
        Cadence de la video, necessaire pour convertir des indices de frame en
        secondes.
    """

    def __init__(self, positions, fps: float):
        if fps <= 0:
            raise ValueError(f"fps doit etre strictement positif, recu {fps!r}.")
        self.positions: list[Optional[CourtPosition]] = list(positions)
        self.fps = float(fps)

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, index):
        return self.positions[index]

    def __iter__(self) -> Iterator[Optional[CourtPosition]]:
        return iter(self.positions)

    # ------------------------------------------------------------------
    # Etat
    # ------------------------------------------------------------------
    @property
    def observed_indices(self) -> list[int]:
        """Indices des frames ou l'objet a effectivement ete detecte."""
        return [i for i, position in enumerate(self.positions) if position is not None]

    @property
    def coverage(self) -> float:
        """Proportion de frames ou l'objet a ete detecte."""
        if not self.positions:
            return 0.0
        return len(self.observed_indices) / len(self.positions)

    def duration_between(self, first_index: int, second_index: int) -> float:
        """Duree en secondes separant deux frames."""
        return abs(second_index - first_index) / self.fps

    # ------------------------------------------------------------------
    # Comblement des absences
    # ------------------------------------------------------------------
    def fill_gaps(self, max_gap: int = 6) -> "Trajectory":
        """Interpole lineairement les absences courtes.

        La justification est physique : un objet suit une trajectoire continue,
        donc entre deux detections proches, l'interpolation lineaire est une
        estimation acceptable.

        Le plafond `max_gap` est essentiel. Combler une absence de trois frames
        est raisonnable ; combler quarante frames reviendrait a inventer une
        trajectoire entiere et a la faire passer pour une mesure. Au-dela du
        plafond, le trou reste ouvert et les calculs en aval l'ignorent.
        """
        filled = list(self.positions)
        observed = self.observed_indices

        for start, end in zip(observed, observed[1:]):
            gap = end - start - 1
            if gap <= 0 or gap > max_gap:
                continue

            origin = self.positions[start]
            target = self.positions[end]

            for offset in range(1, gap + 1):
                ratio = offset / (gap + 1)
                filled[start + offset] = CourtPosition(
                    origin.across + ratio * (target.across - origin.across),
                    origin.along + ratio * (target.along - origin.along),
                )

        return Trajectory(filled, self.fps)

    # ------------------------------------------------------------------
    # Echantillonnage
    # ------------------------------------------------------------------
    def sample_indices(self, step: int) -> list[int]:
        """Indices retenus pour mesurer un deplacement.

        On ne compare pas deux frames consecutives : sur un trentieme de
        seconde, le tremblement de la boite englobante depasse le deplacement
        reel. On echantillonne donc tous les `step` frames.

        La derniere detection est toujours retenue, meme si elle tombe a moins
        de `step` frames de la precedente. Sans cela, tout deplacement survenant
        dans la fenetre finale serait perdu, ce qui biaiserait systematiquement
        la distance cumulee vers le bas.
        """
        observed = self.observed_indices
        if len(observed) < 2:
            return observed

        samples = [observed[0]]
        for index in observed[1:]:
            if index - samples[-1] >= step:
                samples.append(index)

        if samples[-1] != observed[-1]:
            samples.append(observed[-1])

        return samples

    def segments(self, step: int):
        """Genere les couples (indice_debut, indice_fin) entre echantillons."""
        samples = self.sample_indices(step)
        return list(zip(samples, samples[1:]))
