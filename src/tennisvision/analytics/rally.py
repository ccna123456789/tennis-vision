"""Analyse du deroulement de l'echange."""

from __future__ import annotations

from dataclasses import dataclass

# Deplacement minimal sur l'axe du court pour qu'une frame compte comme un
# mouvement, en metres. En deca, on est dans le bruit de detection.
MOTION_THRESHOLD = 0.10

# Ecart minimal entre deux coups, en frames. A 30 fps, 8 frames font un quart
# de seconde : aucun echange reel n'enchaine deux frappes plus vite.
MIN_FRAMES_BETWEEN_SHOTS = 8


@dataclass(frozen=True)
class Shot:
    """Un coup detecte, repere par la frame ou la balle change de sens."""

    frame_index: int
    direction: str  # 'vers le fond proche' ou 'vers le fond eloigne'

    def as_dict(self) -> dict:
        return {"frame": self.frame_index, "direction": self.direction}


def detect_shots(trajectory, min_separation: int = MIN_FRAMES_BETWEEN_SHOTS) -> list[Shot]:
    """Detecte les coups par inversion du sens de la balle.

    Le principe : une frappe renvoie la balle dans l'autre camp, donc inverse
    le signe de sa vitesse le long de l'axe filet-fond de court. On repere ces
    changements de signe.

    Limite assumee : la methode ne distingue pas une frappe d'un autre evenement
    qui inverserait ce sens, comme une balle repoussee par le filet. Avec une
    couverture de detection partielle, elle sous-estime plutot qu'elle
    ne surestime.
    """
    observed = [(i, trajectory[i]) for i in trajectory.observed_indices]
    if len(observed) < 3:
        return []

    shots: list[Shot] = []
    previous_sign = 0
    last_shot_frame = -min_separation

    for (_, previous_position), (index, position) in zip(observed, observed[1:]):
        delta = position.along - previous_position.along
        if abs(delta) < MOTION_THRESHOLD:
            continue

        sign = 1 if delta > 0 else -1

        if previous_sign != 0 and sign != previous_sign:
            if index - last_shot_frame >= min_separation:
                shots.append(
                    Shot(
                        frame_index=index,
                        direction="vers le fond proche" if sign > 0 else "vers le fond eloigne",
                    )
                )
                last_shot_frame = index

        previous_sign = sign

    return shots
