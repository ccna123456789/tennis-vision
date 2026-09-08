"""Distances et vitesses derivees d'une trajectoire."""

from __future__ import annotations

from dataclasses import dataclass, field

# Bornes physiques, pas des reglages empiriques. Elles servent a ecarter les
# artefacts de suivi : quand le tracker permute deux identites, le deplacement
# apparent entre deux frames devient enorme.
MAX_HUMAN_SPEED = 12.0   # m/s, au-dela d'un sprint humain
MAX_BALL_SPEED = 90.0    # m/s, soit 324 km/h, au-dela du record de service

MS_TO_KMH = 3.6


@dataclass
class MotionSummary:
    """Resume cinematique d'un objet sur la sequence analysee."""

    label: str
    distance_m: float = 0.0
    speeds_ms: list = field(default_factory=list)
    rejected_segments: int = 0

    @property
    def mean_speed_ms(self) -> float:
        return sum(self.speeds_ms) / len(self.speeds_ms) if self.speeds_ms else 0.0

    @property
    def peak_speed_ms(self) -> float:
        return max(self.speeds_ms) if self.speeds_ms else 0.0

    @property
    def mean_speed_kmh(self) -> float:
        return self.mean_speed_ms * MS_TO_KMH

    @property
    def peak_speed_kmh(self) -> float:
        return self.peak_speed_ms * MS_TO_KMH

    def as_dict(self) -> dict:
        return {
            "distance_m": round(self.distance_m, 2),
            "mean_speed_kmh": round(self.mean_speed_kmh, 2),
            "peak_speed_kmh": round(self.peak_speed_kmh, 2),
            "measurements": len(self.speeds_ms),
            "rejected_segments": self.rejected_segments,
        }


def summarise_motion(trajectory, label: str, step: int = 5, speed_limit: float = MAX_HUMAN_SPEED) -> MotionSummary:
    """Calcule distance parcourue et vitesses le long d'une trajectoire.

    Chaque segment dont la vitesse depasse `speed_limit` est ecarte, et compte
    dans `rejected_segments` : ce compteur rend le filtrage visible plutot que
    silencieux, ce qui permet de reperer une trajectoire globalement douteuse.
    """
    summary = MotionSummary(label=label)

    for start, end in trajectory.segments(step):
        origin = trajectory[start]
        target = trajectory[end]
        if origin is None or target is None:
            continue

        distance = origin.distance_to(target)
        elapsed = trajectory.duration_between(start, end)
        if elapsed <= 0:
            continue

        speed = distance / elapsed
        if speed > speed_limit:
            summary.rejected_segments += 1
            continue

        summary.distance_m += distance
        summary.speeds_ms.append(speed)

    return summary
