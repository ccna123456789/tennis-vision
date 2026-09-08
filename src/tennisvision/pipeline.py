"""Orchestration de l'analyse.

Ce module ne contient aucune logique metier : il enchaine les composants et
assemble le rapport. Toute la geometrie est dans `geometry`, toute la
cinematique dans `analytics`. Cette separation permet de tester les calculs
sans jamais lancer de detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .analytics import MAX_BALL_SPEED, MAX_HUMAN_SPEED, detect_shots, summarise_motion
from .detection import BallDetector, PlayerDetector
from .geometry import CourtHomography
from .io import DetectionCache, probe, read_frames, write_frames
from .tracking import Trajectory
from .viz import BALL_COLOUR, PLAYER_COLOURS, CourtRadar, draw_detection, draw_stat_panel


@dataclass
class AnalysisConfig:
    """Parametres d'une analyse."""

    video: str
    calibration: str
    player_weights: str = "yolov8n.pt"
    ball_weights: str = "yolov8x.pt"
    player_confidence: float = 0.25
    ball_confidence: float = 0.05
    max_frames: int | None = 90
    sample_step: int = 5
    max_ball_gap: int = 6
    cache_dir: str = "data/cache"
    output_video: str = "data/output/analysis.avi"
    output_report: str = "data/output/report.json"
    use_cache: bool = True


@dataclass
class AnalysisResult:
    """Ce que produit une analyse."""

    video: dict = field(default_factory=dict)
    ball: dict = field(default_factory=dict)
    players: dict = field(default_factory=dict)
    shots: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "video": self.video,
            "ball": self.ball,
            "players": self.players,
            "shots": {"count": len(self.shots), "events": [s.as_dict() for s in self.shots]},
        }


class MatchAnalyser:
    """Enchaine detection, projection, analyse et rendu."""

    def __init__(self, config: AnalysisConfig, root: Path | None = None):
        self.config = config
        self.root = root or Path.cwd()
        self.cache = DetectionCache(self._resolve(config.cache_dir), enabled=config.use_cache)

    def _resolve(self, value) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    # ------------------------------------------------------------------
    def run(self, render: bool = True, log=print) -> AnalysisResult:
        config = self.config
        video_path = self._resolve(config.video)

        info = probe(video_path)
        frames = read_frames(video_path, limit=config.max_frames)
        log(f"Video     : {len(frames)} frames a {info.fps:.1f} fps ({len(frames)/info.fps:.1f} s)")

        homography = CourtHomography.load(self._resolve(config.calibration))
        log(f"Court     : homographie chargee (erreur de reprojection {homography.reprojection_error():.3f} m)")

        player_frames = self._detect_players(frames, video_path, log)
        ball_detections = self._detect_ball(frames, video_path, log)

        # --- Identification des deux joueurs -----------------------------
        detector = self._player_detector
        player_ids = detector.identify_players(player_frames, homography)
        player_frames = detector.keep_only(player_frames, player_ids)
        log(f"Joueurs   : pistes retenues {player_ids}")

        # --- Projection en coordonnees metriques -------------------------
        player_trajectories = {
            track_id: Trajectory(
                [self._position_of(frame, track_id, homography) for frame in player_frames],
                info.fps,
            )
            for track_id in player_ids
        }

        raw_ball = Trajectory(
            [
                homography.to_court(d.box.ground_contact) if d else None
                for d in ball_detections
            ],
            info.fps,
        )
        ball = raw_ball.fill_gaps(max_gap=config.max_ball_gap)
        log(
            f"Balle     : detectee sur {raw_ball.coverage:.0%} des frames, "
            f"{ball.coverage:.0%} apres comblement des absences courtes"
        )

        # --- Analyses ----------------------------------------------------
        result = AnalysisResult()
        result.video = {
            "file": Path(video_path).name,
            "frames_analysed": len(frames),
            "fps": round(info.fps, 2),
            "duration_s": round(len(frames) / info.fps, 2),
        }

        for index, (track_id, trajectory) in enumerate(sorted(player_trajectories.items())):
            summary = summarise_motion(
                trajectory, label=f"joueur {index + 1}",
                step=config.sample_step, speed_limit=MAX_HUMAN_SPEED,
            )
            result.players[f"player_{index + 1}"] = {"track_id": track_id, **summary.as_dict()}

        ball_summary = summarise_motion(
            ball, label="balle", step=3, speed_limit=MAX_BALL_SPEED
        )
        result.ball = {
            "raw_detection_rate": round(raw_ball.coverage, 3),
            "filled_detection_rate": round(ball.coverage, 3),
            **ball_summary.as_dict(),
        }
        result.shots = detect_shots(ball)

        # --- Sorties -----------------------------------------------------
        report_path = self._resolve(config.output_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        report_path.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")

        if render:
            self._render(frames, player_frames, ball_detections, player_trajectories, ball, info, log)

        return result

    # ------------------------------------------------------------------
    def _detect_players(self, frames, video_path, log):
        config = self.config
        self._player_detector = PlayerDetector(config.player_weights, config.player_confidence)

        key = self.cache.key(video_path, config.player_weights, config.player_confidence, len(frames))
        cached = self.cache.load("players", key)
        if cached is not None:
            log("Joueurs   : detections relues depuis le cache")
            return cached

        log("Joueurs   : detection en cours...")
        detections = [self._player_detector.detect(frame) for frame in frames]
        self.cache.store("players", key, detections)
        return detections

    def _detect_ball(self, frames, video_path, log):
        config = self.config
        key = self.cache.key(video_path, config.ball_weights, config.ball_confidence, len(frames))
        cached = self.cache.load("ball", key)
        if cached is not None:
            log("Balle     : detections relues depuis le cache")
            return cached

        log("Balle     : detection en cours (modele lourd, patience)...")
        detector = BallDetector(config.ball_weights, config.ball_confidence)
        detections = [detector.detect(frame) for frame in frames]
        self.cache.store("ball", key, detections)
        return detections

    @staticmethod
    def _position_of(detections, track_id, homography):
        for detection in detections:
            if detection.track_id == track_id:
                return homography.to_court(detection.box.ground_contact)
        return None

    # ------------------------------------------------------------------
    def _render(self, frames, player_frames, ball_detections, trajectories, ball, info, log):
        log("Rendu     : composition de la video annotee...")
        radar = CourtRadar(frames[0].shape)
        ordered_ids = sorted(trajectories)

        for index, frame in enumerate(frames):
            for detection in player_frames[index]:
                rank = ordered_ids.index(detection.track_id) if detection.track_id in ordered_ids else 0
                colour = PLAYER_COLOURS[rank % len(PLAYER_COLOURS)]
                draw_detection(frame, detection, colour, f"Joueur {rank + 1}")

            if ball_detections[index] is not None:
                draw_detection(frame, ball_detections[index], BALL_COLOUR, "Balle")

            radar.draw_panel(frame)
            radar.draw_trail(frame, ball.positions[: index + 1], BALL_COLOUR)
            radar.draw_marker(frame, ball[index], BALL_COLOUR, radius=4)

            lines = []
            for rank, track_id in enumerate(ordered_ids):
                position = trajectories[track_id][index]
                colour = PLAYER_COLOURS[rank % len(PLAYER_COLOURS)]
                radar.draw_marker(frame, position, colour, label=str(rank + 1))
                covered = summarise_motion(
                    Trajectory(trajectories[track_id].positions[: index + 1], info.fps),
                    label="", step=self.config.sample_step, speed_limit=MAX_HUMAN_SPEED,
                )
                lines.append((f"Joueur {rank + 1} : {covered.distance_m:5.1f} m parcourus", colour))
            draw_stat_panel(frame, lines)

        output = self._resolve(self.config.output_video)
        write_frames(frames, output, fps=info.fps)
        log(f"Rendu     : video ecrite dans {output}")
