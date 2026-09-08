"""Vue du dessus du court, incrustee dans la video.

Le dessin derive entierement du modele de court : les traits sont ceux que
`CourtModel.markings()` decrit en metres, convertis a l'echelle du panneau.
Aucune coordonnee de pixel n'est ecrite en dur, si bien qu'une correction de
dimension reglementaire se propage automatiquement au rendu.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..geometry import CourtModel


class CourtRadar:
    """Panneau vue du dessus ou reporter les positions reelles.

    Parameters
    ----------
    frame_shape:
        Dimensions (hauteur, largeur) de la video, pour placer le panneau.
    height:
        Hauteur du panneau en pixels. La largeur en decoule, pour respecter le
        rapport reel du court.
    """

    def __init__(self, frame_shape, height: int = 420, margin_px: int = 28, inset: int = 24):
        self.model = CourtModel()

        frame_height, frame_width = frame_shape[:2]

        # Le panneau garde le rapport largeur/longueur reel du court, augmente
        # d'une bande autour des lignes pour que les joueurs places derriere
        # leur ligne de fond restent visibles.
        self.padding_m = 2.5
        span_across = self.model.width + 2 * self.padding_m
        span_along = self.model.length + 2 * self.padding_m

        self.scale = height / span_along
        self.panel_height = int(height)
        self.panel_width = int(span_across * self.scale)

        self.right = frame_width - margin_px
        self.left = self.right - self.panel_width
        self.top = margin_px
        self.bottom = self.top + self.panel_height
        self.inset = inset

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------
    def to_pixel(self, position) -> tuple[int, int]:
        """Convertit une position metrique en pixel du panneau.

        La vue du dessus n'a plus de perspective : une simple mise a l'echelle
        suffit, contrairement a la projection depuis l'image.
        """
        across = position.across if hasattr(position, "across") else position[0]
        along = position.along if hasattr(position, "along") else position[1]

        x = self.left + (across + self.padding_m) * self.scale
        y = self.top + (along + self.padding_m) * self.scale
        return (int(round(x)), int(round(y)))

    def clamp(self, pixel) -> tuple[int, int]:
        """Ramene un point dans les bornes du panneau.

        Un joueur nettement derriere sa ligne sort du panneau ; l'afficher au
        bord informe mieux que de le faire disparaitre.
        """
        x = min(max(pixel[0], self.left + 3), self.right - 3)
        y = min(max(pixel[1], self.top + 3), self.bottom - 3)
        return (int(x), int(y))

    # ------------------------------------------------------------------
    # Rendu
    # ------------------------------------------------------------------
    def draw_panel(self, frame, opacity: float = 0.94):
        """Dessine le fond et le marquage du court sur la frame."""
        overlay = frame.copy()
        cv2.rectangle(
            overlay, (self.left, self.top), (self.right, self.bottom), (46, 34, 22), cv2.FILLED
        )
        cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

        cv2.rectangle(frame, (self.left, self.top), (self.right, self.bottom), (210, 210, 210), 1)

        for segment in self.model.markings():
            start = self.to_pixel(segment.start)
            end = self.to_pixel(segment.end)
            is_net = segment.name == "filet"
            colour = (120, 200, 255) if is_net else (235, 235, 235)
            cv2.line(frame, start, end, colour, 2 if is_net else 1, cv2.LINE_AA)

        return frame

    def draw_marker(self, frame, position, colour, radius: int = 6, label: str | None = None):
        """Place un marqueur a une position metrique."""
        if position is None:
            return frame

        x, y = self.clamp(self.to_pixel(position))
        cv2.circle(frame, (x, y), radius + 1, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), radius, colour, -1, cv2.LINE_AA)

        if label:
            cv2.putText(
                frame, label, (x + radius + 4, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA,
            )

        return frame

    def draw_trail(self, frame, positions, colour, length: int = 12):
        """Trace la trainee des `length` dernieres positions connues."""
        recent = [p for p in positions[-length:] if p is not None]
        if len(recent) < 2:
            return frame

        points = [self.clamp(self.to_pixel(p)) for p in recent]
        for index in range(1, len(points)):
            fade = index / len(points)
            faded = tuple(int(channel * fade) for channel in colour)
            cv2.line(frame, points[index - 1], points[index], faded, 1, cv2.LINE_AA)

        return frame
