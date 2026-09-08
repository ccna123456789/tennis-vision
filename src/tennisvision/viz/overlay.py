"""Annotations dessinees sur la video."""

from __future__ import annotations

import cv2

PLAYER_COLOURS = [(90, 200, 255), (120, 230, 130)]  # ambre clair, vert clair
BALL_COLOUR = (60, 240, 255)
PANEL_BG = (38, 28, 20)


def draw_detection(frame, detection, colour, label: str):
    """Encadre une detection et l'etiquette au-dessus de sa boite."""
    left, top, right, bottom = (int(v) for v in detection.box.as_xyxy())

    cv2.rectangle(frame, (left, top), (right, bottom), colour, 2, cv2.LINE_AA)

    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(
        frame, (left, top - text_height - 8), (left + text_width + 8, top), colour, cv2.FILLED
    )
    cv2.putText(
        frame, label, (left + 4, top - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA,
    )
    return frame


def draw_stat_panel(frame, lines, origin=(24, 24), width: int = 300):
    """Affiche un bandeau de statistiques en direct dans un coin de l'image."""
    if not lines:
        return frame

    x, y = origin
    line_height = 24
    height = line_height * len(lines) + 18

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), PANEL_BG, cv2.FILLED)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (210, 210, 210), 1)

    for index, (text, colour) in enumerate(lines):
        cv2.putText(
            frame, text, (x + 12, y + 26 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA,
        )

    return frame
