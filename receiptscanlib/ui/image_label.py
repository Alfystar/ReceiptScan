"""
Modulo che contiene la classe ImageLabel per la visualizzazione di un'immagine con punti di controllo.
"""

import numpy as np
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt6.QtWidgets import QLabel


class ImageLabel(QLabel):
    """
    Widget per la visualizzazione di un'immagine con punti di controllo interattivi.
    Permette di modificare i punti per la trasformazione prospettica.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.image = None
        self.points = []
        self.dragging_idx = None
        self.radius_dot_draw = 3
        self.radius_select_area = 50
        self.sapphire = QColor(255, 56, 0)  # Colore rosso per i punti 2-4
        self.orange = QColor(0, 165, 255)  # Colore azzurro per il punto 1
        self.setMinimumSize(400, 400)

    def set_image(self, image, points=None):
        """Imposta l'immagine e i punti di controllo."""
        self.image = image.copy()
        h, w = self.image.shape[:2]
        if points is None or len(points) != 4:
            self.points = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
        else:
            self.points = [
                [max(0, min(w - 1, int(p[0]))), max(0, min(h - 1, int(p[1])))]
                for p in points
            ]
        self.update()

    def get_points(self):
        """Restituisce i punti di controllo come array NumPy."""
        return np.array(self.points, dtype=np.float32)

    def paintEvent(self, event):
        """Gestisce il disegno dell'immagine e dei punti di controllo."""
        super().paintEvent(event)
        if self.image is not None:
            h, w = self.image.shape[:2]
            # Calcola il rettangolo di destinazione per mantenere l'aspect ratio
            widget_w, widget_h = self.width(), self.height()
            scale = min(widget_w / w, widget_h / h)
            disp_w, disp_h = int(w * scale), int(h * scale)
            offset_x = (widget_w - disp_w) // 2
            offset_y = (widget_h - disp_h) // 2
            qimg = QImage(self.image.data, w, h, self.image.strides[0], QImage.Format.Format_BGR888)
            pix = QPixmap.fromImage(qimg).scaled(disp_w, disp_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            painter = QPainter(self)
            painter.drawPixmap(offset_x, offset_y, pix)
            # Disegna solo il rettangolo
            pen = QPen(self.sapphire, 2)
            painter.setPen(pen)
            scaled_points = [
                QPoint(int(p[0] * scale) + offset_x, int(p[1] * scale) + offset_y)
                for p in self.points
            ]
            for i in range(4):
                painter.drawLine(scaled_points[i], scaled_points[(i + 1) % 4])
            # Punti
            for i, p in enumerate(scaled_points):
                color = self.orange if i == 0 else self.sapphire
                painter.setPen(QPen(color, 2))
                painter.setBrush(color)
                painter.drawEllipse(p, self.radius_dot_draw, self.radius_dot_draw)
            painter.end()

    def mousePressEvent(self, event):
        """Gestisce il click del mouse sui punti di controllo."""
        if self.image is None:
            return
        h, w = self.image.shape[:2]
        widget_w, widget_h = self.width(), self.height()
        scale = min(widget_w / w, widget_h / h)
        disp_w, disp_h = int(w * scale), int(h * scale)
        offset_x = (widget_w - disp_w) // 2
        offset_y = (widget_h - disp_h) // 2
        x = (event.position().x() - offset_x) / scale
        y = (event.position().y() - offset_y) / scale
        if x < 0 or y < 0 or x > w - 1 or y > h - 1:
            return
        for i, p in enumerate(self.points):
            if np.linalg.norm([x - p[0], y - p[1]]) < self.radius_select_area:
                if event.button() == Qt.MouseButton.LeftButton:
                    self.dragging_idx = i
                elif event.button() == Qt.MouseButton.RightButton:
                    # Reset al vertice originale
                    if i == 0:
                        self.points[i] = [0, 0]
                    elif i == 1:
                        self.points[i] = [w - 1, 0]
                    elif i == 2:
                        self.points[i] = [w - 1, h - 1]
                    elif i == 3:
                        self.points[i] = [0, h - 1]
                    self.update()
                return

    def mouseMoveEvent(self, event):
        """Gestisce il trascinamento dei punti di controllo."""
        if self.image is None or self.dragging_idx is None:
            return
        h, w = self.image.shape[:2]
        widget_w, widget_h = self.width(), self.height()
        scale = min(widget_w / w, widget_h / h)
        disp_w, disp_h = int(w * scale), int(h * scale)
        offset_x = (widget_w - disp_w) // 2
        offset_y = (widget_h - disp_h) // 2
        x = (event.position().x() - offset_x) / scale
        y = (event.position().y() - offset_y) / scale
        x = max(0, min(w - 1, int(x)))
        y = max(0, min(h - 1, int(y)))
        self.points[self.dragging_idx] = [x, y]
        self.update()

    def mouseReleaseEvent(self, event):
        """Gestisce il rilascio del mouse."""
        self.dragging_idx = None
