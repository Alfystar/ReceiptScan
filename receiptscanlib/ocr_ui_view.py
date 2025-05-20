"""
Modulo che contiene la classe OcrUiView per la visualizzazione dell'interfaccia utente dell'applicazione OCR.
Segue il pattern MVC (Model-View-Controller) dove questa classe rappresenta la View.
"""

import os
import logging
import cv2
import numpy as np
from PyQt6.QtCore import Qt, QPoint, QSize, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QIcon
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout,
    QSlider, QListWidget, QListWidgetItem, QTextEdit
)

# Configurazione del logger per questo modulo
logger = logging.getLogger(__name__)


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
        self.radius_select_area = 30
        self.sapphire = QColor(255, 56, 0)  # Colore rosso per i punti 2-4
        self.orange = QColor(0, 165, 255)   # Colore azzurro per il punto 1
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


class PreviewWidget(QWidget):
    """Widget per la visualizzazione dell'anteprima dei file."""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        modifiers = QApplication.keyboardModifiers()
        if (modifiers & Qt.KeyboardModifier.ControlModifier) or (modifiers & Qt.KeyboardModifier.MetaModifier):
            delta = event.angleDelta().y()
            if delta > 0:
                new_size = min(self.parent.preview_size + 8, 120)
            else:
                new_size = max(self.parent.preview_size - 8, 32)
            self.parent.size_slider.setValue(new_size)
        else:
            super().wheelEvent(event)


class OcrUiView(QMainWindow):
    """
    View dell'applicazione OCR secondo il pattern MVC.
    Si occupa esclusivamente della creazione dell'interfaccia grafica.
    """

    # Segnali per collegare la View al Controller
    preview_selected = pyqtSignal(int)
    preview_size_changed = pyqtSignal(int)
    analyze_clicked = pyqtSignal()
    analyze_all_clicked = pyqtSignal()
    text_comment_changed = pyqtSignal(str)

    def __init__(self):
        """Inizializza la finestra principale dell'applicazione."""
        super().__init__()
        self.setWindowTitle("OCR Receipt Annotator")
        self.preview_size = 60  # dimensione preview regolabile
        self.preview_items = []
        self.init_ui()

    def init_ui(self):
        """Inizializza l'interfaccia utente."""
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        # Preview verticale immagini
        preview_layout = self._setup_preview_panel()
        self.preview_widget = PreviewWidget(self)
        self.preview_widget.setLayout(preview_layout)
        main_layout.addWidget(self.preview_widget, 0)

        # Sinistra: immagine originale con punti di controllo
        self.img_label = ImageLabel()
        main_layout.addWidget(self.img_label, 2)

        # Centro: immagine wrappata e campo commenti
        center_layout = self._setup_center_panel()
        center_widget = QWidget()
        center_widget.setLayout(center_layout)
        main_layout.addWidget(center_widget, 1)

        # Destra: textarea OCR scrollabile
        right_panel = self._setup_right_panel()
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        main_layout.addWidget(right_widget, 1)

        # Sotto: barra navigazione
        nav_layout = self._setup_navigation_bar()

        # Layout finale
        layout = QVBoxLayout()
        layout.addLayout(main_layout)
        layout.addLayout(nav_layout)
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)
        self.resize(1000, 700)

    def _setup_preview_panel(self):
        """Configura il pannello sinistro con le anteprime delle immagini."""
        preview_layout = QVBoxLayout()

        # Titolo della colonna di anteprima
        preview_title_label = QLabel("Elenco File")
        preview_title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(preview_title_label)

        # Lista dei file con anteprima
        self.preview_list = QListWidget()
        self.preview_list.setMaximumWidth(180)
        self.preview_list.setIconSize(QSize(self.preview_size, self.preview_size))
        self.preview_list.setMinimumWidth(140)
        self.preview_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.preview_list.setSpacing(8)
        self.preview_list.setStyleSheet("QListWidget::item { margin-bottom: 8px; }")
        self.preview_list.currentRowChanged.connect(self._on_preview_selected)
        preview_layout.addWidget(self.preview_list)

        # Slider per la dimensione delle anteprime
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setMinimum(32)
        self.size_slider.setMaximum(120)
        self.size_slider.setValue(self.preview_size)
        self.size_slider.valueChanged.connect(self._on_preview_size_changed)
        preview_layout.addWidget(self.size_slider)

        # Leggenda
        legend_label = QLabel("<b>Leggenda:</b><br><span style='color: #00A5FF;'>Punto 1 (azzurro)</span><br><span style='color: #FF3800;'>Punti 2-4 (rossi)</span><br>• Trascina i punti col mouse<br>• Click destro per reset")
        legend_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        legend_label.setWordWrap(False)
        legend_label.setStyleSheet("background-color: #f0f0f0; padding: 8px; border-radius: 3px; min-width: 170px;")
        preview_layout.addWidget(legend_label)

        return preview_layout

    def _setup_center_panel(self):
        """Configura il pannello centrale con l'immagine elaborata e i commenti."""
        center_layout = QVBoxLayout()

        # Area per visualizzare l'immagine elaborata
        self.wrapped_img_label = QLabel()
        self.wrapped_img_label.setMinimumSize(300, 300)
        self.wrapped_img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.wrapped_img_label.setStyleSheet("border: 1px solid gray;")
        center_layout.addWidget(self.wrapped_img_label, 3)

        # Area per i commenti
        comments_layout = QVBoxLayout()
        comments_label = QLabel("<b>Commenti</b>")
        comments_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        comments_layout.addWidget(comments_label)

        self.comments_edit = QTextEdit()
        self.comments_edit.setPlaceholderText("Inserisci qui i tuoi commenti sull'immagine...")
        self.comments_edit.setMinimumHeight(100)
        self.comments_edit.textChanged.connect(self._on_comment_changed)
        comments_layout.addWidget(self.comments_edit)
        center_layout.addLayout(comments_layout, 1)

        return center_layout

    def _setup_right_panel(self):
        """Configura il pannello destro con i risultati OCR."""
        right_panel = QVBoxLayout()

        ocr_label = QLabel("<b>Analisi OCR/LLM</b>")
        ocr_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        right_panel.addWidget(ocr_label)

        self.ocr_text = QTextEdit()
        self.ocr_text.setReadOnly(True)
        self.ocr_text.setPlaceholderText("Qui verrà mostrato il risultato dell'analisi OCR...")
        right_panel.addWidget(self.ocr_text, 1)

        return right_panel

    def _setup_navigation_bar(self):
        """Configura la barra di navigazione in fondo."""
        nav_layout = QHBoxLayout()

        self.analyze_btn = QPushButton("Start Analyze")
        self.analyze_btn.clicked.connect(self._on_analyze_clicked)

        self.analyze_all_btn = QPushButton("Start Analyze All")
        self.analyze_all_btn.clicked.connect(self._on_analyze_all_clicked)

        nav_layout.addStretch(1)
        nav_layout.addWidget(self.analyze_btn)
        nav_layout.addWidget(self.analyze_all_btn)

        return nav_layout

    # Metodi gestione eventi interni che emettono segnali per il controller
    def _on_preview_selected(self, row):
        """Gestisce la selezione di un'anteprima dalla lista."""
        self.preview_selected.emit(row)

    def _on_preview_size_changed(self, value):
        """Gestisce il cambio di dimensione delle anteprime."""
        self.preview_size = value
        self.preview_list.setIconSize(QSize(value, value))
        self.preview_size_changed.emit(value)

    def _on_analyze_clicked(self):
        """Gestisce il click sul pulsante Analyze."""
        self.analyze_clicked.emit()

    def _on_analyze_all_clicked(self):
        """Gestisce il click sul pulsante Analyze All."""
        self.analyze_all_clicked.emit()

    def _on_comment_changed(self):
        """Gestisce il cambio del testo nei commenti."""
        self.text_comment_changed.emit(self.comments_edit.toPlainText())

    # Metodi pubblici per aggiornare l'interfaccia (chiamati dal Controller)
    def set_image_files(self, image_dir, image_files):
        """Imposta la lista dei file immagine."""
        self.preview_list.clear()
        self.preview_items = []

        for fname in image_files:
            img_path = os.path.join(image_dir, fname)
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                qimg = QImage(img.data, w, h, img.strides[0], QImage.Format.Format_BGR888)
                pix = QPixmap.fromImage(qimg).scaled(self.preview_size, self.preview_size,
                                                 Qt.AspectRatioMode.KeepAspectRatio,
                                                 Qt.TransformationMode.SmoothTransformation)
                item = QListWidgetItem(QIcon(pix), fname)
                self.preview_list.addItem(item)
                self.preview_items.append(item)

    def set_current_image(self, img, points, filename):
        """Imposta l'immagine corrente con i suoi punti di controllo."""
        self.img_label.set_image(img, points)
        self.setWindowTitle(f"OCR Receipt Annotator - {filename}")

    def set_current_index(self, idx):
        """Imposta l'indice corrente nella lista di anteprime."""
        if self.preview_list.currentRow() != idx:
            self.preview_list.setCurrentRow(idx)

    def set_wrapped_image(self, pixmap):
        """Imposta l'immagine elaborata."""
        if pixmap:
            scaled_pixmap = pixmap.scaled(self.wrapped_img_label.size(),
                                      Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation)
            self.wrapped_img_label.setPixmap(scaled_pixmap)
        else:
            self.wrapped_img_label.clear()
            self.wrapped_img_label.setText("Nessuna immagine elaborata")

    def set_ocr_text(self, text):
        """Imposta il testo OCR."""
        self.ocr_text.setText(text)

    def set_comment_text(self, text):
        """Imposta il testo del commento."""
        # Blocca i segnali per evitare che il cambiamento generi un evento
        self.comments_edit.blockSignals(True)
        self.comments_edit.setText(text)
        self.comments_edit.blockSignals(False)

    def update_preview_icon(self, idx, pixmap):
        """Aggiorna l'icona di anteprima per un indice specifico."""
        if 0 <= idx < len(self.preview_items):
            self.preview_items[idx].setIcon(QIcon(pixmap))
            self.preview_list.update()

    def get_current_points(self):
        """Ottiene i punti di controllo correnti dall'ImageLabel."""
        return self.img_label.get_points()
