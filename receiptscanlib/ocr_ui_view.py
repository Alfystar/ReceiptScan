"""
Modulo che contiene la classe OcrUiView per la visualizzazione dell'interfaccia utente dell'applicazione OCR.
Segue il pattern MVC (Model-View-Controller) dove questa classe rappresenta la View.
"""

import logging
import os

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QPoint, QSize, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QIcon
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout,
    QSlider, QListWidget, QListWidgetItem, QTextEdit, QFrame
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
            # Aggiorna direttamente la dimensione dell'anteprima senza usare lo slider
            self.parent.preview_size = new_size
            self.parent.preview_list.setIconSize(QSize(new_size, new_size))
            self.parent.preview_size_changed.emit(new_size)
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

        # Contenitore per i blocchi di analisi con effetto ombra e sfumatura
        analysis_container = QWidget()
        analysis_container.setObjectName("analysisContainer")
        analysis_container.setStyleSheet("""
            #analysisContainer {
                background-color: #f8f8f8;
                border-radius: 8px;
                border: 1px solid #ddd;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
        """)
        analysis_layout = QVBoxLayout(analysis_container)

        # Layout orizzontale per tutti i pannelli di analisi
        analysis_panels_layout = QHBoxLayout()

        # Pannello sinistra: immagine originale con punti di controllo
        orig_image_panel = QWidget()
        orig_image_layout = QVBoxLayout(orig_image_panel)
        orig_image_layout.setContentsMargins(0, 5, 0, 0)  # Margini minimi per avere spazio per il titolo

        # Titolo immagine originale
        orig_title_label = QLabel("<b>Immagine Originale</b>")
        orig_title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        orig_image_layout.addWidget(orig_title_label)

        self.img_label = ImageLabel()
        orig_image_layout.addWidget(self.img_label)

        analysis_panels_layout.addWidget(orig_image_panel, 2)  # Mantiene la proporzione 2

        # Linea verticale tra i pannelli
        panel_line1 = QFrame()
        panel_line1.setFrameShape(QFrame.Shape.VLine)
        panel_line1.setFrameShadow(QFrame.Shadow.Sunken)
        panel_line1.setStyleSheet("color: #aaa;")
        analysis_panels_layout.addWidget(panel_line1, 0)

        # Centro: immagine wrappata e campo commenti
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 5, 0, 0)  # Margini minimi per avere spazio per il titolo

        # Titolo immagine elaborata
        center_title_label = QLabel("<b>Immagine Elaborata</b>")
        center_title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center_layout.addWidget(center_title_label)

        # Immagine wrappata
        self.wrapped_img_label = QLabel()
        self.wrapped_img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.wrapped_img_label.setMinimumSize(300, 300)
        center_layout.addWidget(self.wrapped_img_label, 3)  # Proporzione maggiore per l'immagine

        # Campo commenti con dimensione ridotta
        comments_layout = QVBoxLayout()
        comments_label = QLabel("<b>Commenti</b>")
        comments_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        comments_layout.addWidget(comments_label)

        self.comment_text = QTextEdit()
        self.comment_text.setPlaceholderText("Aggiungi un commento...")
        self.comment_text.setMaximumHeight(100)  # Altezza massima limitata
        self.comment_text.textChanged.connect(self._on_text_comment_changed)
        comments_layout.addWidget(self.comment_text)

        center_layout.addLayout(comments_layout, 1)  # Proporzione minore per i commenti

        analysis_panels_layout.addWidget(center_widget, 1)  # Il centro ha proporzione 1

        # Linea verticale tra centro e destra
        panel_line2 = QFrame()
        panel_line2.setFrameShape(QFrame.Shape.VLine)
        panel_line2.setFrameShadow(QFrame.Shadow.Sunken)
        panel_line2.setStyleSheet("color: #aaa;")
        analysis_panels_layout.addWidget(panel_line2, 0)

        # Destra: textarea OCR scrollabile
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 5, 0, 0)  # Margini minimi per avere spazio per il titolo

        # Titolo testo OCR
        ocr_title_label = QLabel("<b>Testo OCR</b>")
        ocr_title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(ocr_title_label)

        # Campo testo OCR
        self.ocr_text = QTextEdit()
        self.ocr_text.setReadOnly(True)
        right_layout.addWidget(self.ocr_text)

        analysis_panels_layout.addWidget(right_widget, 1)  # Il pannello destro ha proporzione 1

        # Aggiungiamo il layout dei pannelli al contenitore di analisi
        analysis_layout.addLayout(analysis_panels_layout)

        # Pulsante analizza (all'interno del container di analisi)
        self.analyze_btn = QPushButton("Analizza")
        self.analyze_btn.clicked.connect(self._on_analyze_clicked)
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        analyze_btn_layout = QHBoxLayout()
        analyze_btn_layout.addStretch(1)
        analyze_btn_layout.addWidget(self.analyze_btn)
        analyze_btn_layout.addStretch(1)
        analysis_layout.addLayout(analyze_btn_layout)

        # Aggiungiamo il contenitore di analisi al layout principale
        main_layout.addWidget(analysis_container, 5)  # Proporzione maggiore per tutto il blocco di analisi

        # Sotto: barra navigazione solo con "Analizza Tutto" (all'esterno del contenitore)
        nav_layout = QHBoxLayout()
        self.analyze_all_btn = QPushButton("Analizza Tutto")
        self.analyze_all_btn.clicked.connect(self._on_analyze_all_clicked)
        self.analyze_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        nav_layout.addStretch(1)
        nav_layout.addWidget(self.analyze_all_btn)

        # Layout finale
        layout = QVBoxLayout()
        layout.addLayout(main_layout)
        layout.addLayout(nav_layout)
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)
        self.resize(1400, 700)

    def _setup_preview_panel(self):
        """Configura il pannello sinistro con le anteprime delle immagini."""
        preview_layout = QVBoxLayout()

        # Titolo della colonna di anteprima
        preview_title_label = QLabel("<b>Elenco File</b>")
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

        # Leggenda
        legend_label = QLabel(
            "<b>Leggenda:</b><br><span style='color: #00A5FF;'>Punto 1 (azzurro)</span><br><span style='color: #FF3800;'>Punti 2-4 (rossi)</span><br>• Trascina i punti col mouse<br>• Click destro per reset<br>• Zoom con Ctrl+rotellina")
        legend_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        legend_label.setWordWrap(False)
        legend_label.setStyleSheet("background-color: #f0f0f0; padding: 8px; border-radius: 3px; min-width: 170px;")
        preview_layout.addWidget(legend_label)

        return preview_layout

    def _setup_center_panel(self):
        """Configura il pannello centrale con immagine wrappata e commenti."""
        center_layout = QVBoxLayout()

        # Immagine wrappata
        self.wrapped_img_label = QLabel()
        self.wrapped_img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.wrapped_img_label.setMinimumSize(300, 300)
        center_layout.addWidget(self.wrapped_img_label, 3)  # Proporzione maggiore per l'immagine

        # Campo commenti con dimensione ridotta
        comments_layout = QVBoxLayout()
        comments_label = QLabel("<b>Commenti</b>")
        comments_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        comments_layout.addWidget(comments_label)

        self.comment_text = QTextEdit()
        self.comment_text.setPlaceholderText("Aggiungi un commento...")
        self.comment_text.setMaximumHeight(100)  # Altezza massima limitata
        self.comment_text.textChanged.connect(self._on_text_comment_changed)
        comments_layout.addWidget(self.comment_text)

        center_layout.addLayout(comments_layout, 1)  # Proporzione minore per i commenti

        return center_layout

    def _setup_right_panel(self):
        """Configura il pannello destro con il testo OCR scrollabile."""
        right_layout = QVBoxLayout()

        # Campo testo OCR
        self.ocr_text = QTextEdit()
        self.ocr_text.setReadOnly(True)
        right_layout.addWidget(self.ocr_text)

        return right_layout

    def _on_preview_selected(self, index):
        """Gestisce la selezione di un'anteprima."""
        self.preview_selected.emit(index)

    def _on_analyze_clicked(self):
        """Gestisce il click sul pulsante 'Analizza'."""
        self.analyze_clicked.emit()

    def _on_analyze_all_clicked(self):
        """Gestisce il click sul pulsante 'Analizza Tutto'."""
        self.analyze_all_clicked.emit()

    def _on_text_comment_changed(self):
        """Gestisce la modifica del testo nei commenti."""
        self.text_comment_changed.emit(self.comment_text.toPlainText())

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
        self.comment_text.blockSignals(True)
        self.comment_text.setText(text)
        self.comment_text.blockSignals(False)

    def update_preview_icon(self, idx, pixmap):
        """Aggiorna l'icona di anteprima per un indice specifico."""
        if 0 <= idx < len(self.preview_items):
            self.preview_items[idx].setIcon(QIcon(pixmap))
            self.preview_list.update()

    def get_current_points(self):
        """Ottiene i punti di controllo correnti dall'ImageLabel."""
        return self.img_label.get_points()




