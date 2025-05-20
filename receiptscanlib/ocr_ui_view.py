"""
Modulo che contiene la classe OcrUiView per la visualizzazione dell'interfaccia utente dell'applicazione OCR.
Segue il pattern MVC (Model-View-Controller) dove questa classe rappresenta la View.
"""

import logging
import os

import cv2
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QIcon
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout,
    QListWidget, QListWidgetItem, QTextEdit, QFrame, QSpacerItem, QSizePolicy
)

# Importiamo le classi dai rispettivi moduli nella directory ui
from receiptscanlib.ui.image_label import ImageLabel
from receiptscanlib.ui.preview_widget import PreviewWidget
from receiptscanlib.ui.transaction_details_widget import TransactionDetailsWidget

# Configurazione del logger per questo modulo
logger = logging.getLogger(__name__)


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
            }
        """)
        analysis_layout = QVBoxLayout(analysis_container)

        # Layout orizzontale per tutti i pannelli di analisi
        analysis_panels_layout = QHBoxLayout()
        analysis_panels_layout.setSpacing(0)  # Elimina lo spazio tra i pannelli, lasciando solo le linee

        # Creiamo widget con larghezza fissa per ogni pannello
        column_width = 400  # Larghezza fissa per ogni colonna

        # Pannello sinistra: immagine originale con punti di controllo
        orig_image_panel = QWidget()
        orig_image_panel.setFixedWidth(column_width)  # Imposta larghezza fissa
        orig_image_layout = QVBoxLayout(orig_image_panel)
        orig_image_layout.setContentsMargins(0, 5, 0, 0)  # Margini minimi per avere spazio per il titolo

        # Titolo immagine originale più grande
        orig_title_label = QLabel("<b>Immagine Originale</b>")
        orig_title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        orig_title_label.setStyleSheet("font-size: 14px;")
        orig_image_layout.addWidget(orig_title_label)

        # Spaziatore verticale
        vspacer = QSpacerItem(20, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        orig_image_layout.addItem(vspacer)

        # Immagine originale con altezza massima
        self.img_label = ImageLabel()
        self.img_label.setMinimumHeight(550)  # Aumenta l'altezza minima
        orig_image_layout.addWidget(self.img_label, 1)  # Stretching verticale massimo

        analysis_panels_layout.addWidget(orig_image_panel)  # Rimuovi la proporzione

        # Linea verticale tra i pannelli con margini
        separator_container1 = QWidget()
        separator_layout1 = QHBoxLayout(separator_container1)
        separator_layout1.setContentsMargins(8, 0, 8, 0)  # Aggiunge margini a destra e sinistra

        panel_line1 = QFrame()
        panel_line1.setFrameShape(QFrame.Shape.VLine)
        panel_line1.setFrameShadow(QFrame.Shadow.Sunken)
        panel_line1.setStyleSheet("color: #aaa;")
        separator_layout1.addWidget(panel_line1)

        analysis_panels_layout.addWidget(separator_container1, 0)

        # Centro: immagine wrappata e campo commenti
        center_widget = QWidget()
        center_widget.setFixedWidth(column_width)  # Imposta larghezza fissa
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 5, 0, 0)  # Margini minimi per avere spazio per il titolo

        # Titolo immagine elaborata più grande
        center_title_label = QLabel("<b>Immagine Elaborata</b>")
        center_title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center_title_label.setStyleSheet("font-size: 14px;")
        center_layout.addWidget(center_title_label)

        # Spaziatore verticale
        vspacer2 = QSpacerItem(20, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        center_layout.addItem(vspacer2)

        # Immagine wrappata
        self.wrapped_img_label = QLabel()
        self.wrapped_img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.wrapped_img_label.setMinimumSize(300, 300)
        center_layout.addWidget(self.wrapped_img_label, 3)  # Proporzione maggiore per l'immagine

        # Rimuoviamo la linea orizzontale visto che ora abbiamo il rettangolo dei commenti
        # che funge già da separatore visivo

        # Campo commenti con dimensione ridotta in un rettangolo
        comments_container = QWidget()
        comments_container.setObjectName("commentsContainer")
        comments_container.setStyleSheet("""
            #commentsContainer {
                background-color: #f0f0f0;
                border-radius: 6px;
                border: 1px solid #ddd;
                margin-top: 8px;
            }
        """)
        comments_layout = QVBoxLayout(comments_container)
        comments_layout.setContentsMargins(10, 10, 10, 10)

        comments_label = QLabel("<b>Commenti</b>")
        comments_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        comments_label.setStyleSheet("font-size: 13px;")
        comments_layout.addWidget(comments_label)

        self.comment_text = QTextEdit()
        self.comment_text.setPlaceholderText("Aggiungi un commento...")
        self.comment_text.setMaximumHeight(100)  # Altezza massima limitata
        self.comment_text.textChanged.connect(self._on_text_comment_changed)
        comments_layout.addWidget(self.comment_text)

        center_layout.addWidget(comments_container, 1)  # Proporzione minore per i commenti

        analysis_panels_layout.addWidget(center_widget, 1)  # Il centro ha proporzione 1

        # Linea verticale tra centro e destra con margini
        separator_container2 = QWidget()
        separator_layout2 = QHBoxLayout(separator_container2)
        separator_layout2.setContentsMargins(8, 0, 8, 0)  # Aggiunge margini a destra e sinistra

        panel_line2 = QFrame()
        panel_line2.setFrameShape(QFrame.Shape.VLine)
        panel_line2.setFrameShadow(QFrame.Shadow.Sunken)
        panel_line2.setStyleSheet("color: #aaa;")
        separator_layout2.addWidget(panel_line2)

        analysis_panels_layout.addWidget(separator_container2, 0)

        # Destra: textarea OCR scrollabile
        right_widget = QWidget()
        right_widget.setFixedWidth(column_width)  # Imposta larghezza fissa
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 5, 0, 0)  # Margini minimi per avere spazio per il titolo

        # Titolo testo OCR
        ocr_title_label = QLabel("<b>Testo OCR</b>")
        ocr_title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ocr_title_label.setStyleSheet("font-size: 14px;")
        right_layout.addWidget(ocr_title_label)

        # Spaziatore verticale
        vspacer3 = QSpacerItem(20, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        right_layout.addItem(vspacer3)

        # Campo testo OCR
        self.ocr_text = QTextEdit()
        self.ocr_text.setReadOnly(True)
        right_layout.addWidget(self.ocr_text)

        # Dettagli transazione in un rettangolo sotto il testo OCR
        details_container = QWidget()
        details_container.setObjectName("detailsContainer")
        details_container.setStyleSheet("""
            #detailsContainer {
                background-color: #f0f0f0;
                border-radius: 6px;
                border: 1px solid #ddd;
                margin-top: 12px;
            }
        """)
        details_layout = QVBoxLayout(details_container)
        details_layout.setContentsMargins(10, 10, 10, 10)
        self.transaction_details = TransactionDetailsWidget()
        details_layout.addWidget(self.transaction_details)
        right_layout.addWidget(details_container)

        analysis_panels_layout.addWidget(right_widget)  # Rimuovi la proporzione

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

    # Metodi pubblici per leggere/scrivere i dettagli transazione
    def get_transaction_details(self):
        return self.transaction_details.get_details()

    def set_transaction_details(self, details):
        self.transaction_details.set_details(details)

