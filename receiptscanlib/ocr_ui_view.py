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
    QListWidget, QListWidgetItem, QTextEdit, QFrame, QSpacerItem, QSizePolicy,
    QStatusBar
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

        # Imposta l'icona della finestra
        self.resize(1500, 800)

        self.init_ui()
        self.apply_light_theme()

    def init_ui(self):
        """Inizializza l'interfaccia utente."""
        main_widget = QWidget()
        main_layout = QVBoxLayout()  # Layout principale verticale per contenere tutto

        # Layout orizzontale per contenere i pannelli principali
        horizontal_layout = QHBoxLayout()

        # Preview verticale immagini
        preview_layout = self._setup_preview_panel()
        self.preview_widget = PreviewWidget(self)
        self.preview_widget.setLayout(preview_layout)
        horizontal_layout.addWidget(self.preview_widget, 0)

        # Contenitore per i blocchi di analisi con effetto ombra e sfumatura
        analysis_container = QWidget()
        analysis_container.setObjectName("analysisContainer")
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
        separator_container1.setFixedWidth(20)  # Larghezza fissa per il contenitore della linea
        separator_layout1 = QVBoxLayout(separator_container1)
        separator_layout1.setContentsMargins(9, 0, 9, 0)  # Margini per centrare la linea

        separator_line1 = QFrame()
        separator_line1.setFrameShape(QFrame.Shape.VLine)
        separator_line1.setFrameShadow(QFrame.Shadow.Sunken)
        separator_layout1.addWidget(separator_line1)

        analysis_panels_layout.addWidget(separator_container1)

        # Pannello centrale: immagine elaborata e commento
        wrapped_image_panel = QWidget()
        wrapped_image_panel.setFixedWidth(column_width)  # Imposta larghezza fissa
        wrapped_image_layout = QVBoxLayout(wrapped_image_panel)
        wrapped_image_layout.setContentsMargins(0, 5, 0, 0)  # Margini minimi

        # Titolo immagine elaborata
        wrapped_title_label = QLabel("<b>Immagine Elaborata</b>")
        wrapped_title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        wrapped_title_label.setStyleSheet("font-size: 14px;")
        wrapped_image_layout.addWidget(wrapped_title_label)

        # Spaziatore verticale
        wrapped_image_layout.addItem(QSpacerItem(20, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))

        # Label per visualizzare l'immagine elaborata (circa 75% dell'altezza)
        self.wrapped_image_label = QLabel()
        self.wrapped_image_label.setMinimumSize(300, 400)
        self.wrapped_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.wrapped_image_label.setStyleSheet("""
            border: 1px solid #ddd;
            border-radius: 4px;
            background-color: #f8f8f8;
        """)
        wrapped_image_layout.addWidget(self.wrapped_image_label, 3)  # Proporzione 3 per l'immagine (75%)

        # Label per il commento
        comment_label = QLabel("Commento:")
        comment_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        wrapped_image_layout.addWidget(comment_label)

        # TextEdit per il commento (circa 25% dell'altezza)
        self.comment_text = QTextEdit()
        self.comment_text.textChanged.connect(self._on_comment_changed)
        self.comment_text.setMaximumHeight(150)  # Limita l'altezza massima
        wrapped_image_layout.addWidget(self.comment_text, 1)  # Proporzione 1 per il commento (25%)

        analysis_panels_layout.addWidget(wrapped_image_panel)

        # Linea verticale tra i pannelli con margini
        separator_container2 = QWidget()
        separator_container2.setFixedWidth(20)  # Larghezza fissa per il contenitore della linea
        separator_layout2 = QVBoxLayout(separator_container2)
        separator_layout2.setContentsMargins(9, 0, 9, 0)  # Margini per centrare la linea

        separator_line2 = QFrame()
        separator_line2.setFrameShape(QFrame.Shape.VLine)
        separator_line2.setFrameShadow(QFrame.Shadow.Sunken)
        separator_layout2.addWidget(separator_line2)

        analysis_panels_layout.addWidget(separator_container2)

        # Pannello destra: testo OCR e dettagli della transazione (invertiti come richiesto)
        transaction_panel = QWidget()
        transaction_panel.setFixedWidth(column_width)  # Imposta larghezza fissa
        transaction_layout = QVBoxLayout(transaction_panel)
        transaction_layout.setContentsMargins(0, 5, 0, 0)  # Margini minimi
        transaction_layout.setSpacing(8)  # Spaziatura ridotta tra gli elementi

        # Titolo dettagli transazione
        ocr_title_label = QLabel("<b>Testo OCR e Dettagli</b>")
        ocr_title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ocr_title_label.setStyleSheet("font-size: 14px;")
        transaction_layout.addWidget(ocr_title_label)

        # Label per il testo OCR
        ocr_label = QLabel("Testo OCR Riconosciuto:")
        ocr_label.setStyleSheet("font-weight: bold; margin-top: 5px;")
        transaction_layout.addWidget(ocr_label)

        # TextEdit per mostrare il testo OCR con più spazio
        self.ocr_text = QTextEdit()
        self.ocr_text.setReadOnly(True)
        self.ocr_text.setMinimumHeight(350)  # Aumentiamo l'altezza minima
        transaction_layout.addWidget(self.ocr_text, 1)  # Proporzione 1 con stretching

        # Aggiungiamo uno stretcher per spingere il TransactionDetailsWidget verso il basso
        transaction_layout.addStretch(1)  # Corretto da 0.5 a 1 - addStretch accetta solo valori interi

        # Label per i dettagli della transazione
        details_label = QLabel("Dettagli Transazione:")
        details_label.setStyleSheet("font-weight: bold; margin-top: 5px;")
        transaction_layout.addWidget(details_label)

        # Widget per i dettagli della transazione
        self.transaction_details = TransactionDetailsWidget()
        self.transaction_details.setMaximumHeight(200)  # Limitiamo l'altezza massima
        transaction_layout.addWidget(self.transaction_details)  # Nessuna proporzione, usa l'altezza preferita

        analysis_panels_layout.addWidget(transaction_panel)

        # Aggiungi i pannelli al container principale
        analysis_layout.addLayout(analysis_panels_layout)

        # Pulsante per analizzare l'immagine corrente (dentro il container)
        self.analyze_button = QPushButton("Analizza Immagine Corrente")
        self.analyze_button.setMinimumHeight(40)
        self.analyze_button.clicked.connect(lambda: self.analyze_clicked.emit())
        analysis_layout.addWidget(self.analyze_button, 0, Qt.AlignmentFlag.AlignRight)
        analysis_layout.setContentsMargins(10, 10, 10, 10)

        # Aggiungiamo il container all'interfaccia principale
        horizontal_layout.addWidget(analysis_container, 1)  # Proporzione 1 per il container di analisi
        main_layout.addLayout(horizontal_layout, 1)  # Il layout orizzontale prende la maggior parte dello spazio

        # Pulsante per analizzare tutte le immagini (fuori dal container, sotto tutto)
        self.analyze_all_button = QPushButton("Analizza Tutte le Immagini")
        self.analyze_all_button.setMinimumHeight(40)
        self.analyze_all_button.setMinimumWidth(300)
        self.analyze_all_button.clicked.connect(lambda: self.analyze_all_clicked.emit())
        main_layout.addWidget(self.analyze_all_button, 0, Qt.AlignmentFlag.AlignRight)
        main_layout.setContentsMargins(10, 10, 10, 10)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Aggiungi statusbar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

    def _setup_preview_panel(self):
        """Configura il pannello delle anteprime."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Lista delle anteprime
        self.preview_list = QListWidget()
        self.preview_list.setIconSize(QSize(self.preview_size, self.preview_size))
        self.preview_list.setMinimumWidth(self.preview_size + 20)  # Imposta la larghezza minima
        self.preview_list.currentRowChanged.connect(self._on_preview_selected)
        layout.addWidget(self.preview_list)

        return layout

    def _on_preview_selected(self, row):
        """Emette il segnale preview_selected quando un'anteprima viene selezionata."""
        self.preview_selected.emit(row)

    def _on_comment_changed(self):
        """Emette il segnale text_comment_changed quando il testo del commento viene cambiato."""
        self.text_comment_changed.emit(self.comment_text.toPlainText())

    def set_image_files(self, image_dir, image_files):
        """
        Imposta l'elenco dei file e crea le anteprime.

        Args:
            image_dir: Directory contenente le immagini
            image_files: Lista dei nomi dei file
        """
        self.preview_list.clear()
        self.preview_items = []

        for idx, fname in enumerate(image_files):
            img_path = os.path.join(image_dir, fname)
            img = cv2.imread(img_path)

            if img is not None:
                h, w = img.shape[:2]
                qimg = QImage(img.data, w, h, img.strides[0], QImage.Format.Format_BGR888)
                pixmap = QPixmap.fromImage(qimg)

                # Ridimensiona la pixmap per l'anteprima
                icon_pixmap = pixmap.scaled(self.preview_size, self.preview_size,
                                           Qt.AspectRatioMode.KeepAspectRatio,
                                           Qt.TransformationMode.SmoothTransformation)

                # Crea l'item con l'icona
                item = QListWidgetItem(QIcon(icon_pixmap), f"{idx+1}: {fname}")
                self.preview_list.addItem(item)
                self.preview_items.append(item)

    def update_preview_icon(self, idx, pixmap):
        """
        Aggiorna l'icona di un'anteprima.

        Args:
            idx: Indice dell'anteprima da aggiornare
            pixmap: Nuova pixmap per l'icona
        """
        if 0 <= idx < len(self.preview_items):
            self.preview_items[idx].setIcon(QIcon(pixmap))

    def set_current_image(self, img, points, fname):
        """
        Imposta l'immagine corrente e i punti di controllo.

        Args:
            img: Immagine OpenCV
            points: Punti di controllo per il warping
            fname: Nome del file
        """
        self.img_label.set_image(img, points)
        self.setWindowTitle(f"OCR Receipt Annotator - {fname}")

    def set_current_index(self, idx):
        """
        Imposta l'indice corrente nella lista delle anteprime.

        Args:
            idx: Indice da selezionare
        """
        if 0 <= idx < self.preview_list.count():
            self.preview_list.setCurrentRow(idx)

    def set_wrapped_image(self, pixmap):
        """
        Imposta l'immagine elaborata.

        Args:
            pixmap: QPixmap dell'immagine elaborata o None
        """
        if pixmap is not None:
            # Ridimensiona la pixmap per adattarla alla label mantenendo le proporzioni
            scaled_pixmap = pixmap.scaled(self.wrapped_image_label.width(),
                                         self.wrapped_image_label.height(),
                                         Qt.AspectRatioMode.KeepAspectRatio,
                                         Qt.TransformationMode.SmoothTransformation)
            self.wrapped_image_label.setPixmap(scaled_pixmap)
        else:
            self.wrapped_image_label.clear()
            self.wrapped_image_label.setText("Nessuna immagine elaborata disponibile")

    def set_ocr_text(self, text):
        """
        Imposta il testo OCR.

        Args:
            text: Testo OCR da visualizzare
        """
        self.ocr_text.setPlainText(text)

    def set_comment_text(self, text):
        """
        Imposta il testo del commento.

        Args:
            text: Testo del commento da visualizzare
        """
        self.comment_text.setPlainText(text)

    def get_current_points(self):
        """Restituisce i punti di controllo correnti."""
        return self.img_label.get_points()

    def set_analyze_buttons_enabled(self, enabled):
        """
        Imposta lo stato di abilitazione dei pulsanti di analisi.

        Args:
            enabled: Se True, i pulsanti saranno abilitati, altrimenti disabilitati
        """
        self.analyze_button.setEnabled(enabled)
        self.analyze_all_button.setEnabled(enabled)

    def set_status_message(self, message, timeout=0):
        """
        Imposta un messaggio nella barra di stato.

        Args:
            message: Il messaggio da visualizzare
            timeout: Timeout in ms (0 = nessun timeout)
        """
        self.statusBar.showMessage(message, timeout)

    def apply_light_theme(self):
        """Applica un tema chiaro all'interfaccia."""
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #F8F9FA;
                color: #212529;
            }
            
            QLabel {
                color: #212529;
            }
            
            QPushButton {
                background-color: #4285F4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            
            QPushButton:hover {
                background-color: #3367D6;
            }
            
            QPushButton:disabled {
                background-color: #B0B0B0;
                color: #F0F0F0;
            }
            
            QTextEdit {
                background-color: white;
                border: 1px solid #DFE1E5;
                border-radius: 4px;
                padding: 4px;
            }
            
            QListWidget {
                background-color: white;
                border: 1px solid #DFE1E5;
                border-radius: 4px;
                padding: 4px;
            }
            
            QListWidget::item:selected {
                background-color: #E8F0FE;
                color: #1967D2;
            }
            
            QFrame[frameShape="4"], QFrame[frameShape="5"] {
                color: #DFE1E5;
            }
            
            #analysisContainer {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #DFE1E5;
            }
        """)
