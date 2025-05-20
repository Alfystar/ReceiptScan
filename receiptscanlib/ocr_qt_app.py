import argparse
import os
import signal
import sys
import logging  # Aggiunto logging

import cv2
import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtCore import Qt, QPoint, QSize
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QIcon
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QSlider, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import QThread, pyqtSignal

# Modificata l'importazione per essere relativa al package
from .image_processor import warp_image
from .ocr_analyzer import init_ocr_model, analyze_image_with_ocr  # Import OCR functions

# Configurazione del logger per questo modulo
logger = logging.getLogger(__name__)


class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.image = None
        self.points = []
        self.dragging_idx = None
        self.radius_dot_draw = 3
        self.radius_select_area = 30
        self.sapphire = QColor(255, 56, 0)
        self.orange = QColor(0, 165, 255)
        self.setMinimumSize(400, 400)

    def set_image(self, image, points=None):
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
        return np.array(self.points, dtype=np.float32)

    def paintEvent(self, event):
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
        self.dragging_idx = None


class MainWindow(QMainWindow):
    def __init__(self, image_dir):
        super().__init__()
        self.setWindowTitle("OCR Receipt Annotator")
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        self.image_files.sort()
        self.current_idx = 0  # Indice dell'immagine corrente
        self.perimeters = {}  # filename -> 4 points
        self.preview_size = 60  # dimensione preview regolabile
        self.processing = {}  # filename -> bool (in analisi)
        self.wrapped_images = {}  # filename -> QPixmap of wrapped image
        self.ocr_results = {}  # filename -> str (risultato OCR)

        # Inizializza il modello OCR
        if not init_ocr_model():
            logger.error("Impossibile inizializzare il modello OCR. Le funzionalità OCR non saranno disponibili.")

        self.init_ui()
        self.load_image()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        # Preview verticale immagini
        class PreviewWidget(QWidget):
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

        self.preview_list = QListWidget()
        self.preview_list.setMaximumWidth(180)
        self.preview_list.setIconSize(QSize(self.preview_size, self.preview_size))
        self.preview_list.setMinimumWidth(140)
        self.preview_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.preview_list.setSpacing(8)
        self.preview_list.setStyleSheet("QListWidget::item { margin-bottom: 8px; }")
        self.preview_items = []
        for fname in self.image_files:
            img_path = os.path.join(self.image_dir, fname)
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                qimg = QImage(img.data, w, h, img.strides[0], QImage.Format.Format_BGR888)
                pix = QPixmap.fromImage(qimg).scaled(self.preview_size, self.preview_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                item = QListWidgetItem(QIcon(pix), fname)
                self.preview_list.addItem(item)
                self.preview_items.append(item)
        self.preview_list.currentRowChanged.connect(self.on_preview_selected)
        # Slider per regolare la dimensione delle preview
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setMinimum(32)
        self.size_slider.setMaximum(120)
        self.size_slider.setValue(self.preview_size)
        self.size_slider.valueChanged.connect(self.update_preview_size)
        preview_layout = QVBoxLayout()
        # Aggiungi un titolo alla colonna di anteprima
        preview_title_label = QLabel("Elenco File")
        preview_title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Allinea il testo al centro
        preview_layout.addWidget(preview_title_label)  # Aggiungi il titolo al layout
        preview_layout.addWidget(self.preview_list)
        preview_layout.addWidget(self.size_slider)
        self.preview_widget = PreviewWidget(self)
        self.preview_widget.setLayout(preview_layout)
        main_layout.addWidget(self.preview_widget, 0)
        # Sinistra: immagine
        self.img_label = ImageLabel()
        main_layout.addWidget(self.img_label, 2)
        # Centro: immagine wrappata
        self.wrapped_img_label = QLabel()  # Usiamo QLabel semplice per ora
        self.wrapped_img_label.setMinimumSize(300, 300)  # Dimensione minima
        self.wrapped_img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.wrapped_img_label.setStyleSheet("border: 1px solid gray;")  # Per visibilità
        main_layout.addWidget(self.wrapped_img_label, 1)  # Aggiunto con fattore di stretch 1
        # Destra: placeholder
        right_panel = QVBoxLayout()
        self.ocr_label = QLabel("<b>Analisi OCR/LLM</b>\n(qui verrà mostrato il risultato)")
        self.ocr_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.ocr_label.setWordWrap(True)  # Per andare a capo automaticamente
        right_panel.addWidget(self.ocr_label)
        right_panel.addStretch(1)
        main_layout.addLayout(right_panel, 1)
        # Sotto: barra navigazione (Start Analyze, Start Analyze All)
        nav_layout = QHBoxLayout()
        self.analyze_btn = QPushButton("Start Analyze")
        self.analyze_btn.clicked.connect(lambda: self.start_analyze(idx=None))  # Modificato per passare idx=None
        self.analyze_all_btn = QPushButton("Start Analyze All")
        self.analyze_all_btn.clicked.connect(self.start_analyze_all)
        nav_layout.addStretch(1)
        nav_layout.addWidget(self.analyze_btn)
        nav_layout.addWidget(self.analyze_all_btn)
        # Layout finale
        layout = QVBoxLayout()
        layout.addLayout(main_layout)
        layout.addLayout(nav_layout)
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)
        self.resize(1000, 700)

    def update_preview_size(self, value):
        self.preview_size = value
        self.preview_list.setIconSize(QSize(self.preview_size, self.preview_size))
        # Aggiorna le icone
        for idx, fname in enumerate(self.image_files):
            img_path = os.path.join(self.image_dir, fname)
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                qimg = QImage(img.data, w, h, img.strides[0], QImage.Format.Format_BGR888)
                pix = QPixmap.fromImage(qimg).scaled(self.preview_size, self.preview_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                self.preview_items[idx].setIcon(QIcon(pix))
        self.preview_list.update()

    def update_wrapped_image_display(self, fname=None):
        if fname is None:
            fname = self.image_files[self.current_idx]

        if fname in self.wrapped_images:
            pixmap = self.wrapped_images[fname]
            # Scala il pixmap per adattarlo al label mantenendo l'aspect ratio
            scaled_pixmap = pixmap.scaled(self.wrapped_img_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.wrapped_img_label.setPixmap(scaled_pixmap)
        else:
            self.wrapped_img_label.clear()
            self.wrapped_img_label.setText("Nessuna immagine elaborata")

    def set_processing(self, fname, processing=True):
        self.processing[fname] = processing
        idx = self.image_files.index(fname)
        # Rimossa la logica dello spinner
        img_path = os.path.join(self.image_dir, fname)
        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
            qimg = QImage(img.data, w, h, img.strides[0], QImage.Format.Format_BGR888)
            pix = QPixmap.fromImage(qimg).scaled(self.preview_size, self.preview_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.preview_items[idx].setIcon(QIcon(pix))
            # Assicurati che nessun widget personalizzato (come lo spinner) sia impostato
            self.preview_list.setItemWidget(self.preview_items[idx], None)
        self.preview_list.update()

    def load_image(self):
        if not self.image_files:
            self.img_label.setText("Nessuna immagine trovata")
            return
        fname = self.image_files[self.current_idx]
        img_path = os.path.join(self.image_dir, fname)
        img = cv2.imread(img_path)
        points = self.perimeters.get(fname)
        if points is None:
            img_path_for_size = os.path.join(self.image_dir, fname)
            img_for_size = cv2.imread(img_path_for_size)
            if img_for_size is not None:
                h_orig, w_orig = img_for_size.shape[:2]
                points = [[0, 0], [w_orig - 1, 0], [w_orig - 1, h_orig - 1], [0, h_orig - 1]]
                self.perimeters[fname] = points
            else:  # Fallback se l'immagine non può essere letta per le dimensioni
                points = [[0, 0], [300, 0], [300, 300], [0, 300]]  # Valori di default arbitrari
                self.perimeters[fname] = points
        self.img_label.set_image(img, points)
        self.setWindowTitle(f"OCR Receipt Annotator - {fname}")
        self.preview_list.setCurrentRow(self.current_idx)
        self.update_wrapped_image_display()  # Aggiorna display immagine wrappata

    def save_current_perimeter(self):
        fname = self.image_files[self.current_idx]
        current_points = self.img_label.get_points().tolist()
        logger.debug(f"save_current_perimeter for {fname} (current_idx: {self.current_idx}): {current_points}")
        self.perimeters[fname] = current_points

    def on_preview_selected(self, row):
        if row < 0 or row >= len(self.image_files):
            return
        self.save_current_perimeter()
        self.current_idx = row
        self.load_image()
        # Mostra il risultato OCR/LLM relativo a questo file, se presente
        fname = self.image_files[self.current_idx]
        if fname in self.ocr_results:
            self.ocr_label.setText(self.ocr_results[fname])
        else:
            self.ocr_label.setText("<b>Analisi OCR/LLM</b>\n(Nessun risultato per questo file o analisi non eseguita)")
        self.update_wrapped_image_display()  # Aggiorna display immagine wrappata


    class OcrWorker(QThread):
        finished = pyqtSignal(str, int, str)  # fname, idx, ocr_result

        def __init__(self, fname, idx, wrapped_img_cv):
            super().__init__()
            self.fname = fname
            self.idx = idx
            self.wrapped_img_cv = wrapped_img_cv

        def run(self):
            # Eseguo OCR in background
            logger.info(f"Avvio analisi OCR in background per {self.fname}...")
            ocr_text_result = analyze_image_with_ocr(self.wrapped_img_cv)
            logger.info(f"Analisi OCR per {self.fname} completata in background.")
            # Emetto il segnale con i risultati
            self.finished.emit(self.fname, self.idx, ocr_text_result)

    def ocr_completed(self, fname, idx, ocr_text_result):
        self.ocr_results[fname] = ocr_text_result  # Salva il risultato OCR
        self._update_ocr_text(fname, idx, ocr_text_result)  # Aggiorna il testo OCR
        self.set_processing(fname, False)  # Reimposta lo stato di elaborazione
        logger.info(f"Analisi OCR per {fname} completata e interfaccia aggiornata.")

    def start_analyze(self, idx=None):
        actual_idx_for_processing = idx

        if actual_idx_for_processing is None:
            # Chiamata dal pulsante "Start Analyze" (tramite lambda, idx è None)
            actual_idx_for_processing = self.current_idx
            logger.debug(f"Start Analyze button clicked for current image {actual_idx_for_processing}. Saving perimeter.")
            self.save_current_perimeter()
        else:
            # Chiamata da start_analyze_all con un indice specifico
            logger.debug(f"start_analyze called for specific index {actual_idx_for_processing}.")

        fname = self.image_files[actual_idx_for_processing]
        self.set_processing(fname, True)  # Imposta lo stato di elaborazione

        # Ottieni le coordinate. Se chiamato da bottone, save_current_perimeter le ha appena aggiornate.
        coords = self.perimeters.get(fname)
        logger.debug(f"Coords for {fname} before None check: {coords}")

        if coords is None:
            # Inizializza le coordinate se non esistono (es. prima analisi per questa immagine)
            img_path_for_coords = os.path.join(self.image_dir, fname)
            img_for_coords = cv2.imread(img_path_for_coords)
            if img_for_coords is not None:
                h, w = img_for_coords.shape[:2]
                coords = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
                self.perimeters[fname] = coords  # Salva le coordinate di default
                logger.debug(f"Coords for {fname} initialized to default: {coords}")
            else:
                logger.error(f"Impossibile leggere l'immagine {fname} per inizializzare le coordinate. Analisi annullata.")
                self.set_processing(fname, False)  # Assicura che lo stato di processing sia resettato
                if actual_idx_for_processing == self.current_idx:
                    self.ocr_label.setText("Errore: Impossibile leggere l'immagine.")
                return

        # Esegui la trasformazione prospettica
        img_path = os.path.join(self.image_dir, fname)
        logger.debug(f"Calling warp_image for {fname} with coords: {coords}")
        wrapped_img_cv = warp_image(img_path, coords)

        if wrapped_img_cv is not None:
            logger.debug(f"warp_image for {fname} returned an image.")
            # Assicura che l'array NumPy sia contiguo
            if not wrapped_img_cv.flags['C_CONTIGUOUS']:
                wrapped_img_cv = np.ascontiguousarray(wrapped_img_cv)

            h_w, w_w = wrapped_img_cv.shape[:2]
            if h_w > 0 and w_w > 0:
                q_wrapped_img = QImage(wrapped_img_cv.data, w_w, h_w, wrapped_img_cv.strides[0], QImage.Format.Format_BGR888)
                if q_wrapped_img.isNull():
                    logger.error(f"QImage creata da wrapped_img_cv per {fname} è nulla.")
                    self.wrapped_images.pop(fname, None)
                    ocr_text_result = "Errore: QImage creata da wrapped_img_cv è nulla."
                    self.ocr_results[fname] = ocr_text_result
                    self._update_ocr_text(fname, actual_idx_for_processing, ocr_text_result)
                    self.set_processing(fname, False)
                    return
                else:
                    self.wrapped_images[fname] = QPixmap.fromImage(q_wrapped_img)
                    logger.debug(f"QPixmap for {fname} created and stored.")
                self._update_image_display(fname, actual_idx_for_processing)

                # Crea e avvia il thread OCR
                self.ocr_thread = self.OcrWorker(fname, actual_idx_for_processing, wrapped_img_cv.copy())
                self.ocr_thread.finished.connect(self.ocr_completed)
                self.ocr_thread.start()
                return  # Ritorna qui, il resto avverrà nel callback

            else:
                logger.error(f"Immagine wrappata per {fname} ha dimensioni non valide: {w_w}x{h_w}")
                self.wrapped_images.pop(fname, None)
                ocr_text_result = "Errore: Immagine wrappata non valida."
        else:
            logger.debug(f"warp_image for {fname} returned None.")
            self.wrapped_images.pop(fname, None)
            ocr_text_result = "Errore: warp_image ha fallito."

        self.ocr_results[fname] = ocr_text_result  # Salva anche in caso di errore di warp
        self._update_ocr_text(fname, actual_idx_for_processing, ocr_text_result)  # Aggiorna il testo OCR
        self.set_processing(fname, False)  # Reimposta lo stato di elaborazione


    def _update_image_display(self, fname_processed, processed_idx):
        logger.debug(f"_update_image_display per {fname_processed}, processed_idx: {processed_idx}, current_idx: {self.current_idx}")
        if processed_idx == self.current_idx:
            self.update_wrapped_image_display(fname_processed)
        self.set_processing(fname_processed, False)  # Reimposta l'icona nella lista di anteprima

    def _update_ocr_text(self, fname_processed, processed_idx, ocr_text_to_display):
        logger.debug(f"_update_ocr_text per {fname_processed}, processed_idx: {processed_idx}, current_idx: {self.current_idx}")
        if processed_idx == self.current_idx:
            self.ocr_label.setText(ocr_text_to_display)

    def _finish_analysis_gui_update(self, fname_processed, processed_idx, ocr_text_to_display):
        self._update_image_display(fname_processed, processed_idx)
        self._update_ocr_text(fname_processed, processed_idx, ocr_text_to_display)
        logger.info(f"Analisi (GUI Update) completata per: {fname_processed}")

    def start_analyze_all(self):
        if not self.image_files:
            logger.info("Nessun file da analizzare in start_analyze_all.")
            return

        if not hasattr(self, 'ocr_results'):
            self.ocr_results = {}

        # 1. Salva eventuali modifiche al perimetro dell'immagine attualmente visualizzata
        #    prima di iniziare l'analisi di tutti i file.
        if self.current_idx < len(self.image_files):
            current_fname_before_all = self.image_files[self.current_idx]
            logger.debug(f"Prima di 'Analyze All', salvataggio perimetro per l'immagine corrente: {current_fname_before_all} (idx: {self.current_idx})")
            self.save_current_perimeter()
        else:
            logger.warning("current_idx non valido prima di start_analyze_all, impossibile salvare il perimetro corrente.")
            # Considera se ritornare o procedere con cautela

        for i in range(len(self.image_files)):
            fname_to_analyze = self.image_files[i]
            logger.debug(f"start_analyze_all: Inizio analisi per l'immagine {i}: {fname_to_analyze}")
            self.start_analyze(idx=i)

        logger.info("Analisi di tutti i file completata.")

        # 2. Aggiorna il display (immagine wrappata e testo OCR) per l'immagine
        #    che è correntemente selezionata dopo il ciclo di analisi.
        if self.current_idx < len(self.image_files):
            current_display_fname = self.image_files[self.current_idx]
            logger.debug(f"Dopo 'Analyze All', aggiornamento display per l'immagine corrente: {current_display_fname} (idx: {self.current_idx})")

            self.update_wrapped_image_display(current_display_fname)  # Aggiorna l'immagine wrappata

            if hasattr(self, 'ocr_results') and current_display_fname in self.ocr_results:
                self.ocr_label.setText(self.ocr_results[current_display_fname])
            else:
                # Imposta un testo di fallback se non ci sono risultati OCR
                self.ocr_label.setText("<b>Analisi OCR/LLM</b>\n(Risultato non disponibile o analisi non eseguita per l'immagine corrente)")
        else:
            logger.warning("current_idx non valido dopo start_analyze_all, impossibile aggiornare il display.")

def handle_sigint(sig, frame):
    logger.info("Terminazione richiesta dall'utente (Ctrl+C). Uscita...")
    QTimer.singleShot(0, QApplication.quit)

# Timer per forzare la gestione dei segnali anche se la finestra non è in focus
def keep_alive():
    # Timer che ogni 200ms chiama processEvents per permettere la gestione di SIGINT
    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(200)
    return timer

if __name__ == "__main__":
    # 1. Gestione degli argomenti della riga di comando
    parser = argparse.ArgumentParser(description="OCR Receipt Annotator")
    parser.add_argument('--dir', type=str, default="test_receipt", help="Directory contenente le immagini delle ricevute.")
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Imposta il livello di logging (default: INFO)'
    )
    args = parser.parse_args()

    # Configurazione del logging
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Livello di log non valido: {args.log_level}')
    logging.basicConfig(level=numeric_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 2. Gestione del segnale SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, handle_sigint)

    # 3. Creazione dell'applicazione Qt
    app = QApplication(sys.argv) # sys.argv viene passato a QApplication

    # 4. Timer per la gestione dei segnali (opzionale, ma utile per SIGINT in GUI)
    _ka_timer = keep_alive() # Assicura che il timer sia mantenuto in vita

    # 5. Creazione e visualizzazione della finestra principale
    window = MainWindow(args.dir) # Passa la directory delle immagini alla finestra
    window.show()

    # 6. Avvio del loop di eventi dell'applicazione
    sys.exit(app.exec())
