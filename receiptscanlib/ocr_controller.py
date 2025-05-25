"""
Modulo che contiene il controller per l'applicazione OCR secondo il pattern MVC.
Questo controller collega la View (OcrUiView) con il Model (dati e logica).
"""

import logging
import os
import threading

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QObject, Qt
from PyQt6.QtGui import QImage, QPixmap

from .image_processor import warp_image
from .ocr_analyzer import init_ocr_model, analyze_image_with_ocr
from .ocr_ui_view import OcrUiView

# Configurazione del logger per questo modulo
logger = logging.getLogger(__name__)


class OcrWorker(QThread):
    """Worker thread per eseguire l'analisi OCR in background."""

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


class ModelInitWorker(QThread):
    """Worker thread per l'inizializzazione del modello OCR in background."""

    finished = pyqtSignal(bool)  # success

    def run(self):
        # Inizializza il modello OCR in background
        logger.info("Inizializzazione del modello OCR in background...")
        success = init_ocr_model()
        logger.info(f"Inizializzazione del modello OCR completata con risultato: {success}")
        # Emetti il segnale con il risultato
        self.finished.emit(success)


class OcrAppController(QObject):
    """
    Controller per l'applicazione OCR secondo il pattern MVC.
    Gestisce la logica dell'applicazione e collega la View con i dati.
    """

    def __init__(self, image_dir):
        """
        Inizializza il controller dell'applicazione.

        Args:
            image_dir: Directory contenente le immagini
        """
        super().__init__()
        self.view = OcrUiView()

        # Dati del modello
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        self.image_files.sort()
        self.current_idx = 0
        self.perimeters = {}  # filename -> 4 points
        self.processing = {}  # filename -> bool (in analisi)
        self.wrapped_images = {}  # filename -> QPixmap of wrapped image
        self.ocr_results = {}  # filename -> str (risultato OCR)
        self.user_comments = {}  # filename -> str (commenti utente)

        # Flag per l'inizializzazione del modello
        self.model_initialized = False

        # Collega i segnali della view alle funzioni del controller
        self._connect_signals()

        # Configura la view con i dati iniziali
        self._setup_view()

        # Disabilita i pulsanti di analisi fino a quando il modello non è inizializzato
        self.view.set_analyze_buttons_enabled(False)

        # Inizializza il modello OCR in un thread separato
        self.init_thread = ModelInitWorker()
        self.init_thread.finished.connect(self.on_model_initialized)
        self.init_thread.start()

    def _connect_signals(self):
        """Collega i segnali della view alle funzioni del controller."""
        self.view.preview_selected.connect(self.on_preview_selected)
        self.view.preview_size_changed.connect(self.update_preview_size)
        self.view.analyze_clicked.connect(self.start_analyze)
        self.view.analyze_all_clicked.connect(self.start_analyze_all)
        self.view.text_comment_changed.connect(self.save_current_comment)

    def on_model_initialized(self, success):
        """Callback chiamato quando l'inizializzazione del modello OCR è completata."""
        self.model_initialized = success
        if success:
            logger.info("Modello OCR inizializzato con successo. Pulsanti di analisi abilitati.")
            self.view.set_analyze_buttons_enabled(True)
            self.view.set_status_message("Modello AI pronto. È possibile eseguire l'analisi.")
        else:
            logger.error("Impossibile inizializzare il modello OCR. Le funzionalità OCR non saranno disponibili.")
            self.view.set_status_message("Errore: modello AI non disponibile.")

    def _setup_view(self):
        """Configura la view con i dati iniziali."""
        self.view.set_image_files(self.image_dir, self.image_files)
        self.view.set_status_message("Caricamento modello AI in corso...")
        self.load_image()  # Carica la prima immagine

    def show(self):
        """Mostra la view."""
        self.view.show()

    def load_image(self):
        """Carica l'immagine corrente e aggiorna la view."""
        if not self.image_files:
            logger.warning("Nessuna immagine trovata.")
            return

        fname = self.image_files[self.current_idx]
        img_path = os.path.join(self.image_dir, fname)
        img = cv2.imread(img_path)

        if img is None:
            logger.error(f"Impossibile leggere l'immagine {img_path}")
            return

        points = self.perimeters.get(fname)
        if points is None:
            h_orig, w_orig = img.shape[:2]
            points = [[0, 0], [w_orig - 1, 0], [w_orig - 1, h_orig - 1], [0, h_orig - 1]]
            self.perimeters[fname] = points

        # Aggiorna la view con i dati correnti
        self.view.set_current_image(img, points, fname)
        self.view.set_current_index(self.current_idx)
        self.update_wrapped_image_display()

        # Carica i commenti salvati per l'immagine corrente
        self.view.set_comment_text(self.user_comments.get(fname, ""))

        # Aggiorna il testo OCR
        if fname in self.ocr_results:
            self.view.set_ocr_text(self.ocr_results[fname])
        else:
            self.view.set_ocr_text("Nessun risultato per questo file o analisi non eseguita")

        # Aggiorna lo stato dei pulsanti di analisi in base allo stato del modello
        self.view.set_analyze_buttons_enabled(self.model_initialized and not self.processing.get(fname, False))

    def update_preview_size(self, value):
        """Aggiorna la dimensione delle anteprime."""
        # La view gestisce già l'aggiornamento visivo
        for idx, fname in enumerate(self.image_files):
            img_path = os.path.join(self.image_dir, fname)
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                qimg = QImage(img.data, w, h, img.strides[0], QImage.Format.Format_BGR888)
                pix = QPixmap.fromImage(qimg).scaled(value, value,
                                                     Qt.AspectRatioMode.KeepAspectRatio,
                                                     Qt.TransformationMode.SmoothTransformation)
                self.view.update_preview_icon(idx, pix)

    def update_wrapped_image_display(self, fname=None):
        """Aggiorna l'immagine wrappata nella view."""
        if fname is None:
            fname = self.image_files[self.current_idx]

        if fname in self.wrapped_images:
            self.view.set_wrapped_image(self.wrapped_images[fname])
        else:
            self.view.set_wrapped_image(None)

    def set_processing(self, fname, processing=True):
        """Imposta lo stato di elaborazione per un file e aggiorna la view."""
        self.processing[fname] = processing
        idx = self.image_files.index(fname)

        img_path = os.path.join(self.image_dir, fname)
        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
            qimg = QImage(img.data, w, h, img.strides[0], QImage.Format.Format_BGR888)
            pix = QPixmap.fromImage(qimg).scaled(self.view.preview_size, self.view.preview_size,
                                                 Qt.AspectRatioMode.KeepAspectRatio,
                                                 Qt.TransformationMode.SmoothTransformation)
            self.view.update_preview_icon(idx, pix)

    def save_current_perimeter(self):
        """Salva il perimetro corrente."""
        fname = self.image_files[self.current_idx]
        current_points = self.view.get_current_points().tolist()
        logger.debug(f"save_current_perimeter for {fname} (current_idx: {self.current_idx}): {current_points}")
        self.perimeters[fname] = current_points

    def save_current_comment(self, text=None):
        """Salva il commento corrente."""
        fname = self.image_files[self.current_idx]
        if text is not None:
            self.user_comments[fname] = text
        else:
            # Se non viene passato un testo, ottienilo dalla view
            self.user_comments[fname] = self.view.comment_text.toPlainText()
        logger.debug(f"Comment saved for {fname}: {self.user_comments[fname]}")

    def on_preview_selected(self, row):
        """Gestisce la selezione di un'anteprima dalla lista."""
        if row < 0 or row >= len(self.image_files):
            return

        self.save_current_perimeter()
        self.save_current_comment()
        self.current_idx = row
        self.load_image()

    def ocr_completed(self, fname, idx, ocr_text_result):
        """Callback chiamato quando l'analisi OCR è completata."""
        self.ocr_results[fname] = ocr_text_result
        self._update_ocr_text(fname, idx, ocr_text_result)
        self.set_processing(fname, False)
        logger.info(f"Analisi OCR per {fname} completata e interfaccia aggiornata.")

    def start_analyze(self, idx=None):
        """Avvia l'analisi OCR per un'immagine specifica o quella corrente."""
        actual_idx_for_processing = idx if idx is not None else self.current_idx

        if actual_idx_for_processing == self.current_idx:
            # Se stiamo processando l'immagine corrente, salva il perimetro
            self.save_current_perimeter()

        fname = self.image_files[actual_idx_for_processing]
        self.set_processing(fname, True)

        # Ottieni le coordinate
        coords = self.perimeters.get(fname)
        logger.debug(f"Coords for {fname} before None check: {coords}")

        if coords is None:
            # Inizializza le coordinate se non esistono
            img_path_for_coords = os.path.join(self.image_dir, fname)
            img_for_coords = cv2.imread(img_path_for_coords)
            if img_for_coords is not None:
                h, w = img_for_coords.shape[:2]
                coords = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
                self.perimeters[fname] = coords
                logger.debug(f"Coords for {fname} initialized to default: {coords}")
            else:
                logger.error(f"Impossibile leggere l'immagine {fname} per inizializzare le coordinate. Analisi annullata.")
                self.set_processing(fname, False)
                if actual_idx_for_processing == self.current_idx:
                    self.view.set_ocr_text("Errore: Impossibile leggere l'immagine.")
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
                self.ocr_thread = OcrWorker(fname, actual_idx_for_processing, wrapped_img_cv.copy())
                self.ocr_thread.finished.connect(self.ocr_completed)
                self.ocr_thread.start()
                return

            else:
                logger.error(f"Immagine wrappata per {fname} ha dimensioni non valide: {w_w}x{h_w}")
                self.wrapped_images.pop(fname, None)
                ocr_text_result = "Errore: Immagine wrappata non valida."
        else:
            logger.debug(f"warp_image for {fname} returned None.")
            self.wrapped_images.pop(fname, None)
            ocr_text_result = "Errore: warp_image ha fallito."

        self.ocr_results[fname] = ocr_text_result
        self._update_ocr_text(fname, actual_idx_for_processing, ocr_text_result)
        self.set_processing(fname, False)

    def _update_image_display(self, fname_processed, processed_idx):
        """Aggiorna il display dell'immagine wrappata."""
        logger.debug(f"_update_image_display per {fname_processed}, processed_idx: {processed_idx}, current_idx: {self.current_idx}")
        if processed_idx == self.current_idx:
            self.update_wrapped_image_display(fname_processed)
        self.set_processing(fname_processed, False)

    def _update_ocr_text(self, fname_processed, processed_idx, ocr_text_to_display):
        """Aggiorna il testo OCR nella view."""
        logger.debug(f"_update_ocr_text per {fname_processed}, processed_idx: {processed_idx}, current_idx: {self.current_idx}")
        if processed_idx == self.current_idx:
            self.view.set_ocr_text(ocr_text_to_display)

    def start_analyze_all(self):
        """Avvia l'analisi OCR per tutte le immagini."""
        if not self.image_files:
            logger.info("Nessun file da analizzare in start_analyze_all.")
            return

        # Salva eventuali modifiche al perimetro dell'immagine attuale
        if self.current_idx < len(self.image_files):
            current_fname_before_all = self.image_files[self.current_idx]
            logger.debug(f"Prima di 'Analyze All', salvataggio perimetro per l'immagine corrente: {current_fname_before_all} (idx: {self.current_idx})")
            self.save_current_perimeter()
        else:
            logger.warning("current_idx non valido prima di start_analyze_all, impossibile salvare il perimetro corrente.")

        # Analizza tutte le immagini
        for i in range(len(self.image_files)):
            fname_to_analyze = self.image_files[i]
            logger.debug(f"start_analyze_all: Inizio analisi per l'immagine {i}: {fname_to_analyze}")
            self.start_analyze(idx=i)

        logger.info("Analisi di tutti i file completata.")

        # Aggiorna il display per l'immagine corrente
        if self.current_idx < len(self.image_files):
            current_display_fname = self.image_files[self.current_idx]
            logger.debug(f"Dopo 'Analyze All', aggiornamento display per l'immagine corrente: {current_display_fname} (idx: {self.current_idx})")

            self.update_wrapped_image_display(current_display_fname)

            if current_display_fname in self.ocr_results:
                self.view.set_ocr_text(self.ocr_results[current_display_fname])
            else:
                self.view.set_ocr_text("Risultato non disponibile o analisi non eseguita per l'immagine corrente")
        else:
            logger.warning("current_idx non valido dopo start_analyze_all, impossibile aggiornare il display.")
