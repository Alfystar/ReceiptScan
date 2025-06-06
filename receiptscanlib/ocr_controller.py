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
from .ocr_analyzer import init_ocr_model, analyze_image_with_ocr, analyze_receipt_structured_llm, unload_ocr_model, init_llm_model, unload_llm_model
from .ocr_ui_view import OcrUiView

# Configurazione del logger per questo modulo
logger = logging.getLogger(__name__)


class OcrWorker(QThread):
    """Worker thread per eseguire l'analisi OCR in background."""

    finished = pyqtSignal(str, int, str)  # fname, idx, ocr_result

    def __init__(self, fname, idx, wrapped_img_cv, comment_user=""):
        super().__init__()
        self.fname = fname
        self.idx = idx
        self.wrapped_img_cv = wrapped_img_cv
        self.comment_user = comment_user  # Commento dell'utente, se necessario

    def run(self):
        # Eseguo OCR in background
        logger.info(f"Avvio analisi OCR in background per {self.fname}...")
        ocr_text_result, ocr_summary_dict = analyze_receipt_structured_llm(self.wrapped_img_cv, self.comment_user)
        logger.info(f"Analisi OCR per {self.fname} completata in background.")
        # Emetto il segnale con i risultati
        self.finished.emit(self.fname, self.idx, ocr_text_result)


class LlmWorker(QThread):
    """Worker thread per eseguire l'analisi LLM in background."""

    finished = pyqtSignal(str, int, dict)  # fname, idx, structured_data

    def __init__(self, fname, idx, ocr_text="", comment_user=""):
        super().__init__()
        self.fname = fname
        self.idx = idx
        self.ocr_text = ocr_text
        self.comment_user = comment_user

    def run(self):
        # Eseguo LLM in background
        logger.info(f"Avvio analisi LLM in background per {self.fname}...")
        try:
            from datetime import datetime
            structured_data = {
                "negozio": "Esempio Negozio",
                "data": datetime.now().strftime("%Y-%m-%d"),
                "importo_totale": "42.00",
                "valuta": "EUR",
                "items": [
                    {"descrizione": "Articolo di esempio", "quantita": 2, "prezzo": "21.00"}
                ]
            }
            logger.info(f"Analisi LLM per {self.fname} completata in background.")
        except Exception as e:
            logger.error(f"Errore durante l'analisi LLM: {e}")
            structured_data = {"error": str(e)}

        # Emetto il segnale con i risultati
        self.finished.emit(self.fname, self.idx, structured_data)


class ModelInitWorker(QThread):
    """Worker thread per l'inizializzazione del modello OCR in background."""

    finished = pyqtSignal(bool)  # success

    def run(self):
        # Inizializza il modello OCR in background
        logger.info("Inizializzazione del modello OCR in background...")
        try:
            logger.info("Check modelli OCR...")
            init_ocr_model()
            unload_ocr_model()
            logger.info("Check modelli LLM...")
            init_llm_model()
            unload_llm_model()
            success = True
        except Exception as e:
            logger.error(f"Errore durante l'inizializzazione del modello OCR: {e}")
            success = False
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
        self.wrapped_cv_images = {}  # filename -> CV2 image (per avere a disposizione l'immagine ritagliata)
        self.ocr_results = {}  # filename -> str (risultato OCR)
        self.llm_results = {}  # filename -> dict (risultato LLM strutturato)
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
        # Segnali esistenti
        self.view.preview_selected.connect(self.on_preview_selected)
        self.view.preview_size_changed.connect(self.update_preview_size)
        self.view.analyze_clicked.connect(self.start_analyze)  # Mantenuto per retrocompatibilità
        self.view.analyze_all_clicked.connect(self.start_analyze_all)  # Mantenuto per retrocompatibilità
        self.view.text_comment_changed.connect(self.save_current_comment)

        # Nuovi segnali
        self.view.crop_image_clicked.connect(self.crop_image)
        self.view.crop_all_clicked.connect(self.crop_all_images)
        self.view.analyze_ocr_clicked.connect(self.analyze_ocr)
        self.view.analyze_all_ocr_clicked.connect(self.analyze_all_ocr)
        self.view.analyze_llm_clicked.connect(self.analyze_llm)
        self.view.analyze_all_llm_clicked.connect(self.analyze_all_llm)

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

        # Aggiorna lo stato del pulsante LLM in base alla disponibilità del testo OCR
        self.view.update_llm_buttons_state()

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

        # Aggiorna lo stato del pulsante LLM per questa immagine
        if idx == self.current_idx:
            self.view.update_llm_buttons_state()

        # Verifica se tutti i file hanno un risultato OCR per abilitare il pulsante "all" LLM
        self.update_all_llm_button_state()

    def llm_completed(self, fname, idx, structured_data):
        """Callback chiamato quando l'analisi LLM è completata."""
        self.llm_results[fname] = structured_data
        # Aggiorna l'interfaccia con i dati strutturati
        if idx == self.current_idx:
            self.view.set_status_message(f"Analisi LLM completata per {fname}", 5000)

        self.set_processing(fname, False)
        logger.info(f"Analisi LLM per {fname} completata e interfaccia aggiornata.")

    def update_all_llm_button_state(self):
        """Aggiorna lo stato del pulsante LLM 'all' in base alla disponibilità dei testi OCR."""
        all_have_ocr = all(fname in self.ocr_results for fname in self.image_files)
        self.view.update_all_llm_button_state(all_have_ocr)

    def crop_image(self):
        """Ritaglia l'immagine corrente usando i punti di controllo."""
        self.save_current_perimeter()
        fname = self.image_files[self.current_idx]
        self.set_processing(fname, True)
        self.view.set_status_message(f"Ritaglio dell'immagine {fname} in corso...")

        # Esegui la trasformazione prospettica
        self._crop_image_internal(self.current_idx)

        self.set_processing(fname, False)
        self.view.set_status_message(f"Ritaglio dell'immagine {fname} completato", 3000)

    def crop_all_images(self):
        """Ritaglia tutte le immagini usando i rispettivi punti di controllo."""
        self.save_current_perimeter()  # Salva il perimetro dell'immagine corrente

        self.view.set_status_message("Ritaglio di tutte le immagini in corso...")

        # Ritaglia ogni immagine
        for idx in range(len(self.image_files)):
            fname = self.image_files[idx]
            self.set_processing(fname, True)
            self._crop_image_internal(idx)
            self.set_processing(fname, False)

        # Aggiorna l'interfaccia per l'immagine corrente
        self.update_wrapped_image_display()
        self.view.set_status_message("Ritaglio di tutte le immagini completato", 3000)

    def _crop_image_internal(self, idx):
        """Implementazione interna per il ritaglio dell'immagine all'indice specificato."""
        fname = self.image_files[idx]
        img_path = os.path.join(self.image_dir, fname)

        # Ottieni i punti del perimetro
        coords = self.perimeters.get(fname)
        if coords is None:
            # Inizializza le coordinate se non esistono
            img_for_coords = cv2.imread(img_path)
            if img_for_coords is not None:
                h, w = img_for_coords.shape[:2]
                coords = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
                self.perimeters[fname] = coords
            else:
                logger.error(f"Impossibile leggere l'immagine {fname} per inizializzare le coordinate")
                return False

        # Esegui il warping
        wrapped_img_cv = warp_image(img_path, coords)

        if wrapped_img_cv is not None:
            # Assicura che l'array NumPy sia contiguo
            if not wrapped_img_cv.flags['C_CONTIGUOUS']:
                wrapped_img_cv = np.ascontiguousarray(wrapped_img_cv)

            h_w, w_w = wrapped_img_cv.shape[:2]
            if h_w > 0 and w_w > 0:
                # Salva l'immagine CV2 per l'analisi OCR
                self.wrapped_cv_images[fname] = wrapped_img_cv.copy()

                # Converti in QImage e QPixmap per la visualizzazione
                q_wrapped_img = QImage(wrapped_img_cv.data, w_w, h_w, wrapped_img_cv.strides[0], QImage.Format.Format_BGR888)
                if not q_wrapped_img.isNull():
                    self.wrapped_images[fname] = QPixmap.fromImage(q_wrapped_img)
                    if idx == self.current_idx:
                        self.update_wrapped_image_display(fname)
                    return True
                else:
                    logger.error(f"QImage creata da wrapped_img_cv per {fname} è nulla")
            else:
                logger.error(f"Immagine wrappata per {fname} ha dimensioni non valide: {w_w}x{h_w}")
        else:
            logger.error(f"warp_image per {fname} ha restituito None")

        return False

    def analyze_ocr(self):
        """Analizza l'immagine corrente con OCR dopo averla ritagliata se necessario."""
        self.save_current_perimeter()
        fname = self.image_files[self.current_idx]

        # Se l'immagine non è stata ancora ritagliata, ritagliala prima
        if fname not in self.wrapped_cv_images:
            logger.info(f"Immagine {fname} non ancora ritagliata, eseguo ritaglio prima dell'analisi OCR")
            if not self._crop_image_internal(self.current_idx):
                self.view.set_status_message(f"Errore durante il ritaglio dell'immagine {fname}", 5000)
                return

        self.set_processing(fname, True)
        self.view.set_status_message(f"Analisi OCR dell'immagine {fname} in corso...")

        # Ottieni il commento utente
        comment_user = self.user_comments.get(fname, "")

        # Avvia l'analisi OCR in un thread separato
        wrapped_img_cv = self.wrapped_cv_images[fname]
        self.ocr_thread = OcrWorker(fname, self.current_idx, wrapped_img_cv, comment_user)
        self.ocr_thread.finished.connect(self.ocr_completed)
        self.ocr_thread.start()

    def analyze_all_ocr(self):
        """Analizza tutte le immagini con OCR dopo averle ritagliate se necessario."""
        self.save_current_perimeter()  # Salva il perimetro dell'immagine corrente

        self.view.set_status_message("Analisi OCR di tutte le immagini in corso...")

        # Analizza ogni immagine
        for idx in range(len(self.image_files)):
            fname = self.image_files[idx]

            # Se l'immagine non è stata ancora ritagliata, ritagliala prima
            if fname not in self.wrapped_cv_images:
                logger.info(f"Immagine {fname} non ancora ritagliata, eseguo ritaglio prima dell'analisi OCR")
                if not self._crop_image_internal(idx):
                    logger.error(f"Errore durante il ritaglio dell'immagine {fname}, salto l'analisi OCR")
                    continue

            self.set_processing(fname, True)

            # Ottieni il commento utente
            comment_user = self.user_comments.get(fname, "")

            # Avvia l'analisi OCR in un thread separato
            wrapped_img_cv = self.wrapped_cv_images[fname]
            ocr_thread = OcrWorker(fname, idx, wrapped_img_cv, comment_user)
            ocr_thread.finished.connect(self.ocr_completed)
            ocr_thread.start()
            # Attendiamo un po' per non sovraccaricare il sistema
            ocr_thread.wait(100)  # ms

    def analyze_llm(self):
        """Analizza il testo OCR corrente con LLM per estrarre dati strutturati."""
        fname = self.image_files[self.current_idx]

        # Verifica che ci sia un testo OCR da analizzare
        if fname not in self.ocr_results or not self.ocr_results[fname].strip():
            self.view.set_status_message("Nessun testo OCR disponibile per l'analisi LLM", 5000)
            return

        ocr_text = self.ocr_results[fname]
        comment_user = self.user_comments.get(fname, "")

        self.set_processing(fname, True)
        self.view.set_status_message(f"Analisi LLM del testo OCR per {fname} in corso...")

        # Avvia l'analisi LLM in un thread separato
        llm_thread = LlmWorker(fname, self.current_idx, ocr_text, comment_user)
        llm_thread.finished.connect(self.llm_completed)
        llm_thread.start()

    def analyze_all_llm(self):
        """Analizza tutti i testi OCR con LLM per estrarre dati strutturati."""
        # Verifica che ci siano testi OCR da analizzare
        missing_ocr = [fname for fname in self.image_files if fname not in self.ocr_results]
        if missing_ocr:
            missing_str = ", ".join(missing_ocr[:3])
            if len(missing_ocr) > 3:
                missing_str += f" e altri {len(missing_ocr) - 3}"
            self.view.set_status_message(f"Mancano risultati OCR per: {missing_str}", 5000)
            return

        self.view.set_status_message("Analisi LLM di tutti i testi OCR in corso...")

        # Analizza ogni testo OCR
        for idx, fname in enumerate(self.image_files):
            ocr_text = self.ocr_results[fname]
            comment_user = self.user_comments.get(fname, "")

            self.set_processing(fname, True)

            # Avvia l'analisi LLM in un thread separato
            llm_thread = LlmWorker(fname, idx, ocr_text, comment_user)
            llm_thread.finished.connect(self.llm_completed)
            llm_thread.start()
            # Attendiamo un po' per non sovraccaricare il sistema
            llm_thread.wait(100)  # ms

    def start_analyze(self, idx=None):
        """
        Avvia l'analisi OCR per un'immagine specifica o quella corrente.
        Mantenuto per retrocompatibilità.
        """
        if idx is None:
            self.analyze_ocr()  # Usa la nuova funzione
        else:
            actual_idx = idx
            fname = self.image_files[actual_idx]

            # Se l'immagine non è stata ancora ritagliata, ritagliala prima
            if fname not in self.wrapped_cv_images:
                self._crop_image_internal(actual_idx)

            self.set_processing(fname, True)

            # Ottieni il commento utente
            comment_user = self.user_comments.get(fname, "")

            # Avvia l'analisi OCR in un thread separato
            wrapped_img_cv = self.wrapped_cv_images[fname]
            ocr_thread = OcrWorker(fname, actual_idx, wrapped_img_cv, comment_user)
            ocr_thread.finished.connect(self.ocr_completed)
            ocr_thread.start()

    def start_analyze_all(self):
        """
        Avvia l'analisi OCR per tutte le immagini.
        Mantenuto per retrocompatibilità.
        """
        self.analyze_all_ocr()  # Usa la nuova funzione

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

# TODO: splittare le fasi AI in base ai pulsanti premuti
# TODO: aggiungere sotto la colonna degli scontrini il pulsante "Export OFX" che esporta in un file OFX e allineare i pulsanti all a quelli singoli