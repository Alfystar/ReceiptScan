import argparse
import signal
import sys
import logging

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

# Il logger principale è configurato qui.
# I moduli della libreria otterranno i loro logger specifici.

def handle_sigint(sig, frame):
    logging.info("Terminazione richiesta dall'utente (Ctrl+C). Uscita...")
    QTimer.singleShot(0, QApplication.quit)

def keep_alive():
    timer = QTimer()
    # Timer che ogni 200ms chiama processEvents per permettere la gestione di SIGINT
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

    # Configurazione del logging globale
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        # Non usare logger qui perché potrebbe non essere ancora configurato
        print(f'Livello di log non valido: {args.log_level}', file=sys.stderr)
        sys.exit(1)
    logging.basicConfig(level=numeric_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Importa il controller dalla libreria DOPO aver configurato il logging,
    # così i logger della libreria ereditano la configurazione.
    try:
        from receiptscanlib.ocr_controller import OcrAppController
    except ImportError as e:
        logging.critical(f"Errore nell'importare OcrAppController da receiptscanlib.ocr_controller: {e}. "
                         f"Assicurati che la directory 'receiptscanlib' e i file al suo interno esistano e siano corretti.")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"Errore generico durante l'importazione di OcrAppController: {e}")
        sys.exit(1)

    # 2. Gestione del segnale SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, handle_sigint)

    # 3. Creazione dell'applicazione Qt
    app = QApplication(sys.argv)

    # 4. Timer per la gestione dei segnali
    _ka_timer = keep_alive()

    # 5. Creazione e visualizzazione della finestra principale
    try:
        controller = OcrAppController(args.dir)
        controller.view.show()  # Mostra la view del controller
    except Exception as e:
        logging.critical(f"Errore durante la creazione o la visualizzazione dell'applicazione: {e}")
        sys.exit(1)

    # 6. Avvio del loop di eventi dell'applicazione
    sys.exit(app.exec())

