import logging
import threading
from PIL import Image
import numpy as np
import cv2 # Per cv2.cvtColor

# Variabili globali per il modello, il processore, il device e il lock
model = None
processor = None
device = None
ocr_lock = threading.Lock()

logger = logging.getLogger(__name__)

def init_ocr_model():
    """
    Inizializza il modello OCR e il processore da Hugging Face.
    Da chiamare una volta all'avvio dell'applicazione.
    """
    global model, processor, device
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        logger.info("Inizializzazione del modello OCR (stepfun-ai/GOT-OCR-2.0-hf)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Utilizzo del device per OCR: {device}")

        model_name = "stepfun-ai/GOT-OCR-2.0-hf"
        # Usare device_map può essere utile per modelli grandi, altrimenti .to(device) dopo il caricamento
        model = AutoModelForImageTextToText.from_pretrained(model_name, trust_remote_code=True)
        model.to(device) # Sposta il modello sul device scelto
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        logger.info("Modello OCR inizializzato con successo.")
        return True
    except ImportError:
        logger.error("Le librerie 'torch' o 'transformers' non sono installate. "
                     "Installale con: pip install torch transformers sentencepiece Pillow accelerate")
        return False
    except Exception as e:
        logger.error(f"Errore durante l'inizializzazione del modello OCR: {e}")
        model = None
        processor = None
        return False # Assicura che False sia restituito in caso di eccezione generica

def analyze_image_with_ocr(image_np_bgr):
    """
    Analizza un'immagine (NumPy array BGR) usando il modello OCR inizializzato.
    Utilizza un mutex per la thread-safety.

    Args:
        image_np_bgr (numpy.ndarray): L'immagine in formato BGR (da OpenCV).

    Returns:
        str: Il testo analizzato, o un messaggio di errore.
    """
    global model, processor, device
    if model is None or processor is None:
        logger.error("Modello OCR non inizializzato. Chiamare init_ocr_model() prima.")
        return "Errore: Modello OCR non inizializzato."

    with ocr_lock: # Acquisisce il lock per l'accesso esclusivo al modello
        try:
            logger.info("Inizio analisi OCR sull'immagine.")

            # Converti l'immagine da OpenCV BGR (NumPy) a PIL RGB
            image_rgb = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            inputs = processor(images=pil_image, return_tensors="pt").to(device)

            generated_ids = model.generate(
                **inputs, # Passa pixel_values e attention_mask (se presente)
                tokenizer=processor.tokenizer, # Come nell'esempio originale
                do_sample=False,
                max_new_tokens=4096
                # Rimuoviamo bos_token_id e eos_token_id espliciti,
                # lasciando che generate li gestisca tramite il tokenizer o la config del modello.
            )

            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            logger.info("Analisi OCR completata con successo.")
            return output_text.strip()
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error("Errore CUDA out of memory durante l'analisi OCR. "
                             "Prova a ridurre la dimensione dell'immagine o usa una GPU con più memoria.")
                return "Errore OCR: CUDA out of memory."
            logger.error(f"Errore di runtime durante l'analisi OCR: {e}")
            return f"Errore OCR (Runtime): {str(e)}"
        except Exception as e:
            logger.error(f"Errore generico durante l'analisi OCR: {e}", exc_info=True)
            return f"Errore OCR (Generico): {str(e)}"

if __name__ == '__main__':
    # Piccolo test (eseguire solo se lo script è chiamato direttamente)
    # Assicurati di avere un'immagine di test e che il modello possa essere scaricato.
    logging.basicConfig(level=logging.INFO)
    logger.info("Esecuzione test ocr_analyzer...")
    if init_ocr_model():
        logger.info("Modello inizializzato per il test.")
        dummy_image_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        dummy_image_bgr[:, :, 0] = 255
        dummy_image_bgr[:, :, 1] = 100
        dummy_image_bgr[:, :, 2] = 50

        logger.info("Analisi immagine dummy (non aspettarti testo sensato)...")
        text_from_dummy = analyze_image_with_ocr(dummy_image_bgr)
        logger.info(f"Testo estratto dall'immagine dummy: '{text_from_dummy}'")
    else:
        logger.error("Inizializzazione del modello fallita durante il test.")
