import logging
import threading

import cv2  # Per cv2.cvtColor
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, GotOcr2Processor, GotOcr2ForConditionalGeneration

# Constanti
OCR_MODEL_NAME = "stepfun-ai/GOT-OCR-2.0-hf"

# Variabili globali per il modello, il processore, il device e il lock
model: GotOcr2ForConditionalGeneration = None
processor: GotOcr2Processor = None
device: str = None
ocr_lock = threading.Lock()

logger = logging.getLogger(__name__)


def init_ocr_model():
    """
    Inizializza il modello OCR e il processore da Hugging Face.
    Da chiamare una volta all'avvio dell'applicazione.
    """
    global model, processor, device
    try:
        logger.info(f"Inizializzazione del modello OCR ({OCR_MODEL_NAME})...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Utilizzo del device per OCR: {device}")

        # Usare device_map può essere utile per modelli grandi, altrimenti .to(device) dopo il caricamento
        model = AutoModelForImageTextToText.from_pretrained(OCR_MODEL_NAME, trust_remote_code=True)
        model.to(device)  # Sposta il modello sul device scelto
        processor = AutoProcessor.from_pretrained(OCR_MODEL_NAME, trust_remote_code=True)

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
        return False  # Assicura che False sia restituito in caso di eccezione generica


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

    with ocr_lock:  # Acquisisce il lock per l'accesso esclusivo al modello
        try:
            logger.info("Inizio analisi OCR sull'immagine.")

            # Converti l'immagine da OpenCV BGR (NumPy) a PIL RGB
            image_rgb = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            logger.info(f"Analisi OCR: Dimensioni immagine PIL prima del processore: {pil_image.size}")  # Log delle dimensioni

            inputs = processor(images=pil_image, return_tensors="pt", format=True).to(device)
            logger.debug(
                f"Analisi OCR: Input tensori pronti per il modello (device: {device}). Shape pixel_values: {inputs.get('pixel_values').shape if inputs.get('pixel_values') is not None else 'N/A'}")

            logger.info("Analisi OCR: Chiamata a model.generate...")
            generated_ids = model.generate(
                **inputs,  # Passa pixel_values e attention_mask (se presente)
                do_sample=False,
                tokenizer=processor.tokenizer,
                stop_strings="<|im_end|>",
                max_new_tokens=512  # Ridotto drasticamente da 4096 per test
                # Rimuoviamo bos_token_id e eos_token_id espliciti,
                # lasciando che generate li gestisca tramite il tokenizer o la config del modello.
            )
            logger.info("Analisi OCR: model.generate completato.")

            output_text = processor.batch_decode(generated_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

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

# TODO: testare questa funzione con immagini di scontrini reali e commenti generati dal prompt:
# vorrei provare a farlo con questo modello, senza usarne un altro, così da mantenere tutta la computazione in locale
# le domande sono sempre le stesse e mi servirebbe una risposta JSON così da poterla parsare facilmente. Le domande sono:
#
#
# 1. in che data è stato fatto lo scontrino
# 2 .Quanto è la spesa totale
# 3. è stato fatto in contanti o tramite pagamento elettronico
# 4. In che valuta è tra (lista di valute)
# In oltre, riceve oltre alla foto anche un commento umano scritto che da ulteriori informazioni su quello scontrino.
#
# L'output dovrebbe essere il testo letto e la risposta in formato json alle domande the ti ho appena scritto
# con domanda all'llm in inglese e risposta in base alla lingua del commento/scontrino.
# TODO: aggingiere che ad ogni nuova chiamata venga resettata la memoria del modello, così da evitare problemi di memoria

def analyze_receipt_structured(image_np_bgr, comment=""):
    """
    Analizza uno scontrino e restituisce sia il testo completo che i dati strutturati in JSON.

    Args:
        image_np_bgr (numpy.ndarray): L'immagine dello scontrino in formato BGR
        comment (str, optional): Commento aggiuntivo con informazioni sullo scontrino

    Returns:
        tuple: (testo_completo, json_risposta)
    """
    global model, processor, device
    if model is None or processor is None:
        logger.error("Modello OCR non inizializzato. Chiamare init_ocr_model() prima.")
        return "Errore: Modello OCR non inizializzato.", "{}"

    with ocr_lock:
        try:
            logger.info("Inizio analisi strutturata dello scontrino")

            # Converti l'immagine da OpenCV BGR a PIL RGB
            image_rgb = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            # Prima estrai il testo completo
            basic_prompt = "Transcribe all text from this receipt."
            basic_inputs = processor(images=pil_image, text=basic_prompt, return_tensors="pt", format=True).to(device)

            logger.info("Generazione testo completo dello scontrino...")
            text_ids = model.generate(
                **basic_inputs,
                do_sample=False,
                tokenizer=processor.tokenizer,
                stop_strings="<|im_end|>",
                max_new_tokens=512
            )
            full_text = \
            processor.batch_decode(text_ids[:, basic_inputs["input_ids"].shape[1]:], skip_special_tokens=True)[
                0].strip()

            # Ora estrai le informazioni strutturate con un prompt specifico
            json_prompt = f"""Analyze this receipt and extract the following information in JSON format:
            Additional context: {comment}

            Return ONLY a valid JSON with this structure:
            {{
              "date": "date of the receipt",
              "total": "total amount",
              "payment_method": "cash or electronic",
              "currency": "EUR or USD or GBP or other currency"
            }}
            """

            json_inputs = processor(images=pil_image, text=json_prompt, return_tensors="pt", format=True).to(device)

            logger.info("Generazione risposta strutturata in JSON...")
            json_ids = model.generate(
                **json_inputs,
                do_sample=False,
                tokenizer=processor.tokenizer,
                stop_strings="<|im_end|>",
                max_new_tokens=256
            )

            json_text = \
            processor.batch_decode(json_ids[:, json_inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

            logger.info("Analisi strutturata completata")
            return full_text, json_text

        except Exception as e:
            logger.error(f"Errore durante l'analisi strutturata: {e}", exc_info=True)
            return f"Errore: {str(e)}", "{}"

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
