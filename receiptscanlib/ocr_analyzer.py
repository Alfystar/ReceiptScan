import json
import logging
import threading

import cv2  # Per cv2.cvtColor
import numpy as np
import torch
from PIL import Image
import transformers
from transformers import (AutoProcessor, AutoModelForImageTextToText, GotOcr2Processor,
                          GotOcr2ForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM,
                          pipeline)
from adapters import AutoAdapterModel
from peft import PeftModel  # Aggiungi questo import all'inizio del file

# Constanti
# https://huggingface.co/docs/transformers/en/model_doc/got_ocr2
OCR_MODEL_NAME = "stepfun-ai/GOT-OCR-2.0-hf"
OCR_ADAPTER_NAME = "Effectz-AI/GOT-OCR2_0_Invoice_MD"  # Se hai un adapter specifico, altrimenti puoi omettere questa riga

LLM_MODEL_ID = "chuanli11/Llama-3.2-3B-Instruct-uncensored" # "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1"

gpu_lock = threading.RLock()  # Lock per la GPU, se necessario
# ocr_lock = threading.RLock()
# llm_lock = threading.RLock()  # Lock per la pipeline LLM

# Variabili globali per il modello LLM
llm_model: AutoModelForCausalLM = None
llm_tokenizer: AutoTokenizer = None
llm_pipeline: pipeline = None

# Variabili globali per il modello OCR
ocr_model: GotOcr2ForConditionalGeneration = None
ocr_processor: GotOcr2Processor = None
ocr_device: str = None

logger = logging.getLogger(__name__)


def enough_gpu_memory(required_gb=3):
    if not torch.cuda.is_available():
        return False
    torch.cuda.empty_cache()  # Libera la memoria non usata da PyTorch
    free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
    free_mem_gb = free_mem / (1024 ** 3)
    return free_mem_gb > required_gb


def init_llm_model():
    global llm_tokenizer, llm_pipeline, llm_model
    try:
        # Test preliminare della GPU e configurazione del dispositivo
        if torch.cuda.is_available():
            try:
                # Testiamo se possiamo accedere alla GPU senza errori
                dummy_tensor = torch.zeros(1).cuda()
                dummy_tensor = dummy_tensor * 2  # Operazione semplice per testare la GPU
                del dummy_tensor  # Puliamo
                torch.cuda.empty_cache()
                device_map = "auto"  # Utilizziamo la divisione automatica tra CPU e GPU
                logger.info("Utilizzo GPU per il modello LLM")
            except Exception as e:
                logger.warning(f"GPU disponibile ma ha generato un errore: {e}. Utilizzo CPU come fallback per LLM.")
                device_map = "cpu"
        else:
            device_map = "cpu"
            logger.info("GPU non disponibile, utilizzo CPU per il modello LLM")

        # Configurazione per evitare problemi di clock e threading
        torch.set_num_threads(1)  # Limita i thread per ridurre problemi di sincronizzazione

        # Carica prima il tokenizer (meno intensivo in memoria)
        llm_tokenizer = transformers.AutoTokenizer.from_pretrained(
            LLM_MODEL_ID,
            use_fast=True,  # Tokenizer veloce dove possibile
            padding_side='left'  # Migliora la performance per text generation
        )

        # Configura il pad token correttamente
        if llm_tokenizer.pad_token_id is None:
            llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id

        # Memoria configurata in modo dinamico in base al dispositivo
        if device_map == "auto":
            max_mem = {0: "3GiB", "cpu": "8GiB"}  # Valori cautelativi
        else:
            max_mem = None  # Non specificato quando si usa solo CPU

        # Carica il modello con configurazioni ottimizzate
        load_options = {
            "device_map": device_map,
            "low_cpu_mem_usage": True,
            "max_memory": max_mem
        }

        # Aggiungiamo la precisione ridotta solo se usiamo la GPU
        if device_map == "auto":
            # Determinare il tipo di precisione supportato dalla GPU
            if torch.cuda.is_bf16_supported():
                load_options["torch_dtype"] = torch.bfloat16
                logger.info("Utilizzo di bfloat16 per il modello LLM")
            else:
                load_options["torch_dtype"] = torch.float16
                logger.info("Utilizzo di float16 per il modello LLM")

        # Carica il modello con le opzioni configurate
        llm_model = transformers.AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID,
            **load_options
        )

        # Crea la pipeline con configurazioni di sicurezza
        llm_pipeline = transformers.pipeline(
            "text-generation",
            model=llm_model,
            tokenizer=llm_tokenizer,
            max_new_tokens=1024,
            do_sample=False,  # Genera output più deterministici
            return_full_text=False,  # Per risparmiare memoria
            pad_token_id=llm_tokenizer.pad_token_id
        )

        return True
    except Exception as e:
        logger.error(f"Errore durante l'inizializzazione del modello LLM: {e}", exc_info=True)
        llm_tokenizer, llm_pipeline, llm_model = None, None, None
        return False


def unload_llm_model():
    global llm_tokenizer, llm_pipeline, llm_model
    if llm_model is None or llm_tokenizer is None or llm_pipeline is None:
        logger.warning("Il modello LLM non è caricato, unload non necessario.")
    del llm_pipeline, llm_tokenizer, llm_model
    llm_pipeline = None
    llm_tokenizer = None
    llm_model = None
    torch.cuda.empty_cache()


def init_ocr_model():
    global ocr_model, ocr_processor, ocr_device
    try:
        # Impostiamo il dispositivo con un meccanismo di fallback più robusto
        if torch.cuda.is_available():
            try:
                # Prima testiamo se possiamo accedere alla GPU senza errori
                dummy_tensor = torch.zeros(1).cuda()
                dummy_tensor = dummy_tensor * 2  # Semplice operazione per verificare che la GPU funzioni
                del dummy_tensor  # Puliamo
                torch.cuda.empty_cache()
                ocr_device = "cuda"
                logger.info("Utilizzo GPU per il modello OCR")
            except Exception as e:
                logger.warning(f"GPU disponibile ma ha generato un errore: {e}. Utilizzo CPU come fallback.")
                ocr_device = "cpu"
        else:
            ocr_device = "cpu"
            logger.info("GPU non disponibile, utilizzo CPU per il modello OCR")

        # Configurazione per evitare problemi di clock e threading
        torch.set_num_threads(1)  # Limitare i thread può aiutare con problemi di sincronizzazione

        # Carica il processore prima del modello
        ocr_processor = AutoProcessor.from_pretrained(OCR_MODEL_NAME, trust_remote_code=True, use_fast=True)

        # Carichiamo il modello con configurazione migliorata
        load_options = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,  # Riduce l'utilizzo di memoria CPU durante il caricamento
        }

        # Aggiungiamo opzioni specifiche per la GPU se usata
        if ocr_device == "cuda":
            load_options["torch_dtype"] = torch.float16  # Utilizziamo la precisione ridotta per risparmiare memoria

        ocr_model = AutoModelForImageTextToText.from_pretrained(OCR_MODEL_NAME, **load_options)
        ocr_model.to(ocr_device)

        return True
    except Exception as e:
        logger.error(f"Errore durante l'inizializzazione del modello OCR: {e}", exc_info=True)
        ocr_model, ocr_processor, ocr_device = None, None, None
        return False


def unload_ocr_model():
    global ocr_model, ocr_processor, ocr_device
    if ocr_model is None or ocr_processor is None:
        logger.warning("Il modello OCR non è caricato, unload non necessario.")
    del ocr_model, ocr_processor
    ocr_model = None
    ocr_processor = None
    ocr_device = None
    torch.cuda.empty_cache()

def analyze_image_with_ocr(image_np_bgr) -> str:
    """
    Analizza un'immagine (NumPy array BGR) usando il modello OCR inizializzato.
    Utilizza un mutex per la thread-safety.

    Args:
        image_np_bgr (numpy.ndarray): L'immagine in formato BGR (da OpenCV).

    Returns:
        str: Il testo analizzato, o un messaggio di errore.
    """
    global ocr_model, ocr_processor, ocr_device
    with gpu_lock:  # Acquisisce il lock per l'accesso esclusivo al modello
        try:
            init_ocr_model()
            logger.info("Inizio analisi OCR sull'immagine.")

            # Converti l'immagine da OpenCV BGR (NumPy) a PIL RGB
            image_rgb = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            logger.info(f"Analisi OCR: Dimensioni immagine PIL prima del processore: {pil_image.size}")  # Log delle dimensioni

            inputs = ocr_processor(images=pil_image, return_tensors="pt", format=True).to(ocr_device)
            logger.debug(f"Analisi OCR: Input tensori pronti per il modello (device: {ocr_device}).")
            logger.debug(f"Shape pixel_values: {inputs.get('pixel_values').shape if inputs.get('pixel_values') is not None else 'N/A'}")

            logger.info("Analisi OCR: Chiamata a ocr_model.generate...")
            generated_ids = ocr_model.generate(
                **inputs,  # Passa pixel_values e attention_mask (se presente)
                do_sample=False,
                tokenizer=ocr_processor.tokenizer,
                stop_strings="<|im_end|>",
                max_new_tokens=4096
            )
            logger.info("Analisi OCR: model.generate completato.")

            output_text = ocr_processor.batch_decode(generated_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

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
        finally:
            unload_ocr_model()


def analyze_receipt_structured_llm(image_np_bgr, comment="", thinking="off"):
    """
    Analizza uno scontrino e restituisce sia il testo completo che i dati strutturati in JSON,
    usando un LLM per la seconda fase.
    """
    global llm_pipeline
    # 1. Estrai il testo con il tuo OCR
    full_text = analyze_image_with_ocr(image_np_bgr)
    if not full_text or "Errore" in full_text:
        return full_text, {}

    # 2. Prepara il prompt per il LLM
    comment = comment.strip() if comment else "No additional comment provided."
    system_prompt = f"detailed thinking {thinking}"
    #  { "date": "date of the receipt", "total": "total amount", "payment_method": "cash or electronic", "currency": "EUR or USD or GBP or other currency" }
    user_prompt = f"""Following text is an OCR of a receipt:

{full_text}

And this is additional context provided by the user: {comment}

Extract information necessary to fill this JSON and return ONLY it with a json valid text that contains the following fields:
{{
"date": "date of the receipt in format dd-mm-yyyy",
"total": "total amount",
"payment_method": "cash or electronic",
"currency": "EUR or USD or GBP or other currency"
}}

No other text or explanation. Be sure the response start with a single opening curly brace '{{' and ends with a single closing curly brace '}}'.
"""
    logger.debug(f"Prompt per il LLM:\n{user_prompt}")
    # 3. Chiamata al LLM
    chat_input = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    with gpu_lock:  # Protegge l'accesso alla pipeline LLM
        try:
            init_llm_model()
            llm_output = llm_pipeline(chat_input)
        except Exception as e:
            logger.error(f"Errore durante l'analisi LLM: {e}", exc_info=True)
            return full_text, {"error": "LLM analysis failed", "details": str(e)}
        finally:
            unload_llm_model()
    llm_result = llm_output[0]
    if isinstance(llm_result, dict) and "generated_text" in llm_result:
        llm_text = llm_result["generated_text"].strip()
    else:
        llm_text = str(llm_result).strip()

    # 4. Parsing del JSON
    try:
        # Cerca la prima parentesi graffa per estrarre solo il JSON
        json_start = llm_text.find("{")
        json_end = llm_text.rfind("}") + 1
        if json_end == 0:
            # Nessuna graffa chiusa trovata, aggiungila
            json_text = llm_text[json_start:] + "}"
        else:
            json_text = llm_text[json_start:json_end]
        receipt_info = json.loads(json_text)
        logger.debug(f"JSON estratto dal modello LLM: {receipt_info}")
    except Exception as e:
        receipt_info = {"error": "Invalid JSON", "raw_text": llm_text}
        logger.error(f"Errore durante il parsing del JSON estratto dal modello LLM: {e}, {receipt_info}", exc_info=True)

    return full_text, receipt_info


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
