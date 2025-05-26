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
    llm_tokenizer = transformers.AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id
    max_mem = {0: "4GiB", "cpu": "8GiB"}  # Adatta il valore a seconda della tua GPU
    llm_model = transformers.AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Split automatico tra CPU e GPU
        max_memory = max_mem
    )
    llm_pipeline = transformers.pipeline(
        "text-generation",
        model=llm_model,
        tokenizer=llm_tokenizer,
        max_new_tokens=512
    )


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
    ocr_device = "cuda" if torch.cuda.is_available() else "cpu"
    ocr_model = AutoModelForImageTextToText.from_pretrained(OCR_MODEL_NAME, trust_remote_code=True)
    ocr_model.to(ocr_device)
    ocr_processor = AutoProcessor.from_pretrained(OCR_MODEL_NAME, trust_remote_code=True, use_fast=True)


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

Extract information necessary to fill this JSON and return ONLY IT, no other text or explanation:
{{
"date": "date of the receipt in format dd-mm-yyyy",
"total": "total amount",
"payment_method": "cash or electronic",
"currency": "EUR or USD or GBP or other currency"
}}
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
        llm_text = llm_result["generated_text"][-1]['content'].strip()
    else:
        llm_text = str(llm_result).strip()

    # 4. Parsing del JSON
    try:
        # Cerca la prima parentesi graffa per estrarre solo il JSON
        json_start = llm_text.find("{")
        json_text = llm_text[json_start:]
        receipt_info = json.loads(json_text)
        logger.debug(f"JSON estratto dal modello LLM: {receipt_info}")
    except Exception as e:
        receipt_info = {"error": "Invalid JSON", "raw_text": llm_text}

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
