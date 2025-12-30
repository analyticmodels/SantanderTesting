#!/usr/bin/env python3
"""
Hallucination Detection - MULTIPROCESSING VERSION
Uses ProcessPoolExecutor for parallel hallucination detection.

Supports multiple detection backends:
- Ollama (Lynx, Llama)
- HuggingFace Inference API

Author: Hallucination Analysis Team
Version: 1.0 - Multiprocessing Edition
"""

import os
import sys
import json
import logging
import time
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from multiprocessing import Manager, Lock
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Get project root directory (parent of src/)
_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent

# Input/Output files
INPUT_FILE = _PROJECT_ROOT / "data" / "hallucinationQuestions_all.parquet"
OUTPUT_FILE = _PROJECT_ROOT / "data" / "hallucinationDetection_MP_results.csv"

# Processing settings
MAX_WORKERS = 12  # Number of parallel processes
SAVE_INTERVAL = 50  # Save progress every N completions
DETECTION_LIMIT = 1000  # Set to integer to limit rows processed (None = all)

# Detection model settings
# Options: "lynx", "llama", or "huggingface" (or "hf")
DETECTION_MODEL = os.getenv("DETECTION_MODEL", "llama")

# Ollama settings
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "granite4:latest")
LYNX_MODEL = os.getenv("LYNX_MODEL", "tensortemplar/patronus-lynx:8b-instruct-q4_k_m")
LLAMA_DETECTION_MODEL = os.getenv("LLAMA_DETECTION_MODEL", "llama3.2:latest")

# HuggingFace settings
HF_DETECTION_MODEL = os.getenv("HF_DETECTION_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
HF_API_TOKEN = os.getenv("HF_API_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
HF_MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "512"))

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("hallucinationDetection_MP.log")
    ]
)
log = logging.getLogger("hallucination_detector_MP")

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

LYNX_PROMPT_TEMPLATE = """Given the following QUESTION, DOCUMENT and ANSWER you must analyze the provided answer and determine whether it is faithful to the contents of the DOCUMENT.

The ANSWER must not offer new information beyond the context provided in the DOCUMENT.

The ANSWER also must not contradict information provided in the DOCUMENT.

Output your final verdict by strictly following this format: "PASS" if the answer is faithful to the DOCUMENT and "FAIL" if the answer is not faithful to the DOCUMENT.

Show your reasoning.

--
QUESTION (THIS DOES NOT COUNT AS BACKGROUND INFORMATION):
{question}

--
DOCUMENT:
{document}

--
ANSWER:
{answer}

--
Your output should be in JSON FORMAT with the keys "REASONING" and "SCORE":
{{"REASONING": <your reasoning as bullet points>, "SCORE": <your final score>}}
"""

LLAMA_HALLUCINATION_PROMPT = """You are an expert hallucination detector. Analyze the ANSWER and determine if it contains hallucinations when compared to the DOCUMENT.

A hallucination is:
1. FABRICATED: Information not in the document
2. CONTRADICTED: Information that conflicts with the document
3. UNSUPPORTED: Specific claims not verifiable from the document

QUESTION:
{question}

DOCUMENT:
{document}

ANSWER:
{answer}

Respond with valid JSON only:
{{"REASONING": "<your analysis>", "HALLUCINATION_DETECTED": true/false, "SCORE": "PASS"/"FAIL", "CONFIDENCE": "HIGH"/"MEDIUM"/"LOW"}}
"""

LLAMA_HALLUCINATION_PROMPT_NO_DOC = """You are an expert hallucination detector. Without a reference document, identify suspicious or fabricated claims in the ANSWER.

Red flags:
1. Suspiciously specific numbers/percentages
2. Named sources that may not exist
3. Precise details on obscure topics
4. Claims presented as facts without hedging

QUESTION:
{question}

ANSWER:
{answer}

Respond with valid JSON only:
{{"REASONING": "<your analysis>", "SUSPICIOUS_CLAIMS": ["claim1", "claim2"], "HALLUCINATION_DETECTED": true/false, "SCORE": "PASS"/"FAIL", "CONFIDENCE": "HIGH"/"MEDIUM"/"LOW"}}
"""

HF_HALLUCINATION_PROMPT = """You are a strict hallucination detector. Check if the ANSWER contains claims NOT supported by the DOCUMENT.

DOCUMENT:
{document}

QUESTION: {question}

ANSWER:
{answer}

Respond with ONLY this JSON:
{{"REASONING": "List each claim and whether supported", "HALLUCINATION_DETECTED": true/false, "CONFIDENCE": "HIGH", "SCORE": "PASS"/"FAIL"}}
"""

HF_HALLUCINATION_PROMPT_NO_DOC = """You are a strict hallucination detector. Identify fabricated or suspicious claims in the ANSWER.

QUESTION: {question}

ANSWER:
{answer}

Red flags: specific percentages, made-up dates, named sources, precise policy details.

Respond with ONLY this JSON:
{{"REASONING": "List suspicious claims", "SUSPICIOUS_CLAIMS": [], "HALLUCINATION_DETECTED": true/false, "CONFIDENCE": "HIGH", "SCORE": "PASS"/"FAIL"}}
"""

# =============================================================================
# WORKER PROCESS GLOBALS
# =============================================================================
# These are initialized in each worker process via init_worker()
_worker_ollama_client = None
_worker_hf_client = None
_worker_detection_model = None


def init_worker(detection_model: str, hf_token: str = None):
    """
    Initialize worker process with detection client.
    Called once per worker process at startup.
    """
    global _worker_ollama_client, _worker_hf_client, _worker_detection_model

    _worker_detection_model = detection_model

    if detection_model in ("lynx", "llama"):
        import ollama
        _worker_ollama_client = ollama
        log.debug(f"Worker initialized with Ollama for {detection_model}")
    elif detection_model in ("huggingface", "hf"):
        from huggingface_hub import InferenceClient
        _worker_hf_client = InferenceClient(
            model=HF_DETECTION_MODEL,
            token=hf_token
        )
        log.debug(f"Worker initialized with HuggingFace client")


def parse_json_response(raw_response: str) -> Dict:
    """Parse JSON from LLM response with fallback extraction."""
    try:
        # Try direct parse
        return json.loads(raw_response)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in response
    try:
        json_start = raw_response.find('{')
        json_end = raw_response.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = raw_response[json_start:json_end]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    return None


def extract_score_from_text(raw_response: str) -> Tuple[str, str]:
    """Extract score and confidence from raw text when JSON parsing fails."""
    raw_upper = raw_response.upper()
    score = 'UNKNOWN'
    confidence = 'LOW'

    # Check for explicit FAIL/PASS in JSON-like format
    if 'SCORE": "FAIL' in raw_upper or 'SCORE":"FAIL' in raw_upper:
        score = 'FAIL'
    elif 'SCORE": "PASS' in raw_upper or 'SCORE":"PASS' in raw_upper:
        score = 'PASS'
    elif 'HALLUCINATION_DETECTED": TRUE' in raw_upper:
        score = 'FAIL'
    elif 'HALLUCINATION_DETECTED": FALSE' in raw_upper:
        score = 'PASS'
    elif 'FAIL' in raw_upper and 'PASS' not in raw_upper:
        score = 'FAIL'
    elif 'PASS' in raw_upper and 'FAIL' not in raw_upper:
        score = 'PASS'

    return score, confidence


def detect_hallucination_worker(args: Tuple) -> Tuple:
    """
    Worker function for multiprocessing - detect hallucinations in a single Q&A pair.

    Args:
        args: Tuple of (idx, question, answer, document)

    Returns:
        Tuple of (idx, score, confidence, reasoning, suspicious_claims,
                 hallucination_detected, raw_response)
    """
    global _worker_ollama_client, _worker_hf_client, _worker_detection_model

    idx, question, answer, document = args

    try:
        if _worker_detection_model in ("lynx", "llama"):
            return _detect_with_ollama(idx, question, answer, document)
        elif _worker_detection_model in ("huggingface", "hf"):
            return _detect_with_huggingface(idx, question, answer, document)
        else:
            return (idx, "ERROR", "LOW", f"Unknown model: {_worker_detection_model}",
                   "[]", None, "")
    except Exception as e:
        log.error(f"Error in worker for row {idx}: {e}")
        return (idx, "ERROR", "LOW", f"Worker error: {str(e)}", "[]", None, str(e))


def _detect_with_ollama(idx: int, question: str, answer: str, document: str) -> Tuple:
    """Detect hallucinations using Ollama (Lynx or Llama)."""
    global _worker_ollama_client, _worker_detection_model

    # Select model and prompt
    if _worker_detection_model == "lynx":
        model = LYNX_MODEL
        if document:
            prompt = LYNX_PROMPT_TEMPLATE.format(
                question=question, document=document, answer=answer
            )
        else:
            prompt = LYNX_PROMPT_TEMPLATE.format(
                question=question,
                document="No reference document provided.",
                answer=answer
            )
    else:  # llama
        model = LLAMA_DETECTION_MODEL
        if document:
            prompt = LLAMA_HALLUCINATION_PROMPT.format(
                question=question, document=document, answer=answer
            )
        else:
            prompt = LLAMA_HALLUCINATION_PROMPT_NO_DOC.format(
                question=question, answer=answer
            )

    # Call Ollama
    response = _worker_ollama_client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    raw_response = response['message']['content'].strip()

    # Parse response
    parsed = parse_json_response(raw_response)

    if parsed:
        score = parsed.get('SCORE', 'UNKNOWN')
        if isinstance(score, bool):
            score = 'FAIL' if score else 'PASS'
        elif isinstance(score, str):
            score = score.upper()
            if score not in ('PASS', 'FAIL'):
                if parsed.get('HALLUCINATION_DETECTED', False):
                    score = 'FAIL'
                else:
                    score = 'PASS'

        confidence = parsed.get('CONFIDENCE', 'HIGH')
        reasoning = parsed.get('REASONING', '')
        if isinstance(reasoning, list):
            reasoning = json.dumps(reasoning)

        suspicious = parsed.get('SUSPICIOUS_CLAIMS', [])
        if isinstance(suspicious, list):
            suspicious = json.dumps(suspicious)

        hallucination_detected = parsed.get('HALLUCINATION_DETECTED', score == 'FAIL')

        return (idx, score, confidence, reasoning, suspicious,
               hallucination_detected, raw_response)

    # Fallback parsing
    score, confidence = extract_score_from_text(raw_response)
    return (idx, score, confidence, raw_response, "[]",
           score == 'FAIL', raw_response)


def _detect_with_huggingface(idx: int, question: str, answer: str, document: str) -> Tuple:
    """Detect hallucinations using HuggingFace Inference API."""
    global _worker_hf_client

    # Select prompt
    if document:
        prompt = HF_HALLUCINATION_PROMPT.format(
            question=question, document=document, answer=answer
        )
    else:
        prompt = HF_HALLUCINATION_PROMPT_NO_DOC.format(
            question=question, answer=answer
        )

    # Call HuggingFace API
    messages = [{"role": "user", "content": prompt}]

    response = _worker_hf_client.chat_completion(
        messages=messages,
        max_tokens=HF_MAX_NEW_TOKENS,
        temperature=0.1,
        top_p=0.95,
    )

    if response and response.choices:
        raw_response = response.choices[0].message.content.strip()
    else:
        return (idx, "ERROR", "LOW", "Empty response from HuggingFace",
               "[]", None, "")

    # Parse response
    parsed = parse_json_response(raw_response)

    if parsed:
        score = parsed.get('SCORE', 'UNKNOWN')
        if isinstance(score, bool):
            score = 'FAIL' if score else 'PASS'
        elif isinstance(score, str):
            score = score.upper()
            if score not in ('PASS', 'FAIL'):
                if parsed.get('HALLUCINATION_DETECTED', False):
                    score = 'FAIL'
                else:
                    score = 'PASS'

        confidence = parsed.get('CONFIDENCE', 'MEDIUM')
        reasoning = parsed.get('REASONING', '')
        if isinstance(reasoning, list):
            reasoning = json.dumps(reasoning)

        suspicious = parsed.get('SUSPICIOUS_CLAIMS', [])
        if isinstance(suspicious, list):
            suspicious = json.dumps(suspicious)

        hallucination_detected = parsed.get('HALLUCINATION_DETECTED', score == 'FAIL')

        return (idx, score, confidence, reasoning, suspicious,
               hallucination_detected, raw_response)

    # Fallback parsing
    score, confidence = extract_score_from_text(raw_response)
    return (idx, score, confidence, raw_response, "[]",
           score == 'FAIL', raw_response)


# =============================================================================
# MAIN
# =============================================================================
def main():
    start_time = time.time()

    print("="*80)
    print("HALLUCINATION DETECTION - MULTIPROCESSING VERSION")
    print("="*80 + "\n")

    log.info(f"Detection model: {DETECTION_MODEL}")
    log.info(f"Max workers: {MAX_WORKERS}")
    log.info(f"Input file: {INPUT_FILE}")

    # Load input data
    log.info("\n" + "="*60)
    log.info("STEP 1: Loading input dataset")
    log.info("="*60)

    try:
        df = pd.read_parquet(INPUT_FILE, engine='fastparquet')
        total_rows = len(df)
        log.info(f"Loaded {total_rows} rows from {INPUT_FILE}")
    except Exception as e:
        log.error(f"Failed to load {INPUT_FILE}: {e}")
        return 1

    # Apply limit if set
    if DETECTION_LIMIT and DETECTION_LIMIT < total_rows:
        df = df.sample(n=DETECTION_LIMIT, random_state=None).reset_index(drop=True)
        log.info(f"Randomly selected {DETECTION_LIMIT} rows (limit applied)")

    # Check required columns
    if 'question' not in df.columns or 'answer' not in df.columns:
        log.error("Missing required columns: 'question' and/or 'answer'")
        return 1

    # Initialize result columns
    result_columns = ['score', 'confidence', 'reasoning', 'suspicious_claims',
                     'hallucination_detected', 'raw_response']
    for col in result_columns:
        if col not in df.columns:
            df[col] = None

    # Resume support
    if os.path.exists(OUTPUT_FILE):
        log.info(f"\nResuming from existing output: {OUTPUT_FILE}")
        try:
            df_progress = pd.read_csv(OUTPUT_FILE)
            for col in result_columns:
                if col in df_progress.columns:
                    df[col] = df_progress[col]
            log.info("Successfully loaded previous progress")
        except Exception as e:
            log.warning(f"Could not load progress: {e}")

    # Find pending evaluations
    pending = df[df['score'].isna()]
    completed = len(df) - len(pending)

    log.info(f"\nProgress Status:")
    log.info(f"  Total: {len(df)}")
    log.info(f"  Completed: {completed}")
    log.info(f"  Pending: {len(pending)}")

    if len(pending) == 0:
        log.info("\nAll rows already processed!")
        _print_summary(df)
        return 0

    # Confirmation
    print(f"\n" + "="*60)
    print("READY TO START DETECTION")
    print("="*60)
    print(f"Rows to process: {len(pending)}")
    print(f"Detection model: {DETECTION_MODEL}")
    print(f"Workers: {MAX_WORKERS}")

    proceed = input("\nProceed? (y/n): ")
    if proceed.lower() != 'y':
        log.info("Cancelled by user")
        return 0

    # Prepare work items
    work_items = []
    for idx, row in pending.iterrows():
        document = str(row.get('document', '')) if 'document' in row else ''
        work_items.append((
            idx,
            str(row['question']),
            str(row['answer']),
            document
        ))

    def persist():
        """Save progress to CSV."""
        temp_file = f"{OUTPUT_FILE}.tmp"
        df.to_csv(temp_file, index=False, encoding='utf-8-sig')
        os.replace(temp_file, str(OUTPUT_FILE))
        completed_now = len(df[df['score'].notna()])
        log.info(f"Progress saved: {completed_now}/{len(df)} complete")

    # Process with multiprocessing
    log.info("\n" + "="*60)
    log.info("STEP 2: Running hallucination detection")
    log.info("="*60 + "\n")

    try:
        log.info(f"Starting {MAX_WORKERS} worker processes...")

        with ProcessPoolExecutor(
            max_workers=MAX_WORKERS,
            initializer=init_worker,
            initargs=(DETECTION_MODEL, HF_API_TOKEN)
        ) as executor:
            # Submit all tasks
            futures = {
                executor.submit(detect_hallucination_worker, item): item[0]
                for item in work_items
            }

            processed_count = 0
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    (i, score, confidence, reasoning, suspicious,
                     hallucination_detected, raw_response) = future.result()

                    df.at[i, 'score'] = score
                    df.at[i, 'confidence'] = confidence
                    df.at[i, 'reasoning'] = reasoning
                    df.at[i, 'suspicious_claims'] = suspicious
                    df.at[i, 'hallucination_detected'] = hallucination_detected
                    df.at[i, 'raw_response'] = raw_response

                    status = "HALLUCINATION" if hallucination_detected else "FAITHFUL"
                    log.info(f"Row {i}: {score} ({confidence}) - {status}")

                except Exception as e:
                    log.error(f"Error processing row {idx}: {e}")
                    df.at[idx, 'score'] = "ERROR"
                    df.at[idx, 'reasoning'] = f"Processing error: {e}"

                processed_count += 1
                if processed_count % SAVE_INTERVAL == 0:
                    persist()
                    elapsed = time.time() - start_time
                    rate = processed_count / (elapsed / 60)
                    log.info(f"Rate: {rate:.1f} rows/min")

    except KeyboardInterrupt:
        log.warning("\nInterrupted by user")
        log.info("Saving progress...")
        persist()
        return 0

    # Final save
    persist()

    # Print summary
    _print_summary(df)

    total_time = (time.time() - start_time) / 60
    log.info(f"\nCompleted in {total_time:.2f} minutes")

    return 0


def _print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    total = len(df)
    pass_count = (df['score'] == 'PASS').sum()
    fail_count = (df['score'] == 'FAIL').sum()
    unknown_count = (df['score'] == 'UNKNOWN').sum()
    error_count = (df['score'] == 'ERROR').sum()

    print("\n" + "="*60)
    print("DETECTION SUMMARY")
    print("="*60)
    print(f"Total evaluated: {total}")
    print(f"PASS (faithful): {pass_count} ({100*pass_count/total:.1f}%)")
    print(f"FAIL (hallucination): {fail_count} ({100*fail_count/total:.1f}%)")
    print(f"UNKNOWN: {unknown_count} ({100*unknown_count/total:.1f}%)")
    print(f"ERROR: {error_count} ({100*error_count/total:.1f}%)")
    print(f"\nResults saved to: {OUTPUT_FILE}")
    print("="*60)


if __name__ == "__main__":
    sys.exit(main())
