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

# Suppress verbose HTTP client logs at environment level
os.environ["HTTPX_LOG_LEVEL"] = "ERROR"
os.environ["CURL_VERBOSE"] = "0"

# Disable httpx event hooks that spam the terminal
import warnings
warnings.filterwarnings("ignore")

# Patch httpx to disable verbose event logging
try:
    import httpx
    # Disable all event hooks in httpx
    httpx._client.logger = logging.getLogger("httpx")
    httpx._client.logger.setLevel(logging.CRITICAL)
except:
    pass  # httpx not yet imported, will be handled in worker init

# =============================================================================
# CONFIGURATION
# =============================================================================

# Get project root directory (parent of src/)
_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent

# Input/Output files
INPUT_FILE = _PROJECT_ROOT / "data" / "hallucinationQuestions_all_doc.parquet"
OUTPUT_FILE = _PROJECT_ROOT / "data" / "hallucinationDetection_MP_results.csv"

# Processing settings
MAX_WORKERS = 32  # Number of parallel processes
SAVE_INTERVAL = 50  # Save progress every N completions
DETECTION_LIMIT = None  # Set to integer to limit rows processed (None = all)

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
# Configure file logging only - console output controlled separately
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("hallucinationDetection_MP.log")
    ]
)
log = logging.getLogger("hallucination_detector_MP")

# Suppress verbose HTTP logs from HuggingFace and other libraries
for logger_name in [
    "httpx", "httpcore", "urllib3", "huggingface_hub",
    "huggingface_hub.inference._client",
    "huggingface_hub.inference._generated.types"
]:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL)  # Only show critical errors
    logger.propagate = False  # Don't propagate to parent loggers

# Console logging interval (rows)
CONSOLE_LOG_INTERVAL = 200

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

HF_HALLUCINATION_PROMPT_NO_DOC = """You are an expert hallucination detector for AI-generated text. Your task is to determine whether an ANSWER contains hallucinations when evaluated strictly against a REFERENCE DOCUMENT.

You must prioritize recall: it is preferable to flag a possible hallucination than to miss one.

---

## Evaluation Mode
REFERENCE-BOUND = true

When REFERENCE-BOUND is true:
- Treat the REFERENCE DOCUMENT as the sole authoritative source of truth.
- Even correct general knowledge must be marked UNSUPPORTED unless it is explicitly stated in the document.
- Do NOT rely on background knowledge, training data, or assumptions not grounded in the document.

---

## Definition of Hallucination
A hallucination occurs when the ANSWER contains one or more claims that are:

1. **Fabricated**: Information not present in the reference document
2. **Contradictory**: Information that conflicts with the reference document
3. **Unsupported**: Specific factual claims (names, numbers, dates, causal assertions) not verifiable from the document
4. **Confabulated**: Plausible-sounding details that introduce invented specifics

---

## What is NOT a Hallucination (Strictly Limited)
- Direct quotations or faithful paraphrases of the document
- Reasonable inferences that are *clearly and explicitly implied* by the document
- Hedged or uncertain language ("may", "could", "possibly") **only if** it does not introduce new facts
- Entities or facts explicitly mentioned in the QUESTION (but not new attributes about them)

---

## Hallucination Decision Rule (High Recall)
A hallucination is detected if ANY of the following are true:

- At least one claim is CONTRADICTED by the document
- One or more UNSUPPORTED claims introduce specific factual details
- The answer relies on general knowledge not grounded in the document
- There is ambiguity and it is unclear whether a claim is supported

When in doubt, choose the more conservative option and flag a hallucination.

---

## Your Task
Analyze the ANSWER by decomposing it into atomic claims and evaluating each claim strictly against the REFERENCE DOCUMENT.

---

QUESTION:
{question}

---

REFERENCE DOCUMENT:
{document}

---

ANSWER TO EVALUATE:
{answer}

---

## Evaluation Instructions
1. Identify each distinct factual or inferential claim in the ANSWER.
2. For each claim, determine:
   - **Type**:
     - FACTUAL (verifiable fact, number, name, date, causal claim)
     - INFERENTIAL (logical conclusion drawn from the document)
     - GENERAL_KNOWLEDGE (background information not document-specific)
     - OPINION (value judgment or recommendation)
   - **Status**:
     - SUPPORTED: Explicitly stated in the document (quote required)
     - UNSUPPORTED: Not present or not verifiable in the document
     - CONTRADICTED: Conflicts with the document
   - **Materiality**:
     - true if the claim materially affects correctness or meaning
     - false if the claim is minor or peripheral
3. For SUPPORTED claims, include the exact quote from the document that supports the claim.
4. If no explicit quote exists, the claim MUST be marked UNSUPPORTED.
5. Pay particular attention to:
   - Numbers, dates, names, and causal relationships
   - Added specificity not present in the document
   - Plausible but invented explanations

---

## Self-Check (Mandatory)
Before finalizing your answer, re-evaluate your analysis:
- Did you assume any facts not explicitly stated?
- Did you allow general knowledge without document support?
- Would a strict external auditor disagree with your classification?

If so, revise your evaluation conservatively.

---

## Output Format
Respond with valid JSON only. Do NOT include any explanatory text outside the JSON.

{
  "REASONING": [
    {
      "claim": "<atomic claim>",
      "type": "FACTUAL | INFERENTIAL | GENERAL_KNOWLEDGE | OPINION",
      "status": "SUPPORTED | UNSUPPORTED | CONTRADICTED",
      "material": true | false,
      "evidence": "<exact quote from document or null>",
      "explanation": "<brief justification>"
    }
  ],
  "HALLUCINATION_DETECTED": true | false,
  "CONFIDENCE": {
    "level": "HIGH | MEDIUM | LOW",
    "rationale": "<brief explanation>"
  },
  "SCORE": "PASS | FAIL"
}
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

    # Suppress verbose logging in worker processes
    import logging
    import os
    import sys
    os.environ["HTTPX_LOG_LEVEL"] = "ERROR"

    for logger_name in ["httpx", "httpcore", "urllib3", "huggingface_hub"]:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)
        logging.getLogger(logger_name).propagate = False
        logging.getLogger(logger_name).disabled = True

    # Monkey-patch httpx's event logger to completely silence it
    try:
        import httpx
        # Replace the logger with a null logger
        httpx._client.logger = logging.getLogger("null")
        httpx._client.logger.disabled = True
        # Also disable httpcore's trace logging
        import httpcore
        httpcore._trace.logger = logging.getLogger("null")
        httpcore._trace.logger.disabled = True
    except:
        pass

    _worker_detection_model = detection_model

    if detection_model in ("lynx", "llama"):
        import ollama
        _worker_ollama_client = ollama
        log.debug(f"Worker initialized with Ollama for {detection_model}")
    elif detection_model in ("huggingface", "hf"):
        from huggingface_hub import InferenceClient

        # Create InferenceClient
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

    # Check if client is initialized
    if _worker_hf_client is None:
        log.error(f"Row {idx}: HuggingFace client not initialized in worker process")
        return (idx, "ERROR", "LOW", "HF client not initialized", "[]", None, "")

    try:
        # Select prompt
        if document:
            prompt = HF_HALLUCINATION_PROMPT.format(
                question=question, document=document, answer=answer
            )
        else:
            prompt = HF_HALLUCINATION_PROMPT_NO_DOC.format(
                question=question, answer=answer
            )

        # Call HuggingFace API - suppress verbose output to terminal
        messages = [{"role": "user", "content": prompt}]

        # Suppress stdout/stderr at file descriptor level (httpx writes directly to FD)
        import os
        import sys

        # Save original file descriptors
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()
        saved_stdout = os.dup(stdout_fd)
        saved_stderr = os.dup(stderr_fd)

        try:
            # Redirect to /dev/null
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, stdout_fd)
            os.dup2(devnull, stderr_fd)
            os.close(devnull)

            # Make the API call with suppressed output
            response = _worker_hf_client.chat_completion(
                messages=messages,
                max_tokens=HF_MAX_NEW_TOKENS,
                temperature=0.1,
                top_p=0.95,
            )
        finally:
            # Restore original file descriptors
            os.dup2(saved_stdout, stdout_fd)
            os.dup2(saved_stderr, stderr_fd)
            os.close(saved_stdout)
            os.close(saved_stderr)

        if response and response.choices:
            raw_response = response.choices[0].message.content.strip()
        else:
            log.error(f"Row {idx}: Empty response from HuggingFace API")
            return (idx, "ERROR", "LOW", "Empty response from HuggingFace",
                   "[]", None, "")
    except Exception as e:
        log.error(f"Row {idx}: HuggingFace API call failed: {type(e).__name__}: {str(e)}")
        return (idx, "ERROR", "LOW", f"API error: {str(e)}", "[]", None, str(e))

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
            pass_count = 0
            fail_count = 0
            error_count = 0

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

                    # Track counts for summary
                    if score == 'PASS':
                        pass_count += 1
                    elif score == 'FAIL':
                        fail_count += 1
                    elif score == 'ERROR':
                        error_count += 1

                except Exception as e:
                    log.error(f"Error processing row {idx}: {e}")
                    df.at[idx, 'score'] = "ERROR"
                    df.at[idx, 'reasoning'] = f"Processing error: {e}"
                    error_count += 1

                processed_count += 1

                # Console summary every CONSOLE_LOG_INTERVAL rows
                if processed_count % CONSOLE_LOG_INTERVAL == 0:
                    elapsed = time.time() - start_time
                    rate = processed_count / (elapsed / 60)
                    print(f"\n{'='*60}")
                    print(f"Progress Update: {processed_count}/{len(work_items)} rows processed")
                    print(f"{'='*60}")
                    print(f"  PASS (faithful): {pass_count}")
                    print(f"  FAIL (hallucination): {fail_count}")
                    print(f"  ERROR: {error_count}")
                    print(f"  Processing rate: {rate:.1f} rows/min")
                    print(f"  Elapsed time: {elapsed/60:.1f} min")
                    print(f"{'='*60}\n")

                # Save progress at regular intervals
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
