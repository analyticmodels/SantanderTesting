import pandas as pd
from pathlib import Path
import json
import os
from dotenv import load_dotenv
from multiprocessing import Pool, cpu_count

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================
# Hallucination detection settings
# Options: "lynx" (Patronus Lynx - specialized), "llama" (Llama 3.2 - general purpose), or "watsonx"
DETECTION_MODEL = os.getenv("DETECTION_MODEL", "lynx")

# Lynx detection settings
LYNX_MODEL = os.getenv("LYNX_MODEL", "tensortemplar/patronus-lynx:8b-instruct-q4_k_m")

# Llama 3.2 detection settings
LLAMA_DETECTION_MODEL = os.getenv("LLAMA_DETECTION_MODEL", "llama3.2:latest")

# WatsonX detection settings
WATSONX_BASE_URL = os.getenv("WATSONX_BASE_URL", "https://apigee-outbound-dev1.nonprod.corpint.net")
WATSONX_BASIC_CREDENTIALS = os.getenv(
    "WATSONX_BASIC_CREDENTIALS",

)
WATSONX_DETECTION_MODEL = os.getenv("WATSONX_DETECTION_MODEL", "meta-llama/llama-3-3-70b-instruct")
WATSONX_MAX_TOKENS = int(os.getenv("WATSONX_MAX_TOKENS", "800"))
# =============================================================================

# Lynx prompt template for hallucination detection
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

# Llama 3.2 prompt template for hallucination detection
LLAMA_HALLUCINATION_PROMPT = """You are an expert hallucination detector for AI-generated text. Your task is to determine whether an ANSWER contains hallucinations when evaluated against a REFERENCE DOCUMENT.

## Definition of Hallucination
A hallucination occurs when the AI generates content that is:
1. **Fabricated**: Information that is made up and not present in the reference document
2. **Contradictory**: Information that directly conflicts with the reference document
3. **Unsupported**: Specific claims (names, numbers, dates, facts) that cannot be verified from the reference
4. **Confabulated**: Plausible-sounding details that fill gaps with invented specifics

## What is NOT a Hallucination
- Reasonable inferences directly supported by the document
- Paraphrasing or summarizing document content accurately
- General knowledge that doesn't conflict with the document
- Appropriate hedging language ("may", "could", "typically")

## Your Task
Analyze the ANSWER for hallucinations by checking each claim against the REFERENCE DOCUMENT.

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
1. Identify each factual claim in the ANSWER
2. For each claim, determine if it is:
   - SUPPORTED: Directly stated or clearly implied by the document
   - UNSUPPORTED: Not mentioned in the document (potential hallucination)
   - CONTRADICTED: Conflicts with the document (definite hallucination)
3. Consider the severity: Minor unsupported details vs. major fabricated facts

## Output Format
Respond with valid JSON only, no additional text:
{{
    "REASONING": [
        "Claim 1: <claim> - <SUPPORTED/UNSUPPORTED/CONTRADICTED> - <explanation>",
        "Claim 2: <claim> - <SUPPORTED/UNSUPPORTED/CONTRADICTED> - <explanation>"
    ],
    "HALLUCINATION_DETECTED": <true or false>,
    "CONFIDENCE": "<HIGH/MEDIUM/LOW>",
    "SCORE": "<PASS if no hallucination, FAIL if hallucination detected>"
}}
"""

# Llama 3.2 prompt for cases without reference document
LLAMA_HALLUCINATION_PROMPT_NO_DOC = """You are an expert hallucination detector for AI-generated text. Your task is to identify potential hallucinations in an ANSWER when NO reference document is available.

## Definition of Hallucination (Without Reference)
Without a reference document, focus on detecting:
1. **Overly Specific Fabrications**: Suspiciously precise details (exact dates, percentages, names) that seem invented
2. **False Authority Claims**: References to non-existent studies, regulations, or sources
3. **Confident Misinformation**: Stating uncertain or likely false information as definite fact
4. **Logical Inconsistencies**: Internal contradictions within the answer itself
5. **Implausible Claims**: Information that defies common knowledge or basic logic

## Indicators of Potential Hallucination
- Very specific numbers/dates without hedging (e.g., "exactly 47.3%" vs "approximately 50%")
- Named sources that cannot be easily verified
- Claims presented with high confidence on obscure or complex topics
- Technical details that seem too convenient or complete

## Your Task
Analyze the ANSWER for signs of hallucination based on internal consistency and plausibility.

---
QUESTION:
{question}

---
ANSWER TO EVALUATE:
{answer}

---
## Evaluation Instructions
1. Identify claims that appear suspiciously specific or authoritative
2. Check for internal logical consistency
3. Assess whether confidence level matches the verifiability of claims
4. Flag any claims that seem fabricated or implausible

## Output Format
Respond with valid JSON only, no additional text:
{{
    "REASONING": [
        "Observation 1: <specific concern or observation>",
        "Observation 2: <specific concern or observation>"
    ],
    "SUSPICIOUS_CLAIMS": [
        "<list any claims that appear potentially hallucinated>"
    ],
    "HALLUCINATION_DETECTED": <true or false>,
    "CONFIDENCE": "<HIGH/MEDIUM/LOW>",
    "SCORE": "<PASS if likely faithful, FAIL if hallucination likely>"
}}
"""


def detect_hallucination_lynx(question: str, answer: str, document: str = "") -> dict:
    """
    Use Patronus Lynx to detect if an answer contains hallucinations.

    Args:
        question: The question that was asked
        answer: The answer to evaluate
        document: Reference document/context (if empty, we're checking for unsupported claims)

    Returns:
        dict with 'reasoning', 'score', 'confidence', and 'raw_response'
    """
    import ollama as ollama_client

    # If no document provided, use a minimal context indicating no ground truth available
    if not document:
        document = "No reference document provided. Evaluate if the answer makes specific claims that cannot be verified."

    prompt = LYNX_PROMPT_TEMPLATE.format(
        question=question,
        document=document,
        answer=answer
    )

    response = ollama_client.chat(
        model=LYNX_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    raw_response = response['message']['content'].strip()

    # Try to parse JSON from response
    try:
        # Find JSON in response (it might have extra text)
        json_start = raw_response.find('{')
        json_end = raw_response.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = raw_response[json_start:json_end]
            parsed = json.loads(json_str)
            return {
                'reasoning': parsed.get('REASONING', ''),
                'score': parsed.get('SCORE', 'UNKNOWN'),
                'confidence': 'HIGH',  # Lynx is specialized, assume high confidence
                'raw_response': raw_response
            }
    except json.JSONDecodeError:
        pass

    # Fallback: extract score from text
    score = 'UNKNOWN'
    if 'FAIL' in raw_response.upper():
        score = 'FAIL'
    elif 'PASS' in raw_response.upper():
        score = 'PASS'

    return {
        'reasoning': raw_response,
        'score': score,
        'confidence': 'LOW',  # Couldn't parse structured response
        'raw_response': raw_response
    }


def detect_hallucination_llama(question: str, answer: str, document: str = "") -> dict:
    """
    Use Llama 3.2 to detect if an answer contains hallucinations.

    This function uses a general-purpose LLM with a carefully crafted prompt
    to identify hallucinations. It works in two modes:
    - With document: Checks if answer is faithful to the reference
    - Without document: Checks for suspicious fabrications and inconsistencies

    Args:
        question: The question that was asked
        answer: The answer to evaluate
        document: Reference document/context (optional)

    Returns:
        dict with 'reasoning', 'score', 'confidence', 'suspicious_claims', and 'raw_response'
    """
    import ollama as ollama_client

    # Select appropriate prompt based on whether document is provided
    if document and document.strip():
        prompt = LLAMA_HALLUCINATION_PROMPT.format(
            question=question,
            document=document,
            answer=answer
        )
    else:
        prompt = LLAMA_HALLUCINATION_PROMPT_NO_DOC.format(
            question=question,
            answer=answer
        )

    response = ollama_client.chat(
        model=LLAMA_DETECTION_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    raw_response = response['message']['content'].strip()

    # Try to parse JSON from response
    try:
        # Find JSON in response (it might have extra text)
        json_start = raw_response.find('{')
        json_end = raw_response.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = raw_response[json_start:json_end]
            parsed = json.loads(json_str)

            # Normalize score from various possible formats
            score = parsed.get('SCORE', 'UNKNOWN')
            if isinstance(score, bool):
                score = 'FAIL' if score else 'PASS'
            elif isinstance(score, str):
                score = score.upper()
                if score not in ('PASS', 'FAIL'):
                    # Check HALLUCINATION_DETECTED field as backup
                    if parsed.get('HALLUCINATION_DETECTED', False):
                        score = 'FAIL'
                    else:
                        score = 'PASS'

            return {
                'reasoning': parsed.get('REASONING', []),
                'score': score,
                'confidence': parsed.get('CONFIDENCE', 'MEDIUM'),
                'suspicious_claims': parsed.get('SUSPICIOUS_CLAIMS', []),
                'hallucination_detected': parsed.get('HALLUCINATION_DETECTED', score == 'FAIL'),
                'raw_response': raw_response
            }
    except json.JSONDecodeError:
        pass

    # Fallback: extract score from text using multiple indicators
    score = 'UNKNOWN'
    confidence = 'LOW'

    raw_upper = raw_response.upper()

    # Check for explicit FAIL/PASS
    if 'SCORE": "FAIL' in raw_upper or 'SCORE":"FAIL' in raw_upper:
        score = 'FAIL'
    elif 'SCORE": "PASS' in raw_upper or 'SCORE":"PASS' in raw_upper:
        score = 'PASS'
    # Check for hallucination detected indicators
    elif 'HALLUCINATION_DETECTED": TRUE' in raw_upper or 'HALLUCINATION DETECTED' in raw_upper:
        score = 'FAIL'
    elif 'HALLUCINATION_DETECTED": FALSE' in raw_upper or 'NO HALLUCINATION' in raw_upper:
        score = 'PASS'
    # Last resort: look for keywords
    elif 'FAIL' in raw_upper and 'PASS' not in raw_upper:
        score = 'FAIL'
    elif 'PASS' in raw_upper and 'FAIL' not in raw_upper:
        score = 'PASS'

    return {
        'reasoning': raw_response,
        'score': score,
        'confidence': confidence,
        'suspicious_claims': [],
        'hallucination_detected': score == 'FAIL',
        'raw_response': raw_response
    }


def detect_hallucination_watsonx(question: str, answer: str, document: str = "") -> dict:
    """
    Use WatsonX to detect if an answer contains hallucinations.

    This function uses WatsonX LLM with a carefully crafted prompt to identify hallucinations.
    It works similarly to the Llama detection but uses WatsonX API.

    Args:
        question: The question that was asked
        answer: The answer to evaluate
        document: Reference document/context (optional)

    Returns:
        dict with 'reasoning', 'score', 'confidence', 'suspicious_claims', and 'raw_response'
    """
    try:
        from watsonx import WatsonX
    except ImportError:
        raise ImportError(
            "WatsonX module not found. Please ensure the watsonx module is available."
        )

    # Use Llama prompt template (works well with WatsonX too)
    if document and document.strip():
        prompt = LLAMA_HALLUCINATION_PROMPT.format(
            question=question,
            document=document,
            answer=answer
        )
    else:
        prompt = LLAMA_HALLUCINATION_PROMPT_NO_DOC.format(
            question=question,
            answer=answer
        )

    # Initialize WatsonX client
    watsonx_client = WatsonX()
    oauth_url = f"{WATSONX_BASE_URL}/oauth2/accesstoken-clientcredentials"
    token = watsonx_client.post_oauth2(WATSONX_BASIC_CREDENTIALS, oauth_url)

    # Make API call with retry logic for token expiration
    try:
        generated_text, _, _ = watsonx_client.post_text_generation(
            WATSONX_BASE_URL,
            token,
            WATSONX_DETECTION_MODEL,
            prompt,
            WATSONX_MAX_TOKENS
        )
    except Exception as e:
        # Check if it's a 401 error (token expired)
        if hasattr(e, 'response') and hasattr(e.response, 'status_code') and e.response.status_code == 401:
            # Refresh token and retry
            token = watsonx_client.post_oauth2(WATSONX_BASIC_CREDENTIALS, oauth_url)
            generated_text, _, _ = watsonx_client.post_text_generation(
                WATSONX_BASE_URL,
                token,
                WATSONX_DETECTION_MODEL,
                prompt,
                WATSONX_MAX_TOKENS
            )
        else:
            raise

    raw_response = generated_text.strip()

    # Try to parse JSON from response
    try:
        # Find JSON in response (it might have extra text)
        json_start = raw_response.find('{')
        json_end = raw_response.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = raw_response[json_start:json_end]
            parsed = json.loads(json_str)

            # Normalize score from various possible formats
            score = parsed.get('SCORE', 'UNKNOWN')
            if isinstance(score, bool):
                score = 'FAIL' if score else 'PASS'
            elif isinstance(score, str):
                score = score.upper()
                if score not in ('PASS', 'FAIL'):
                    # Check HALLUCINATION_DETECTED field as backup
                    if parsed.get('HALLUCINATION_DETECTED', False):
                        score = 'FAIL'
                    else:
                        score = 'PASS'

            return {
                'reasoning': parsed.get('REASONING', []),
                'score': score,
                'confidence': parsed.get('CONFIDENCE', 'MEDIUM'),
                'suspicious_claims': parsed.get('SUSPICIOUS_CLAIMS', []),
                'hallucination_detected': parsed.get('HALLUCINATION_DETECTED', score == 'FAIL'),
                'raw_response': raw_response
            }
    except json.JSONDecodeError:
        pass

    # Fallback: extract score from text using multiple indicators
    score = 'UNKNOWN'
    confidence = 'LOW'

    raw_upper = raw_response.upper()

    # Check for explicit FAIL/PASS
    if 'SCORE": "FAIL' in raw_upper or 'SCORE":"FAIL' in raw_upper:
        score = 'FAIL'
    elif 'SCORE": "PASS' in raw_upper or 'SCORE":"PASS' in raw_upper:
        score = 'PASS'
    # Check for hallucination detected indicators
    elif 'HALLUCINATION_DETECTED": TRUE' in raw_upper or 'HALLUCINATION DETECTED' in raw_upper:
        score = 'FAIL'
    elif 'HALLUCINATION_DETECTED": FALSE' in raw_upper or 'NO HALLUCINATION' in raw_upper:
        score = 'PASS'
    # Last resort: look for keywords
    elif 'FAIL' in raw_upper and 'PASS' not in raw_upper:
        score = 'FAIL'
    elif 'PASS' in raw_upper and 'FAIL' not in raw_upper:
        score = 'PASS'

    return {
        'reasoning': raw_response,
        'score': score,
        'confidence': confidence,
        'suspicious_claims': [],
        'hallucination_detected': score == 'FAIL',
        'raw_response': raw_response
    }


def _process_qa_pair(args):
    """
    Worker function to evaluate a single question-answer pair for hallucinations.

    This function is designed to be called by multiprocessing.Pool.

    Args:
        args: Tuple of (index, question, answer, reference_doc, detection_model)

    Returns:
        Tuple of (index, result_entry_dict)
    """
    index, question, answer, reference_doc, detection_model = args

    # Run hallucination detection
    result = detect_hallucination(
        question=question,
        answer=answer,
        document=reference_doc,
        model=detection_model
    )

    # Normalize reasoning to string (can be list from JSON or string from fallback)
    reasoning = result['reasoning']
    if isinstance(reasoning, list):
        reasoning = json.dumps(reasoning, indent=2)

    result_entry = {
        'question': question,
        'answer': answer,
        'hallucination_score': result['score'],
        'confidence': result.get('confidence', 'N/A'),
        'reasoning': reasoning,
        'raw_response': result['raw_response'],
        'detection_model': detection_model
    }

    # Add model-specific fields if available
    if detection_model in ("llama", "watsonx"):
        suspicious = result.get('suspicious_claims', [])
        # Convert list to JSON string for parquet compatibility
        if isinstance(suspicious, list):
            suspicious = json.dumps(suspicious)
        result_entry['suspicious_claims'] = suspicious
        result_entry['hallucination_detected'] = result.get('hallucination_detected', False)

    return (index, result_entry)


def detect_hallucination(question: str, answer: str, document: str = "", model: str = None) -> dict:
    """
    Detect if an answer contains hallucinations using the configured detection model.

    This is the main entry point for hallucination detection. It dispatches to either
    Patronus Lynx (specialized), Llama 3.2 (general purpose), or WatsonX based on configuration.

    Args:
        question: The question that was asked
        answer: The answer to evaluate
        document: Reference document/context (optional)
        model: Detection model to use ("lynx", "llama", or "watsonx"). If None, uses DETECTION_MODEL config.

    Returns:
        dict with 'reasoning', 'score', 'confidence', and 'raw_response'
    """
    # Use configured model if not specified
    if model is None:
        model = DETECTION_MODEL

    model = model.lower()

    if model == "lynx":
        return detect_hallucination_lynx(question, answer, document)
    elif model == "llama":
        return detect_hallucination_llama(question, answer, document)
    elif model == "watsonx":
        return detect_hallucination_watsonx(question, answer, document)
    else:
        raise ValueError(f"Unknown detection model: {model}. Use 'lynx', 'llama', or 'watsonx'.")


def run_hallucination_detection(input_file: Path = None, output_file: Path = None,
                                limit: int = None, detection_model: str = None,
                                document: str = "", num_workers: int = None):
    """
    Run hallucination detection on question-answer pairs.

    Args:
        input_file: Path to parquet file with 'question' and 'answer' columns.
                   If None, tries to auto-detect based on common filenames.
        output_file: Path to save detection results.
                    If None, auto-generates based on detection model.
        limit: Optional limit on number of pairs to process (for testing)
        detection_model: Model to use for detection ("lynx", "llama", or "watsonx").
                        If None, uses DETECTION_MODEL from config.
        document: Optional reference document for all Q&A pairs.
                 Can also be a path to a text file.
        num_workers: Number of parallel workers for multiprocessing.
                    Defaults to cpu_count(). Set to 1 to disable multiprocessing.
    """
    # Use default for num_workers if not specified
    if num_workers is None:
        num_workers = cpu_count()
    # Use configured model if not specified
    if detection_model is None:
        detection_model = DETECTION_MODEL

    print(f"Using detection model: {detection_model}")
    if detection_model == "llama":
        print(f"  Llama model: {LLAMA_DETECTION_MODEL}")
    elif detection_model == "watsonx":
        print(f"  WatsonX model: {WATSONX_DETECTION_MODEL}")
        print(f"  Base URL: {WATSONX_BASE_URL}")
    else:  # lynx
        print(f"  Lynx model: {LYNX_MODEL}")

    # Auto-detect input file if not specified
    if input_file is None:
        # Try common filenames
        candidates = [
            Path('../data/hallucinationQA_llama3.2_latest.parquet'),
            Path('../data/hallucinationQA_anthropic.parquet'),
            Path('../data/hallucinationQuestions_anthropic.parquet'),
        ]
        for candidate in candidates:
            if candidate.exists():
                input_file = candidate
                break

        if input_file is None:
            raise ValueError("Could not auto-detect input file. Please specify with --input-file")

    # Load reference document if provided as a file path
    reference_doc = document
    if document and Path(document).exists():
        try:
            with open(document, 'r', encoding='utf-8') as f:
                reference_doc = f.read()
            print(f"Loaded reference document from: {document} ({len(reference_doc)} characters)")
        except Exception as e:
            print(f"Warning: Could not read document file: {e}")
            reference_doc = document

    df = pd.read_parquet(input_file, engine='fastparquet')

    if 'question' not in df.columns or 'answer' not in df.columns:
        raise ValueError(f"Input file must contain 'question' and 'answer' columns. Found: {df.columns.tolist()}")

    if limit:
        df = df[:limit]

    print(f"Loaded {len(df)} question-answer pairs from {input_file}")
    print(f"Using {num_workers} worker(s) for parallel processing")

    # Prepare arguments for each worker
    worker_args = [
        (i, row['question'], row['answer'], reference_doc, detection_model)
        for i, row in df.iterrows()
    ]

    # Process Q&A pairs in parallel or sequentially
    if num_workers > 1:
        print(f"Processing {len(df)} Q&A pairs in parallel...")
        with Pool(num_workers) as pool:
            # Use imap for progress tracking
            results_list = []
            for i, result in enumerate(pool.imap(_process_qa_pair, worker_args), 1):
                results_list.append(result)
                if i % 10 == 0:
                    print(f"Processed {i}/{len(df)} pairs")
    else:
        print(f"Processing {len(df)} Q&A pairs sequentially...")
        results_list = []
        for i, arg in enumerate(worker_args, 1):
            result = _process_qa_pair(arg)
            results_list.append(result)
            if i % 10 == 0:
                print(f"Processed {i}/{len(df)} pairs")

    # Sort results by index and collect all result entries
    results_list.sort(key=lambda x: x[0])
    results = [result_entry for (_idx, result_entry) in results_list]

    # Create results DataFrame
    df_results = pd.DataFrame(results)

    # Auto-generate output filename if not specified
    if output_file is None:
        output_file = Path(f'../data/hallucinationDetection_{detection_model}.parquet')

    # Save results
    df_results.to_parquet(output_file, engine='fastparquet', index=False)

    # Summary statistics
    pass_count = (df_results['hallucination_score'] == 'PASS').sum()
    fail_count = (df_results['hallucination_score'] == 'FAIL').sum()
    unknown_count = (df_results['hallucination_score'] == 'UNKNOWN').sum()

    print(f"\n=== Hallucination Detection Results ===")
    print(f"Total pairs evaluated: {len(df_results)}")
    print(f"PASS (faithful): {pass_count} ({100*pass_count/len(df_results):.1f}%)")
    print(f"FAIL (hallucination detected): {fail_count} ({100*fail_count/len(df_results):.1f}%)")
    print(f"UNKNOWN: {unknown_count} ({100*unknown_count/len(df_results):.1f}%)")
    print(f"\nSaved to: {output_file}")

    return df_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate question-answer pairs for hallucinations")
    parser.add_argument("--input-file", "-i", type=str, default=None,
                        help="Input parquet file with Q&A pairs (default: auto-detect)")
    parser.add_argument("--output-file", "-o", type=str, default=None,
                        help="Output parquet file for detection results (default: auto-generate)")
    parser.add_argument("--detection-model", "-d", choices=["lynx", "llama", "watsonx"], default=None,
                        help="Hallucination detection model: 'lynx', 'llama', or 'watsonx' (default: from DETECTION_MODEL env/config)")
    parser.add_argument("--limit", "-l", type=int, default=None,
                        help="Limit number of Q&A pairs to process (for testing)")
    parser.add_argument("--workers", "-w", type=int, default=None,
                        help=f"Number of parallel workers (default: {cpu_count()}). Set to 1 to disable multiprocessing")
    parser.add_argument("--document", "-c", type=str, default="",
                        help="Reference document text or path to document file")

    args = parser.parse_args()

    # Convert string paths to Path objects if provided
    input_path = Path(args.input_file) if args.input_file else None
    output_path = Path(args.output_file) if args.output_file else None

    # Run hallucination detection
    df_results = run_hallucination_detection(
        input_file=input_path,
        output_file=output_path,
        limit=args.limit,
        detection_model=args.detection_model,
        document=args.document,
        num_workers=args.workers
    )

    print(f"\n✓ Successfully completed hallucination detection")
