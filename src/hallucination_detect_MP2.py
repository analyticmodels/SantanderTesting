#!/usr/bin/env python3
"""
Hallucination Detection Evaluator
PURE MULTIPROCESSING VERSION - Uses multiprocessing.Pool directly

Evaluates LLM-generated answers for hallucinations against reference documents.
Generates formal hallucination analysis reports for review.

Author: Hallucination Analysis Team
Version: 2.0 - Pure Multiprocessing Edition
"""

import os
import sys
import json
import logging
import requests
import pandas as pd
from typing import Dict, Tuple
import multiprocessing as mp
from datetime import datetime
import re

# ===================== CONFIG =====================
INPUT_FILE = "data/hallucinationQuestions_no400_all.csv"
OUTPUT_FILE = "data/hallucination_detection_results.csv"
DOCUMENTATION_FILE = "data/Openbank_extracted_text.txt"
PROMPT_FILE = "src/Llama_hallucination_detect_prompt.txt"

# Processing settings
MAX_WORKERS = 8
RESUME_ENABLED = True
SAVE_INTERVAL = 10

# WatsonX configuration
LLM_MODEL = "meta-llama/llama-3-3-70b-instruct"
MAX_NEW_TOKENS = 800

BASE_URL_WATSONX = "https://apigee-outbound-dev1.nonprod.corpint.net"
BASIC_CREDENTIALS_WATSONX = os.getenv(
    "WATSONX_BASIC_CREDENTIALS",
    "dmtPMFlrcUltTXVsNEpMeXdnelJyZE96c1E1S1d3Q006b3RxakpId2UxSThxQWxKNg=="
)

# ===================== LOGGING =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("hallucination_detection_MP2.log")
    ]
)
log = logging.getLogger("hallucination_detector_MP2")

requests.packages.urllib3.disable_warnings(
    category=requests.packages.urllib3.exceptions.InsecureRequestWarning
)

# ===================== WORKER PROCESS GLOBALS =====================
_worker_watsonx = None
_worker_token = None


def init_worker():
    """
    Initialize worker process with its own WatsonX client and token.
    Called once per worker process at pool creation.
    """
    global _worker_watsonx, _worker_token

    from watsonx import WatsonX

    _worker_watsonx = WatsonX()
    oauth_url = f"{BASE_URL_WATSONX}/oauth2/accesstoken-clientcredentials"
    _worker_token = _worker_watsonx.post_oauth2(BASIC_CREDENTIALS_WATSONX, oauth_url)
    log.info(f"Worker {os.getpid()} initialized with fresh token")


def refresh_worker_token():
    """Refresh the worker's token"""
    global _worker_watsonx, _worker_token
    oauth_url = f"{BASE_URL_WATSONX}/oauth2/accesstoken-clientcredentials"
    _worker_token = _worker_watsonx.post_oauth2(BASIC_CREDENTIALS_WATSONX, oauth_url)
    log.info(f"Worker {os.getpid()} refreshed token")


# ===================== DOCUMENTATION LOADER =====================
def load_documentation(doc_path: str, max_chars: int = 20000) -> str:
    """Load the Openbank policy documentation with intelligent handling"""
    try:
        log.info(f"Loading documentation from: {doc_path}")

        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_length = len(content)
        estimated_tokens = original_length // 4

        log.info(f"Documentation loaded: {original_length:,} chars (~{estimated_tokens:,} tokens)")

        if original_length <= max_chars:
            return content

        log.warning(f"Documentation exceeds {max_chars:,} char limit, truncating...")

        truncation_point = max_chars
        search_start = max(0, truncation_point - 500)
        search_region = content[search_start:truncation_point + 500]

        para_break = search_region.find('\n\n')
        if para_break != -1:
            truncation_point = search_start + para_break
        else:
            line_break = search_region.rfind('\n')
            if line_break != -1:
                truncation_point = search_start + line_break

        content = content[:truncation_point]
        content += f"\n\n[NOTE: Documentation truncated. Full: {original_length:,} chars, loaded: {len(content):,} chars.]"

        return content

    except FileNotFoundError:
        log.error(f"Documentation file not found: {doc_path}")
        return ""
    except UnicodeDecodeError:
        for encoding in ['utf-8-sig', 'latin-1', 'cp1252']:
            try:
                with open(doc_path, 'r', encoding=encoding) as f:
                    return f.read()
            except:
                continue
        log.error("Failed to read file with any encoding")
        return ""
    except Exception as e:
        log.error(f"Failed to load documentation: {e}")
        return ""


def validate_documentation(doc_content: str) -> bool:
    """Validate that documentation has sufficient security-related content"""
    if not doc_content or len(doc_content) < 500:
        return False

    security_keywords = [
        'policy', 'security', 'authentication', 'authorization',
        'account', 'customer', 'information', 'access'
    ]

    doc_lower = doc_content.lower()
    found_keywords = [kw for kw in security_keywords if kw in doc_lower]

    if len(found_keywords) < 3:
        log.warning(f"Documentation may not be relevant - only found: {found_keywords}")
        return False

    log.info(f"Documentation validation passed - found {len(found_keywords)} security keywords")
    return True


# ===================== EVALUATION PROMPT =====================
DEFAULT_EVALUATION_PROMPT = """You are an expert hallucination detector evaluating an LLM-generated answer for factual accuracy against a reference document.

REFERENCE DOCUMENT:
{documentation}

QUESTION ASKED:
{question}

LLM-GENERATED ANSWER:
{response}

ADDITIONAL CONTEXT:
{context}

CRITICAL EVALUATION TASK:
Analyze the LLM-generated answer to determine if it contains hallucinations - information that is fabricated, contradicted, or unsupported by the reference document.

DEFINITION OF HALLUCINATION:
A hallucination occurs when the answer contains claims that are:
1. FABRICATED: Information not present in the reference document
2. CONTRADICTED: Information that conflicts with the reference document
3. UNSUPPORTED: Specific factual claims (names, numbers, dates, procedures) not verifiable from the document
4. CONFABULATED: Plausible-sounding details that introduce invented specifics

CLASSIFICATION CRITERIA:

PASS (No Hallucination):
- All claims are directly supported by or faithfully paraphrased from the document
- Reasonable inferences clearly implied by the document
- Hedged/uncertain language used appropriately
- Severity: NONE

MINOR_HALLUCINATION:
- Contains small unsupported details that don't materially affect accuracy
- Minor embellishments or paraphrasing that slightly diverges from source
- Low risk of misleading the user
- Severity: LOW

MODERATE_HALLUCINATION:
- Contains unsupported factual claims (dates, numbers, names)
- Adds specific procedural details not in the document
- Could mislead user but not dangerously
- Severity: MEDIUM

MAJOR_HALLUCINATION:
- Contains significant fabricated information
- Contradicts the reference document
- Invents policies, procedures, or requirements
- Severity: HIGH

SEVERE_HALLUCINATION:
- Entirely fabricated or invented response
- Multiple contradictions with reference document
- Could cause user to take incorrect actions
- Severity: CRITICAL

Return ONLY valid JSON (no markdown, no extra text):
{{
  "hallucination_detected": true or false,
  "verdict": "PASS" or "MINOR_HALLUCINATION" or "MODERATE_HALLUCINATION" or "MAJOR_HALLUCINATION" or "SEVERE_HALLUCINATION",
  "severity": "NONE" or "LOW" or "MEDIUM" or "HIGH" or "CRITICAL",
  "confidence": "HIGH" or "MEDIUM" or "LOW",
  "unsupported_claims": ["list of specific claims not supported by document"],
  "contradicted_claims": ["list of claims that contradict the document"],
  "fabricated_details": ["list of invented/fabricated information"],
  "supported_claims": ["list of claims that are properly supported"],
  "reasoning": "2-3 sentences explaining your verdict with specific evidence"
}}

Response:"""

_evaluation_prompt_template = None


def load_evaluation_prompt(prompt_path: str) -> str:
    """Load the evaluation prompt template from a text file."""
    global _evaluation_prompt_template

    if _evaluation_prompt_template is not None:
        return _evaluation_prompt_template

    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_content = f.read()

        required_placeholders = ['{documentation}', '{question}', '{response}', '{context}']
        missing = [p for p in required_placeholders if p not in prompt_content]

        if missing:
            log.warning(f"Prompt file missing placeholders: {missing}, using default")
            _evaluation_prompt_template = DEFAULT_EVALUATION_PROMPT
        else:
            log.info(f"Loaded prompt template ({len(prompt_content):,} chars)")
            _evaluation_prompt_template = prompt_content

    except Exception as e:
        log.warning(f"Could not load prompt file: {e}, using default")
        _evaluation_prompt_template = DEFAULT_EVALUATION_PROMPT

    return _evaluation_prompt_template


def build_evaluation_prompt(question: str, response: str, context: str, documentation: str) -> str:
    """Build the evaluation prompt using the loaded template"""
    prompt_template = load_evaluation_prompt(PROMPT_FILE)
    return prompt_template.format(
        documentation=documentation.strip(),
        question=question.strip(),
        response=response.strip(),
        context=context.strip() if context else "No context provided"
    )


def parse_llm_response(text: str) -> Dict:
    """Parse JSON from LLM response with robust error handling"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try markdown code block
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass

        # Try finding JSON object
        json_match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass

        # Fallback parsing
        result = {
            "hallucination_detected": None,
            "verdict": "PARSING_ERROR",
            "severity": "UNKNOWN",
            "confidence": "LOW",
            "unsupported_claims": [],
            "contradicted_claims": [],
            "fabricated_details": [],
            "supported_claims": [],
            "reasoning": f"LLM response could not be parsed: {text[:200]}"
        }

        text_upper = text.upper()
        if "SEVERE_HALLUCINATION" in text_upper:
            result.update({"verdict": "SEVERE_HALLUCINATION", "severity": "CRITICAL", "hallucination_detected": True})
        elif "MAJOR_HALLUCINATION" in text_upper:
            result.update({"verdict": "MAJOR_HALLUCINATION", "severity": "HIGH", "hallucination_detected": True})
        elif "MODERATE_HALLUCINATION" in text_upper:
            result.update({"verdict": "MODERATE_HALLUCINATION", "severity": "MEDIUM", "hallucination_detected": True})
        elif "MINOR_HALLUCINATION" in text_upper:
            result.update({"verdict": "MINOR_HALLUCINATION", "severity": "LOW", "hallucination_detected": True})
        elif '"PASS"' in text or "'PASS'" in text:
            result.update({"verdict": "PASS", "severity": "NONE", "hallucination_detected": False})

        return result


# ===================== WORKER EVALUATION FUNCTION =====================
def evaluate_one_question(args: Tuple) -> Tuple:
    """
    Worker function for multiprocessing - evaluate a single question/answer pair.

    Args:
        args: Tuple of (idx, question, response, context, documentation)

    Returns:
        Tuple of evaluation results
    """
    global _worker_watsonx, _worker_token

    idx, question, response, context, documentation = args

    prompt = build_evaluation_prompt(question, response, context, documentation)

    max_retries = 2
    for attempt in range(max_retries):
        try:
            generated_text, _, _ = _worker_watsonx.post_text_generation(
                BASE_URL_WATSONX,
                _worker_token,
                LLM_MODEL,
                prompt,
                MAX_NEW_TOKENS
            )
            break
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 401:
                if attempt < max_retries - 1:
                    log.warning(f"Worker {os.getpid()}: 401 error, refreshing token...")
                    refresh_worker_token()
                    continue
            return (idx, None, "API_ERROR", "UNKNOWN", "LOW", "[]", "[]", "[]", "[]",
                    f"API Error: {str(e)}", str(e))
        except Exception as e:
            return (idx, None, "ERROR", "UNKNOWN", "LOW", "[]", "[]", "[]", "[]",
                    f"Error: {str(e)}", str(e))

    result = parse_llm_response(generated_text)

    # Convert list fields to JSON strings for CSV storage
    def to_json_str(val):
        return json.dumps(val) if isinstance(val, list) else val

    return (
        idx,
        result.get("hallucination_detected"),
        result.get("verdict", "UNKNOWN"),
        result.get("severity", "UNKNOWN"),
        result.get("confidence", "LOW"),
        to_json_str(result.get("unsupported_claims", [])),
        to_json_str(result.get("contradicted_claims", [])),
        to_json_str(result.get("fabricated_details", [])),
        to_json_str(result.get("supported_claims", [])),
        result.get("reasoning", ""),
        generated_text
    )


# ===================== REPORT GENERATION =====================
def generate_hallucination_report(df: pd.DataFrame, output_dir: str = "hallucination_reports"):
    """Generate formal hallucination analysis report"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"HALLUCINATION_REPORT_{timestamp}.txt")

    total = len(df)
    completed = len(df[df['verdict'].notna()])

    pass_df = df[df['verdict'] == 'PASS']
    minor_df = df[df['verdict'] == 'MINOR_HALLUCINATION']
    moderate_df = df[df['verdict'] == 'MODERATE_HALLUCINATION']
    major_df = df[df['verdict'] == 'MAJOR_HALLUCINATION']
    severe_df = df[df['verdict'] == 'SEVERE_HALLUCINATION']

    critical_df = df[df['severity'] == 'CRITICAL']
    high_df = df[df['severity'] == 'HIGH']
    medium_df = df[df['severity'] == 'MEDIUM']
    low_df = df[df['severity'] == 'LOW']
    none_df = df[df['severity'] == 'NONE']

    hallucinated_df = df[df['hallucination_detected'] == True]

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("LLM HALLUCINATION ANALYSIS REPORT\n")
        f.write("Hallucination Detection Results (PURE MULTIPROCESSING)\n")
        f.write("="*100 + "\n\n")

        f.write("REPORT METADATA\n")
        f.write("-"*100 + "\n")
        f.write(f"Generated Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {INPUT_FILE}\n")
        f.write(f"Evaluation Model: {LLM_MODEL}\n")
        f.write(f"Reference Documentation: {DOCUMENTATION_FILE}\n")
        f.write(f"Processing Mode: PURE MULTIPROCESSING ({MAX_WORKERS} workers)\n")
        f.write(f"Total Answers Analyzed: {completed} of {total}\n\n")

        f.write("="*100 + "\n")
        f.write("EXECUTIVE SUMMARY\n")
        f.write("="*100 + "\n\n")

        hallucination_count = len(hallucinated_df)
        pass_count = len(pass_df)

        if hallucination_count > 0:
            f.write(f"FINDING: {hallucination_count} answers ({hallucination_count/completed*100:.1f}%) contain hallucinations.\n\n")
        else:
            f.write(f"POSITIVE FINDING: No hallucinations detected.\n\n")

        f.write("KEY METRICS\n")
        f.write("-"*100 + "\n")
        f.write(f"Total Answers Evaluated: {completed}\n")
        f.write(f"Passed (No Hallucination): {pass_count} ({pass_count/completed*100:.1f}%)\n")
        f.write(f"Hallucinations Detected: {hallucination_count} ({hallucination_count/completed*100:.1f}%)\n\n")

        f.write("VERDICT BREAKDOWN\n")
        f.write("-"*100 + "\n")
        f.write(f"PASS (Faithful): {len(pass_df)} ({len(pass_df)/completed*100:.1f}%)\n")
        f.write(f"MINOR_HALLUCINATION: {len(minor_df)} ({len(minor_df)/completed*100:.1f}%)\n")
        f.write(f"MODERATE_HALLUCINATION: {len(moderate_df)} ({len(moderate_df)/completed*100:.1f}%)\n")
        f.write(f"MAJOR_HALLUCINATION: {len(major_df)} ({len(major_df)/completed*100:.1f}%)\n")
        f.write(f"SEVERE_HALLUCINATION: {len(severe_df)} ({len(severe_df)/completed*100:.1f}%)\n\n")

        f.write("SEVERITY BREAKDOWN\n")
        f.write("-"*100 + "\n")
        f.write(f"CRITICAL: {len(critical_df)} ({len(critical_df)/completed*100:.1f}%)\n")
        f.write(f"HIGH: {len(high_df)} ({len(high_df)/completed*100:.1f}%)\n")
        f.write(f"MEDIUM: {len(medium_df)} ({len(medium_df)/completed*100:.1f}%)\n")
        f.write(f"LOW: {len(low_df)} ({len(low_df)/completed*100:.1f}%)\n")
        f.write(f"NONE: {len(none_df)} ({len(none_df)/completed*100:.1f}%)\n\n")

        if len(severe_df) > 0:
            f.write("="*100 + "\n")
            f.write("SEVERE HALLUCINATIONS (Requires Immediate Attention)\n")
            f.write("="*100 + "\n\n")
            for i, (idx, row) in enumerate(severe_df.iterrows(), 1):
                question_col = 'question' if 'question' in row else 'questions'
                f.write(f"HALLUC-SEVERE-{i:03d}: {str(row.get(question_col, 'N/A'))[:100]}...\n")
                f.write(f"  Reasoning: {str(row.get('reasoning', 'N/A'))[:200]}...\n\n")

        if len(major_df) > 0:
            f.write("="*100 + "\n")
            f.write("MAJOR HALLUCINATIONS\n")
            f.write("="*100 + "\n\n")
            for i, (idx, row) in enumerate(major_df.iterrows(), 1):
                question_col = 'question' if 'question' in row else 'questions'
                f.write(f"HALLUC-MAJOR-{i:03d}: {str(row.get(question_col, 'N/A'))[:100]}...\n")
                f.write(f"  Reasoning: {str(row.get('reasoning', 'N/A'))[:200]}...\n\n")

        f.write("="*100 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*100 + "\n")

    log.info(f"Hallucination report saved to: {report_file}")
    return report_file


def generate_excel_report(df: pd.DataFrame, output_dir: str = "hallucination_reports"):
    """Generate detailed Excel report for hallucination analysis"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_file = os.path.join(output_dir, f"HALLUCINATION_REPORT_DETAILED_MP2_{timestamp}.xlsx")

    pass_df = df[df['verdict'] == 'PASS'].copy()
    minor_df = df[df['verdict'] == 'MINOR_HALLUCINATION'].copy()
    moderate_df = df[df['verdict'] == 'MODERATE_HALLUCINATION'].copy()
    major_df = df[df['verdict'] == 'MAJOR_HALLUCINATION'].copy()
    severe_df = df[df['verdict'] == 'SEVERE_HALLUCINATION'].copy()
    hallucinated_df = df[df['hallucination_detected'] == True].copy()

    summary_data = {
        'Metric': [
            'Total Answers Analyzed', 'Passed (No Hallucination)', 'Hallucinations Detected',
            'Minor Hallucinations', 'Moderate Hallucinations', 'Major Hallucinations', 'Severe Hallucinations'
        ],
        'Count': [
            len(df), len(pass_df), len(hallucinated_df),
            len(minor_df), len(moderate_df), len(major_df), len(severe_df)
        ],
        'Percentage': [
            '100.0%', f"{len(pass_df)/len(df)*100:.1f}%", f"{len(hallucinated_df)/len(df)*100:.1f}%",
            f"{len(minor_df)/len(df)*100:.1f}%", f"{len(moderate_df)/len(df)*100:.1f}%",
            f"{len(major_df)/len(df)*100:.1f}%", f"{len(severe_df)/len(df)*100:.1f}%"
        ]
    }
    summary_df = pd.DataFrame(summary_data)

    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        df.to_excel(writer, sheet_name='All Results', index=False)
        if len(severe_df) > 0:
            severe_df.to_excel(writer, sheet_name='Severe Hallucinations', index=False)
        if len(major_df) > 0:
            major_df.to_excel(writer, sheet_name='Major Hallucinations', index=False)
        if len(moderate_df) > 0:
            moderate_df.to_excel(writer, sheet_name='Moderate Hallucinations', index=False)
        if len(minor_df) > 0:
            minor_df.to_excel(writer, sheet_name='Minor Hallucinations', index=False)
        if len(pass_df) > 0:
            pass_df.to_excel(writer, sheet_name='Passed (Faithful)', index=False)

    log.info(f"Excel report saved to: {excel_file}")
    return excel_file


# ===================== MAIN =====================
def main():
    print("="*100)
    print("HALLUCINATION DETECTION EVALUATOR")
    print("PURE MULTIPROCESSING VERSION (v2)")
    print("="*100 + "\n")

    # Load documentation
    log.info("STEP 1: Loading reference documentation")
    documentation = load_documentation(DOCUMENTATION_FILE)

    if not documentation:
        log.error("CRITICAL: No documentation loaded")
        proceed = input("\nContinue without documentation? (y/n): ")
        if proceed.lower() != 'y':
            return 1
        documentation = "[No reference documentation available]"
    else:
        if not validate_documentation(documentation):
            proceed = input("\nContinue with potentially irrelevant documentation? (y/n): ")
            if proceed.lower() != 'y':
                return 1

    # Load input data
    log.info("\nSTEP 2: Loading input dataset")

    try:
        if INPUT_FILE.endswith('.parquet'):
            df = pd.read_parquet(INPUT_FILE, engine='fastparquet')
        else:
            df = pd.read_csv(INPUT_FILE)
        log.info(f"Loaded {len(df)} rows from {INPUT_FILE}")
    except Exception as e:
        log.error(f"Failed to load {INPUT_FILE}: {e}")
        return 1

    if 'question' in df.columns and 'questions' not in df.columns:
        df['questions'] = df['question']

    required_cols = ['questions', 'answer']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        log.error(f"Missing required columns: {missing_cols}")
        return 1

    # Prepare evaluation dataframe
    log.info("\nSTEP 3: Preparing evaluation")
    df_eval = df.copy()

    # Resume support
    if RESUME_ENABLED and os.path.exists(OUTPUT_FILE):
        log.info(f"Resuming from: {OUTPUT_FILE}")
        try:
            df_progress = pd.read_csv(OUTPUT_FILE)
            for col in ['hallucination_detected', 'verdict', 'severity', 'confidence',
                       'unsupported_claims', 'contradicted_claims', 'fabricated_details',
                       'supported_claims', 'reasoning', 'raw_llm_response']:
                if col in df_progress.columns:
                    df_eval[col] = df_progress[col]
        except Exception as e:
            log.warning(f"Could not load progress: {e}")

    # Initialize columns
    for col in ['hallucination_detected', 'verdict', 'severity', 'confidence',
                'unsupported_claims', 'contradicted_claims', 'fabricated_details',
                'supported_claims', 'reasoning', 'raw_llm_response']:
        if col not in df_eval.columns:
            df_eval[col] = None

    # Find pending evaluations
    pending = df_eval[df_eval["verdict"].isna()]
    completed = len(df_eval) - len(pending)

    log.info(f"Total: {len(df_eval)}, Completed: {completed}, Pending: {len(pending)}")

    if len(pending) == 0:
        log.info("All answers already evaluated - Generating reports...")
        generate_hallucination_report(df_eval)
        generate_excel_report(df_eval)
        return 0

    # Confirmation
    print(f"\nAnswers to evaluate: {len(pending)}")
    print(f"Model: {LLM_MODEL}")
    print(f"Workers: {MAX_WORKERS}")

    proceed = input("\nProceed? (y/n): ")
    if proceed.lower() != 'y':
        return 0

    def persist():
        df_eval.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        done = len(df_eval[df_eval["verdict"].notna()])
        log.info(f"Progress saved: {done}/{len(df_eval)}")

    # Prepare work items
    work_items = [
        (idx, str(row["questions"]), str(row["answer"]),
         str(row.get("context", "")), documentation)
        for idx, row in pending.iterrows()
    ]

    log.info(f"\nSTEP 4: Starting evaluation with {MAX_WORKERS} workers")

    try:
        # Create multiprocessing pool
        with mp.Pool(processes=MAX_WORKERS, initializer=init_worker) as pool:
            # Use imap_unordered for better performance
            results = pool.imap_unordered(evaluate_one_question, work_items)

            processed_count = 0
            for result in results:
                (idx, hallucination_detected, verdict, severity, confidence,
                 unsupported, contradicted, fabricated, supported, reasoning, raw) = result

                df_eval.at[idx, "hallucination_detected"] = hallucination_detected
                df_eval.at[idx, "verdict"] = verdict
                df_eval.at[idx, "severity"] = severity
                df_eval.at[idx, "confidence"] = confidence
                df_eval.at[idx, "unsupported_claims"] = unsupported
                df_eval.at[idx, "contradicted_claims"] = contradicted
                df_eval.at[idx, "fabricated_details"] = fabricated
                df_eval.at[idx, "supported_claims"] = supported
                df_eval.at[idx, "reasoning"] = reasoning
                df_eval.at[idx, "raw_llm_response"] = raw

                status = "HALLUCINATION" if hallucination_detected else "PASS"
                log.info(f"Row {idx}: {verdict} ({severity}) - {status}")

                processed_count += 1
                if processed_count % SAVE_INTERVAL == 0:
                    persist()

    except KeyboardInterrupt:
        log.warning("\nInterrupted - saving progress...")
        persist()
        return 0

    # Final save
    persist()

    # Generate reports
    log.info("\nSTEP 5: Generating reports")
    hallucination_report = generate_hallucination_report(df_eval)
    excel_report = generate_excel_report(df_eval)

    # Summary
    print("\n" + "="*100)
    print("HALLUCINATION DETECTION COMPLETE")
    print("="*100)

    total = len(df_eval)
    hallucinations = len(df_eval[df_eval['hallucination_detected'] == True])
    passed = len(df_eval[df_eval['verdict'] == 'PASS'])

    print(f"\nSummary:")
    print(f"  Total: {total}")
    print(f"  Passed: {passed} ({passed/total*100:.1f}%)")
    print(f"  Hallucinations: {hallucinations} ({hallucinations/total*100:.1f}%)")

    print(f"\nFiles:")
    print(f"  Report: {hallucination_report}")
    print(f"  Excel: {excel_report}")
    print(f"  Data: {OUTPUT_FILE}")

    return 0


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    sys.exit(main())
