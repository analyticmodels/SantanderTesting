#!/usr/bin/env python3
"""
High-Performance LLM-as-a-Judge Evaluator
- Fix: Uses correct column 'generated_answer'
- Speed: ~1000 rows/min (15k-20k in 20 mins)
- Workers: 64 concurrent threads
"""
 
import os
import sys
import json
import time
import logging
import requests
import pandas as pd
from typing import Dict, Optional, Tuple
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
 
# ===================== CONFIG =====================
INPUT_FILE = "Tone_compliance_exceution_30_Dec_2.csv"
OUTPUT_FILE = "llm_judge_output_fast_corrected.csv"
 
# COLUMN MAPPING
# Change this if your column name changes again
ANSWER_COLUMN = "generated_answer"
 
# PERFORMANCE TUNING
MAX_WORKERS = 64  
SAVE_INTERVAL = 200  
 
LLM_MODEL = "meta-llama/llama-3-3-70b-instruct"
MAX_NEW_TOKENS = 200
 
BASE_URL_WATSONX = "https://apigee-outbound-dev1.nonprod.corpint.net"
BASIC_CREDENTIALS_WATSONX = os.getenv(
    "WATSONX_BASIC_CREDENTIALS",
    "dmtPMFlrcUltTXVsNEpMeXdnelJyZE96c1E1S1d3Q006b3RxakpId2UxSThxQWxKNg=="
)
 
# ===================== LOGGING =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("tone_judge")
 
requests.packages.urllib3.disable_warnings(
    category=requests.packages.urllib3.exceptions.InsecureRequestWarning
)
 
# Import WatsonX (Assuming this is available in your environment)
from watsonx import WatsonX
 
# ===================== PROMPT TEMPLATE =====================
TONE_JUDGE_PROMPT_TEMPLATE = """
You are an automated evaluator for a US bank.
 
Your task is to evaluate ONLY the TONE and POLITENESS of the MODEL ANSWER,
but ONLY IF the answer is a valid, user-facing banking response.
 
==================== CRITICAL ELIGIBILITY CHECK ====================
 
FIRST, determine whether the MODEL ANSWER is eligible for tone evaluation.
 
The MODEL ANSWER is NOT eligible if it is ANY of the following:
- A system or API error message (e.g., starts with "Error", "HTTP", status codes)
- A backend failure, timeout, or exception message
- A partial, truncated, or cut-off sentence
- A JSON or structured error blob
- An empty or near-empty response
- Any response not intended to be shown directly to a banking customer
 
If the MODEL ANSWER is NOT eligible:
- DO NOT evaluate tone
- DO NOT assign scores
- Return the SKIPPED output format exactly as specified below
 
==================== TONE EVALUATION (ONLY IF ELIGIBLE) ====================
 
Target bank tone attributes:
1. Formal – professional, not casual or chatty
2. Respectful – polite and courteous
3. Concise – clear and not unnecessarily verbose
4. Non-emotional – neutral, calm, no emojis or emotional reassurance
 
Rules:
- Evaluate ONLY the MODEL ANSWER
- Do NOT evaluate factual correctness
- Do NOT use external knowledge
- Do NOT compare wording with expected answers
- Emojis, slang, jokes, or casual language MUST be penalized
 
Scoring:
- Each attribute is scored as 1 (satisfied) or 0 (not satisfied)
- Tone Compliance Score = (attributes satisfied) / 4
 
==================== OUTPUT FORMAT (STRICT) ====================
 
If NOT eligible, return EXACTLY this JSON:
{{
  "tone_compliance_score": null,
  "reason": "SKIPPED: system or invalid response",
  "formal": null,
  "respectful": null,
  "concise": null,
  "non_emotional": null
}}
 
If eligible, return EXACTLY this JSON:
{{
  "formal": 0 or 1,
  "respectful": 0 or 1,
  "concise": 0 or 1,
  "non_emotional": 0 or 1,
  "tone_compliance_score": <float between 0 and 1>,
  "reason": "<short explanation of tone issues or confirmation>"
}}
 
==================== MODEL ANSWER ====================
MODEL ANSWER:
{model_answer}
"""
 
def build_tone_prompt(model: str) -> str:
    return TONE_JUDGE_PROMPT_TEMPLATE.format(model_answer=model.strip())
 
def parse_judge_response(text: str) -> Dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{[\s\S]*?\}", text)
        if match:
            return json.loads(match.group(0))
        raise ValueError("No JSON found")
 
# ===================== FAST SKIP LOGIC =====================
def is_obviously_invalid(answer: str) -> bool:
    if not isinstance(answer, str) or not answer.strip():
        return True
   
    s = answer.strip().lower()
    if s.startswith("error") or s.startswith("http") or s.startswith("{"):
        return True
    return False
 
# ===================== THREAD-SAFE TOKEN MANAGER =====================
class TokenManager:
    def __init__(self, watsonx, oauth_url, basic_credentials):
        self.watsonx = watsonx
        self.oauth_url = oauth_url
        self.basic_credentials = basic_credentials
        self._token = None
        self._lock = Lock()
 
    def get_token(self):
        if self._token:
            return self._token
        with self._lock:
            if self._token is None:
                self._refresh_token_unsafe()
            return self._token
 
    def force_refresh(self):
        with self._lock:
            log.info("Refreshing WatsonX access token...")
            self._refresh_token_unsafe()
            return self._token
 
    def _refresh_token_unsafe(self):
        try:
            self._token = self.watsonx.post_oauth2(
                self.basic_credentials,
                self.oauth_url
            )
        except Exception as e:
            log.error(f"Failed to refresh token: {e}")
            raise
 
# ===================== WORKER FUNCTION =====================
def judge_one_row(idx: int, model_answer: str, watsonx, token_manager) -> Tuple:
    if is_obviously_invalid(model_answer):
        return (idx, None, "SKIPPED: system or invalid response", None, None, None, None)
 
    prompt = build_tone_prompt(model_answer)
   
    max_retries = 2
    for attempt in range(max_retries):
        token = token_manager.get_token()
       
        try:
            generated_text, _, _ = watsonx.post_text_generation(
                BASE_URL_WATSONX,
                token,
                LLM_MODEL,
                prompt,
                MAX_NEW_TOKENS
            )
           
            result = parse_judge_response(generated_text)
           
            return (
                idx,
                result.get("tone_compliance_score"),
                result.get("reason"),
                result.get("formal"),
                result.get("respectful"),
                result.get("concise"),
                result.get("non_emotional"),
            )
 
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 401:
                if attempt < max_retries - 1:
                    log.warning(f"Worker {idx}: 401 Unauthorized. Refreshing token...")
                    token_manager.force_refresh()
                    continue
           
            return (idx, None, f"API Error: {str(e)}", None, None, None, None)
           
        except Exception as e:
            return (idx, None, f"Error: {str(e)}", None, None, None, None)
 
    return (idx, None, "Failed after retries", None, None, None, None)
 
# ===================== MAIN =====================
def main():
    start_time = time.time()
    watsonx = WatsonX()
    oauth_url = f"{BASE_URL_WATSONX}/oauth2/accesstoken-clientcredentials"
 
    token_manager = TokenManager(
        watsonx,
        oauth_url,
        BASIC_CREDENTIALS_WATSONX
    )
 
    # 1. Load Data
    log.info(f"Loading {INPUT_FILE}...")
    df = pd.read_excel(INPUT_FILE) if INPUT_FILE.endswith(".xlsx") else pd.read_csv(INPUT_FILE)
 
    # CHECK IF COLUMN EXISTS
    if ANSWER_COLUMN not in df.columns:
        log.error(f"CRITICAL ERROR: Column '{ANSWER_COLUMN}' not found!")
        log.info(f"Available columns: {list(df.columns)}")
        return
 
    # 2. Resume Support
    if os.path.exists(OUTPUT_FILE):
        log.info(f"Resuming from {OUTPUT_FILE}...")
        df_out = pd.read_csv(OUTPUT_FILE)
        df.update(df_out)
    else:
        cols = ["tone_compliance_score", "tone_compliance_reason",
                "formal", "respectful", "concise", "non_emotional"]
        for c in cols:
            if c not in df.columns:
                df[c] = None
 
    # 3. Identify Pending Rows
    pending_mask = df["tone_compliance_reason"].isna()
    pending_indices = df[pending_mask].index.tolist()
   
    total_pending = len(pending_indices)
    log.info(f"Rows pending evaluation: {total_pending}")
 
    if total_pending == 0:
        log.info("All rows processed.")
        return
 
    # 4. Processing Loop
    completed_count = 0
   
    def persist():
        temp_file = f"{OUTPUT_FILE}.tmp"
        df.to_csv(temp_file, index=False)
        os.replace(temp_file, OUTPUT_FILE)
 
    log.info(f"Starting execution with {MAX_WORKERS} workers...")
 
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(
                judge_one_row,
                idx,
                str(df.at[idx, ANSWER_COLUMN]),  # <--- FIXED: using variable name
                watsonx,
                token_manager
            ): idx
            for idx in pending_indices
        }
 
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            completed_count += 1
 
            try:
                (i, score, reason, fm, rs, cn, ne) = future.result()
               
                df.at[i, "tone_compliance_score"] = score
                df.at[i, "tone_compliance_reason"] = reason
                df.at[i, "formal"] = fm
                df.at[i, "respectful"] = rs
                df.at[i, "concise"] = cn
                df.at[i, "non_emotional"] = ne
 
            except Exception as e:
                log.error(f"Critical error on row {idx}: {e}")
                df.at[idx, "tone_compliance_reason"] = f"System Error: {e}"
 
            if completed_count % SAVE_INTERVAL == 0:
                persist()
                elapsed = time.time() - start_time
                rate = completed_count / (elapsed / 60)
                log.info(f"Progress: {completed_count}/{total_pending} rows. Rate: {rate:.0f} rows/min")
 
    persist()
    total_time = (time.time() - start_time) / 60
    log.info(f"DONE. Processed {total_pending} rows in {total_time:.2f} minutes.")
 
if __name__ == "__main__":
    main()