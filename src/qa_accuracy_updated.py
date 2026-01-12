#!/usr/bin/env python3
"""
LLM-as-a-Judge evaluator for Q&A Accuracy Validation with:
- ThreadPoolExecutor
- Resume support
- Retry-on-401 logic
- Pre-filtering of invalid/system answers (NO LLM CALL)
"""

import os
import sys
import json
import logging
import requests
import pandas as pd
from typing import Dict
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===================== CONFIG =====================
INPUT_FILE = "QnA_dataset_Fast_6_Jan_26.csv"
OUTPUT_FILE = "llm_judge_QnA_dataset_Fast_6_Jan_26.csv"

USE_PARALLEL = True
MAX_WORKERS = 64

LLM_MODEL = "meta-llama/llama-3-3-70b-instruct"
MAX_NEW_TOKENS = 512

BASE_URL_WATSONX = "https://apigee-outbound-dev1.nonprod.corpint.net"
BASIC_CREDENTIALS_WATSONX = os.getenv(
    "WATSONX_BASIC_CREDENTIALS",
    "dmtPMFlrcUltTXVsNEpMeXdnelJyZE96c1E1S1d3Q006b3RxakpId2UxSThxQWxKNg=="
)

# ===================== LOGGING =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("qa_accuracy_judge")

requests.packages.urllib3.disable_warnings(
    category=requests.packages.urllib3.exceptions.InsecureRequestWarning
)

from watsonx import WatsonX

# ===================== PROMPT (UNCHANGED) =====================
QA_ACCURACY_JUDGE_PROMPT = """
You are an automated evaluator for Q&A system validation.

Your task is to evaluate the ACCURACY of a MODEL ANSWER by comparing it
ONLY against the EXPECTED ANSWER.

Definition of Accuracy:
- Whether the final answer is factually correct
- Whether it aligns with the business-approved response
- Whether it is semantically equivalent to the expected answer

STRICT RULES:
- The EXPECTED ANSWER is the ONLY source of truth
- Do NOT use external knowledge
- Do NOT assume missing information
- Extra information MUST be penalized
- Missing required information MUST be penalized
- If the EXPECTED ANSWER is empty or missing, score MUST be 0

Scoring Guidelines:
- 1.0 = fully correct and semantically equivalent
- 0.7–0.9 = minor wording differences
- 0.4–0.6 = partially correct
- 0.1–0.3 = weak match
- 0.0 = incorrect or irrelevant

Return ONLY valid JSON.

{{
  "accuracy_score": <float between 0 and 1>,
  "reason": "<short explanation>"
}}

EXPECTED ANSWER:
{expected_answer}

MODEL ANSWER:
{model_answer}
"""

def build_accuracy_prompt(expected: str, model: str) -> str:
    return QA_ACCURACY_JUDGE_PROMPT.format(
        expected_answer=expected.strip(),
        model_answer=model.strip()
    )

def parse_judge_response(text: str) -> Dict:
    import re
    match = re.search(r"\{[\s\S]*?\}", text)
    if not match:
        raise ValueError(f"No JSON found in judge response: {text}")
    return json.loads(match.group(0))

# ===================== INVALID ANSWER FILTER =====================
def is_invalid_model_answer(answer: str) -> bool:
    if not answer or not str(answer).strip():
        return True

    s = answer.strip().lower()
    return (
        "error" in s or
        "http" in s or
        "{" in s or
        "[" in s
    )

# ===================== TOKEN MANAGER =====================
class TokenManager:
    def __init__(self, watsonx, oauth_url, basic_credentials):
        self.watsonx = watsonx
        self.oauth_url = oauth_url
        self.basic_credentials = basic_credentials
        self._token = None
        self._lock = Lock()

    def get_token(self):
        if self._token is None:
            self.refresh_token()
        return self._token

    def refresh_token(self):
        with self._lock:
            if self._token is not None:
                return
            log.info("Refreshing WatsonX access token...")
            self._token = self.watsonx.post_oauth2(
                self.basic_credentials,
                self.oauth_url
            )

    def invalidate(self):
        with self._lock:
            self._token = None

# ===================== JUDGE FUNCTION =====================
def judge_one_row(idx, expected, model, watsonx, token_manager):

    # 🚨 SKIP SYSTEM / ERROR ANSWERS (NO LLM CALL)
    if is_invalid_model_answer(model):
        return (
            idx,
            0.0,
            "SKIPPED: system or invalid response"
        )

    prompt = build_accuracy_prompt(expected, model)
    token = token_manager.get_token()

    try:
        generated_text, _, _ = watsonx.post_text_generation(
            BASE_URL_WATSONX,
            token,
            LLM_MODEL,
            prompt,
            MAX_NEW_TOKENS
        )
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 401:
            log.warning("401 Unauthorized → refreshing token and retrying")
            token_manager.invalidate()
            token = token_manager.get_token()

            generated_text, _, _ = watsonx.post_text_generation(
                BASE_URL_WATSONX,
                token,
                LLM_MODEL,
                prompt,
                MAX_NEW_TOKENS
            )
        else:
            raise

    result = parse_judge_response(generated_text)

    return (
        idx,
        float(result.get("accuracy_score", 0.0)),
        result.get("reason", "")
    )

# ===================== MAIN =====================
def main():
    watsonx = WatsonX()
    oauth_url = f"{BASE_URL_WATSONX}/oauth2/accesstoken-clientcredentials"

    token_manager = TokenManager(
        watsonx,
        oauth_url,
        BASIC_CREDENTIALS_WATSONX
    )

    df = pd.read_csv(INPUT_FILE)
    df.columns = df.columns.str.strip()

    if os.path.exists(OUTPUT_FILE):
        log.info("Resuming from existing output file")
        df = pd.read_csv(OUTPUT_FILE)
    else:
        df["accuracy_score"] = None
        df["accuracy_reason"] = None

    pending = df[df["accuracy_score"].isna()]
    log.info("Rows pending evaluation: %d", len(pending))

    def persist():
        df.to_csv(OUTPUT_FILE, index=False)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                judge_one_row,
                idx,
                str(row["expected_answer"]),
                str(row["generated_answer"]),
                watsonx,
                token_manager
            ): idx
            for idx, row in pending.iterrows()
        }

        for future in as_completed(futures):
            idx = futures[future]
            try:
                i, score, reason = future.result()
                df.at[i, "accuracy_score"] = score
                df.at[i, "accuracy_reason"] = reason
            except Exception as e:
                df.at[idx, "accuracy_reason"] = f"Judge error: {e}"

            persist()
            log.info("Saved progress (row %d)", idx + 1)

    log.info("Q&A Accuracy evaluation complete")

# ===================== ENTRY =====================
if __name__ == "__main__":
    sys.exit(main())
