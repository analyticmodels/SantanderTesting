import requests
import os
import pandas as pd
import urllib3
import concurrent.futures
import threading
import time
 
# --- CONFIGURATION ---
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
 
AUTH_TOKEN_CRED = "Yk8xQ3BPeW1NaWVKVm5CSEdla3BDaEFoVjdreTl1dUE6eGpuVnZqTEo1TmRQZGRtYg=="
TOKEN_URL = "https://apigee-test1.nonprod.corpint.net/oauth2/accesstoken-clientcredentials"
API_URL = "https://apigee-test1.nonprod.corpint.net/rag/retrieve-generate/idx-vector-openbank-voicebot"
 
INPUT_FILE = "C:/Users/SDYER/repos/SantanderTesting/data/hallucinationQuestions_all.csv"
OUTPUT_FILE = "C:/Users/SDYER/repos/SantanderTesting/data/hallucinationQuestions_responses_all.csv"
 
# --- TOKEN MANAGER CLASS (Thread-Safe) ---
class TokenManager:
    """
    Manages the authentication token safely across multiple threads.
    If a 401 occurs, it refreshes the token and allows threads to retry.
    """
    def __init__(self):
        self.access_token = None
        self.lock = threading.Lock()
        # Initialize with a token immediately
        self.refresh_token()
 
    def get_token(self):
        return self.access_token
 
    def refresh_token(self):
        """Fetches a new token. Uses a lock to prevent 50 threads from refreshing at once."""
        with self.lock:
            try:
                # print("  [Auth] Refreshing Access Token...")
                headers = {
                    "Authorization": f"Basic {AUTH_TOKEN_CRED}",
                    "Content-Type": "application/x-www-form-urlencoded"
                }
                data = {"grant_type": "client_credentials"}
                response = requests.post(TOKEN_URL, headers=headers, data=data, verify=False, timeout=10)
               
                if response.status_code == 200:
                    self.access_token = response.json().get("access_token")
                    # print("  [Auth] Token refreshed successfully.")
                else:
                    print(f"  [Auth] CRITICAL: Token refresh failed {response.status_code}")
            except Exception as e:
                print(f"  [Auth] Error refreshing token: {e}")
 
# Initialize Global Token Manager
token_manager = TokenManager()
 
# --- WORKER FUNCTION ---
def process_single_row(row):
    """
    Process a single row. If 401 is encountered, refresh token and retry.
    """
    # 1. Identify the question column (case insensitive safe-guard)
    question = row.get("question") or row.get("Question") or row.get("QUESTION")
    if not question:
        row["generated_answer"] = "Error: Question column missing"
        row["context"] = ""
        return row
 
    payload = {"prompt": question}
   
    # Retry logic
    max_retries = 2
    for attempt in range(max_retries + 1):
       
        # Get current token
        token = token_manager.get_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
 
        try:
            # Timeout is important so threads don't hang forever
            response = requests.post(API_URL, headers=headers, json=payload, verify=False, timeout=30)
 
            # --- SUCCESS ---
            if response.status_code == 200:
                data = response.json()
                answer = data.get("completion", "No answer available")
                references = data.get("references", [])
               
                # Context extraction
                context_parts = []
                if isinstance(references, list):
                    for ref in references:
                        if isinstance(ref, dict):
                            context_parts.append(str(ref.get("document_chunk", ref.get("content", ""))))
                        elif isinstance(ref, str):
                            context_parts.append(ref)
               
                row["generated_answer"] = answer
                row["context"] = " ".join(context_parts)
                return row # Success, exit function
 
            # --- 401 UNAUTHORIZED (EXPIRED TOKEN) ---
            elif response.status_code == 401:
                # Only refresh if we haven't used up our retries
                if attempt < max_retries:
                    print(f"  [401 Detected] Refreshing token and retrying for: {question[:30]}...")
                    token_manager.refresh_token()
                    time.sleep(1) # Short pause to let token update settle
                    continue # Retry the loop
                else:
                    row["generated_answer"] = "Error 401: Unauthorized (Retry Limit Exceeded)"
                    row["context"] = ""
                    return row
 
            # --- OTHER ERRORS ---
            else:
                row["generated_answer"] = f"Error {response.status_code}: {response.text[:100]}"
                row["context"] = ""
                return row
 
        except Exception as e:
            if attempt < max_retries:
                continue
            row["generated_answer"] = f"Exception: {str(e)}"
            row["context"] = ""
            return row
           
    return row
 
# --- MAIN PARALLEL PROCESSOR ---
def process_csv_parallel(input_csv, output_csv, max_workers=50):
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return
 
    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)
    # Strip whitespace from column names to avoid key errors
    df.columns = df.columns.str.strip()
   
    total_rows = len(df)
    print(f"Loaded {total_rows} rows. Starting parallel execution with {max_workers} threads...")
 
    results = []
    start_time = time.time()
 
    # ThreadPoolExecutor manages the concurrency
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Convert dataframe rows to dictionaries and submit to executor
        # FIXED: Removed stray 'Y' from to_dict()
        futures = [executor.submit(process_single_row, row.to_dict()) for _, row in df.iterrows()]
       
        # Process results as they complete
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            # FIXED: Removed stray 'Y' after try:
            try:
                result_row = future.result()
                results.append(result_row)
                completed += 1
               
                # Progress logging every 100 records
                if completed % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    print(f"Progress: {completed}/{total_rows} ({rate:.2f} reqs/sec)")
            except Exception as exc:
                print(f"Row generated an exception: {exc}")
 
    # Save Results
    print("Processing complete. Saving to CSV...")
    result_df = pd.DataFrame(results)
   
    # Optional: Reorder columns to ensure Input -> Output flow
    cols = list(result_df.columns)
    if "question" in cols and "generated_answer" in cols:
        # Move generated answer next to question if possible, or just save as is
        pass
 
    result_df.to_csv(output_csv, index=False)
   
    total_time = time.time() - start_time
    print(f"Done! Saved to {output_csv}")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
 
if __name__ == "__main__":
    # max_workers=50 is usually safe for high throughput.
    # If you get server connection errors, lower it to 20.
    process_csv_parallel(INPUT_FILE, OUTPUT_FILE, max_workers=50) 