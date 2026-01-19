import requests
import urllib3
import json
 
# Suppress warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
 
# --- CONFIGURATION ---
auth_token = "Yk8xQ3BPeW1NaWVKVm5CSEdla3BDaEFoVjdreTl1dUE6eGpuVnZqTEo1TmRQZGRtYg=="
token_url = "https://apigee-test1.nonprod.corpint.net/oauth2/accesstoken-clientcredentials"
api_url = "https://apigee-test1.nonprod.corpint.net/rag/retrieve-generate/idx-vector-sbna-contact-center-4"
 
def get_access_token():
    """Get a fresh access token"""
    token_headers = {
        "Authorization": f"Basic {auth_token}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    token_data = {"grant_type": "client_credentials"}
   
    print("1. Fetching Access Token...")
    response = requests.post(token_url, headers=token_headers, data=token_data, verify=False)
   
    if response.status_code == 200:
        print("   Success!")
        return response.json().get("access_token")
    else:
        raise Exception(f"Auth failed: {response.status_code}, {response.text}")
 
def call_api_safe(question, token):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {"prompt": question}
 
    print(f"2. Sending Question: '{question}'")
    response = requests.post(api_url, headers=headers, json=payload, verify=False)
 
    print("\n" + "="*50)
    print("3. RAW API RESPONSE (Copy this if it fails)")
    print("="*50)
   
    # 1. Print Raw Data immediately
    try:
        data = response.json()
        print(json.dumps(data, indent=4)) # Pretty print the JSON
    except:
        print("Response is not JSON:", response.text)
        return
 
    # 2. Safe Extraction
    print("\n" + "="*50)
    print("4. EXTRACTED DATA")
    print("="*50)
 
    # Get Answer
    answer = data.get("completion", "No answer found")
    print(f"ANSWER:\n{answer}\n")
 
    # Get Context (CRASH PROOF LOGIC)
    references = data.get("references", [])
   
    context_parts = []
   
    if isinstance(references, list):
        for i, ref in enumerate(references):
            # Check if it is a Dictionary (Standard RAG)
            if isinstance(ref, dict):
                chunk = ref.get("document_chunk", "")
                context_parts.append(str(chunk))
            # Check if it is a String (Simple List)
            elif isinstance(ref, str):
                context_parts.append(ref)
            else:
                context_parts.append(f"[Unknown Type: {type(ref)}]")
    else:
        print(f"WARNING: 'references' is not a list. It is: {type(references)}")
 
    full_context = " ".join(context_parts)
    print(f"CONTEXT (Length: {len(full_context)}):\n{full_context[:500]}... [truncated]")
 
if __name__ == "__main__":
    TEST_QUESTION = "What is the monthly fee for a small business checking account?"
   
    try:
        token = get_access_token()
        call_api_safe(TEST_QUESTION, token)
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")