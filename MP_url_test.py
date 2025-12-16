import requests
import os
import pandas as pd
# from dotenv import load_dotenv
from datetime import datetime
from multiprocessing import Pool, cpu_count


# Retrieve the credential from environment variables
auth_token = "QURWNEZxeUZQdDNaclVsbHpBTWhsSmo3SG1QRnNPZWc6dEZtVFhHUGlLYW5OdlFXSw=="
if not auth_token:
    raise Exception("Missing AUTH_TOKEN in environment variables")

# Auth details
token_url = "https://apigee-staging.nonprod.corpint.net/oauth2/accesstoken-clientcredentials"
token_headers = {
    "Authorization": f"Basic {auth_token}",
    "Content-Type": "application/x-www-form-urlencoded"
}
token_data = {"grant_type": "client_credentials"}

# API details
api_url = "https://apigee-staging.nonprod.corpint.net/rag/retrieve-generate/idx-vector-openbank-voicebot"


def get_access_token():
    """Get a fresh access token"""
    response = requests.post(token_url, headers=token_headers, data=token_data, verify=False)
    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        raise Exception(f"Auth failed: {response.status_code}, {response.text}")


def call_api(question, token):
    """Call RAG API and extract answer + context"""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {"prompt": question}

    response = requests.post(api_url, headers=headers, json=payload, verify=False)

    if response.status_code == 200 :
        data = response.json()
        answer = data.get("completion", "No answer available")
        references = data.get("references", [])
        context = " ".join(ref.get("document_chunk", "") for ref in references)
        return answer, context
    else:
        print(response.json())
        data = response.json()
        answer = data.get("errors", [{}])[0].get("description", {}).get("error", "")
        answer = str(response.status_code) +": " +  answer
        print("answer:", answer)
        return answer , ""


def process_row(args):
    """Process a single row - worker function for multiprocessing"""
    idx, row_dict, total = args
    question = row_dict["New_Question"]

    try:
        token = get_access_token()
        answer, context = call_api(question, token)
    except Exception as e:
        answer, context = f"Error: {e}", ""

    row_dict.update({
        "answer": answer,
        "context": context,
        "execution_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    print(f"Processed {idx+1}/{total}: {question} \n {answer} \n {context}")
    return row_dict


def process_csv(input_csv, output_csv, num_workers=None):
    """Process CSV and save results using multiprocessing"""
    if input_csv.lower().endswith(".csv"):
        df = pd.read_csv(input_csv)
    elif input_csv.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(input_csv)
    else:
        raise ValueError(f"Unsupported file format: {input_csv}")

    if num_workers is None:
        num_workers = min(cpu_count(), len(df))

    # Prepare arguments for each worker
    work_items = [(idx, row.to_dict(), len(df)) for idx, row in df.iterrows()]

    print(f"Processing {len(df)} rows with {num_workers} workers...")

    with Pool(processes=num_workers) as pool:
        results = pool.map(process_row, work_items)

    # Save to CSV
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    process_csv(r"qa_Amb_4793.xlsx", "output_qa_Amb_4793_run_2.csv")