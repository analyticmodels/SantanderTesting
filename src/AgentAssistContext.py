import requests
import urllib3
import json
 
# Suppress warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
 
# --- CONFIGURATION ---
# auth_token = "Yk8xQ3BPeW1NaWVKVm5CSEdla3BDaEFoVjdreTl1dUE6eGpuVnZqTEo1TmRQZGRtYg=="
# token_url = "https://apigee-test1.nonprod.corpint.net/oauth2/accesstoken-clientcredentials"
# api_url = "https://apigee-test1.nonprod.corpint.net/rag/retrieve-generate/idx-vector-sbna-contact-center-4"

class Agent_assist_context:
    def __init__(self,question, auth_token = "Yk8xQ3BPeW1NaWVKVm5CSEdla3BDaEFoVjdreTl1dUE6eGpuVnZqTEo1TmRQZGRtYg==",
                 token_url = "https://apigee-test1.nonprod.corpint.net/oauth2/accesstoken-clientcredentials", 
                 api_url = "https://apigee-test1.nonprod.corpint.net/rag/retrieve-generate/idx-vector-sbna-contact-center-4",
                 n_chunks=10):
        self.question = question
        self.auth_token = auth_token
        self.token_url = token_url
        self.n_chunks = n_chunks
        self.api_url = api_url
        self.token = self.get_access_token()

    def set_question(self,question):
        self.question = question

    def get_question(self):
        return self.question

    def get_access_token(self,auth_token=None,token_url=None):
        """Get a fresh access token"""
        if auth_token is None:
            auth_token = self.auth_token
        if token_url is None:
            token_url = self.token_url
        token_headers = {
            "Authorization": f"Basic {auth_token}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        token_data = {"grant_type": "client_credentials"}
    
        #print("1. Fetching Access Token...")
        response = requests.post(token_url, headers=token_headers, data=token_data, verify=False)
    
        if response.status_code == 200:
            #print("   Success!")
            return response.json().get("access_token")
        else:
            raise Exception(f"Auth failed: {response.status_code}, {response.text}")
 
    def get_context(self, question=None, token=None):
        if question is None:
            question = self.question
        n_chunks=self.n_chunks
        if token is None:
            if self.token is None: 
                try:
                    token = self.get_access_token()
                except BaseException as e:
                    print(f"unable to get access token: {e}")
                    return
            else:
                token = self.token
        api_url = self.api_url
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        payload = {"prompt": question}
    
        response = requests.post(api_url, headers=headers, json=payload, verify=False)

        try:
            data = response.json()
        except:
             print("Response is not JSON:", response.text)
             return
    
        #answer = data.get("completion", "No answer found")

        references = data.get("references", [])
    
        context_parts = []
        full_context = ""
        if isinstance(references, list):
            for i, ref in enumerate(references):
                # Check if it is a Dictionary (Standard RAG)
                if isinstance(ref, dict):
                    chunk = ref.get("document_chunk", "")
                    score = ref.get("score",0)
                    context_parts.append([chunk,score])
                    #print (f"score={score}")
                if i == (n_chunks-1): break
        else:
            print(f"WARNING: 'references' is not a list. It is: {type(references)}")
        if context_parts == []:
            return ""
        else:
            sorted_context_parts = sorted(context_parts, key=lambda item: item[1],reverse=True)
            for j in range(n_chunks):
                full_context = full_context + " " + context_parts[j][0]
                #print(sorted_context_parts[j][0][:10])
                #print(f"Context scores\n{context_parts[j][1]}\n")

            #print()
            return full_context
 
if __name__ == "__main__":
    TEST_QUESTION = "What is the monthly fee for a small business checking account?"
   
    context_getter = Agent_assist_context(TEST_QUESTION)
    try:
        #context_getter.get_access_token()
        context = context_getter.get_context()
        print("\nContext passed out:\n")
        print(context[:10])
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
    #print(context)