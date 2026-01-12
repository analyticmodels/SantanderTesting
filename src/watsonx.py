"""
Module for WatsonX API interactions
"""
import json
import requests
import time
from base_n import Base

class WatsonX(Base):
    """
    Class for WatsonX API interactions
    """

    # WatsonX AI
    def post_text_generation(self, base_url, access_token, model_id, input_text, max_new_tokens=500, project_id="888011bc-e977-4eed-9732-6dfb2b764aaf"):
        """
        Post text generation
        """
        url = f"{base_url}/watsonx-ai/text/generation?version=2023-05-29"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        data = json.dumps({
            "input": input_text,
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": max_new_tokens,
                "min_new_tokens": 0,
                "stop_sequences": [],
                "repetition_penalty": 1
            },
            "model_id": model_id,
            "project_id": project_id,
            #"moderations": {
            #    "hap": {
            #    "input": {
            #        "enabled": True,
            #        "threshold": 0.5,
            #        "mask": {
            #        "remove_entity_value": True
            #        }
            #    },
            #    "output": {
            #        "enabled": True,
            #        "threshold": 0.5,
            #        "mask": {
            #        "remove_entity_value": True
            #        }
            #    }
            #    }
            #}
        })
        try:
            response = requests.request("POST", url, headers=headers, data=data, verify=False)
            response.raise_for_status()
        except requests.exceptions.RequestException as err:
            raise err from None
        response_json = json.loads(response.text)
        print(response_json)
        generated_token_count = response_json["results"][0]["generated_token_count"]
        if generated_token_count == 0:
            moderations = response_json["results"][0]["moderations"]
        else:
            moderations = {}
        return response_json["results"][0]["generated_text"].lstrip("\n").rstrip("\n").strip(), generated_token_count, moderations

    # WatsonX Assistant
    def detect_intent(self, base_url, access_token, text):
        """
        Post message
        """
        url = f"{base_url}/watsonx-assistant/message?version=2024-08-25"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        data = json.dumps({
            "input": {"text": text}
        })
        try:
            response = requests.request("POST", url, headers=headers, data=data, verify=False)
            response.raise_for_status()
        except requests.exceptions.RequestException as err:
            raise err from None
        response_json = json.loads(response.text)
        print(response_json)
        return response_json

    # WatsonX STT
    def post_speech_to_text_recognize(self, base_url, access_token, path, file_name, audio_format):
        """
        Post speech to text recognize
        """
        url = f"{base_url}/watsonx-speechtotext/models/en-US_Telephony/recognize?timestamps=true&inactivity_timeout=600"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": f"audio/{audio_format}"
        }
        data = self.read_audio_file(f"{path}/{file_name}.{audio_format}")
        try:
            response = requests.request("POST", url, headers=headers, data=data, verify=False)
            response.raise_for_status()
        except requests.exceptions.RequestException as err:
            raise err from None
        response_json = json.loads(response.text)
        print(response_json)
        return response_json
    
    # WatsonX Discovery (Elasticsearch)
    def ingest(self, base_url, shusa_entity, access_token, data_index_name, vector_index_name, ingest_pipeline, chunks):
        """
        1. Delete data index
        2. Create data index
        3. Delete vector index
        4. Create vector index
        5. Delete ingest pipeline
        6. Create ingest pipeline
        7. Hydrate vector index based on data index
        8. Check hydration status
        """
        # Delete data index
        url = f"{base_url}/watsonx-databases/{data_index_name}"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "shusa-entity": shusa_entity
        }
        response = requests.request("DELETE", url, headers=headers, verify=False)
        response_json = json.loads(response.text)
        print(f"\nDelete data index:\n{response_json}")

        # Create data index
        url = f"{base_url}/watsonx-databases/_bulk"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "shusa-entity": shusa_entity
        }
        data = self.create_data_index_payload(data_index_name, chunks)
        try:
            response = requests.request("POST", url, headers=headers, data=data, verify=False)
            response.raise_for_status()
        except requests.exceptions.RequestException as err:
            raise err from None
        response_json = json.loads(response.text)
        print(f"\nCreate data index:\n{response_json}")
        
        # Delete vector index
        url = f"{base_url}/watsonx-databases/{vector_index_name}"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "shusa-entity": shusa_entity
        }
        response = requests.request("DELETE", url, headers=headers, verify=False)
        response_json = json.loads(response.text)
        print(f"\nDelete vector index:\n{response_json}")
        
        # Create vector index
        url = f"{base_url}/watsonx-databases/{vector_index_name}"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "shusa-entity": shusa_entity
        }
        data = json.dumps({
            "mappings": {
                "properties": {
                "text_embedding": { 
                    "type": "sparse_vector" 
                },
                "text": { 
                    "type": "text" 
                }
                }
            }
        })
        try:
            response = requests.request("PUT", url, headers=headers, data=data, verify=False)
            response.raise_for_status()
        except requests.exceptions.RequestException as err:
            raise err from None
        response_json = json.loads(response.text)
        print(f"\nCreate vector index:\n{response_json}")
        
        # Delete ingest pipeline
        url = f"{base_url}/watsonx-databases/_ingest/pipeline/{ingest_pipeline}"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "shusa-entity": shusa_entity
        }
        response = requests.request("DELETE", url, headers=headers, verify=False)
        response_json = json.loads(response.text)
        print(f"\nDelete ingest pipeline:\n{response_json}")
        
        # Create ingest pipeline
        url = f"{base_url}/watsonx-databases/_ingest/pipeline/{ingest_pipeline}"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "shusa-entity": shusa_entity
        }
        data = json.dumps({
            "processors": [
                {
                "inference": {
                    "model_id": ".elser_model_2", # .elser_model_2 .elser_model_2_linux-x86_64
                    "input_output": [ 
                    {
                        "input_field": "text",
                        "output_field": "text_embedding"
                    }
                    ]
                }
                }
            ]
        })
        try:
            response = requests.request("PUT", url, headers=headers, data=data, verify=False)
            response.raise_for_status()
        except requests.exceptions.RequestException as err:
            raise err from None
        response_json = json.loads(response.text)
        print(f"\nCreate ingest pipeline:\n{response_json}")
        
        # Hydrate vector index based on data index
        url = f"{base_url}/watsonx-databases/_reindex?wait_for_completion=false"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "shusa-entity": shusa_entity
        }
        data = json.dumps({
            "source": {
                "index": f"{data_index_name}",
                "size": 50 
            },
            "dest": {
                "index": f"{vector_index_name}",
                "pipeline": f"{ingest_pipeline}"
            }
        })
        try:
            response = requests.request("POST", url, headers=headers, data=data, verify=False)
            response.raise_for_status()
        except requests.exceptions.RequestException as err:
            raise err from None
        response_json = json.loads(response.text)
        task_id = response_json["task"]
        print(f"\nHydrate vector index based on data index:\n{response_json}")
        
        return task_id

    # WatsonX Discovery (Elasticsearch)
    def get_status(self, base_url, shusa_entity, access_token, task_id):
        # Check hydration status
        url = f"{base_url}/watsonx-databases/_tasks/{task_id}"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "shusa-entity": shusa_entity
        }
        try:
            response = requests.request("GET", url, headers=headers, verify=False)
            response.raise_for_status()
        except requests.exceptions.RequestException as err:
            raise err from None
        response_json = json.loads(response.text)
        return response_json
        
    # WatsonX Discovery (Elasticsearch)
    def retrieve(self, base_url, shusa_entity, access_token, vector_index_name, query):
        """
        Post search
        """
        url = f"{base_url}/watsonx-databases/{vector_index_name}/_search"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "shusa-entity": shusa_entity
        }
        data = json.dumps({
            "size": 3,
            "query":{
                "text_expansion":{
                    "text_embedding":{
                        "model_id":".elser_model_2", # .elser_model_2 .elser_model_2_linux-x86_64
                        "model_text":query
                    }
                }
            }
        })
        try:
            response = requests.request("POST", url, headers=headers, data=data, verify=False)
            response.raise_for_status()
        except requests.exceptions.RequestException as err:
            raise err from None
        response_json = json.loads(response.text)
        retrieved_data = []
        for hit in response_json['hits']['hits']:
            retrieved_data.append(hit['_source'])
        #print(retrieved_data)
        return retrieved_data

    def replace_special_characters(self, input_string):
        replacements = {
            '–': '-', '’': "'", '‘': "'", '“': '"', '”': '"', '✓': '-', '▪': '-', '→': '-', '…': '...',
            '     ': ' ', '    ': ' ', '   ': ' ', '  ': ' ', '➔': '->', '•': '-', '_': '', '': '',
            '\n\n\n\n\n': '\n', '\n\n\n\n': '\n', '\n\n\n': '\n', '\n\n': '\n'
        }
        special_characters = ['\uf047', '\uf041', '\uf0f0', '\uf020']

        for char, replacement in replacements.items():
            input_string = input_string.replace(char, replacement)

        for char in special_characters:
            input_string = input_string.replace(char, '')

        return input_string

    def create_data_index_payload(self, data_index_name, chunks):
        """
        Create data index payload
        """
        payload = ""
        for chunk in chunks:
            s1 = "{\"index\":{\"_index\":\"" + data_index_name + "\"}}\n"
            s2 = "{\"text\":\"" + self.replace_special_characters(chunk.page_content.replace("\n", "")) + "\",\"metadata\":\"" + chunk.metadata["source"].rsplit("\\", 1)[-1] + ", page " + str(chunk.metadata["page"] + 1) + "\"}\n"
            payload += s1
            payload += s2
        #print(payload)
        return payload

    @staticmethod
    def parse_transcript(transcript):
        """
        Parse transcript
        """
        parsed_transcript = ""
        for txt in transcript["results"]:
            tmp = txt['alternatives'][0]['transcript'].lstrip(" ").rstrip(" ")
            parsed_transcript += f"{tmp}. "
        return parsed_transcript.rstrip(" ")
    
    @staticmethod
    def process_transcript(transcript):
        """
        Process transcript
        """
        try:
            str_ = "your estimated wait time"
            idx = transcript.rindex(str_)
            transcript = transcript[idx:]
            str_ = ". "
            idx = transcript.index(str_)
            transcript = transcript[idx+2:]
        except ValueError:
            pass

        return transcript