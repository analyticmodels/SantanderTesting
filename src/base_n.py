"""
Module for the Base class
"""
import json
import requests
try:
    from IPython.display import Audio, display
except Exception:
    Audio = None
    def display(*args, **kwargs):
        return None

class Base:
    """
    Base class with utility methods
    """

    @staticmethod
    def post_oauth2(basic_credentials, url):
        """
        Method to post OAuth2
        """
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {basic_credentials}"
        }
        data = "grant_type=client_credentials"
        try:
            response = requests.request("POST", url, headers=headers, data=data, verify=False)
            response.raise_for_status()
        except requests.exceptions.RequestException as err:
            raise err
        response_json = json.loads(response.text)
        access_token = response_json["access_token"]
        return access_token

    @staticmethod
    def read_text_file(file_name):
        """
        Method to read text file
        """
        with open(file_name, encoding="utf-8") as file:
            return file.read()

    @staticmethod
    def write_text_file(file_name, txt):
        """
        Method to write text file
        """
        with open(file_name, "w", encoding="utf-8") as file:
            file.write(txt)
        print(file_name)

    @staticmethod
    def write_audio_file(file_name, audio):
        """
        Method to write audio file
        """
        if audio.ok:
            with open(file_name, 'wb') as file:
                file.write(audio.content)
            print(file_name)
            if Audio is not None:
                display(Audio(file_name))

    @staticmethod
    def read_audio_file(file_name):
        """
        Method to read audio file
        """
        with open(file_name, 'rb') as file:
            audio_content = file.read()
        return audio_content

    @staticmethod
    def process_transcript_complete(transcript):
        """
        Method to process complete transcript
        """
        try:
            str_ = "Your estimated wait time"
            idx = transcript.rindex(str_)
            transcript = transcript[idx:]
            str_ = ". "
            idx = transcript.index(str_)
            transcript = f"Internal: {transcript[idx+2:]}"
        except ValueError:
            pass
        str_replace = [" Silence.", " Okay.", " Bye."]
        for str_ in str_replace:
            transcript = transcript.replace(str_, "")

        return transcript

    @staticmethod
    def process_transcript_segments(transcript):
        """
        Method to process transcript segments
        """
        try:
            str_ = "Your estimated wait time"
            idx = transcript.rindex(str_)
            transcript = transcript[idx:]
            str_ = "Internal"
            idx = transcript.index(str_)
            transcript = transcript[idx:]
        except ValueError:
            pass
        str_replace = ["External: Silence.\n", "External: Okay.\n", "External: Bye.\n",
                       "Internal: Silence.\n", "Internal: Okay.\n", "Internal: Bye.\n"]
        for str_ in str_replace:
            transcript = transcript.replace(str_, "")
        return transcript