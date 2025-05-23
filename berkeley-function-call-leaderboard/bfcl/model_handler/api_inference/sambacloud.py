from bfcl.model_handler.api_inference.openai import OpenAIHandler
from openai import OpenAI, RateLimitError
import os
import time
from bfcl.model_handler.utils import (
    retry_with_backoff,
)

class SambaCloudHandler(OpenAIHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
    
    def _init_client(self):
        base_url = "https://api.sambanova.ai/v1"
        self.client = OpenAI(base_url=base_url, api_key=os.getenv("SAMBACLOUD_API_KEY"))

    @retry_with_backoff(error_type=RateLimitError)
    def generate_with_backoff(self, **kwargs):
        max_retries = 3
        retry_count = 0
        start_time = time.time()
        api_response = self.client.chat.completions.create(**kwargs)
        while retry_count < max_retries:
            api_response = self.client.chat.completions.create(**kwargs)
            if hasattr(api_response, 'error') and api_response.error:
                retry_count += 1
                print(f"Sleeping for 20 seconds...")
                time.sleep(20)
                continue
            break
        end_time = time.time()

        return api_response, end_time - start_time
