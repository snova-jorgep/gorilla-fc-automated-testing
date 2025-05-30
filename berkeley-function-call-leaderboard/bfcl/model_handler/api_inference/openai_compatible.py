from bfcl.model_handler.api_inference.openai import OpenAIHandler
from bfcl.model_handler.api_inference.deepseek import DeepSeekAPIHandler
from openai import OpenAI, RateLimitError
import os
import time
from bfcl.model_handler.utils import (
    retry_with_backoff,
)

class OpenaiCompatibleHandler(OpenAIHandler):
    def __init__(self, model_name, temperature) -> None:
        self.compatible_provider=model_name.split('/')[0]
        model_name = '/'.join([sec for sec in model_name.split('/')[1:]])
        super().__init__(model_name, temperature)
    
    def _init_client(self):
        if self.compatible_provider == "sambanova":
            base_url = "https://api.sambanova.ai/v1"
            self.client = OpenAI(base_url=base_url, api_key=os.getenv("SAMBACLOUD_API_KEY"))
        elif self.compatible_provider == "groq":
            base_url = "https://api.groq.com/openai/v1"
            self.client = OpenAI(base_url=base_url, api_key=os.getenv("GROQ_API_KEY"))
        elif self.compatible_provider == "cerebras":
            base_url = "https://api.cerebras.ai/v1"
            self.client = OpenAI(base_url=base_url, api_key=os.getenv("CEREBRAS_API_KEY"))
        elif self.compatible_provider == "fireworks":
            base_url = "https://api.fireworks.ai/inference/v1"
            self.client = OpenAI(base_url=base_url, api_key=os.getenv("FIREWORKS_API_KEY"))
        elif self.compatible_provider == "together":
            base_url = "https://api.together.xyz/v1"
            self.client = OpenAI(base_url=base_url, api_key=os.getenv("TOGETHER_API_KEY"))
        else:
            raise(Exception(f"{self.compatible_provider} not implemented"))

    @retry_with_backoff(error_type=RateLimitError)
    def generate_with_backoff(self, **kwargs):
        max_retries = 10
        retry_count = 0
        start_time = time.time()
        while retry_count < max_retries:
            try:
                api_response = self.client.chat.completions.create(**kwargs)
                # If there's no error, break out of the loop
                if not (hasattr(api_response, 'error') and api_response.error):
                    break
            except Exception as e:
                print(f"Error occurred: {str(e)}")
                
            # Calculate exponential backoff (base 2)
            sleep_time = min(2 ** retry_count, 60)  # Cap at 60 seconds to avoid too long waits
            print(f"Attempt {retry_count + 1} failed. Sleeping for {sleep_time} seconds...")
            time.sleep(sleep_time)
            retry_count += 1
        
        end_time = time.time()
        return api_response, end_time - start_time