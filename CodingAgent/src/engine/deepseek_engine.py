import os
import requests
from requests.exceptions import HTTPError, Timeout, RequestException
import backoff

@backoff.on_exception(
    backoff.expo, 
    (HTTPError, Timeout, RequestException), 
    max_tries=5
)
def deepseek_chat_engine(api_url, api_key, model, messages, temperature, top_p):
    """Send a chat completion request to DeepSeek API."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": 2000,
        "stream": False
    }

    response = requests.post(api_url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()  # Raise HTTPError for bad responses
    return response.json()


class DeepSeekEngine:
    """Engine for interacting with DeepSeek API."""

    def __init__(self, llm_engine_name, api_key=None):
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.llm_engine_name = llm_engine_name  # E.g., 'deepseek-chat', 'deepseek-reasoner'
        
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("API Key is required. Please provide it or set it in the 'DEEPSEEK_API_KEY' environment variable.")

    def respond(self, user_input, temperature, top_p):
        """Process user input and get response from DeepSeek API."""
        messages = [
            {"role": turn["role"], "content": turn["content"]}
            for turn in user_input
        ]

        try:
            response = deepseek_chat_engine(
                self.api_url,
                self.api_key,
                self.llm_engine_name,
                messages,
                temperature,
                top_p
            )
        except (HTTPError, Timeout, RequestException) as e:
            print(f"ERROR: Unable to call DeepSeek API. Reason: {e}")
            return "ERROR", 0, 0

        # Extract relevant response data
        output_text = response["choices"][0]["message"]["content"]
        input_tokens = response["usage"]["prompt_tokens"]
        output_tokens = response["usage"]["completion_tokens"]

        return output_text, input_tokens, output_tokens