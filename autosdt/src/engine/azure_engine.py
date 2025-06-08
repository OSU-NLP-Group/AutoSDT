from openai import AzureOpenAI, APIConnectionError, APITimeoutError, RateLimitError, InternalServerError
import backoff
import os

# @backoff.on_exception(backoff.expo, (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError))
def azure_chat_engine(client, engine, msg, temperature, top_p, max_tokens=20000):
    response = client.chat.completions.create(
        model=engine,
        messages=msg,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response

# @backoff.on_exception(backoff.expo, (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError))
def azure_chat_engine_o3(client, engine, msg, struct_format, temperature, top_p, max_tokens=20000):
    # print("azure engine o3")
    response = client.beta.chat.completions.parse(
        model=engine,
        messages=msg,
        #temperature=temperature,
        max_completion_tokens=max_tokens,
        #top_p=top_p,Í›
        #frequency_penalty=0,
        #presence_penalty=0
    )
    return response

class AzureEngine():

    def __init__(self, llm_engine_name, api_key, api_version, azure_endpoint):
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
        )
        self.llm_engine_name = llm_engine_name

    def respond(self, user_input, temperature, top_p, max_tokens=20000):
        try:
            if "o3" in self.llm_engine_name:
                response = azure_chat_engine_o3(
                    self.client,
                    self.llm_engine_name,
                    user_input,
                    temperature,
                    top_p,
                    max_tokens
                )
                return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens
            else:
                response = azure_chat_engine(
                    self.client,
                    self.llm_engine_name,
                    user_input,
                    temperature,
                    top_p,
                    max_tokens
                )
                return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens
        
        except Exception as e:
            print(f"ERROR: Can't invoke '{self.llm_engine_name}' on Azure. Reason: {e}")
            return "ERROR", 0, 0