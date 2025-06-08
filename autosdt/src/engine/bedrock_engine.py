from botocore.exceptions import ClientError, ReadTimeoutError
from botocore.client import Config

import boto3

def bedrock_converse_engine(client, engine, msg, temperature, top_p, maxTokens=20000):
    # import pdb; pdb.set_trace()
    response = client.converse(
        modelId=engine,
        messages=msg,
        inferenceConfig={"maxTokens": maxTokens, "temperature": temperature, "topP": top_p},
    )
    # response = client.messages.with_raw_response.create(
    #     model=engine,
    #     messages=msg,
    #     inferenceConfig={"maxTokens": maxTokens, "temperature": temperature, "topP": top_p},
    # )

    return response

class BedrockEngine():

    def __init__(self, llm_engine_name):
        self.client = boto3.client(
            "bedrock-runtime", 
            region_name="us-east-2", 
            config=Config(retries={"total_max_attempts": 3}, read_timeout=1200)
        )
        self.llm_engine_name = llm_engine_name

    def respond(self, user_input, temperature, top_p, max_tokens=20000):
        conversation = [
            {"role": turn["role"], "content": [{"text": turn["content"]}]}
            for turn in user_input
        ]

        try:
            response = bedrock_converse_engine(
                self.client, 
                self.llm_engine_name, 
                conversation,
                temperature,
                top_p,
                max_tokens
            )
        except (ClientError, Exception) as e:
            print(f"ERROR: Can't invoke '{self.llm_engine_name}'. Reason: {e}")
            return "ERROR", 0, 0

        return response["output"]["message"]["content"][0]["text"], response["usage"]["inputTokens"], response["usage"]["outputTokens"]
    

if __name__ == "__main__":
    eng = BedrockEngine("us.anthropic.claude-3-7-sonnet-20250219-v1:0")
    # eng = BedrockEngine("us.anthropic.claude-3-5-sonnet-20241022-v2:0")
    import time
    # calculate the time taken to run the respond function
    start_time = time.time()
    print(eng.respond([{"role": "user", "content": "Hello, how are you? Please generate text as long as possible until you reach the max tokens limit"}], 0.7, 0.9, 4000))  # Example usage
    print("--- %s seconds ---" % (time.time() - start_time))  # calculate the time taken to run the respond function