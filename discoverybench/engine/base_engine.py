class LLMEngine():
    def __init__(self, llm_engine_name, api_key=None, api_version=None, azure_endpoint=None, port=8000):
        self.llm_engine_name = llm_engine_name
        self.engine = None
        self.port = port
        
        # Azure engine
        if llm_engine_name.startswith("azure"):
            llm_engine_name = llm_engine_name.split("azure_")[-1]
            from engine.azure_engine import AzureEngine
            self.engine = AzureEngine(llm_engine_name, api_key, api_version, azure_endpoint)
        
        # # Structured Azure engine
        # elif llm_engine_name.startswith("struct"):
        #     llm_engine_name = llm_engine_name.split("struct_azure_")[-1]
        #     from engine.structured_engine import StructuredAzureEngine
        #     self.engine = StructuredAzureEngine(llm_engine_name, api_key, api_version, azure_endpoint)

        # OpenAI engine
        elif llm_engine_name.startswith("gpt") or llm_engine_name.startswith("o1"):
            from engine.openai_engine import OpenaiEngine
            self.engine = OpenaiEngine(llm_engine_name)
        
        # # DeepSeek engine
        # elif llm_engine_name.startswith("deepseek") or llm_engine_name.startswith("r1"):
        #     from engine.deepseek_engine import DeepSeekEngine
        #     self.engine = DeepSeekEngine(llm_engine_name)
        
        # Bedrock engine
        elif llm_engine_name.startswith("vllm"):
            llm_engine_name = llm_engine_name.replace("vllm_", "")
            from engine.vllm_engine import VLLMEngine
            self.engine = VLLMEngine(llm_engine_name, port=port)
        else:
            from engine.bedrock_engine import BedrockEngine
            self.engine = BedrockEngine(llm_engine_name)

    def respond(self, user_input, temperature, top_p, max_tokens):
        return self.engine.respond(user_input, temperature, top_p, max_tokens)
    
    def respond_structured(self, user_input, struct_format, temperature, top_p, max_tokens):
        return self.engine.respond_structured(user_input, struct_format, temperature, top_p, max_tokens)
