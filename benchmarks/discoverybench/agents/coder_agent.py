from agents.coder_utils import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_together import Together
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from agents.qwen_client import QwenClient
import os
import json
from utils.dv_log import DVLogger
import uuid
from langchain_core.callbacks import FileCallbackHandler, StdOutCallbackHandler
import string

# Add Qwen imports
from transformers import AutoModelForCausalLM, AutoTokenizer
from abc import ABC
from langchain.llms.base import LLM
# from vllm import LLM
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
import traceback
import torch
from langchain.schema import SystemMessage, HumanMessage

# Set environment variable for CUDA debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Qwen model wrapper class
class Qwen(LLM, ABC):
    max_token: int = 10000
    temperature: float = 0.7  # Stable temperature
    top_p: float = 0.9
    history_len: int = 3
    model_name: str = ""
    
    # Declare these as private attributes for Pydantic compatibility
    _model = None
    _tokenizer = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        try:
            print(f"Loading model: {model_name}")
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",  # 明确指定 float16
                device_map="auto",
            )
            self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            # import pdb; pdb.set_trace()
            # Set pad token if not set
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
                
            print(f"Model loaded successfully on device: {next(self._model.parameters()).device}")
                
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            traceback.print_exc()
            raise

    @property
    def _llm_type(self) -> str:
        return "Qwen"

    @property
    def _history_len(self) -> int:
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        try:
            # Clear GPU cache before generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"\n=== Input prompt ===")
            # print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
            print("=" * 50)
            
            messages = [
                {"role": "user", "content": prompt}
            ]
            # import pdb; pdb.set_trace()
            # Apply chat template with error handling
            try:
                text = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                print(f"Error applying chat template: {e}")
                # Fallback to simple formatting
                text = f"<|user|>\n{prompt}\n<|assistant|>\n"
            # import pdb; pdb.set_trace()
            # Tokenize with careful handling
            model_inputs = self._tokenizer(
                [text], 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048  # 减小最大长度
            )
            # import pdb; pdb.set_trace()
            # Move to device and check for errors
            try:
                model_inputs = {k: v.to(self._model.device) for k, v in model_inputs.items()}
            except Exception as e:
                print(f"Error moving inputs to device: {e}")
                # Try CPU fallback
                device = torch.device("cpu")
                print(f"Falling back to CPU device")
                self._model = self._model.to(device)
                model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            
            # Generate with careful parameters
            try:
                with torch.no_grad():  # Disable gradients for inference
                    generated_ids = self._model.generate(
                        **model_inputs,
                        max_new_tokens=512
                    )
            except RuntimeError as e:
                if "CUDA error" in str(e) or "device-side assert" in str(e):
                    print(f"CUDA error occurred: {e}")
                else:
                    raise
            
            # Extract the generated tokens
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs['input_ids'], generated_ids)
            ]

            # Decode with safe handling
            try:
                response = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            except Exception as e:
                print(f"Error decoding response: {e}")
                response = "I encountered an error while generating the response."
            
            print(f"\n=== Model response ===")
            # print(response)
            # print("=" * 50)
            return response
            
        except Exception as e:
            print(f"Error in _call: {str(e)}")
            traceback.print_exc()
            # Return a safe fallback response
            return "I encountered an error. Please try a simpler query or check the system logs."

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"max_token": self.max_token,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "history_len": self.history_len,
                "model_name": self.model_name}

# uncomment the following line to enable debug mode
# langchain.debug = True


def get_prompt_data(
        prompt_config: str = None
):
    if prompt_config is None and os.environ.get("PROMPT_CONFIG") is None:
        raise ValueError("PROMPT_CONFIG not set and prompt_config not provided")
    else:
        prompt_config = prompt_config or os.environ.get("PROMPT_CONFIG")

    with open(prompt_config, "r") as file:
        return json.load(file)

def replace_env_vars(value):
    """替换字符串中的环境变量"""
    if not isinstance(value, str):
        return value
    template = string.Template(value)
    return template.safe_substitute(os.environ)

class BaseAgent():
    def __init__(
        self,
        model_config: str = None,
        api_config: str = None,
        model_name: str = "gpt-4o",
        log_file: str = "output.log",
        max_iterations: int = 5,
        use_api: bool = False,
        api_base: str = None
    ):
        self.logfile = log_file
        self.logger = DVLogger(f"{model_name}_{uuid.uuid4()}", log_file)
        self.file_handler = FileCallbackHandler(self.logfile)
        self.stdout_handler = StdOutCallbackHandler()
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.use_api = use_api
        self.api_base = api_base

        # Load model config
        if model_config is None and os.environ.get("MODEL_CONFIG") is None:
            raise ValueError("MODEL_CONFIG not set and model_config not provided")
        else:
            model_config = model_config or os.environ.get("MODEL_CONFIG")

        with open(model_config, "r") as file:
            self.model_config = json.load(file)

        # Load API config
        if api_config is None and os.environ.get("API_CONFIG") is None:
            raise ValueError("API_CONFIG not set and api_config not provided")
        else:
            api_config = api_config or os.environ.get("API_CONFIG")

        with open(api_config, "r") as file:
            self.api_config = json.load(file)

        # Get model config
        if model_name not in self.model_config["models"]:
            raise ValueError(f"Model {model_name} not found in model config")
        model_info = self.model_config["models"][model_name]

        # Get API config
        api = model_info.get("api", "openai")
        if api not in self.api_config["apis"]:
            raise ValueError(f"API {api} not found in API config")
        api_info = self.api_config["apis"][api]

        # Get API key (skip if using local API)
        api_key = None
        if not self.use_api:
            api_key = replace_env_vars(api_info.get("api_key"))
            if not api_key:
                raise ValueError(f"API key not found for {api}")

        # Get model
        self.model = self.get_model(
            api=api,
            api_key=api_key,
            model=model_name,
            use_api=self.use_api,
            api_base=self.api_base
        )

    def get_model(
        self,
        api,
        api_key,
        model,
        **kwargs
    ):
        use_api = kwargs.get("use_api", False)
        api_base = kwargs.get("api_base", None)

        if use_api:
            if api == "qwen":
                return QwenClient(api_base=api_base)
            else:
                raise ValueError(f"API {api} not supported for local API service")

        if api == "openai":
            return ChatOpenAI(
                model_name=model,
                openai_api_key=api_key,
                temperature=0.7,
                max_tokens=10000,
                callbacks=[self.file_handler, self.stdout_handler]
            )
        elif api == "azure":
            return AzureChatOpenAI(
                model_name=model,
                openai_api_key=api_key,
                azure_endpoint=api_base,
                api_version="2024-02-15-preview",
                temperature=0.7,
                max_tokens=10000,
                callbacks=[self.file_handler, self.stdout_handler]
            )
        elif api == "anthropic":
            return ChatAnthropic(
                model_name=model,
                anthropic_api_key=api_key,
                temperature=0.7,
                max_tokens=10000,
                callbacks=[self.file_handler, self.stdout_handler]
            )
        elif api == "together":
            return Together(
                model_name=model,
                together_api_key=api_key,
                temperature=0.7,
                max_tokens=10000,
                callbacks=[self.file_handler, self.stdout_handler]
            )
        elif api == "google":
            return ChatGoogleGenerativeAI(
                model_name=model,
                google_api_key=api_key,
                temperature=0.7,
                max_tokens=10000,
                callbacks=[self.file_handler, self.stdout_handler]
            )
        elif api == "qwen":
            return Qwen(model_name=model)
        else:
            raise ValueError(f"API {api} not supported")

    def generate(self, dataset_paths, query):
        try:
            # 构建系统提示和用户输入
            system_message = SystemMessage(content="You are a discovery agent who can execute a python code only once to answer a query based on one or more datasets. The datasets will be present in the current directory.")
            user_message = HumanMessage(content=f"Load all datasets using python using provided paths. Paths: {dataset_paths}. {query}")
            import pdb; pdb.set_trace()
            # 调用模型
            response = self.model.invoke([system_message, user_message])
            
            # 处理响应
            if hasattr(response, "generations"):
                # 如果是 ChatResult 格式
                output = response.generations[0].message.content
            elif hasattr(response, "content"):
                # 如果是 AIMessage 格式
                output = response.content
            else:
                # 其他情况
                output = str(response)
                
            self.logger.log_json({"response": output})
        except Exception as e:
            error_msg = str(e)
            print("Execution Stopped due to : ", error_msg)
            self.logger.logger.error(f"Execution Stopped due to : {error_msg}")
        self.logger.close()