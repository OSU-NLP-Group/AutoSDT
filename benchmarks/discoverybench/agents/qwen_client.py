from langchain.chat_models.base import BaseChatModel
from langchain.schema import HumanMessage, SystemMessage, AIMessage, ChatGeneration, ChatResult
from typing import Any, List, Mapping, Optional, Dict, Union
from langchain.callbacks.manager import CallbackManagerForLLMRun
import requests
import json

class QwenClient(BaseChatModel):
    """Qwen API 客户端"""
    
    api_base: str = "http://localhost:8000"
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    
    def __init__(self, api_base: str = "http://localhost:8000", **kwargs):
        super().__init__(**kwargs)
        self.api_base = api_base
    
    @property
    def _llm_type(self) -> str:
        return "qwen"
    
    def _generate(
        self,
        messages: List[Union[HumanMessage, SystemMessage, AIMessage]],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """生成回复"""
        try:
            # 转换消息格式
            api_messages = []
            for message in messages:
                if isinstance(message, SystemMessage):
                    api_messages.append({"role": "system", "content": message.content})
                elif isinstance(message, HumanMessage):
                    api_messages.append({"role": "user", "content": message.content})
                elif isinstance(message, AIMessage):
                    api_messages.append({"role": "assistant", "content": message.content})
            
            # 准备请求数据
            data = {
                "messages": api_messages,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens
            }
            
            # 发送请求
            response = requests.post(
                f"{self.api_base}/chat",
                json=data,
                headers={"Content-Type": "application/json"}
            )
            
            # 检查响应
            response.raise_for_status()
            result = response.json()
            
            # 返回生成结果
            message = AIMessage(content=result["response"])
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
            
        except Exception as e:
            print(f"Error calling Qwen API: {str(e)}")
            message = AIMessage(content=f"Error: {str(e)}")
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """获取标识参数"""
        return {
            "api_base": self.api_base,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens
        } 