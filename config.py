import os
from dotenv import load_dotenv
from litellm import completion  # LiteLLM 用于 CrewAI 集成

load_dotenv()

# 配置 DashScope Qwen-Max LLM（使用 OpenAI 兼容模式）
os.environ["DASHSCOPE_API_KEY"] = os.getenv("DASHSCOPE_API_KEY")
LLM_MODEL = "qwen3-max"  # 或 "qwen2.5-max" 如果是最新；DashScope 支持 Qwen 系列
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# CrewAI LLM 配置函数（返回 LiteLLM 实例，确保中文交互）
def get_llm():
    from litellm import LiteLLM
    return LiteLLM(
        model_name=LLM_MODEL,
        api_base=LLM_BASE_URL,
        temperature=0.7,  # 适中温度，确保稳定输出
        max_tokens=2048,
        additional_kwargs={"language": "zh"}  # 强制中文输出（如果支持）
    )