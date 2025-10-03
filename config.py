import os
from dotenv import load_dotenv
from openai import OpenAI  # 使用OpenAI客户端，兼容dashscope

load_dotenv()

# 配置 DashScope Qwen-Max LLM（使用 OpenAI 兼容模式）
os.environ["DASHSCOPE_API_KEY"] = os.getenv("DASHSCOPE_API_KEY")
LLM_MODEL = "qwen3-max"  # 或 "qwen2.5-max" 如果是最新；DashScope 支持 Qwen 系列
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# CrewAI LLM 配置函数（返回 OpenAI 客户端实例，确保中文交互）
def get_llm():
    # 创建OpenAI客户端实例，配置为使用dashscope的兼容模式API
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=LLM_BASE_URL
    )
    return client