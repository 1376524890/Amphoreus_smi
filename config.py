import os
from dotenv import load_dotenv
from crewai import LLM

load_dotenv()

# 配置 DashScope Qwen-Max LLM
LLM_MODEL = "dashscope/qwen3-max"  # 需要指定provider前缀
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# CrewAI LLM 配置函数
# 返回一个CrewAI兼容的LLM实例，确保中文交互

def get_llm():
    # 创建CrewAI兼容的LLM实例
    llm = LLM(
        model=LLM_MODEL,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=LLM_BASE_URL
    )
    return llm