# llm/llm.py
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    openai_api_key="sk-xxxxxxxx",
    model_name="qwen-plus",
    temperature=0.7,
    max_tokens=512,
)