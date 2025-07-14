# llm/qwen_plus.py   （示例修改）
from openai import OpenAI   # 新增

def chat(prompt: str, system: str) -> str:
    client = OpenAI(
        api_key="sk-xxxxxxxxxxxxxxxx",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 千问示例
    )

    resp = client.chat.completions.create(
        model="qwen-plus",   # 或者 gpt-4o 等
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    return resp.choices[0].message.content