"""
Agent 基类
"""
import json, datetime, uuid
from llm.qwen_plus import chat

class Agent:
    def __init__(self, name:str, sys:str, kb:dict):
        self.name   = name
        self.sys    = sys
        self.kb     = kb
        self.memory = []  # 多轮对话可扩展

    def act(self, stage:str, round:int)->dict:
        """生成行为并调用 LLM"""
        prompt = f"""
当前世界阶段：{stage}，第 {round} 轮。
世界观摘要：{json.dumps(self.kb, ensure_ascii=False, indent=0)[:1000]}...
请用一句话描述你本轮的行动意图，并给出预期影响（-1~1）。
返回 JSON：{{"action":"...","impact":float}}
"""
        raw = chat(prompt, self.sys)
        try:
            out = json.loads(raw)
        except:
            out = {"action":raw.strip(), "impact":0.0}
        out.update({
            "agent":self.name,
            "timestamp":datetime.datetime.utcnow().isoformat()
        })
        self.memory.append(out)
        return out