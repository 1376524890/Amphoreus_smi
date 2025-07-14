"""
增强交互版 Agent
- 每轮 3 回合行动
- 实时看到他人上一步
- 一轮结束后生成总结
"""
import json, datetime, uuid, random
from llm.qwen_plus import chat

class Agent:
    def __init__(self, name: str, sys: str, kb: dict):
        self.name = name
        self.sys  = sys
        self.kb   = kb               # 会被外部更新：kb["last_round_summary"]
        self.memory = []             # 本轮 3 步的记忆

    # -------------------------------------------------
    # 1. 单步行动：能看到【实时局面 snapshot】
    # -------------------------------------------------
    def step(self, stage: str, round_id: int, step: int,
             snapshot: list) -> dict:
        """
        snapshot: 本轮前面所有已发生的 step 日志（按时间顺序）
        返回单条日志 dict
        """
        prompt = f"""
世界阶段：{stage} | 轮次：{round_id} | 回合步：{step}
上一轮的最终局面：{self.kb.get("last_round_summary","尚未发生")}
本轮已发生的实时事件：{json.dumps(snapshot, ensure_ascii=False, indent=0)[:1200]}...
你是{self.name}，请用一句话描述你在本回合步的行动，并给出预期影响(-1~1)。
返回 JSON：{{"action":"...","impact":float}}
"""
        raw = chat(prompt, self.sys)
        try:
            out = json.loads(raw)
        except:
            out = {"action": raw.strip(), "impact": round(random.uniform(-0.2, 0.2), 2)}

        out.update({
            "agent": self.name,
            "round": round_id,
            "step": step,
            "timestamp": datetime.datetime.utcnow().isoformat()
        })
        self.memory.append(out)
        return out

    # -------------------------------------------------
    # 2. 一轮 3 回合的完整行动（外部调用）
    # -------------------------------------------------
    def play_round(self, agents_order: list, stage: str, round_id: int) -> list:
        """
        agents_order: 本回合行动顺序的 Agent 实例列表
        返回本轮所有 step 日志
        """
        snapshot = []                       # 实时累积
        for step in range(1, 4):            # 3 回合
            for ag in agents_order:
                one_step = ag.step(stage, round_id, step, snapshot)
                snapshot.append(one_step)
        return snapshot

    # -------------------------------------------------
    # 3. 一轮结束 → 生成最终总结（供下一轮当背景）
    # -------------------------------------------------
    @staticmethod
    def round_summary(logs: list) -> str:
        """
        logs: 本轮全部 step 日志
        返回人类可读的一段总结（可再交给 LLM 精炼）
        """
        impacts = [l["impact"] for l in logs]
        pos, neg = sum(i for i in impacts if i > 0), sum(i for i in impacts if i < 0)
        actions = "; ".join([l["action"] for l in logs])
        summary = (f"本轮共{len(logs)}步：正向影响{pos:.2f}，负向影响{neg:.2f}。 "
                   f"主要事件：{actions[:300]}...")
        return summary