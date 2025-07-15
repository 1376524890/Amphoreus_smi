"""
增强交互版主循环
- 每轮 11×3 = 33 步
- 实时 snapshot 传播
- 每轮结束后生成总结并写回 kb
"""
import json, os, datetime, random
from config import MAX_ROUND, STAGES
from agents.agent_templates import TEMPLATES
from agents.agent import Agent
from utils.similarity import is_converged
from utils.reward import calc_reward
# from langchain_community.autonomous_agents import GroupChat
from langchain_experimental.autonomous_agents import GroupChat

# ---------- 工具 ----------
def load_kb(path="world/world.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def save_log(log, rnd):
    os.makedirs("log", exist_ok=True)
    with open(f"log/run_{rnd:03d}.json", "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

def save_summary(summary):
    with open("log/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

# ---------- 主循环 ----------
def main():
    # 1. 加载世界观
    kb = load_kb()
    kb.update(load_kb("world/world_supplement.json"))
    kb["last_round_summary"] = "世界刚诞生，混沌未分。"   # 初始背景

    # 2. 创建 11 个 Agent
    agents = [Agent(name=k, sys=TEMPLATES[k], kb=kb) for k in TEMPLATES]
    agent_executors = [ag.agent_executor for ag in agents]

    # 3. 初始化多智能体群聊
    MAX_ROUND = 15
    group = GroupChat(
        agents=agent_executors,
        speaker_selection_method="round_robin",
        max_round=MAX_ROUND,
    )

    # 4. 启动群聊
    initial_topic = f"世界初始话题：{kb['last_round_summary']}"
    print(f"===== 启动多智能体群聊 | 初始话题：{initial_topic} =====")
    final_summary = group.run(initial_topic)

    print("🎉 群聊结束，最终总结：", final_summary)

if __name__ == "__main__":
    main()