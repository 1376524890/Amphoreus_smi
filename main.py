"""
主循环
"""
import json, os, datetime
from config import MAX_ROUND, STAGES
from agents.agent_templates import TEMPLATES
from agents.agent import Agent
from utils.similarity import is_converged
from utils.reward import calc_reward

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

def main():
    kb = load_kb()
    kb.update(load_kb("world/world_supplement.json"))
    agents = [Agent(name=k, sys=TEMPLATES[k], kb=kb) for k in TEMPLATES]

    stage_idx = 0
    all_logs  = []

    for rnd in range(1, MAX_ROUND+1):
        print(f"\n===== Round {rnd} | Stage {STAGES[stage_idx]} =====")
        round_logs = []

        # 1. 每个 Agent 行动
        for ag in agents:
            act = ag.act(STAGES[stage_idx], rnd)
            act["reward"] = calc_reward(ag.name, act)
            round_logs.append(act)

        # 2. 保存本轮日志
        save_log(round_logs, rnd)
        all_logs.extend(round_logs)

        # 3. 管理员筛选 & 更新知识库
        good = [a for a in round_logs if a["impact"]>0.5]
        if good:
            kb.setdefault("supplements",[]).extend(good)
            save_summary(kb["supplements"])

        # 4. 收敛检测
        if is_converged(all_logs):
            print("✅ 已收敛，提前结束")
            break

        # 5. 阶段推进（简单按轮）
        if rnd % 5 == 0 and stage_idx < len(STAGES)-1:
            stage_idx += 1

    print("🎉 模拟结束")

if __name__ == "__main__":
    main()