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

# ---------- 工具 ----------
def load_kb(path="world/world.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_kb(data, path="world/world.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

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

    stage_idx = 0
    all_logs  = []          # 用于收敛检测的全局列表（跨轮）

    for rnd in range(1, MAX_ROUND + 1):
        print(f"\n===== Round {rnd} | Stage {STAGES[stage_idx]} =====")

        # 3. 每回合行动顺序（可随机）
        agents_order = random.sample(agents, len(agents))

        # 4. 执行 3 回合（修复 AttributeError）
        snapshot = agents_order[0].play_round(agents_order, STAGES[stage_idx], rnd)

        # 5. 计算单步奖励（可选）
        for step_log in snapshot:
            step_log["reward"] = calc_reward(step_log["agent"], step_log)

        # 6. 保存本轮 33 步完整日志
        save_log(snapshot, rnd)
        all_logs.extend(snapshot)

        # 7. 生成并广播本轮总结
        summary = Agent.round_summary(snapshot)
        for ag in agents:
            ag.kb["last_round_summary"] = summary

        # 8. 管理员筛选高影响事件入库
        high_impact = [s for s in snapshot if abs(s["impact"]) > 0.5]
        if high_impact:
            kb.setdefault("supplements", []).extend(high_impact)
        save_summary(kb["supplements"])
        save_kb(kb)  # 持久化更新知识库

        # 9. 收敛检测（基于最近 N 步）
        if is_converged(all_logs):
            print("✅ 连续行为相似，已收敛，提前结束")
            break

        # 10. 阶段推进（每 5 轮）
        if rnd % 5 == 0 and stage_idx < len(STAGES) - 1:
            stage_idx += 1
            print(f"🔔 进入新阶段：{STAGES[stage_idx]}")

    print("🎉 模拟结束，最终总结：", summary)

if __name__ == "__main__":
    main()