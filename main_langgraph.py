# main_langgraph.py
import json
from graph.graph import graph
from config import MAX_ROUND, STAGES
from utils.similarity import is_converged
from utils.reward import calc_reward
from kb.loader import load_kb

# ---------- 工具函数 ----------
def save_log(log, rnd):
    import os, datetime
    os.makedirs("log", exist_ok=True)
    with open(f"log/run_{rnd:03d}.json", "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

def save_summary(summary):
    with open("log/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

# ---------- 主循环 ----------
def main():
    # 1. 加载知识库
    kb = load_kb()
    kb.update(load_kb("world/world_supplement.json"))
    kb["last_round_summary"] = "世界刚诞生，混沌未分。"

    stage_idx = 0
    all_logs = []
    high_impact_events = []

    for rnd in range(1, MAX_ROUND + 1):
        print(f"\n===== Round {rnd} | Stage {STAGES[stage_idx]} =====")

        # 2. 初始化本轮状态
        state = {
            "round": rnd,
            "stage": STAGES[stage_idx],
            "snapshot": [],
            "agents_left": 33  # 11智能体×3回合
        }

        # 3. 执行33步图推理
        final_state = None
        for step_out in graph.stream(state):
            # 打印当前步输出
            for node, output in step_out.items():
                print(f"[{node}] {output['snapshot'][-1].content}")
            final_state = output

        # 4. 处理本轮结果
        round_snapshot = [{
            "agent": msg.content.split("：")[0],
            "action": msg.content.split("：")[1],
            "round": rnd,
            "step": i+1,
            "timestamp": msg.additional_kwargs.get("timestamp", str(datetime.datetime.utcnow()))
        } for i, msg in enumerate(final_state["snapshot"])]

        # 5. 计算奖励并保存日志
        for step_log in round_snapshot:
            step_log["reward"] = calc_reward(step_log["agent"], step_log)
        save_log(round_snapshot, rnd)
        all_logs.extend(round_snapshot)

        # 6. 高影响事件筛选
        high_impact = [s for s in round_snapshot if abs(float(s["action"].split("影响：")[-1])) > 0.5]
        if high_impact:
            high_impact_events.extend(high_impact)
            save_summary(high_impact_events)

        # 7. 收敛检测
        if is_converged(all_logs):
            print("✅ 连续行为相似，已收敛，提前结束")
            break

        # 8. 阶段推进
        if rnd % 5 == 0 and stage_idx < len(STAGES) - 1:
            stage_idx += 1
            print(f"🔔 进入新阶段：{STAGES[stage_idx]}")

    print("🎉 模拟结束，最终总结：", kb.get("last_round_summary"))

if __name__ == "__main__":
    import datetime
    main()