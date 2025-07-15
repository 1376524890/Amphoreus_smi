# tools/toolkit.py
from langchain.agents import tool
from utils.reward import calculate_reward

@tool
def write_log(msg: str) -> str:
    """将重要事件写入日志
    Args:
        msg: 需要记录的日志信息
    """
    with open("/home/codeserver/AMPHOREUS/log/tool_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.utcnow().isoformat()}] {msg}\n")
    return "日志已成功记录"

@tool
def calc_reward(action: str, impact: float) -> str:
    """计算行动奖励值
    Args:
        action: 智能体的行动描述
        impact: 行动的预期影响(-1~1)
    """
    reward = calculate_reward(action, impact)
    return f"行动奖励值: {reward:.2f}"

tools = [write_log, calc_reward]