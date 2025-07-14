"""
简单奖励计算
"""
from config import REWARD_WEIGHT as W

def calc_reward(agent_name:str, log:dict)->float:
    """
    这里用两个 proxy：
    1) philosophy 契合度：智能体名是否出现在 action 中(简化)
    2) goal 达成度：|impact|
    """
    philosophy = 1.0 if agent_name in log["action"] else 0.5
    goal       = abs(log.get("impact",0))
    return W["philosophy"]*philosophy + W["goal"]*goal