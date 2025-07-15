# graph/state.py
from typing import List, TypedDict
from langchain_core.messages import BaseMessage

class WorldState(TypedDict):
    round: int                       # 当前轮次
    stage: str                       # 世界阶段
    snapshot: List[BaseMessage]      # 实时公共日志
    agents_left: int                 # 本轮剩余行动次数（11×3=33）