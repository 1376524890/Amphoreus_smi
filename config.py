import os
from typing import List

# ========== 阿里云百炼 ==========
ALI_API_KEY = os.getenv("ALI_API_KEY") or "sk-xxxxxxxxxxxxxxxx"
ALI_MODEL   = "qwen-plus"
ALI_BASE    = "https://bailian.aliyuncs.com/v1"   # 官方 OpenAI 兼容地址

# ========== 迭代控制 ==========
MAX_ROUND      = 30           # 最大迭代轮数
SIM_THRESHOLD  = 0.95         # 余弦相似度收敛阈值
HISTORY_WINDOW = 3            # 检测最近 n 轮相似
REWARD_WEIGHT  = {"philosophy":0.6, "goal":0.4}

# ========== 世界阶段 ==========
STAGES: List[str] = [
    "启蒙世","造物世","黄金世",
    "纷争世","幻灭世","再创世"
]