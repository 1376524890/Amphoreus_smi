from crewai.tools import BaseTool
import random

class CoreflameCollectorTool(BaseTool):
    name: str = "核心火焰收集器"
    description: str = "为一个继承者收集单个核心火焰。输入：current_civilization_score (整数)，heir_factor (str)。输出：collected (布尔值)。"

    def _run(self, current_civilization_score: int, heir_factor: str) -> bool:
        probability = min(current_civilization_score / 100, 1.0) * 0.083
        return random.random() < probability