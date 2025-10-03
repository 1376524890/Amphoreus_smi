from crewai_tools import BaseTool

class BlackTideTool(BaseTool):
    name: str = "黑潮入侵者"
    description: str = "引入毁灭事件。输入：entropy_level (浮点数)。输出：destruction_factor (浮点数，0-1)。"

    def _run(self, entropy_level: float) -> float:
        return min(entropy_level * 0.1 + random.uniform(0.1, 0.5), 1.0)