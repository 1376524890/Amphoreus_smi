from crewai_tools import BaseTool
from typing import Dict

class MemoryInheritanceTool(BaseTool):
    name: str = "记忆继承者"
    description: str = "为迭代继承记忆。输入：previous_loop_memories (字典)，heir_id (整数)。输出：enhanced_strength (整数)。"

    def _run(self, previous_loop_memories: Dict, heir_id: int) -> int:
        base = previous_loop_memories.get(f'heir_{heir_id}', 0)
        return base + random.randint(5, 15)