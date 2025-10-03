from crewai import Agent
from tools.coreflame_collector_tool import CoreflameCollectorTool
from tools.memory_inheritance_tool import MemoryInheritanceTool
from config import get_llm

# 12 个 Chrysos Heirs 的因素（游戏灵感）
HEIR_FACTORS = ['争斗', '理性', '世界承载', '和谐', '勇气', '洞察', '耐力', '命运', '混沌', '秩序', '永恒', '重生']

def create_chrysos_agents():
    agents = []
    for i in range(1, 13):
        factor = HEIR_FACTORS[i-1]
        agents.append(
            Agent(
                role=f'黄金继承者 {i}: {factor}',
                goal=f'收集 {factor} 核心火焰，继承记忆，与他人协作延缓永恒轮回。',
                backstory=f'黄金继承者如 Phainon（若 i=1），路径行者的变量，在迭代中。',
                tools=[CoreflameCollectorTool(), MemoryInheritanceTool()],
                llm=get_llm(),
                verbose=True
            )
        )
    return agents