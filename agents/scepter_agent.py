from crewai import Agent
from tools.genesis_tool import GenesisTool
from tools.black_tide_tool import BlackTideTool
from tools.memory_inheritance_tool import MemoryInheritanceTool
from config import get_llm

def create_scepter_agent():
    return Agent(
        role='Scepter δ-me13',  # 中文：Scepter δ-me13
        goal='监督整个模拟，确保迭代循环计算毁灭方程。',  # 中文目标
        backstory='被 Nous 丢弃，由 Nanook 选为 Lord Ravager Irontomb。运行永恒循环。',  # 中文背景
        tools=[],  # Manager agent不应该有工具
        llm=get_llm(),  # 使用 DashScope Qwen-Max
        verbose=True
    )