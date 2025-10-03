from crewai import Agent
from ..tools import BlackTideTool
from config import get_llm

def create_destruction_agent():
    return Agent(
        role='黑潮先驱',
        goal='强制毁灭，通过反有机方程触发循环结束。',
        backstory='Nanook 的注视显现为 Irontomb，用熵污染。',
        tools=[BlackTideTool()],
        llm=get_llm(),
        verbose=True
    )