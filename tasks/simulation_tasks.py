from crewai import Task
from agents.chrysos_agents import create_chrysos_agents

HEIR_FACTORS = ['争斗', '理性', '世界承载', '和谐', '勇气', '洞察', '耐力', '命运', '混沌', '秩序', '永恒', '重生']

def create_simulation_tasks(cycle_num: int, previous_memories: dict = None, agents: dict = None):
    tasks = [
        Task(
            description=f'在循环 {cycle_num} 中：使用种子 {cycle_num % 100} 启动创世纪。演化基本实体。',
            agent=agents['titan'],
            expected_output='创世纪演化的实体列表。'
        ),
        Task(
            description=f'使用创世纪输出，模拟文明增长。计算分数（人口 + 强度总和）。',
            agent=agents['scepter'],
            expected_output='文明分数（整数）。'
        )
    ]
    # 12 个继承者的并行任务
    chrysos_agents = create_chrysos_agents()
    for i, agent in enumerate(chrysos_agents, start=1):
        factor = HEIR_FACTORS[i-1]
        tasks.append(
            Task(
                description=f'基于分数收集 {factor} 核心火焰。若可用，继承记忆：{previous_memories.get(f"heir_{i}", 0)}。与其他继承者协作。',
                agent=agent,
                expected_output=f'收集 {factor} （布尔值）。'
            )
        )
    tasks.extend([
        Task(
            description='基于熵（分数 / 1000）引入黑潮。计算毁灭因子。',
            agent=agents['destruction'],
            expected_output='毁灭因子（浮点数）。'
        ),
        Task(
            description=f'确定结果：统计收集的核心火焰。若 >=12，进入新纪元 stall（内层循环）。否则，重置。为每个继承者增强下次迭代的记忆。',
            agent=agents['scepter'],
            expected_output='循环总结：{分数}，{火焰计数}，{毁灭}。记忆：带继承者键的字典。'
        )
    ])
    return tasks