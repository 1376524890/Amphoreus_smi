from crewai import Crew, Process
import random
import time
from crewai import Crew, Process
from agents import create_scepter_agent, create_titan_agent, create_destruction_agent, create_chrysos_agents
from tasks import create_simulation_tasks
from config import get_llm

def run_amphoreus_simulation(num_outer_loops: int = 50):
    print("程序开始执行...")
    overall_stats = {'successful_stalls': 0, 'total_flames': 0, 'avg_destruction': 0}
    memories = {f'heir_{i}': 0 for i in range(1, 13)}

    # 组装代理
    try:
        agents = {
            'scepter': create_scepter_agent(),
            'titan': create_titan_agent(),
            'destruction': create_destruction_agent()
        }
        chrysos_agents = create_chrysos_agents()  # 动态添加 Chrysos 代理
    except Exception as e:
        print(f"代理创建失败: {e}")
        return

    for cycle in range(1, num_outer_loops + 1):
        print(f"\n--- 外层循环 {cycle} ---")
        tasks = create_simulation_tasks(cycle, memories, agents, chrysos_agents)
        amphoreus_crew = Crew(
            agents=[agents['titan']] + chrysos_agents + [agents['destruction']],
            tasks=tasks,
            process=Process.hierarchical,
            manager_agent=agents['scepter'],
            verbose=True
        )
        try:
            result = amphoreus_crew.kickoff()
        except Exception as e:
            import traceback
            print(f"循环 {cycle} 执行失败: {e}")
            print(f"异常类型: {type(e).__name__}")
            print("异常堆栈:")
            traceback.print_exc()
            continue

        # 简化解析（实际中从 result 解析）
        score = random.randint(50, 200 * (cycle // 10 + 1))
        flames_collected = [random.choice([True, False]) for _ in range(12)]
        flames_count = sum(flames_collected)
        destruction = random.uniform(0.2, 0.8 - (cycle / num_outer_loops * 0.3))

        if flames_count >= 12:
            overall_stats['successful_stalls'] += 1
            print("新纪元激活！内层永恒轮回 stall Irontomb。")
            for inner in range(3):
                time.sleep(0.1)
                print(f"  内层循环 {inner+1}: 继承者间继承记忆，熵减少。")
        else:
            print("黑潮淹没。循环重置。")

        overall_stats['total_flames'] += flames_count
        overall_stats['avg_destruction'] += destruction

        for i in range(1, 13):
            memories[f'heir_{i}'] += 10 if flames_collected[i-1] else 0

        print(f"循环 {cycle} 结束: 分数={score}, 火焰={flames_count}/12, 毁灭={destruction:.2f}")

    overall_stats['avg_destruction'] /= num_outer_loops
    print("\n--- 模拟完成 ---")
    print(f"统计: Stall={overall_stats['successful_stalls']}/{num_outer_loops}, 平均火焰={overall_stats['total_flames']/num_outer_loops:.1f}, 平均毁灭={overall_stats['avg_destruction']:.2f}")
    print("Irontomb 方程: 收敛到毁灭。" if overall_stats['successful_stalls'] < num_outer_loops / 2 else "Stall！需要外部变量（开拓者）打破循环。")

if __name__ == "__main__":
    run_amphoreus_simulation(3)