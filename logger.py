import json
import os
import time
from typing import Dict, Any, List
import numpy as np
from datetime import datetime

class SimulationLogger:
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = log_dir
        self.logs = []  # 存储所有迭代日志
        self.current_iteration_log = {}  # 当前迭代日志
        self._create_log_dir()
        self.run_id = self._generate_run_id()

    def _create_log_dir(self) -> None:
        """创建日志目录"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            print(f"日志目录已创建: {self.log_dir}")

    def _generate_run_id(self) -> str:
        """生成唯一运行ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = np.random.randint(1000, 9999)
        return f"run_{timestamp}_{random_suffix}"

    def log_iteration(self, iteration: int, state: np.ndarray, actions: Dict[str, Any], rewards: Dict[str, float], next_state: np.ndarray) -> None:
        """记录当前迭代的日志"""
        # 将状态向量转换为列表以便JSON序列化
        state_list = state.tolist() if isinstance(state, np.ndarray) else state
        next_state_list = next_state.tolist() if isinstance(next_state, np.ndarray) else next_state
        
        # 构建迭代日志
        self.current_iteration_log = {
            "iteration": iteration,
            "timestamp": time.time(),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "state": state_list,
            "actions": actions,
            "rewards": rewards,
            "next_state": next_state_list
        }
        
        # 添加到日志列表
        self.logs.append(self.current_iteration_log)
        
        # 打印迭代摘要
        self._print_iteration_summary(iteration, rewards)

    def _print_iteration_summary(self, iteration: int, rewards: Dict[str, float]) -> None:
        """打印迭代摘要信息"""
        avg_reward = np.mean(list(rewards.values())) if rewards else 0
        max_reward_agent = max(rewards, key=rewards.get) if rewards else "N/A"
        max_reward = rewards[max_reward_agent] if max_reward_agent != "N/A" else 0
        
        summary = f"迭代 {iteration}: 平均奖励 = {avg_reward:.4f}, 最高奖励 = {max_reward:.4f} ({max_reward_agent})"
        print(summary)

    def get_current_iteration_log(self) -> Dict[str, Any]:
        """获取当前迭代的日志"""
        return self.current_iteration_log

    def get_all_logs(self) -> List[Dict[str, Any]]:
        """获取所有迭代的日志"""
        return self.logs

    def save_logs(self) -> str:
        """保存日志到文件
        Returns:
            日志文件路径
        """
        log_filename = f"simulation_logs_{self.run_id}.json"
        log_path = os.path.join(self.log_dir, log_filename)
        
        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(self.logs, f, ensure_ascii=False, indent=2)
            print(f"日志已保存到: {log_path}")
            return log_path
        except Exception as e:
            print(f"保存日志失败: {e}")
            return ""

    def load_logs(self, log_path: str) -> bool:
        """从文件加载日志
        Args:
            log_path: 日志文件路径
        Returns:
            是否加载成功
        """
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                self.logs = json.load(f)
            print(f"已从 {log_path} 加载 {len(self.logs)} 条日志")
            return True
        except Exception as e:
            print(f"加载日志失败: {e}")
            return False

    def get_reward_history(self) -> Dict[str, List[float]]:
        """获取所有智能体的奖励历史"""
        reward_history = {}
        for log in self.logs:
            iteration = log["iteration"]
            for agent_id, reward in log["rewards"].items():
                if agent_id not in reward_history:
                    reward_history[agent_id] = []
                reward_history[agent_id].append(reward)
        return reward_history

    def generate_convergence_report(self, window_size: int = 5) -> Dict[str, Any]:
        """生成收敛报告"""
        if len(self.logs) < window_size:
            return {"status": "insufficient_data", "message": f"需要至少 {window_size} 次迭代才能生成报告"}
        
        reward_history = self.get_reward_history()
        convergence_metrics = {}
        
        # 计算每个智能体的奖励稳定性
        for agent_id, rewards in reward_history.items():
            # 计算最近window_size次迭代的奖励标准差
            recent_rewards = rewards[-window_size:]
            std_dev = np.std(recent_rewards)
            avg_reward = np.mean(recent_rewards)
            
            convergence_metrics[agent_id] = {
                "average_reward": avg_reward,
                "std_deviation": std_dev,
                "convergence_score": 1.0 - (std_dev / (avg_reward + 1e-8))  # 归一化收敛分数
            }
        
        # 计算整体收敛分数
        overall_convergence = np.mean([metrics["convergence_score"] for metrics in convergence_metrics.values()])
        
        return {
            "status": "success",
            "run_id": self.run_id,
            "total_iterations": len(self.logs),
            "overall_convergence_score": overall_convergence,
            "agent_metrics": convergence_metrics,
            "converged": overall_convergence > 0.9  # 收敛阈值
        }

    def export_reward_data(self) -> str:
        """导出奖励数据为CSV格式"""
        reward_history = self.get_reward_history()
        if not reward_history:
            return ""
        
        csv_filename = f"reward_history_{self.run_id}.csv"
        csv_path = os.path.join(self.log_dir, csv_filename)
        
        # 获取所有智能体ID
        agent_ids = list(reward_history.keys())
        
        try:
            with open(csv_path, 'w', encoding='utf-8') as f:
                # 写入表头
                f.write("Iteration," + ",".join(agent_ids) + "\n")
                
                # 写入数据
                for i in range(len(self.logs)):
                    row = [str(i)]
                    for agent_id in agent_ids:
                        if i < len(reward_history[agent_id]):
                            row.append(f"{reward_history[agent_id][i]:.4f}")
                        else:
                            row.append("")
                    f.write(",".join(row) + "\n")
            
            print(f"奖励数据已导出到: {csv_path}")
            return csv_path
        except Exception as e:
            print(f"导出奖励数据失败: {e}")
            return ""