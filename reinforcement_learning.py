import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Tuple
import os
from collections import deque
import random

# 设置随机种子以确保可复现性
np.random.seed(42)
torch.manual_seed(42)

class QNetwork(nn.Module):
    """Q网络模型，用于值函数近似"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    """基于DQN的智能体"""
    def __init__(self, state_dim: int, action_dim: int, agent_id: str, learning_rate: float = 0.001, gamma: float = 0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_id = agent_id
        self.gamma = gamma  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.memory = deque(maxlen=10000)  # 经验回放缓冲区

        # 创建策略网络和目标网络
        self.policy_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def act(self, state: np.ndarray) -> int:
        """根据当前状态选择动作"""
        if np.random.rand() <= self.epsilon:
            # 探索：随机选择动作
            return random.randrange(self.action_dim)
        # 利用：选择Q值最大的动作
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return torch.argmax(q_values).item()

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """存储经验到回放缓冲区"""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self) -> None:
        """从经验回放中学习"""
        if len(self.memory) < self.batch_size:
            return

        # 随机采样批次
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 转换为张量
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # 计算当前Q值和目标Q值
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.target_net(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q

        # 计算损失并优化
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self) -> None:
        """更新目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path: str) -> None:
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path: str) -> None:
        """加载模型"""
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())

class MARLTrainingModule:
    """多智能体强化学习训练模块"""
    def __init__(self, agents: Dict[str, Any], state_dim: int, action_dim: int, update_interval: int = 10):
        self.agents = agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.update_interval = update_interval
        self.dqn_agents = self._initialize_dqn_agents()
        self.iteration_count = 0

    def _initialize_dqn_agents(self) -> Dict[str, DQNAgent]:
        """初始化所有智能体的DQN模型"""
        dqn_agents = {}
        for agent_id in self.agents.keys():
            if agent_id.startswith("hero_agent"):  # 只为英雄智能体创建DQN模型
                dqn_agents[agent_id] = DQNAgent(
                    state_dim=self.state_dim,
                    action_dim=self.action_dim,
                    agent_id=agent_id
                )
        return dqn_agents

    def select_actions(self, state: np.ndarray) -> Dict[str, int]:
        """为所有智能体选择动作"""
        actions = {}
        for agent_id, dqn_agent in self.dqn_agents.items():
            actions[agent_id] = dqn_agent.act(state)
        return actions

    def remember_experiences(self, state: np.ndarray, actions: Dict[str, int], rewards: Dict[str, float], next_state: np.ndarray, done: bool) -> None:
        """存储所有智能体的经验"""
        for agent_id, dqn_agent in self.dqn_agents.items():
            if agent_id in actions and agent_id in rewards:
                dqn_agent.remember(
                    state=state,
                    action=actions[agent_id],
                    reward=rewards[agent_id],
                    next_state=next_state,
                    done=done
                )

    def update(self, rewards: Dict[str, float]) -> None:
        """更新所有智能体的策略"""
        # 每个智能体进行经验回放学习
        for agent_id, dqn_agent in self.dqn_agents.items():
            if agent_id in rewards:
                dqn_agent.replay()

        # 定期更新目标网络
        self.iteration_count += 1
        if self.iteration_count % self.update_interval == 0:
            for dqn_agent in self.dqn_agents.values():
                dqn_agent.update_target_network()

    def get_behavior_vector(self) -> np.ndarray:
        """获取当前所有智能体的行为向量，用于收敛检测"""
        behavior_vector = []
        for dqn_agent in self.dqn_agents.values():
            # 提取策略网络的权重作为行为特征
            weights = []
            for param in dqn_agent.policy_net.parameters():
                weights.extend(param.data.numpy().flatten())
            # 归一化权重向量
            weights = np.array(weights)
            weights = weights / (np.linalg.norm(weights) + 1e-8)
            behavior_vector.extend(weights)
        
        # 如果行为向量为空，返回随机向量
        if not behavior_vector:
            return np.random.rand(32)  # 默认向量长度
        
        return np.array(behavior_vector)

    def save_models(self, save_dir: str = "./models") -> None:
        """保存所有智能体的模型"""
        for agent_id, dqn_agent in self.dqn_agents.items():
            model_path = os.path.join(save_dir, f"{agent_id}_model.pth")
            dqn_agent.save_model(model_path)
        print(f"所有模型已保存到 {save_dir}")

    def load_models(self, load_dir: str = "./models") -> None:
        """加载所有智能体的模型"""
        for agent_id, dqn_agent in self.dqn_agents.items():
            model_path = os.path.join(load_dir, f"{agent_id}_model.pth")
            if os.path.exists(model_path):
                dqn_agent.load_model(model_path)
                print(f"已加载 {agent_id} 的模型")
            else:
                print(f"未找到 {agent_id} 的模型文件: {model_path}")

    def get_epsilon_values(self) -> Dict[str, float]:
        """获取所有智能体的当前探索率"""
        return {agent_id: dqn_agent.epsilon for agent_id, dqn_agent in self.dqn_agents.items()}

    def set_epsilon_values(self, epsilon: float) -> None:
        """设置所有智能体的探索率"""
        for dqn_agent in self.dqn_agents.values():
            dqn_agent.epsilon = epsilon

class RewardShapingModule:
    """奖励塑造模块，用于优化原始奖励信号"""
    def __init__(self, worldview_kb, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2):
        self.worldview_kb = worldview_kb
        self.alpha = alpha  # 命途契合度权重
        self.beta = beta    # 阶段目标权重
        self.gamma = gamma  # 协作权重

    def shape_rewards(self, original_rewards: Dict[str, float], actions: Dict[str, Any], current_stage: str) -> Dict[str, float]:
        """塑造奖励信号
        Args:
            original_rewards: 原始奖励
            actions: 智能体动作
            current_stage: 当前阶段
        Returns:
            塑造后的奖励
        """
        shaped_rewards = {}
        stage_info = self.worldview_kb.get_stage_info(current_stage)
        stage_goals = self._get_stage_goals(stage_info)

        # 计算每个智能体的命途契合度
        destiny_fitness = self._calculate_destiny_fitness(actions)

        # 计算协作奖励
        cooperation_reward = self._calculate_cooperation_reward(actions)

        # 计算阶段目标完成度
        stage_progress = self._calculate_stage_progress(current_stage)

        # 整合所有奖励成分
        for agent_id, original_reward in original_rewards.items():
            # 基础奖励
            shaped_reward = original_reward

            # 添加命途契合度奖励
            if agent_id in destiny_fitness:
                shaped_reward += self.alpha * destiny_fitness[agent_id]

            # 添加协作奖励
            shaped_reward += self.gamma * cooperation_reward

            # 限制奖励范围
            shaped_reward = max(0.0, min(1.0, shaped_reward))
            shaped_rewards[agent_id] = shaped_reward

        return shaped_rewards

    def _calculate_destiny_fitness(self, actions: Dict[str, Any]) -> Dict[str, float]:
        """计算智能体动作与命途的契合度"""
        fitness = {}
        for agent_id, action in actions.items():
            agent = self.worldview_kb.get_agent_by_id(agent_id)
            if not agent:
                fitness[agent_id] = 0.0
                continue

            destiny = agent.get("destiny")
            action_type = action.get("type", "")
            abilities = agent.get("abilities", [])

            # 检查动作是否属于智能体能力
            if action_type in abilities:
                fitness[agent_id] = 1.0
            else:
                # 根据命途特性评估动作契合度
                fitness_score = self._evaluate_action_fitness(destiny, action_type)
                fitness[agent_id] = fitness_score

        return fitness

    def _evaluate_action_fitness(self, destiny: str, action_type: str) -> float:
        """评估动作与命途的契合度"""
        # 命途-动作契合度映射
        fitness_map = {
            "秩序": {"制定规则": 1.0, "执行纪律": 0.9, "维护稳定": 0.8, "纠正偏差": 0.85},
            "同谐": {"促进合作": 1.0, "调解冲突": 0.9, "建立联盟": 0.85, "共享资源": 0.8},
            "毁灭": {"发动攻击": 1.0, "破坏设施": 0.9, "引发混乱": 0.8, "削弱敌人": 0.95},
            "存护": {"构建防御": 1.0, "保护盟友": 0.95, "修复设施": 0.8, "建立屏障": 0.85},
            "丰饶": {"创造资源": 1.0, "治愈伤病": 0.9, "促进生长": 0.85, "提升产能": 0.8}
            # 其他命途的映射...
        }

        # 获取命途对应的动作契合度字典
        destiny_fitness = fitness_map.get(destiny, {})
        
        # 返回契合度分数，默认为0.2
        return destiny_fitness.get(action_type, 0.2)

    def _calculate_cooperation_reward(self, actions: Dict[str, Any]) -> float:
        """计算协作奖励"""
        if len(actions) < 2:
            return 0.0

        # 简单协作检测：计算有多少动作是促进合作的
        cooperation_actions = 0
        total_actions = len(actions)

        for action in actions.values():
            action_type = action.get("type", "")
            if action_type in ["促进合作", "共享资源", "建立联盟", "调解冲突", "信息交流"]:
                cooperation_actions += 1

        # 协作比例作为协作奖励
        return cooperation_actions / total_actions

    def _get_stage_goals(self, stage_info: Dict[str, Any]) -> List[str]:
        """获取当前阶段的目标"""
        if not stage_info:
            return []

        stage_description = stage_info.get("描述", "")
        # 简单规则提取目标关键词
        if "建立" in stage_description:
            return ["建立文明", "发展基础"]
        elif "繁荣" in stage_description:
            return ["技术发展", "文化繁荣", "人口增长"]
        elif "战争" in stage_description or "纷争" in stage_description:
            return ["军事发展", "防御准备", "资源储备"]
        elif "毁灭" in stage_description or "衰败" in stage_description:
            return ["生存保障", "资源保护", "重建准备"]
        else:
            return []

    def _calculate_stage_progress(self, current_stage: str) -> float:
        """计算当前阶段目标的完成度"""
        # 在实际应用中，这里应该根据世界状态计算阶段目标的完成度
        # 简化实现：随机返回0.3-0.7之间的进度值
        return np.random.uniform(0.3, 0.7)