import json
import time
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import numpy as np
from typing import List, Dict, Any

# 导入自定义模块
from environment import AmphoreusEnvironment
from agents import AdminAgent, HeroAgent
from reinforcement_learning import MARLTrainingModule
from knowledge_base import WorldviewKnowledgeBase
from logger import SimulationLogger

class AmphoreusSystem:
    def __init__(self):
        # 初始化知识库
        self.worldview_kb = WorldviewKnowledgeBase(
            worldview_path="/home/codeserver/AMPHOREUS/世界观.json",
            personality_path="/home/codeserver/AMPHOREUS/《崩坏：星穹铁道》命途智能体个性设定.md"
        )
        
        # 初始化环境
        self.environment = AmphoreusEnvironment(
            worldview_kb=self.worldview_kb,
            initial_stage="启蒙世"
        )
        
        # 初始化智能体
        self.agents = self._initialize_agents()
        
        # 初始化强化学习模块
        self.training_module = MARLTrainingModule(
            agents=self.agents,
            state_dim=self.environment.state_dim,
            action_dim=self.environment.action_dim
        )
        
        # 初始化日志系统
        self.logger = SimulationLogger(log_dir="./logs")
        
        # 系统状态
        self.current_iteration = 0
        self.converged = False
        self.max_iterations = 100
        self.convergence_threshold = 0.05
        self.previous_behavior_vector = None

    def _initialize_agents(self) -> Dict[str, Any]:
        """初始化所有智能体，包括11个英雄智能体和1个管理员智能体"""
        agents = {}
        
        # 获取命途智能体个性设定
        personalities = self.worldview_kb.get_personalities()
        
        # 创建11个英雄智能体
        for i, (destiny, personality) in enumerate(personalities.items()):
            if i >= 11:  # 限制为11个英雄智能体
                break
            agents[f"hero_agent_{i+1}"] = HeroAgent(
                agent_id=f"hero_agent_{i+1}",
                destiny=destiny,
                personality=personality,
                worldview_kb=self.worldview_kb
            )
        
        # 创建管理员智能体
        agents["admin_agent"] = AdminAgent(
            agent_id="admin_agent",
            worldview_kb=self.worldview_kb,
            convergence_threshold=self.convergence_threshold
        )
        
        return agents

    def _check_convergence(self, current_behavior_vector: np.ndarray) -> bool:
        """检查智能体行为是否收敛"""
        if self.previous_behavior_vector is None:
            self.previous_behavior_vector = current_behavior_vector
            return False
        
        # 计算行为向量相似度（余弦相似度）
        similarity = np.dot(self.previous_behavior_vector, current_behavior_vector) / (
            np.linalg.norm(self.previous_behavior_vector) * np.linalg.norm(current_behavior_vector)
        )
        
        # 如果相似度高于阈值，则认为收敛
        if similarity > (1 - self.convergence_threshold):
            return True
        
        self.previous_behavior_vector = current_behavior_vector
        return False

    def run_simulation(self):
        """运行多智能体强化学习模拟"""
        print("=== 翁法罗斯多智能体模拟系统启动 ===")
        print(f"初始阶段: {self.environment.current_stage}")
        
        while self.current_iteration < self.max_iterations and not self.converged:
            print(f"\n=== 迭代 {self.current_iteration + 1}/{self.max_iterations} ===")
            
            # 获取当前环境状态
            current_state = self.environment.get_state()
            
            # 所有智能体执行动作
            actions = {}
            for agent_id, agent in self.agents.items():
                if agent_id == "admin_agent":
                    continue  # 管理员智能体不参与环境交互
                actions[agent_id] = agent.take_action(current_state)
            
            # 环境执行动作并返回新状态和奖励
            next_state, rewards = self.environment.step(actions)
            
            # 记录当前迭代日志
            self.logger.log_iteration(
                iteration=self.current_iteration,
                state=current_state,
                actions=actions,
                rewards=rewards,
                next_state=next_state
            )
            
            # 管理员智能体评估并更新奖励参数
            admin_agent = self.agents["admin_agent"]
            updated_rewards = admin_agent.evaluate_rewards(
                iteration=self.current_iteration,
                logs=self.logger.get_current_iteration_log()
            )
            
            # 强化学习模块更新策略
            self.training_module.update(updated_rewards)
            
            # 检查收敛
            behavior_vector = self.training_module.get_behavior_vector()
            self.converged = self._check_convergence(behavior_vector)
            
            # 如果收敛，管理员智能体更新世界观
            if self.converged:
                print("检测到行为收敛，更新世界观...")
                new_knowledge = admin_agent.filter_useful_knowledge(self.logger.get_all_logs())
                self.worldview_kb.update_knowledge(new_knowledge)
                
                # 检查是否需要进入下一阶段
                if self.environment.should_progress_stage():
                    self.environment.progress_stage()
                    print(f"进入新阶段: {self.environment.current_stage}")
                    self.converged = False  # 新阶段重置收敛状态
            
            self.current_iteration += 1
        
        print("\n=== 模拟结束 ===")
        print(f"总迭代次数: {self.current_iteration}")
        print(f"最终阶段: {self.environment.current_stage}")
        print(f"收敛状态: {'已收敛' if self.converged else '未收敛'}")
        
        # 保存最终日志和模型
        self.logger.save_logs()
        self.training_module.save_models()

if __name__ == "__main__":
    system = AmphoreusSystem()
    system.run_simulation()