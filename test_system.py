import unittest
import numpy as np
import os
import json
from main import AmphoreusSystem
from environment import AmphoreusEnvironment
from agents import HeroAgent, AdminAgent
from knowledge_base import WorldviewKnowledgeBase
from reinforcement_learning import MARLTrainingModule, RewardShapingModule
from logger import SimulationLogger
from config import config

class TestAmphoreusSystem(unittest.TestCase):
    """AMPHOREUS系统测试类"""
    @classmethod
    def setUpClass(cls):
        """在所有测试开始前执行一次"""
        # 初始化知识库
        cls.kb = WorldviewKnowledgeBase(
            worldview_path=config.paths['worldview_file'],
            personality_path=config.paths['personality_file']
        )
        # 创建测试日志目录
        cls.test_log_dir = "./test_logs"
        os.makedirs(cls.test_log_dir, exist_ok=True)

    def setUp(self):
        """在每个测试开始前执行"""
        # 重置随机种子
        np.random.seed(config.seed)
        # 创建环境实例
        self.environment = AmphoreusEnvironment(self.kb)
        # 创建日志器
        self.logger = SimulationLogger(log_dir=self.test_log_dir)
        # 创建强化学习模块
        self.rl_module = MARLTrainingModule(
            agents={},  # 暂时为空，测试时会动态添加
            state_dim=config.environment['state_dim'],
            action_dim=10  # 假设10个可能的动作
        )
        # 创建奖励塑造模块
        self.reward_shaper = RewardShapingModule(self.kb)

    def tearDown(self):
        """在每个测试结束后执行"""
        # 清理测试日志
        for file in os.listdir(self.test_log_dir):
            file_path = os.path.join(self.test_log_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    def test_environment_initialization(self):
        """测试环境初始化"""
        self.assertEqual(self.environment.current_stage, "dawn")
        self.assertEqual(len(self.environment.titan_states), config.environment['num_titans'])
        self.assertEqual(len(self.environment.civilization_metrics), len(config.environment['civilization_metrics']))
        self.assertIsInstance(self.environment.get_state(), np.ndarray)
        self.assertEqual(self.environment.get_state().shape[0], config.environment['state_dim'])

    def test_environment_state_transition(self):
        """测试环境状态转换"""
        initial_state = self.environment.get_state().copy()
        # 创建一个简单的动作字典
        actions = {
            "hero_agent_1": {"type": "发展技术", "target_titan": 0},
            "hero_agent_2": {"type": "增加人口", "target_titan": 1}
        }
        # 执行一步
        new_state, rewards, done = self.environment.step(actions)
        # 检查状态是否变化
        self.assertFalse(np.array_equal(initial_state, new_state))
        # 检查奖励是否为字典
        self.assertIsInstance(rewards, dict)
        # 检查done是否为布尔值
        self.assertIsInstance(done, bool)

    def test_stage_transition(self):
        """测试阶段转换机制"""
        # 强制设置文明指标以触发阶段转换
        self.environment.civilization_metrics = {
            "technology": 0.8,
            "culture": 0.7,
            "population": 0.6,
            "resources": 0.75,
            "stability": 0.65
        }
        # 检查阶段是否转换
        self.environment.check_stage_transition()
        self.assertNotEqual(self.environment.current_stage, "dawn")

    def test_hero_agent_creation(self):
        """测试英雄智能体创建"""
        # 获取命途列表
        destiny_list = self.kb.get_all_destinies()
        self.assertGreater(len(destiny_list), 0)

        # 创建一个英雄智能体
        agent_id = "hero_agent_test"
        destiny = destiny_list[0]
        agent = HeroAgent(agent_id, destiny, self.kb)

        self.assertEqual(agent.agent_id, agent_id)
        self.assertEqual(agent.destiny, destiny)
        self.assertIsNotNone(agent.personality)
        self.assertIsInstance(agent.memory, list)

    def test_agent_action_selection(self):
        """测试智能体动作选择"""
        # 创建一个英雄智能体
        agent = HeroAgent("test_agent", "秩序", self.kb)
        # 获取环境状态
        state = self.environment.get_state()
        # 生成动作
        action = agent.select_action(state)
        # 检查动作格式
        self.assertIsInstance(action, dict)
        self.assertIn("type", action)
        self.assertIn("target_titan", action)
        self.assertIsInstance(action["target_titan"], int)

    def test_reward_shaping(self):
        """测试奖励塑造功能"""
        # 创建原始奖励
        original_rewards = {
            "hero_agent_1": 0.5,
            "hero_agent_2": 0.4
        }
        # 创建动作
        actions = {
            "hero_agent_1": {"type": "制定规则", "target_titan": 0},
            "hero_agent_2": {"type": "发动攻击", "target_titan": 1}
        }
        # 塑造奖励
        shaped_rewards = self.reward_shaper.shape_rewards(original_rewards, actions, "dawn")
        # 检查结果
        self.assertIsInstance(shaped_rewards, dict)
        self.assertEqual(len(shaped_rewards), len(original_rewards))
        for reward in shaped_rewards.values():
            self.assertGreaterEqual(reward, 0.0)
            self.assertLessEqual(reward, 1.0)

    def test_rl_module_initialization(self):
        """测试强化学习模块初始化"""
        # 创建智能体字典
        agents = {
            "hero_agent_1": HeroAgent("hero_agent_1", "秩序", self.kb),
            "hero_agent_2": HeroAgent("hero_agent_2", "毁灭", self.kb)
        }
        # 初始化RL模块
        rl_module = MARLTrainingModule(
            agents=agents,
            state_dim=config.environment['state_dim'],
            action_dim=10
        )
        # 检查智能体是否正确初始化
        self.assertEqual(len(rl_module.dqn_agents), 2)
        self.assertIn("hero_agent_1", rl_module.dqn_agents)
        self.assertIn("hero_agent_2", rl_module.dqn_agents)

    def test_rl_action_selection(self):
        """测试RL模块动作选择"""
        # 创建智能体字典
        agents = {
            "hero_agent_1": HeroAgent("hero_agent_1", "秩序", self.kb),
            "hero_agent_2": HeroAgent("hero_agent_2", "毁灭", self.kb)
        }
        # 初始化RL模块
        rl_module = MARLTrainingModule(
            agents=agents,
            state_dim=config.environment['state_dim'],
            action_dim=10
        )
        # 获取状态
        state = self.environment.get_state()
        # 选择动作
        actions = rl_module.select_actions(state)
        # 检查动作
        self.assertIsInstance(actions, dict)
        self.assertEqual(len(actions), 2)
        for action in actions.values():
            self.assertIsInstance(action, int)
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, 10)

    def test_logger_functionality(self):
        """测试日志器功能"""
        # 记录测试日志
        episode = 1
        step = 5
        state = self.environment.get_state()
        actions = {
            "hero_agent_1": 0,
            "hero_agent_2": 3
        }
        rewards = {
            "hero_agent_1": 0.6,
            "hero_agent_2": 0.4
        }
        # 记录日志
        self.logger.log_step(episode, step, state, actions, rewards, False)
        # 保存日志
        run_id = self.logger.save_log()
        # 检查日志文件是否存在
        log_file = os.path.join(self.test_log_dir, f"simulation_log_{run_id}.json")
        self.assertTrue(os.path.exists(log_file))
        # 加载日志并验证
        with open(log_file, 'r') as f:
            log_data = json.load(f)
        self.assertEqual(log_data['run_id'], run_id)
        self.assertEqual(len(log_data['steps']), 1)

    def test_admin_agent_functionality(self):
        """测试管理员智能体功能"""
        # 创建管理员智能体
        admin_agent = AdminAgent("admin_agent", self.kb, self.logger)
        # 创建测试日志
        for i in range(10):
            self.logger.log_step(1, i, np.random.rand(config.environment['state_dim']),
                                {"hero_agent_1": i % 10}, {"hero_agent_1": 0.5 + i*0.01}, False)
        # 筛选日志
        filtered_logs = admin_agent.filter_logs()
        self.assertIsInstance(filtered_logs, list)
        # 检查收敛检测
        behavior_vectors = [np.random.rand(32) for _ in range(10)]
        convergence = admin_agent.detect_convergence(behavior_vectors)
        self.assertIsInstance(convergence, bool)

    def test_full_system_integration(self):
        """测试完整系统集成"""
        # 创建完整系统
        system = AmphoreusSystem()
        # 初始化系统
        system.initialize()
        # 运行一个简短的模拟
        num_episodes = 2
        max_steps = 5
        system.run_simulation(num_episodes=num_episodes, max_steps_per_episode=max_steps)
        # 检查结果
        self.assertEqual(len(system.logger.log_data['steps']), num_episodes * max_steps)
        self.assertGreater(len(system.rl_module.dqn_agents), 0)

if __name__ == '__main__':
    # 运行所有测试
    unittest.main()