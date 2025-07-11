import numpy as np

class Config:
    """系统配置参数类"""
    def __init__(self):
        # 系统基本设置
        self.system_name = "AMPHOREUS多智能体强化学习系统"
        self.version = "1.0.0"
        self.seed = 42  # 随机种子，确保可复现性
        self.max_episodes = 1000  # 最大训练回合数
        self.max_steps_per_episode = 200  # 每回合最大步数
        self.save_interval = 100  # 模型保存间隔（回合数）
        self.log_interval = 10  # 日志打印间隔（回合数）
        self.convergence_threshold = 0.95  # 收敛阈值
        self.convergence_window = 50  # 收敛检测窗口大小

        # 环境配置
        self.environment = {
            "state_dim": 15,  # 状态向量维度
            "num_titans": 5,  # 泰坦数量
            "civilization_metrics": ["technology", "culture", "population", "resources", "stability"],
            "black_tide_initial_strength": 0.2,
            "black_tide_increase_rate": 0.01,
            "stage_transition_thresholds": {
                "dawn": 0.3,
                "growth": 0.5,
                "prosperity": 0.7,
                "crisis": 0.6,
                "collapse": 0.3,
                "illusion": 0.8
            }
        }

        # 强化学习配置
        self.rl = {
            "gamma": 0.99,  # 折扣因子
            "learning_rate": 0.001,  # 学习率
            "batch_size": 64,  # 批次大小
            "memory_capacity": 10000,  # 经验回放缓冲区大小
            "epsilon_start": 1.0,  # 初始探索率
            "epsilon_min": 0.01,  # 最小探索率
            "epsilon_decay": 0.995,  # 探索率衰减率
            "target_update_interval": 10,  # 目标网络更新间隔
            "hidden_dim": 64,  # 神经网络隐藏层维度
            "behavior_vector_dim": 32,  # 行为向量维度
            "reward_shaping": {
                "alpha": 0.5,  # 命途契合度权重
                "beta": 0.3,   # 阶段目标权重
                "gamma": 0.2  # 协作权重
            }
        }

        # 智能体配置
        self.agents = {
            "num_hero_agents": 11,  # 英雄智能体数量
            "admin_agent": {
                "evaluation_threshold": 0.7,  # 知识重要性阈值
                "log_filter_window": 100,  # 日志筛选窗口大小
                "worldview_update_interval": 50  # 世界观更新间隔
            },
            # 各命途智能体的特定配置
            "destiny_specific": {
                "秩序": {"action_cooldown": 3, "influence_range": 0.8},
                "同谐": {"action_cooldown": 2, "influence_range": 0.9},
                "毁灭": {"action_cooldown": 4, "influence_range": 0.7},
                "存护": {"action_cooldown": 3, "influence_range": 0.85},
                "丰饶": {"action_cooldown": 2, "influence_range": 0.8},
                "智识": {"action_cooldown": 3, "influence_range": 0.75},
                "虚无": {"action_cooldown": 5, "influence_range": 0.6},
                "繁育": {"action_cooldown": 2, "influence_range": 0.8},
                "贪饕": {"action_cooldown": 3, "influence_range": 0.7},
                "欢愉": {"action_cooldown": 2, "influence_range": 0.75},
                "开拓": {"action_cooldown": 4, "influence_range": 0.8}
            }
        }

        # 文件路径配置
        self.paths = {
            "worldview_file": "./世界观.json",
            "personality_file": "./《崩坏：星穹铁道》命途智能体个性设定.md",
            "models_dir": "./models",
            "logs_dir": "./logs",
            "results_dir": "./results"
        }

    def print_config_summary(self):
        """打印配置摘要"""
        print(f"{self.system_name} v{self.version} 配置摘要:")
        print(f"- 随机种子: {self.seed}")
        print(f"- 最大训练回合数: {self.max_episodes}")
        print(f"- 状态维度: {self.environment['state_dim']}")
        print(f"- 学习率: {self.rl['learning_rate']}")
        print(f"- 折扣因子: {self.rl['gamma']}")
        print(f"- 收敛阈值: {self.convergence_threshold}")
        print(f"- 模型保存路径: {self.paths['models_dir']}")

    def get_destiny_config(self, destiny_name):
        """获取指定命途的配置"""
        return self.agents['destiny_specific'].get(destiny_name, {})

    def update_config(self, new_config):
        """更新配置参数
        Args:
            new_config: 包含新配置参数的字典
        """
        for key, value in new_config.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"警告: 配置参数 {key} 不存在")

# 创建配置实例
config = Config()