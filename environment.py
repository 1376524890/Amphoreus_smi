import json
import numpy as np
from typing import Dict, Any, List

class AmphoreusEnvironment:
    def __init__(self, worldview_kb, initial_stage: str = "启蒙世"):
        self.worldview_kb = worldview_kb
        self.world_data = worldview_kb.worldview_data
        self.current_stage = initial_stage
        self.stage_index = self._get_stage_index(initial_stage)
        self.stages = self.world_data.get("迭代发展阶段", [])
        self.titan_states = self._initialize_titan_states()
        self.civilization_metrics = self._initialize_civilization_metrics()
        self.black_tide_intensity = 0.0  # 黑潮强度，范围0-1
        self.state_dim = 15  # 状态维度，可根据实际需要调整
        self.action_dim = 10  # 动作维度，可根据实际需要调整
        self.stage_transition_thresholds = {
            "启蒙世": 0.7,
            "造物世": 0.6,
            "黄金世": 0.8,
            "纷争世": 0.5,
            "幻灭世": 0.4
        }

    def _get_stage_index(self, stage_name: str) -> int:
        """获取当前阶段在迭代发展阶段中的索引"""
        for i, stage in enumerate(self.stages):
            if stage.get("阶段名称") == stage_name:
                return i
        return 0

    def _initialize_titan_states(self) -> Dict[str, Any]:
        """初始化泰坦神明状态"""
        titan_states = {}
        # 初始化命运三泰坦
        for titan in self.world_data.get("泰坦介绍", {}).get("命运三泰坦", []):
            titan_name = titan.get("名称", "未知泰坦")
            titan_states[titan_name] = {
                "awake": True,
                "power_level": 1.0,
                "influence": 0.5
            }
        # 初始化支柱三泰坦
        for titan in self.world_data.get("泰坦介绍", {}).get("支柱三泰坦", []):
            titan_name = titan.get("名称", "未知泰坦")
            titan_states[titan_name] = {
                "awake": True,
                "power_level": 1.0,
                "influence": 0.5
            }
        # 初始化其他泰坦
        for titan in self.world_data.get("泰坦介绍", {}).get("其他泰坦", []):
            titan_name = titan.get("名称", "未知泰坦")
            titan_states[titan_name] = {
                "awake": False,
                "power_level": 0.3,
                "influence": 0.1
            }
        return titan_states

    def _initialize_civilization_metrics(self) -> Dict[str, float]:
        """初始化文明发展指标"""
        return {
            "technology_level": 0.1,
            "cultural_development": 0.1,
            "population": 0.2,
            "resource_abundance": 0.8,
            "social_stability": 0.5
        }

    def get_state(self) -> np.ndarray:
        """获取当前环境状态向量"""
        # 阶段编码（独热向量）
        stage_onehot = np.zeros(len(self.stages))
        stage_onehot[self.stage_index] = 1
        
        # 泰坦状态摘要（平均活跃度和影响力）
        titan_awake_ratio = np.mean([1.0 if t["awake"] else 0.0 for t in self.titan_states.values()])
        titan_avg_influence = np.mean([t["influence"] for t in self.titan_states.values()])
        
        # 文明指标
        civ_metrics = np.array(list(self.civilization_metrics.values()))
        
        # 黑潮强度
        black_tide = np.array([self.black_tide_intensity])
        
        # 组合所有状态特征
        state = np.concatenate([
            stage_onehot,
            np.array([titan_awake_ratio, titan_avg_influence]),
            civ_metrics,
            black_tide
        ])
        
        return state

    def _update_titan_states(self, actions: Dict[str, Any]) -> None:
        """根据智能体动作更新泰坦状态"""
        # 根据不同命途智能体的动作影响相应泰坦
        for agent_id, action in actions.items():
            agent = self.worldview_kb.get_agent_by_id(agent_id)
            if not agent:
                continue
            
            destiny = agent.get("destiny")
            action_type = action.get("type")
            target_titan = action.get("target_titan")
            
            if not target_titan or target_titan not in self.titan_states:
                continue
            
            # 根据命途和动作类型调整泰坦状态
            if destiny == "秩序":
                if action_type == "加强秩序":
                    self.titan_states[target_titan]["influence"] = min(1.0, self.titan_states[target_titan]["influence"] + 0.1)
            elif destiny == "同谐":
                if action_type == "促进协作":
                    self.titan_states[target_titan]["influence"] = min(1.0, self.titan_states[target_titan]["influence"] + 0.08)
            elif destiny == "毁灭":
                if action_type == "发动攻击":
                    self.titan_states[target_titan]["power_level"] = max(0.0, self.titan_states[target_titan]["power_level"] - 0.15)
            # 其他命途的影响...

    def _update_civilization_metrics(self) -> None:
        """更新文明发展指标"""
        # 基于泰坦状态更新文明指标
        avg_influence = np.mean([t["influence"] for t in self.titan_states.values()])
        
        # 技术水平更新
        self.civilization_metrics["technology_level"] = min(1.0, 
            self.civilization_metrics["technology_level"] + 0.01 * avg_influence)
        
        # 文化发展更新
        self.civilization_metrics["cultural_development"] = min(1.0, 
            self.civilization_metrics["cultural_development"] + 0.008 * avg_influence)
        
        # 人口更新
        pop_growth_rate = 0.02 * (1 - self.black_tide_intensity) * avg_influence
        self.civilization_metrics["population"] = min(1.0, 
            self.civilization_metrics["population"] + pop_growth_rate)
        
        # 资源丰度更新（随人口增长而减少，随技术发展而增加）
        resource_change = 0.01 * self.civilization_metrics["technology_level"] - 0.015 * self.civilization_metrics["population"]
        self.civilization_metrics["resource_abundance"] = max(0.0, 
            self.civilization_metrics["resource_abundance"] + resource_change)
        
        # 社会稳定性更新
        stability_factor = 0.5 * self.civilization_metrics["resource_abundance"] + 0.3 * avg_influence - 0.2 * self.black_tide_intensity
        self.civilization_metrics["social_stability"] = max(0.0, min(1.0, stability_factor))

    def _update_black_tide(self) -> None:
        """更新黑潮强度"""
        # 黑潮强度基于当前阶段和文明状态变化
        if self.current_stage == "纷争世" or self.current_stage == "幻灭世":
            # 在纷争世和幻灭世，黑潮强度增加
            self.black_tide_intensity = min(1.0, self.black_tide_intensity + 0.02)
        else:
            # 在其他阶段，黑潮强度缓慢减弱
            self.black_tide_intensity = max(0.0, self.black_tide_intensity - 0.01)

    def _calculate_rewards(self) -> Dict[str, float]:
        """计算每个智能体的奖励"""
        rewards = {}
        
        # 基础奖励：基于文明发展指标
        base_reward = np.mean(list(self.civilization_metrics.values()))
        
        # 为每个智能体计算特定奖励
        for agent_id in self.worldview_kb.get_all_agent_ids():
            agent = self.worldview_kb.get_agent_by_id(agent_id)
            if not agent:
                continue
            
            destiny = agent.get("destiny")
            reward = base_reward
            
            # 根据命途特性调整奖励
            if destiny == "丰饶":
                # 丰饶命途奖励人口增长和资源丰度
                reward += 0.3 * self.civilization_metrics["population"]
                reward += 0.2 * self.civilization_metrics["resource_abundance"]
            elif destiny == "存护":
                # 存护命途奖励社会稳定性
                reward += 0.5 * self.civilization_metrics["social_stability"]
            elif destiny == "毁灭":
                # 毁灭命途奖励黑潮强度
                reward += 0.4 * self.black_tide_intensity
            elif destiny == "秩序":
                # 秩序命途奖励社会稳定性和技术水平
                reward += 0.25 * self.civilization_metrics["social_stability"]
                reward += 0.25 * self.civilization_metrics["technology_level"]
            # 其他命途的奖励计算...
            
            # 确保奖励在合理范围内
            rewards[agent_id] = max(0.0, min(1.0, reward))
        
        return rewards

    def should_progress_stage(self) -> bool:
        """判断是否应该进入下一阶段"""
        # 如果是最后阶段，不继续前进
        if self.stage_index >= len(self.stages) - 1:
            return False
        
        # 获取当前阶段的转换阈值
        threshold = self.stage_transition_thresholds.get(self.current_stage, 0.6)
        
        # 计算文明发展综合指数
        civ_score = np.mean(list(self.civilization_metrics.values()))
        
        # 如果文明指数达到阈值，或者在幻灭世且黑潮强度足够高，则进入下一阶段
        if (civ_score >= threshold) or (
            self.current_stage == "幻灭世" and self.black_tide_intensity >= 0.8):
            return True
        
        return False

    def progress_stage(self) -> None:
        """进入下一阶段"""
        if self.stage_index < len(self.stages) - 1:
            self.stage_index += 1
            self.current_stage = self.stages[self.stage_index].get("阶段名称")
            # 新阶段重置部分状态
            self._reset_stage_state()
            print(f"已进入新阶段: {self.current_stage}")

    def _reset_stage_state(self) -> None:
        """重置新阶段的状态"""
        # 根据新阶段特性调整泰坦状态
        if self.current_stage == "纷争世":
            # 纷争世：部分泰坦苏醒，黑潮强度增加
            for titan_name in self.titan_states:
                if "纷争" in titan_name or "毁灭" in titan_name:
                    self.titan_states[titan_name]["awake"] = True
                    self.titan_states[titan_name]["power_level"] = min(1.0, self.titan_states[titan_name]["power_level"] + 0.3)
            self.black_tide_intensity = min(1.0, self.black_tide_intensity + 0.2)
        elif self.current_stage == "幻灭世":
            # 幻灭世：大部分泰坦休眠，黑潮强度达到峰值
            for titan_name in self.titan_states:
                if "创生" in titan_name or "丰饶" in titan_name:
                    self.titan_states[titan_name]["awake"] = False
            self.black_tide_intensity = 0.9
        elif self.current_stage == "再创世":
            # 再创世：重置大部分状态
            self.titan_states = self._initialize_titan_states()
            self.civilization_metrics = self._initialize_civilization_metrics()
            self.black_tide_intensity = 0.0

    def step(self, actions: Dict[str, Any]) -> tuple[np.ndarray, Dict[str, float]]:
        """环境一步迭代
        Args:
            actions: 智能体动作字典
        Returns:
            next_state: 下一状态
            rewards: 奖励字典
        """
        # 更新泰坦状态
        self._update_titan_states(actions)
        
        # 更新文明指标
        self._update_civilization_metrics()
        
        # 更新黑潮强度
        self._update_black_tide()
        
        # 计算奖励
        rewards = self._calculate_rewards()
        
        # 获取下一状态
        next_state = self.get_state()
        
        return next_state, rewards

    def reset(self) -> np.ndarray:
        """重置环境到初始状态"""
        self.current_stage = self.stages[0].get("阶段名称")
        self.stage_index = 0
        self.titan_states = self._initialize_titan_states()
        self.civilization_metrics = self._initialize_civilization_metrics()
        self.black_tide_intensity = 0.0
        self.previous_behavior_vector = None
        return self.get_state()