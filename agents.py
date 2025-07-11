import json
import numpy as np
from typing import Dict, Any, List, Optional
from langchain_community.llms import OpenAI
from langchain.chains.llm import LLMChain
from langchain_community.prompts import PromptTemplate
import re

class BaseAgent:
    def __init__(self, agent_id: str, worldview_kb):
        self.agent_id = agent_id
        self.worldview_kb = worldview_kb
        self.memory = []
        self.destiny = "未知"
        self.personality = {}
        self.action_history = []
        self.reward_history = []
        self.llm = OpenAI(temperature=0.7)

    def take_action(self, state: np.ndarray) -> Dict[str, Any]:
        """根据当前状态采取行动"""
        raise NotImplementedError("子类必须实现take_action方法")

    def learn(self, reward: float) -> None:
        """根据奖励进行学习"""
        self.reward_history.append(reward)

    def save_memory(self) -> List[Dict[str, Any]]:
        """保存智能体记忆"""
        return self.memory

    def load_memory(self, memory: List[Dict[str, Any]]) -> None:
        """加载智能体记忆"""
        self.memory = memory

class HeroAgent(BaseAgent):
    def __init__(self, agent_id: str, destiny: str, personality: Dict[str, Any], worldview_kb):
        super().__init__(agent_id, worldview_kb)
        self.destiny = destiny
        self.personality = personality
        self.action_space = self._define_action_space()
        self.prompt_template = self._create_prompt_template()
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def _define_action_space(self) -> List[str]:
        """根据命途定义动作空间"""
        base_actions = ["观察环境", "收集信息", "与其他智能体交流", "使用特殊能力"]
        
        # 根据不同命途添加特定动作
        if self.destiny == "秩序":
            return base_actions + ["制定规则", "执行纪律", "维护稳定", "纠正偏差"]
        elif self.destiny == "同谐":
            return base_actions + ["促进合作", "调解冲突", "建立联盟", "共享资源"]
        elif self.destiny == "毁灭":
            return base_actions + ["发动攻击", "破坏设施", "引发混乱", "削弱敌人"]
        elif self.destiny == "存护":
            return base_actions + ["构建防御", "保护盟友", "修复设施", "建立屏障"]
        elif self.destiny == "丰饶":
            return base_actions + ["创造资源", "治愈伤病", "促进生长", "提升产能"]
        elif self.destiny == "繁育":
            return base_actions + ["培育新个体", "扩张种群", "优化基因", "建立殖民地"]
        elif self.destiny == "虚无":
            return base_actions + ["制造幻觉", "扭曲现实", "解构秩序", "引发怀疑"]
        elif self.destiny == "巡猎":
            return base_actions + ["追踪目标", "精准打击", "潜行侦察", "清除威胁"]
        elif self.destiny == "欢愉":
            return base_actions + ["组织活动", "提升士气", "创造娱乐", "激发情感"]
        elif self.destiny == "均衡":
            return base_actions + ["调节平衡", "化解极端", "协调利益", "维持中立"]
        else:
            return base_actions

    def _create_prompt_template(self) -> PromptTemplate:
        """创建LLM提示模板"""
        personality_desc = self.personality.get("哲学内涵", "")
        behavior准则 = self.personality.get("行为准则", "")
        
        template = f"""
        你是{self.destiny}命途的智能体，性格特点：{personality_desc}
        你的行为准则：{behavior准则}
        当前环境状态：{{state_description}}
        你的可用动作：{self.action_space}
        你的历史动作：{{action_history}}
        请根据你的命途特性和当前环境，从可用动作中选择一个最合适的动作，并说明理由。
        输出格式：动作: [选择的动作] 理由: [你的理由]
        """
        
        return PromptTemplate(
            input_variables=["state_description", "action_history"],
            template=template
        )

    def _state_to_description(self, state: np.ndarray) -> str:
        """将状态向量转换为自然语言描述"""
        # 获取当前阶段
        stage_index = np.argmax(state[:len(self.worldview_kb.worldview_data.get("迭代发展阶段", []))])
        stages = self.worldview_kb.worldview_data.get("迭代发展阶段", [])
        current_stage = stages[stage_index].get("阶段名称", "未知阶段") if stages else "未知阶段"
        
        # 泰坦状态
        titan_awake_ratio = state[len(stages)]
        titan_avg_influence = state[len(stages)+1]
        
        # 文明指标
        tech_level = state[len(stages)+2]
        cultural_dev = state[len(stages)+3]
        population = state[len(stages)+4]
        resources = state[len(stages)+5]
        stability = state[len(stages)+6]
        
        # 黑潮强度
        black_tide = state[len(stages)+7]
        
        description = f"当前阶段：{current_stage}，泰坦苏醒比例：{titan_awake_ratio:.2f}，泰坦平均影响力：{titan_avg_influence:.2f}，"
        description += f"技术水平：{tech_level:.2f}，文化发展：{cultural_dev:.2f}，人口：{population:.2f}，"
        description += f"资源丰度：{resources:.2f}，社会稳定性：{stability:.2f}，黑潮强度：{black_tide:.2f}"
        
        return description

    def take_action(self, state: np.ndarray) -> Dict[str, Any]:
        """根据当前状态采取行动"""
        # 将状态向量转换为自然语言描述
        state_description = self._state_to_description(state)
        
        # 准备历史动作信息
        action_history = "；".join([f"{a['action']}（理由：{a['reason']}）" for a in self.action_history[-3:]])
        if not action_history:
            action_history = "无"
        
        # 使用LLM生成动作决策
        response = self.llm_chain.run(
            state_description=state_description,
            action_history=action_history
        )
        
        # 解析LLM响应
        action_match = re.search(r'动作: (.*?) 理由: (.*)', response)
        if action_match:
            action = action_match.group(1).strip()
            reason = action_match.group(2).strip()
        else:
            # 如果解析失败，随机选择一个动作
            action = np.random.choice(self.action_space)
            reason = "无法解析LLM响应，随机选择动作"
        
        # 记录动作
        action_info = {
            "action": action,
            "reason": reason,
            "state": state_description,
            "timestamp": time.time()
        }
        self.action_history.append(action_info)
        
        # 确定目标泰坦（如果需要）
        target_titan = self._determine_target_titan(action, state_description)
        
        return {
            "type": action,
            "target_titan": target_titan,
            "reason": reason,
            "agent_id": self.agent_id,
            "destiny": self.destiny
        }

    def _determine_target_titan(self, action: str, state_description: str) -> Optional[str]:
        """确定动作的目标泰坦（如果适用）"""
        # 需要目标的动作类型
        target_required_actions = ["发动攻击", "制定规则", "促进合作", "构建防御", "创造资源", "培育新个体"]
        
        if action not in target_required_actions:
            return None
        
        # 获取所有泰坦名称
        titan_names = list(self.worldview_kb.worldview_data.get("泰坦介绍", {}).keys())
        if not titan_names:
            return None
        
        # 简单规则：根据动作类型和命途选择目标泰坦
        if self.destiny == "毁灭" and action == "发动攻击":
            # 毁灭命途优先攻击秩序相关泰坦
            for titan_name in titan_names:
                if "秩序" in titan_name or "公正" in titan_name:
                    return titan_name
        elif self.destiny == "秩序" and action == "制定规则":
            # 秩序命途优先针对混乱相关泰坦
            for titan_name in titan_names:
                if "纷争" in titan_name or "诡计" in titan_name:
                    return titan_name
        
        # 如果没有特定目标，随机选择一个泰坦
        return np.random.choice(titan_names)

class AdminAgent(BaseAgent):
    def __init__(self, agent_id: str, worldview_kb, convergence_threshold: float = 0.05):
        super().__init__(agent_id, worldview_kb)
        self.convergence_threshold = convergence_threshold
        self.evaluation_prompt = PromptTemplate(
            input_variables=["logs", "stage", "civilization_metrics"],
            template="""
            作为管理员智能体，你需要根据以下日志评估各个英雄智能体的表现：
            {logs}
            当前阶段：{stage}
            文明发展指标：{civilization_metrics}
            请为每个智能体打分（0-10分），并说明评分理由。
            输出格式：智能体ID: 分数 理由: ...
            """
        )
        self.evaluation_chain = LLMChain(llm=self.llm, prompt=self.evaluation_prompt)
        self.knowledge_filter_prompt = PromptTemplate(
            input_variables=["logs", "current_worldview"],
            template="""
            从以下日志中筛选有利于世界观收敛的条目，这些条目应该：
            1. 提供新的世界观信息
            2. 反映智能体对世界观的理解
            3. 有助于推动世界发展
            日志：{logs}
            当前世界观：{current_worldview}
            请输出筛选后的条目，每条一个JSON对象，包含"content"和"importance"字段（0-1）。
            """
        )
        self.knowledge_filter_chain = LLMChain(llm=self.llm, prompt=self.knowledge_filter_prompt)

    def evaluate_rewards(self, iteration: int, logs: Dict[str, Any]) -> Dict[str, float]:
        """评估并更新奖励参数"""
        # 提取相关日志信息
        stage = logs.get("state", {}).get("current_stage", "未知阶段")
        civilization_metrics = logs.get("state", {}).get("civilization_metrics", {})
        
        # 格式化日志信息
        formatted_logs = []
        for agent_id, action in logs.get("actions", {}).items():
            formatted_logs.append(f"智能体{agent_id}：{action['action']}（理由：{action['reason']}）")
        formatted_logs = "\n".join(formatted_logs)
        
        # 格式化文明指标
        formatted_civ_metrics = ", ".join([f"{k}：{v:.2f}" for k, v in civilization_metrics.items()])
        
        # 使用LLM进行评估
        response = self.evaluation_chain.run(
            logs=formatted_logs,
            stage=stage,
            civilization_metrics=formatted_civ_metrics
        )
        
        # 解析评估结果
        rewards = {}
        lines = response.split("\n")
        for line in lines:
            if ":" in line and "理由:" in line:
                agent_part, reason_part = line.split("理由:")
                agent_id, score_str = agent_part.split(":")
                agent_id = agent_id.strip()
                score = float(score_str.strip()) / 10.0  # 归一化到0-1
                rewards[agent_id] = score
        
        return rewards

    def filter_useful_knowledge(self, all_logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """筛选有利于收敛的日志条目作为世界观补充"""
        # 格式化日志
        formatted_logs = []
        for i, log in enumerate(all_logs[-5:]):  # 只看最近5次迭代
            formatted_logs.append(f"迭代{i+1}：{log}")
        formatted_logs = "\n".join(formatted_logs)
        
        # 获取当前世界观
        current_worldview = json.dumps(self.worldview_kb.worldview_data, ensure_ascii=False)
        
        # 使用LLM筛选知识
        response = self.knowledge_filter_chain.run(
            logs=formatted_logs,
            current_worldview=current_worldview
        )
        
        # 解析筛选结果
        try:
            # 尝试解析JSON数组
            useful_knowledge = json.loads(response)
            if isinstance(useful_knowledge, list):
                return useful_knowledge
            else:
                return [useful_knowledge]
        except:
            # 如果解析失败，手动提取条目
            knowledge_entries = []
            for line in response.split("\n"):
                if "content" in line and "importance" in line:
                    try:
                        entry = eval(line.strip())
                        if isinstance(entry, dict) and "content" in entry and "importance" in entry:
                            knowledge_entries.append(entry)
                    except:
                        continue
            return knowledge_entries

    def should_update_worldview(self, behavior_changes: List[float]) -> bool:
        """判断是否应该更新世界观"""
        if len(behavior_changes) < 5:
            return False
        
        # 计算最近5次行为变化的平均值
        avg_change = np.mean(behavior_changes[-5:])
        return avg_change < self.convergence_threshold

    def take_action(self, state: np.ndarray) -> Dict[str, Any]:
        """管理员智能体动作（主要是评估和更新）"""
        return {
            "type": "评估与更新",
            "action": "评估智能体表现并更新世界观",
            "agent_id": self.agent_id
        }