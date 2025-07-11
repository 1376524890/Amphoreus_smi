import json
import re
from typing import Dict, Any, List
import markdown
from bs4 import BeautifulSoup

class WorldviewKnowledgeBase:
    def __init__(self, worldview_path: str, personality_path: str):
        self.worldview_path = worldview_path
        self.personality_path = personality_path
        self.worldview_data = self._load_worldview()
        self.personalities = self._load_personalities()
        self.agent_profiles = self._create_agent_profiles()
        self.supplementary_knowledge = []

    def _load_worldview(self) -> Dict[str, Any]:
        """加载世界观JSON文件"""
        try:
            with open(self.worldview_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载世界观文件失败: {e}")
            return {
                "标题": "默认世界观",
                "世界背景": "未知世界",
                "迭代发展阶段": [],
                "泰坦介绍": {},
                "命途介绍": []
            }

    def _load_personalities(self) -> Dict[str, Dict[str, Any]]:
        """解析命途智能体个性设定Markdown文件"""
        personalities = {}
        try:
            with open(self.personality_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
                html_content = markdown.markdown(md_content)
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # 查找所有智能体个性设定部分
                sections = soup.find_all('h3')  # Markdown中的###会被解析为h3
                for section in sections:
                    # 提取命途名称
                    title_text = section.get_text(strip=True)
                    destiny_match = re.search(r'^(\d+)\.\s*([^ ]+)智能体$', title_text)
                    if not destiny_match:
                        continue
                    
                    destiny_name = destiny_match.group(2)
                    personality = {}
                    
                    # 获取当前部分的所有兄弟节点直到下一个h3
                    next_node = section.next_sibling
                    while next_node and next_node.name != 'h3':
                        if next_node.name == 'p':
                            p_text = next_node.get_text(strip=True)
                            # 解析哲学内涵、终极目标、行为准则
                            if p_text.startswith('**哲学内涵**：'):
                                personality['哲学内涵'] = p_text[len('**哲学内涵**：'):]
                            elif p_text.startswith('**终极目标**：'):
                                personality['终极目标'] = p_text[len('**终极目标**：'):]
                            elif p_text.startswith('**行为准则**：'):
                                personality['行为准则'] = p_text[len('**行为准则**：'):]
                        next_node = next_node.next_sibling
                    
                    if personality:
                        personalities[destiny_name] = personality
            
        except Exception as e:
            print(f"加载个性设定文件失败: {e}")
            # 添加默认个性设定
            personalities = {
                "秩序": {"哲学内涵": "坚信绝对秩序", "终极目标": "构建永恒秩序空间", "行为准则": "严格执行规则"},
                "同谐": {"哲学内涵": "秉持分化耦合", "终极目标": "整合为和谐整体", "行为准则": "促进交流合作"}
            }
        
        return personalities

    def _create_agent_profiles(self) -> Dict[str, Dict[str, Any]]:
        """创建智能体配置文件"""
        agent_profiles = {}
        destiny_list = list(self.personalities.keys())
        
        # 为每个命途创建一个智能体
        for i, destiny in enumerate(destiny_list):
            agent_id = f"hero_agent_{i+1}"
            agent_profiles[agent_id] = {
                "agent_id": agent_id,
                "destiny": destiny,
                "personality": self.personalities[destiny],
                "abilities": self._generate_abilities(destiny)
            }
            
            # 限制为11个英雄智能体
            if i >= 10:
                break
        
        return agent_profiles

    def _generate_abilities(self, destiny: str) -> List[str]:
        """根据命途生成智能体能力"""
        ability_map = {
            "秩序": ["规则制定", "秩序维护", "逻辑分析", "偏差纠正"],
            "同谐": ["冲突调解", "合作促进", "资源整合", "情感共鸣"],
            "毁灭": ["强力攻击", "设施破坏", "混乱制造", "弱点洞察"],
            "存护": ["防御构建", "盟友保护", "损伤修复", "屏障生成"],
            "丰饶": ["资源创造", "生命治愈", "生长促进", "产能提升"],
            "繁育": ["种群培育", "基因优化", "殖民地建立", "快速繁殖"],
            "虚无": ["幻觉制造", "现实扭曲", "秩序解构", "怀疑引发"],
            "巡猎": ["目标追踪", "精准打击", "潜行侦察", "威胁清除"],
            "欢愉": ["士气提升", "娱乐创造", "情感激发", "活动组织"],
            "均衡": ["平衡调节", "极端化解", "利益协调", "中立维持"]
        }
        
        return ability_map.get(destiny, ["基础观察", "信息收集", "简单交流"])

    def get_personalities(self) -> Dict[str, Dict[str, Any]]:
        """获取所有命途个性设定"""
        return self.personalities

    def get_agent_profiles(self) -> Dict[str, Dict[str, Any]]:
        """获取所有智能体配置"""
        return self.agent_profiles

    def get_agent_by_id(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取智能体配置"""
        return self.agent_profiles.get(agent_id)

    def get_all_agent_ids(self) -> List[str]:
        """获取所有智能体ID"""
        return list(self.agent_profiles.keys())

    def update_knowledge(self, new_knowledge: List[Dict[str, Any]]) -> None:
        """更新世界观知识库"""
        if not new_knowledge:
            return
        
        # 添加新的知识条目
        self.supplementary_knowledge.extend(new_knowledge)
        
        # 将重要知识整合到主世界观
        important_knowledge = [item for item in new_knowledge if item.get("importance", 0) >= 0.7]
        if important_knowledge:
            # 创建或更新世界观中的"补充知识"部分
            if "补充知识" not in self.worldview_data:
                self.worldview_data["补充知识"] = []
            
            # 去重后添加
            existing_contents = {item["content"] for item in self.worldview_data["补充知识"]}
            for item in important_knowledge:
                if item["content"] not in existing_contents:
                    self.worldview_data["补充知识"].append(item)
                    existing_contents.add(item["content"])
            
            # 保存更新后的世界观
            self._save_worldview()

    def _save_worldview(self) -> None:
        """保存更新后的世界观文件"""
        try:
            with open(self.worldview_path, 'w', encoding='utf-8') as f:
                json.dump(self.worldview_data, f, ensure_ascii=False, indent=2)
            print(f"世界观已更新并保存到 {self.worldview_path}")
        except Exception as e:
            print(f"保存世界观文件失败: {e}")

    def get_stage_info(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """获取指定阶段的信息"""
        for stage in self.worldview_data.get("迭代发展阶段", []):
            if stage.get("阶段名称") == stage_name:
                return stage
        return None

    def get_titan_info(self, titan_name: str) -> Optional[Dict[str, Any]]:
        """获取指定泰坦的信息"""
        for titan_group in self.worldview_data.get("泰坦介绍", {}).values():
            if isinstance(titan_group, list):
                for titan in titan_group:
                    if titan.get("名称") == titan_name:
                        return titan
            elif isinstance(titan_group, dict):
                if titan_group.get("名称") == titan_name:
                    return titan_group
        return None

    def get_destiny_info(self, destiny_name: str) -> Optional[Dict[str, Any]]:
        """获取指定命途的信息"""
        for destiny in self.worldview_data.get("命途介绍", []):
            if destiny.get("命途名称") == destiny_name:
                return destiny
        return None