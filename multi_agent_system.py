import os
from typing import Dict, List, Any, Optional
from langchain.llms import Ollama
from langchain.chat_models import ChatOllama
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.agents import AgentType, initialize_agent, AgentExecutor, BaseMultiActionAgent
from langchain.schema import AgentAction, AgentFinish
from langchain.prompts import StringPromptTemplate
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.utilities import SerpAPIWrapper
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
import threading
import time
import json

# 设置Ollama模型和API
os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
MODEL_NAME = "llama2:13b"

# 知识库类 - 存储和检索世界观信息
class WorldKnowledgeBase:
    def __init__(self, directory="world_knowledge"):
        self.embeddings = OllamaEmbeddings(model=MODEL_NAME)
        self.directory = directory
        self.knowledge_db = None
        self._initialize_db()
        
    def _initialize_db(self):
        # 加载知识库文档
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            # 创建初始文档
            with open(f"{self.directory}/base_world.txt", "w") as f:
                f.write("这是一个未来世界，人类已经在火星建立了殖民地。\n"
                        "地球上的资源日益枯竭，火星成为了人类的第二家园。\n"
                        "火星殖民地由多个势力控制，包括联合太空联盟、火星独立党和地球回归派。\n"
                        "科技已经高度发达，人们普遍使用脑机接口和量子计算设备。\n"
                        "然而，不同势力之间的矛盾日益尖锐，战争一触即发。")
        
        # 加载文档并创建向量数据库
        loader = DirectoryLoader(self.directory, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        # 如果已有数据库，从磁盘加载；否则创建新的
        if os.path.exists(f"{self.directory}/.chroma") and os.listdir(f"{self.directory}/.chroma"):
            self.knowledge_db = Chroma(embedding_function=self.embeddings, 
                                       persist_directory=f"{self.directory}/.chroma")
        else:
            self.knowledge_db = Chroma.from_documents(documents=splits, 
                                                     embedding=self.embeddings,
                                                     persist_directory=f"{self.directory}/.chroma")
        self.knowledge_db.persist()
    
    def add_knowledge(self, source: str, content: str):
        """添加新的知识条目"""
        # 保存到文本文件
        timestamp = int(time.time())
        filename = f"{self.directory}/{source}_{timestamp}.txt"
        with open(filename, "w") as f:
            f.write(content)
        
        # 更新向量数据库
        doc = [Document(page_content=content, metadata={"source": source, "timestamp": timestamp})]
        self.knowledge_db.add_documents(doc)
        self.knowledge_db.persist()
    
    def query_knowledge(self, query: str, k: int = 5) -> List[str]:
        """查询相关知识"""
        docs = self.knowledge_db.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

# 自定义工具 - 访问知识库
def knowledge_tool_factory(knowledge_base: WorldKnowledgeBase):
    def query_world_knowledge(query: str) -> str:
        """查询世界观知识库，获取相关信息"""
        results = knowledge_base.query_knowledge(query)
        return "\n\n".join(results)
    
    return Tool(
        name="WorldKnowledge",
        func=query_world_knowledge,
        description="当需要了解世界观相关信息时使用，例如势力、科技、历史等"
    )

# 自定义工具 - 记录观察
def recorder_tool_factory(agent_name: str, interaction_history: Dict):
    def record_observation(observation: str) -> str:
        """记录智能体的观察或行为"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        if agent_name not in interaction_history:
            interaction_history[agent_name] = []
        interaction_history[agent_name].append({
            "timestamp": timestamp,
            "observation": observation
        })
        return f"已记录: {timestamp} - {agent_name}: {observation}"
    
    return Tool(
        name="RecordObservation",
        func=record_observation,
        description="用于记录你的观察或行为，以便后续参考"
    )

# 智能体类
class CustomAgent:
    def __init__(self, name: str, role: str, knowledge_base: WorldKnowledgeBase, 
                 interaction_history: Dict, tools: Optional[List[Tool]] = None):
        self.name = name
        self.role = role
        self.knowledge_base = knowledge_base
        self.interaction_history = interaction_history
        self.llm = ChatOllama(model=MODEL_NAME)
        
        # 基础工具
        if tools is None:
            self.tools = [
                knowledge_tool_factory(knowledge_base),
                recorder_tool_factory(name, interaction_history)
            ]
        else:
            self.tools = tools
        
        # 初始化记忆
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        
        # 设置系统消息
        system_message = SystemMessage(
            content=f"""你是{name}，{role}。你存在于一个未来世界，人类已经在火星建立了殖民地。
地球上的资源日益枯竭，火星成为了人类的第二家园。
火星殖民地由多个势力控制，包括联合太空联盟、火星独立党和地球回归派。
科技已经高度发达，人们普遍使用脑机接口和量子计算设备。
然而，不同势力之间的矛盾日益尖锐，战争一触即发。

你可以与其他智能体交流，获取信息，做出决策。
你可以使用提供的工具来查询世界观知识或记录你的观察。
保持角色一致性，根据你的角色和目标进行互动。"""
        )
        
        # 初始化代理
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            agent_kwargs={"system_message": system_message}
        )
    
    def interact(self, message: str) -> str:
        """与智能体交互"""
        return self.agent.run(input=message)

# 多智能体管理器
class MultiAgentManager:
    def __init__(self, knowledge_base: WorldKnowledgeBase):
        self.knowledge_base = knowledge_base
        self.interaction_history = {}
        self.agents = {}
        self.agent_threads = {}
        self.running = False
    
    def add_agent(self, name: str, role: str, tools: Optional[List[Tool]] = None):
        """添加智能体"""
        agent = CustomAgent(name, role, self.knowledge_base, self.interaction_history, tools)
        self.agents[name] = agent
        return agent
    
    def start_conversation(self, initial_prompt: str, max_turns: int = 10):
        """启动多智能体对话"""
        self.running = True
        current_speaker = list(self.agents.keys())[0]
        message = initial_prompt
        
        for turn in range(max_turns):
            if not self.running:
                break
                
            print(f"\n--- 第 {turn+1} 轮对话 ---")
            print(f"{current_speaker}:")
            
            # 当前智能体回复
            response = self.agents[current_speaker].interact(message)
            print(response)
            
            # 记录对话
            if "conversation" not in self.interaction_history:
                self.interaction_history["conversation"] = []
            self.interaction_history["conversation"].append({
                "turn": turn+1,
                "speaker": current_speaker,
                "message": message,
                "response": response
            })
            
            # 决定下一个发言者（简化版：轮流发言）
            speaker_index = list(self.agents.keys()).index(current_speaker)
            next_index = (speaker_index + 1) % len(self.agents)
            current_speaker = list(self.agents.keys())[next_index]
            
            # 下一个智能体的输入
            message = f"""你是{current_speaker}，{self.agents[current_speaker].role}。
{self.agents[list(self.agents.keys())[speaker_index]].name}刚刚说："{response}"
请做出回应。"""
            
            # 短暂延迟以便观察
            time.sleep(1)
    
    def stop_conversation(self):
        """停止对话"""
        self.running = False
    
    def get_interaction_history(self):
        """获取交互历史"""
        return self.interaction_history

# 主函数
def main():
    # 初始化知识库
    knowledge_base = WorldKnowledgeBase()
    
    # 初始化多智能体管理器
    manager = MultiAgentManager(knowledge_base)
    
    # 添加13个智能体
    # 联合太空联盟成员
    manager.add_agent("指挥官艾伦", "联合太空联盟火星基地指挥官，负责维护基地安全和运营")
    manager.add_agent("科学家索菲亚", "联合太空联盟首席科学家，研究火星资源开发技术")
    manager.add_agent("工程师马克", "联合太空联盟高级工程师，负责基地设施维护和升级")
    
    # 火星独立党成员
    manager.add_agent("领袖卡洛斯", "火星独立党领袖，致力于火星脱离地球统治")
    manager.add_agent("宣传员娜奥米", "火星独立党宣传负责人，负责舆论引导")
    manager.add_agent("战术家张", "火星独立党军事战术专家，制定抵抗策略")
    
    # 地球回归派成员
    manager.add_agent("代表伊丽莎白", "地球回归派政治代表，主张回归地球统治")
    manager.add_agent("牧师约书亚", "地球回归派精神领袖，宣扬地球至上教义")
    manager.add_agent("安保主管凯特", "地球回归派安保负责人，保护派系成员安全")
    
    # 中立角色
    manager.add_agent("记者阿米尔", "独立记者，报道火星上的各类事件和冲突")
    manager.add_agent("商人陈", "星际商人，在各势力间进行贸易活动")
    manager.add_agent("医生艾米丽", "火星殖民地首席医疗官，负责所有人的健康")
    manager.add_agent("矿工罗德里戈", "火星资源矿工，负责采集火星上的稀有矿物")
    
    # 启动对话
    initial_prompt = "刚刚收到消息，地球方面切断了对火星的物资供应。各派系有什么反应？"
    manager.start_conversation(initial_prompt, max_turns=20)
    
    # 保存交互历史
    with open("interaction_history.json", "w") as f:
        json.dump(manager.get_interaction_history(), f, indent=2)

if __name__ == "__main__":
    main()  