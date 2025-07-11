# 翁法罗斯模拟仿真项目 
## BringStorm
* 创建13个Agent，其中12个小Agent模拟12泰坦，1个大Agent模拟管理员。
* 使用系统prompt的方式给不同Agent设置各自的身份，其中11个Agent的系统提示词固定不变，1个Agent的系统提示词根据上一次迭代结果进行优化更新，管理员Agent的提示词在每次迭代后询问用户是否需要更新，如需更新由用户自行设定
* 搜集并制作可以被所有Agent只读访问的背景知识库作为世界观背景
* 在每次迭代中记录所有agent的输出保存为json文件作为运行日志，Agent的每个动作需参考日志和世界观背景
* 管理员Agent在每一次迭代结束后筛选日志中有利于收敛的条目并将其写入另一个json文件中作为世界观补充
* Agent的每个动作均被写入日志文件中，每轮动作结束后由管理员agent根据日志评价各个agent的奖励参数，并根据目标有无达成决定是否推进时间阶段

* 需注意：
1. 如何判定Agent的死亡
2. 怎样设置模型的收敛条件
3. 怎么判定时间线的推进

## 技术实现
1. 使用LangChain作为框架控制多agent调用

# 翁法罗斯模拟仿真项目技术方案

## 项目架构概述graph TD
    A[用户界面] --> B[管理员Agent]
    B --> C[泰坦Agent 1-12]
    C --> D[知识库系统]
    B --> E[日志系统]
    E --> F[收敛检测器]
    F --> G[知识库更新]
    G --> D
## 核心逻辑设计

### 1. Agent死亡判定机制

**死亡条件**：

- 当Agent A明确发出杀死Agent B的指令
- 目标Agent在日志中有"死亡确认"响应
- 连续3次迭代无有效行动输出
def check_agent_death(agent_action, target_agent_id, logs):
    """判定Agent是否死亡"""
    kill_phrases = ["杀死", "消灭", "终结", "摧毁", "抹除"]
    
    # 检查是否发出杀死指令
    if any(phrase in agent_action for phrase in kill_phrases):
        # 检查目标Agent是否有死亡确认
        target_response = logs.get_last_response(target_agent_id)
        death_confirmations = ["死亡", "消逝", "终结", "不复存在"]
        
        if any(conf in target_response for conf in death_confirmations):
            return True
    
    # 检查是否连续无响应
    if logs.get_inactive_count(target_agent_id) >= 3:
        return True
        
    return False
### 2. 事件驱动的时间推进

**目标事件链设计**：
EVENT_CHAIN = [
    {"id": "era1", "name": "创世之初", "target": "形成基本世界规则"},
    {"id": "era2", "name": "泰坦崛起", "target": "至少8位泰坦获得领域掌控"},
    {"id": "era3", "name": "诸神之战", "target": "3位泰坦陨落"},
    {"id": "era4", "name": "新秩序", "target": "建立稳定世界结构"},
    {"id": "era5", "name": "翁法罗斯", "target": "创建世界中心圣石"}
]
**时间推进逻辑**：
def advance_era(current_era, logs):
    """推进到下一目标时代"""
    current_target = EVENT_CHAIN[current_era]["target"]
    
    # 检查当前目标是否达成
    if check_target_achieved(current_target, logs):
        print(f"目标 {current_target} 已达成! 推进到 {EVENT_CHAIN[current_era+1]['name']}")
        return current_era + 1
    
    return current_era

def check_target_achieved(target, logs):
    """检查目标是否达成"""
    # 使用LLM判断目标完成度
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    
    prompt = PromptTemplate(
        input_variables=["target", "logs"],
        template="根据以下日志内容判断目标'{target}'是否已完全达成? 只回答是或否\n日志:{logs}"
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(target=target, logs=logs.last_iteration())
    
    return "是" in response
### 3. 迭代终止条件
def check_iteration_end(active_agents, current_era):
    """检查迭代是否结束"""
    # 条件1: 所有Agent死亡
    if len(active_agents) == 0:
        return True, "所有Agent死亡"
    
    # 条件2: 完成最终目标
    if current_era >= len(EVENT_CHAIN) - 1:
        return True, "完成最终目标"
        
    return False, ""
## 技术实现细节

### LangChain技术栈组件
from langchain.agents import AgentExecutor, Tool
from langchain.memory import ReadOnlySharedMemory
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
### 知识库构建与检索
# 知识库初始化
def init_knowledge_base(world_data):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(
        texts=world_data, 
        embedding=embeddings,
        persist_directory="./knowledge_db"
    )
    return vectorstore.as_retriever()

# RAG检索工具
world_retriever = init_knowledge_base(load_world_data())
knowledge_tool = Tool(
    name="World Knowledge",
    func=world_retriever.get_relevant_documents,
    description="访问世界背景知识"
)

# Agent内存配置
shared_memory = ReadOnlySharedMemory(memory_key="world_knowledge")
### Agent系统实现
# 泰坦Agent创建
def create_titan_agent(agent_id, system_prompt):
    llm = ChatOpenAI(temperature=0.7, model="gpt-4-turbo")
    
    tools = [
        knowledge_tool,
        Tool(
            name="ActionLogger",
            func=log_action,
            description="记录行动结果"
        )
    ]
    
    agent = initialize_agent(
        tools,
        llm,
        agent="chat-conversational-react-description",
        system_message=system_prompt,
        memory=shared_memory
    )
    return agent

# 管理员Agent
admin_agent = create_admin_agent()

# 系统运行循环
def simulation_loop():
    current_era = 0
    active_agents = all_titans
    
    while True:
        # 所有Agent行动
        for agent in active_agents:
            action = agent.execute(EVENT_CHAIN[current_era])
            log_action(agent.id, action)
            
            # 检查死亡
            if check_agent_death(action):
                active_agents.remove(agent)
                log_death(agent.id)
        
        # 推进时间线
        current_era = advance_era(current_era, action_log)
        
        # 检查迭代结束
        end_iteration, reason = check_iteration_end(active_agents, current_era)
        if end_iteration:
            print(f"迭代结束! 原因: {reason}")
            break
### 知识库动态更新
def update_knowledge_base(logs):
    """从日志提取关键事件更新知识库"""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    # 提取重要事件
    important_events = admin_agent.extract_important_events(logs)
    
    # 处理文本
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.create_documents(important_events)
    
    # 添加到知识库
    vectorstore = Chroma(persist_directory="./knowledge_db", 
                         embedding_function=OpenAIEmbeddings())
    vectorstore.add_documents(docs)
    
    # 更新检索器
    global world_retriever
    world_retriever = vectorstore.as_retriever()
## 关键配置文件示例

### 泰坦提示词模板
titan_template: |
  你是{name}，执掌{domain}之力的泰坦。当前时代：{era_name}
  时代目标：{era_target}
  
  行动准则：
  1. 追求实现当前时代目标
  2. 可与其他泰坦合作或对抗
  3. 你的终极使命是见证翁法罗斯圣石的诞生
  
  知识背景：
  {relevant_knowledge}
  
  上次行动：
  {last_action}
### 管理员提示词模板
admin_template: |
  你是世界观察者，负责维护宇宙平衡。当前时代：{era_name}
  存活泰坦：{active_titans}/{total_titans}
  
  职责：
  1. 监控时代目标进度：{era_target}
  2. 标记重要事件（影响>0.7）
  3. 必要时干预以防止崩溃
  
  可执行命令：
  /advance_era - 强制推进时代
  /inject_event [描述] - 注入新事件
  /terminate - 结束当前迭代
## 性能优化策略

1. **并行处理**：使用`langchain.experimental.plan_and_execute`并行执行Agent动作
2. **缓存机制**：对知识库查询实施Redis缓存
3. **选择性记录**：仅记录影响分数>0.5的事件到日志
4. **增量更新**：知识库每3次迭代更新一次

## 迭代分析报告
{
  "iteration": 42,
  "status": "completed",
  "end_reason": "完成最终目标",
  "duration": "3.2小时",
  "titans_survived": 5,
  "eras_completed": 5,
  "knowledge_updates": 17,
  "key_events": [
    "盖亚创造大地",
    "克洛诺斯推翻乌拉诺斯",
    "宙斯释放独眼巨人",
    "普罗米修斯盗火",
    "翁法罗斯圣石落成"
  ]
}
## 实施建议

1. 使用LangSmith平台监控Agent交互
2. 为每个时代设置2-3个里程碑事件
3. 添加混沌指数作为世界稳定度指标
4. 实现可视化仪表盘展示世界演化过程
    