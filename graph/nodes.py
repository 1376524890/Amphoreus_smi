# graph/nodes.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from graph.state import WorldState
from langchain_core.messages import AIMessage

llm = ChatOpenAI(
    model="qwen-plus",
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    openai_api_key="sk-xxxxxxxx",
    temperature=0.7
)

# ---------- 单个智能体节点 ----------
def make_agent_node(name: str, sys_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt + "\n世界阶段：{stage}\n历史：{snapshot}"),
        ("user", "请给出你在本回合步的一句话行动与预期影响(-1~1)。")
    ])
    chain = prompt | llm

    def _node(state: WorldState, config: RunnableConfig):
        # 调用 LLM
        msg = chain.invoke({
            "stage": state["stage"],
            "snapshot": "\n".join([m.content for m in state["snapshot"]])
        })
        # 把结果包装成消息并追加到快照
        new_message = AIMessage(content=f"{name}：{msg.content}")
        updated_snapshot = state["snapshot"] + [new_message]
        return {
            "snapshot": updated_snapshot,
            "agents_left": state["agents_left"] - 1
        }
    return _node