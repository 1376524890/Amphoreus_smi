# graph/graph.py
from langgraph.graph import StateGraph, START
from graph.state import WorldState
from graph.nodes import make_agent_node
from agents.agent_templates import TEMPLATES   # 10命途模板

# ---------- 创建状态图 ----------
builder = StateGraph(WorldState)

# ---------- 添加11个智能体节点 ----------
for name, tmpl in TEMPLATES.items():
    builder.add_node(name, make_agent_node(name, tmpl.prompt.template))

# ---------- 构建33步顺序边（11智能体×3轮） ----------
order = list(TEMPLATES.keys()) * 3  # 生成[a1,a2,...,a11,a1,a2,...]顺序
for i in range(len(order)-1):
    builder.add_edge(order[i], order[i+1])

# ---------- 设置起始点和终点 ----------
builder.add_edge(START, order[0])
builder.set_finish_point(order[-1])

# ---------- 编译图 ----------
graph = builder.compile()