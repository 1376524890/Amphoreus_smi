# agents/memory.py
from langchain.memory import ConversationBufferMemory

class LongMemory:
    def __init__(self, agent_name):
        self.mem = ConversationBufferMemory(
            memory_key="history",
            human_prefix=f"{agent_name}_User",
            ai_prefix=f"{agent_name}_AI",
            return_messages=True,
        )