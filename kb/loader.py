# kb/loader.py
from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 加载世界知识JSON文件
loader = DirectoryLoader(
    path="/home/codeserver/AMPHOREUS/world",
    glob="*.json",
    loader_cls=JSONLoader,
    loader_kwargs={"jq_schema": ".", "text_content": False}
)

docs = loader.load()

# 创建向量存储
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        openai_api_key="sk-xxxxxxxx"
    ),
    persist_directory="./chroma_db"
)
vectorstore.persist()

# 创建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})