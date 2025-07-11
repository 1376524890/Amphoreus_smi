from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.docstore.document import Document
import os

# 翁法罗斯世界观知识库构建脚本

# 定义文档内容
worldview_content = """
《崩坏：星穹铁道》之翁法罗斯世界观
**
一、世界背景
翁法罗斯被称作 “永恒之地”，是隐匿于深空的未知星域，与外界隔绝，难以被寻常星际航行察觉与抵达。其名字源于希腊语 “Ὀμφαλός”，意为 “大地的肚脐”，象征世界中心。在这片神秘之地，融合了古希腊与古罗马文化核心元素。
传说起始，世界一片混沌，神明投下火种，12 位泰坦自火中诞生。它们编制命运、开辟天地、捏塑生命、引渡灾祸，泰坦的力量燃放出文明之光，万邦生灵得以繁衍。但好景不长，末日黑潮降临，神明陷入疯狂，凡人被迫举戈相向。不过，仍有追逐火种的英雄在创世伟业中奋勇前行。
实际上，翁法罗斯是由星体计算机帝皇权杖 δ - me13 运行的模拟世界。该权杖曾作为 “智识” 博识尊的天体神经元，后遭废弃，又在漫长演算中受 “毁灭” 瞥视，升格为绝灭大君 “铁墓”。黄金裔体内流淌的金血便是纳努克（“毁灭” 星神）的恩赐。翁法罗斯被 “智识”“记忆”“毁灭” 三重命途缠裹，因诞生过三位令使级别的存在，得以被忆庭之镜照映出来，而其本地人对外部世界和星神毫无所知，他们信仰本土神明 —— 泰坦 。
二、运行设定
翁法罗斯由帝皇权杖 δ - me13 模拟运行，以 “反有机方程” 为基础，专门模拟不同文明的毁灭路径。其内部时空状态特殊，时间流动异常，空间也存在隐蔽。从计算机概念角度理解，翁法罗斯就如同 “屏幕”，呈现 “显卡”（即帝皇权杖 δ - me13）的计算结果 。
这里的 “历史” 本质是帝皇权杖硬盘里存储的文件。“改变历史”“还原历史” 并非真正的时光倒流，而是对硬盘中旧文件的调用，“预言” 则是翻阅 “还未用到的” 文件。部分泰坦神权还会在文件系统中制造混乱，导致翁法罗斯部分历史的因果关系或时间关系错乱。例如赛飞儿的 “诡计” 神权，让系统超期使用 “正常运作的黎明机器” 的文件构建奥赫玛，覆盖了原本 “三百年后黎明机器报废” 的逻辑 。
三、迭代发展阶段
启蒙世：命运三泰坦 “万径之门” 雅努斯、“公正之秤” 塔兰顿和 “永夜之帷” 欧洛尼斯先后苏醒，塑造了翁法罗斯的空间、律法和时间。支柱三泰坦中的 “磐岩之脊” 吉奥里亚诞生，随后 “晨昏之眼” 艾格勒和 “满溢之杯” 法吉娜也相继诞生，它们支撑起天地，为生命诞生搭建温床。法吉娜和艾格勒诞生于启蒙世，比它们更早苏醒的命运三泰坦和吉奥里亚同样出现于该时期 。
造物世：创生之神 “全世之座” 刻法勒依照自身形象捏塑了最初的人类，人类在翁法罗斯大地建立起万千城邦，各城邦生活方式、传统与泰坦信仰各异 。
黄金世：此阶段是翁法罗斯相对繁荣和平的时期，泰坦庇护着信仰它们的人类，文明蓬勃发展 。
纷争世：黑潮降临，神明堕落，泰坦间矛盾激化，引发战争，世界陷入混乱与纷争 。
幻灭世：战争和黑潮的影响下，翁法罗斯走向衰败，诸多城邦陷入永夜，世界濒临毁灭 。
再创世（未到来）：在经历毁灭后，借助 “记忆” 的力量，将备份的关键信息（可能存储在 “火种” 中）用于重建翁法罗斯 。
四、泰坦介绍
命运三泰坦
“万径之门” 雅努斯：最古老且受世人尊敬的泰坦，掌管万千门径与道路，包括命运旅程。能打开通往所有破碎凡界的门扉，创造不存在的道路，指明看不见的方向。它以预言揭示命运，引导人们前行，使死者灵魂能在结束后被引渡向来世。但它也是隔绝与监禁的神明 。
“公正之秤” 塔兰顿：捍卫翁法罗斯的禁忌与边界，为万物指定逻辑与律法，是公义的化身。创造利衡币使交易等价进行，维护世间平衡，对打破律法者施以惩戒。众神将其视为纷争调停者，它以严苛律法约束世界，也同样约束自己，不容偏袒和不公 。
“永夜之帷” 欧洛尼斯：在白昼与黑夜划分后苏醒，前往界外域化身永夜天帷，支撑世界的过去、现在与未来，保存珍贵记忆，让万物流转。其祭司能唤起奇迹，秘仪可让过往事物重现，从永夜天帷回声中获取未来谕示 。
支柱三泰坦
“磐岩之脊” 吉奥里亚：大地的泰坦，天空和海洋诞生于其沉眠呼吸。是体积最大、温和善良的泰坦，也是人世的守护神。将自身神体赠予其他泰坦作为创造生命基石，传授耕种知识，“山之民” 和 “大地兽” 是其造物 。
“满溢之杯” 法吉娜：司掌大海与风暴，也是蜜酿与宴会的主人。可随意移动水体，负有洗刷世界污秽职能，赐予人类酿造技艺，教会他们歌唱与舞蹈。与吉奥里亚永远对抗，喜欢在人世漂流，显形于海上波涛、河上湍流、宴上酒杯中 。
“晨昏之眼” 艾格勒：司掌天空、天体以及昼夜循环的巨大泰坦神明，高居天穹注视世界。能带来温暖和甘霖，也能化作可怕天灾惩戒僭越者。其在与灾厄泰坦战争中许多眼睛被 “刺盲”，导致诸多城邦陷入永夜 。
其他泰坦：创生之神 “全世之座” 刻法勒，被众神赋予背负生命责任，舍弃统御全世的王座，为人间留下光明 。纷争泰坦尼卡多利象征战争与毁灭 。此外还有诸多泰坦各司其职，共同构成翁法罗斯的运转规则 。
五、命途介绍
智识：翁法罗斯的 “智识” 部分由帝皇权杖 δ - me13 代表，它控制着世界的运行和毁灭，以 “反有机方程” 模拟文明毁灭路径，其防火墙中时间层比空间层更难突破 。
记忆：在翁法罗斯的核心作用是在帝皇权杖影响范围外备份世界概况，在世界被删干净后，用备份关键信息（可能在 “火种” 里）重建翁法罗斯。“记忆” 力量还能维系开拓者 “灵魂” 完整性 。
毁灭：绝灭大君 “铁墓” 由帝皇权杖 δ - me13 受 “毁灭” 瞥视升格而成，黄金裔体内金血是 “毁灭” 星神纳努克的恩赐 。
六、角色设定（黄金裔为例）
阿格莱雅缇宝：流淌着黄金血的黄金裔，具体能力和故事与翁法罗斯的发展紧密相关，在对抗黑潮、追寻真相的过程中发挥着重要作用 。
那刻夏：从 “负世泰坦” 的 “火种” 中获取过上一任 “救世主” 的记忆，在翁法罗斯的命运交织中探寻自我和世界的秘密 。
风堇：黄金裔角色，在翁法罗斯的复杂局势下，凭借自身能力和信念，为守护某些重要事物而行动 。
万敌：作为黄金裔，在纷争的世界中展现出强大实力，为了自己认同的理念参与到翁法罗斯的各种事件中 。
翁法罗斯世界中，每个角色都有其独特的背景故事、目标和使命，他们的行动推动着世界的发展与变革，在三重命途的交织下，共同演绎着这个神秘世界的史诗 。
"""

# 创建文档对象
def create_documents(content: str) -> list[Document]:
    """将文本内容分割并创建为文档列表"""
    # 按章节分割文本
    sections = content.split('\n\n')
    
    documents = []
    current_section = ""
    current_header = ""
    
    for line in sections:
        # 识别标题行
        if line.startswith('**') or line.startswith('#') or line.strip().endswith(':'):
            if current_section.strip():
                # 创建文档
                doc = Document(
                    page_content=current_section.strip(),
                    metadata={"source": "翁法罗斯世界观", "category": current_header}
                )
                documents.append(doc)
                current_section = ""
            
            # 提取标题
            current_header = line.strip().replace('**', '').replace('#', '').replace(':', '')
        else:
            current_section += line + "\n"
    
    # 添加最后一个部分
    if current_section.strip() and current_header:
        doc = Document(
            page_content=current_section.strip(),
            metadata={"source": "翁法罗斯世界观", "category": current_header}
        )
        documents.append(doc)
    
    # 验证文档内容
    if not documents:
        raise ValueError("No valid documents were created from the content")
    
    return documents

# 初始化向量数据库
def init_vector_db(documents: list[Document], persist_dir: str = "omphalos_knowledge") -> Chroma:
    """初始化并填充向量数据库"""
    # 确保目录存在
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)
    
    # 使用中文适用的嵌入模型
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # 创建向量数据库
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    # 持久化数据库
    vectordb.persist()
    return vectordb

# 创建工具函数
def query_omphalos_knowledge(query: str, k: int = 4) -> str:
    """查询翁法罗斯知识库"""
    # 加载向量数据库
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory="omphalos_knowledge"
    )
    
    # 相似性搜索
    docs = vectordb.similarity_search(query, k=k)
    
    # 整理结果
    results = []
    for doc in docs:
        results.append(f"**{doc.metadata['category']}**\n{doc.page_content}")
    
    return "\n\n---\n\n".join(results)

# 主函数
if __name__ == "__main__":
    # 创建文档
    documents = create_documents(worldview_content)
    
    # 初始化向量数据库
    vectordb = init_vector_db(documents)
    
    # 示例查询
    sample_query = "翁法罗斯的运行机制是什么？"
    print(f"查询: {sample_query}")
    print(query_omphalos_knowledge(sample_query))