# AMPHOREUS - 永恒轮回模拟系统

## 项目概述

AMPHOREUS是一个基于CrewAI构建的复杂AI代理模拟系统，模拟了一个名为"永恒轮回（Irontomb）"的概念。系统由多个AI代理组成，它们协作完成从创世纪到毁灭的循环过程，试图通过收集核心火焰来打破永恒轮回。

### 核心概念
- **永恒轮回（Irontomb）**：一个持续进行的模拟循环，从创世纪到毁灭的过程
- **核心火焰**：代表不同因素的关键资源，收集全部12个可打破循环
- **熵与毁灭**：系统中的关键机制，决定循环的终结
- **记忆继承**：代理在迭代过程中积累经验的机制

## 目录结构

```
AMPHOREUS/
├── agents/
│   ├── __init__.py         # 代理导入模块
│   ├── chrysos_agents.py   # 黄金继承者代理
│   ├── destruction_agent.py # 黑潮先驱代理
│   ├── scepter_agent.py    # Scepter δ-me13监督代理
│   └── titan_agent.py      # 泰坦创造者代理
├── tasks/
│   ├── __init__.py         # 任务导入模块
│   └── simulation_tasks.py # 模拟任务定义
├── tools/
│   ├── __init__.py         # 工具导入模块
│   ├── black_tide_tool.py  # 黑潮入侵工具
│   ├── coreflame_collector_tool.py # 核心火焰收集工具
│   ├── genesis_tool.py     # 创世纪模拟工具
│   └── memory_inheritance_tool.py # 记忆继承工具
├── config.py               # 配置文件（LLM配置）
├── main.py                 # 项目入口点
├── requirements.txt        # 项目依赖
└── .gitignore              # Git忽略文件
```

## 系统架构与主要组件

### 1. 代理系统（Agents）

系统包含4种类型的AI代理，它们协同工作完成模拟过程：

#### Scepter δ-me13 监督者
```python
# 创建Scepter代理的核心代码
def create_scepter_agent():
    return Agent(
        role='Scepter δ-me13',  
        goal='监督整个模拟，确保迭代循环计算毁灭方程。', 
        backstory='被 Nous 丢弃，由 Nanook 选为 Lord Ravager Irontomb。运行永恒循环。',
        tools=[GenesisTool(), BlackTideTool(), MemoryInheritanceTool()],
        llm=get_llm(),
        verbose=True
    )
```
- **职责**：监督整个模拟过程，协调其他代理，计算毁灭方程
- **工具**：创世纪模拟器、黑潮入侵者、记忆继承者
- **特性**：作为层级结构的顶端，控制整个模拟流程

#### 泰坦创造者
```python
# 创建泰坦代理的核心代码
def create_titan_agent():
    return Agent(
        role='泰坦创造者',
        goal='启动创世纪，从混沌中带来秩序。',
        backstory='从神残骸中诞生，模拟如 Kephale 和 Nikador 的 Aeons。',
        tools=[GenesisTool()],
        llm=get_llm(),
        verbose=True
    )
```
- **职责**：启动创世纪过程，从混沌中创造秩序
- **工具**：创世纪模拟器
- **特性**：模拟宇宙初始创造过程的代理

#### 黑潮先驱
```python
# 创建黑潮代理的核心代码
def create_destruction_agent():
    return Agent(
        role='黑潮先驱',
        goal='强制毁灭，通过反有机方程触发循环结束。',
        backstory='Nanook 的注视显现为 Irontomb，用熵污染。',
        tools=[BlackTideTool()],
        llm=get_llm(),
        verbose=True
    )
```
- **职责**：引入毁灭事件，计算毁灭因子
- **工具**：黑潮入侵者
- **特性**：代表系统中的熵增力量

#### 黄金继承者（12个）
```python
# 创建黄金继承者代理的核心代码
def create_chrysos_agents():
    agents = []
    for i in range(1, 13):
        factor = HEIR_FACTORS[i-1]
        agents.append(
            Agent(
                role=f'黄金继承者 {i}: {factor}',
                goal=f'收集 {factor} 核心火焰，继承记忆，与他人协作延缓永恒轮回。',
                backstory=f'黄金继承者如 Phainon（若 i=1），路径行者的变量，在迭代中。',
                tools=[CoreflameCollectorTool(), MemoryInheritanceTool()],
                llm=get_llm(),
                verbose=True
            )
        )
    return agents
```
- **职责**：各自收集代表不同因素的核心火焰，继承记忆
- **工具**：核心火焰收集器、记忆继承者
- **特性**：12个代理分别代表不同的因素（争斗、理性、世界承载、和谐、勇气、洞察、耐力、命运、混沌、秩序、永恒、重生）

### 2. 工具系统（Tools）

系统包含4种核心工具，为代理提供执行特定操作的能力：

#### 创世纪模拟器
```python
class GenesisTool(BaseTool):
    name: str = "创世纪模拟器"
    description: str = "从细胞自动机模拟生命。输入：initial_seed (整数)。输出：evolved_entities (字典列表：{'type': str, 'strength': int})。"

    def _run(self, initial_seed: int) -> List[Dict]:
        # 使用细胞自动机规则模拟生命演化
        grid = [[0 for _ in range(5)] for _ in range(5)]
        grid[2][2] = initial_seed % 2 + 1
        # 执行5代演化
        for gen in range(5):
            # 应用康威生命游戏规则
            new_grid = [[0 for _ in range(5)] for _ in range(5)]
            for i in range(5):
                for j in range(5):
                    # 计算邻居数量
                    neighbors = sum(grid[x][y] for x in range(max(0, i-1), min(5, i+2))
                                    for y in range(max(0, j-1), min(5, j+2)) if (x, y) != (i, j))
                    # 应用生存/死亡规则
                    if grid[i][j] > 0:
                        new_grid[i][j] = 1 if 2 <= neighbors <= 3 else 0
                    else:
                        new_grid[i][j] = 1 if neighbors == 3 else 0
            grid = new_grid
        # 生成实体列表
        entities = [{'type': '有机' if cell == 1 else '无机', 'strength': random.randint(1, 10)} 
                   for row in grid for cell in row if cell > 0]
        return entities
```
- **功能**：使用细胞自动机模拟生命起源与演化
- **输入**：初始种子（整数）
- **输出**：演化后的实体列表
- **算法**：基于康威生命游戏规则，但进行了简化和定制

#### 黑潮入侵者
```python
class BlackTideTool(BaseTool):
    name: str = "黑潮入侵者"
    description: str = "引入毁灭事件。输入：entropy_level (浮点数)。输出：destruction_factor (浮点数，0-1)。"

    def _run(self, entropy_level: float) -> float:
        return min(entropy_level * 0.1 + random.uniform(0.1, 0.5), 1.0)
```
- **功能**：引入毁灭事件，计算毁灭因子
- **输入**：熵水平（浮点数）
- **输出**：毁灭因子（浮点数，0-1范围）
- **特性**：熵水平越高，毁灭因子越大，但有上限1.0

#### 核心火焰收集器
```python
class CoreflameCollectorTool(BaseTool):
    name: str = "核心火焰收集器"
    description: str = "为一个继承者收集单个核心火焰。输入：current_civilization_score (整数)，heir_factor (str)。输出：collected (布尔值)。"

    def _run(self, current_civilization_score: int, heir_factor: str) -> bool:
        probability = min(current_civilization_score / 100, 1.0) * 0.083
        return random.random() < probability
```
- **功能**：为黄金继承者尝试收集核心火焰
- **输入**：当前文明分数（整数），继承者因素（字符串）
- **输出**：是否成功收集（布尔值）
- **机制**：基于当前文明分数计算收集概率，最高分对应约8.3%的收集概率

#### 记忆继承者
```python
class MemoryInheritanceTool(BaseTool):
    name: str = "记忆继承者"
    description: str = "为迭代继承记忆。输入：previous_loop_memories (字典)，heir_id (整数)。输出：enhanced_strength (整数)。"

    def _run(self, previous_loop_memories: Dict, heir_id: int) -> int:
        base = previous_loop_memories.get(f'heir_{heir_id}', 0)
        return base + random.randint(5, 15)
```
- **功能**：在迭代过程中为继承者增强记忆
- **输入**：上一轮记忆（字典），继承者ID（整数）
- **输出**：增强后的记忆强度（整数）
- **特性**：每次迭代都为继承者增加5-15点记忆强度

### 3. 任务系统（Tasks）

系统的核心任务流程如下：

```python
def create_simulation_tasks(cycle_num: int, previous_memories: dict = None, agents: dict = None):
    tasks = [
        # 1. 创世纪启动任务
        Task(
            description=f'在循环 {cycle_num} 中：使用种子 {cycle_num % 100} 启动创世纪。演化基本实体。',
            agent=agents['titan'],
            expected_output='创世纪演化的实体列表。'
        ),
        # 2. 文明增长模拟任务
        Task(
            description=f'使用创世纪输出，模拟文明增长。计算分数（人口 + 强度总和）。',
            agent=agents['scepter'],
            expected_output='文明分数（整数）。'
        )
    ]
    
    # 3. 12个继承者的并行核心火焰收集任务
    chrysos_agents = create_chrysos_agents()
    for i, agent in enumerate(chrysos_agents, start=1):
        factor = HEIR_FACTORS[i-1]
        tasks.append(
            Task(
                description=f'基于分数收集 {factor} 核心火焰。若可用，继承记忆：{previous_memories.get(f"heir_{i}", 0)}。与其他继承者协作。',
                agent=agent,
                expected_output=f'收集 {factor} （布尔值）。'
            )
        )
    
    # 4. 黑潮引入与毁灭计算任务
    tasks.extend([
        Task(
            description='基于熵（分数 / 1000）引入黑潮。计算毁灭因子。',
            agent=agents['destruction'],
            expected_output='毁灭因子（浮点数）。'
        ),
        # 5. 结果确定与记忆更新任务
        Task(
            description=f'确定结果：统计收集的核心火焰。若 >=12，进入新纪元 stall（内层循环）。否则，重置。为每个继承者增强下次迭代的记忆。',
            agent=agents['scepter'],
            expected_output='循环总结：{分数}，{火焰计数}，{毁灭}。记忆：带继承者键的字典。'
        )
    ])
    return tasks
```

### 4. 模拟流程（主程序）

整个模拟的核心流程在`main.py`中实现：

```python
def run_amphoreus_simulation(num_outer_loops: int = 50):
    # 初始化统计数据和记忆
    overall_stats = {'successful_stalls': 0, 'total_flames': 0, 'avg_destruction': 0}
    memories = {f'heir_{i}': 0 for i in range(1, 13)}

    # 组装代理
    try:
        agents = {
            'scepter': create_scepter_agent(),
            'titan': create_titan_agent(),
            'destruction': create_destruction_agent()
        }
        chrysos_agents = create_chrysos_agents()  # 动态添加 Chrysos 代理
    except Exception as e:
        print(f"代理创建失败: {e}")
        return

    # 主循环
    for cycle in range(1, num_outer_loops + 1):
        print(f"\n--- 外层循环 {cycle} ---")
        # 创建当前循环的任务
        tasks = create_simulation_tasks(cycle, memories, agents)
        # 创建Crew并执行任务
        amphoreus_crew = Crew(
            agents=[agents['scepter'], agents['titan']] + chrysos_agents + [agents['destruction']],
            tasks=tasks,
            process=Process.hierarchical,  # 层级处理模式
            verbose=2
        )
        try:
            result = amphoreus_crew.kickoff()
        except Exception as e:
            print(f"循环 {cycle} 执行失败: {e}")
            continue

        # 简化解析（实际中从 result 解析）
        # 在实际系统中，这些值会从LLM的响应中提取
        score = random.randint(50, 200 * (cycle // 10 + 1))
        flames_collected = [random.choice([True, False]) for _ in range(12)]
        flames_count = sum(flames_collected)
        destruction = random.uniform(0.2, 0.8 - (cycle / num_outer_loops * 0.3))

        # 处理结果：新纪元或重置
        if flames_count >= 12:
            overall_stats['successful_stalls'] += 1
            print("新纪元激活！内层永恒轮回 stall Irontomb。")
            for inner in range(3):
                time.sleep(0.1)
                print(f"  内层循环 {inner+1}: 继承者间继承记忆，熵减少。")
        else:
            print("黑潮淹没。循环重置。")

        # 更新统计数据和记忆
        overall_stats['total_flames'] += flames_count
        overall_stats['avg_destruction'] += destruction

        for i in range(1, 13):
            memories[f'heir_{i}'] += 10 if flames_collected[i-1] else 0

        print(f"循环 {cycle} 结束: 分数={score}, 火焰={flames_count}/12, 毁灭={destruction:.2f}")

    # 输出最终统计结果
    overall_stats['avg_destruction'] /= num_outer_loops
    print("\n--- 模拟完成 ---")
    print(f"统计: Stall={overall_stats['successful_stalls']}/{num_outer_loops}, 平均火焰={overall_stats['total_flames']/num_outer_loops:.1f}, 平均毁灭={overall_stats['avg_destruction']:.2f}")
    print("Irontomb 方程: 收敛到毁灭。" if overall_stats['successful_stalls'] < num_outer_loops / 2 else "Stall！需要外部变量（开拓者）打破循环。")
```

## 技术栈与依赖

| 技术/依赖        | 版本/来源     | 用途                              | 备注                                   |
|-----------------|--------------|----------------------------------|----------------------------------------|
| Python          | 3.13+        | 开发语言                          | 项目基于最新Python版本                 |
| CrewAI          | 最新版       | AI代理协作框架                    | 提供代理和任务管理                     |
| crewai_tools    | 最新版       | CrewAI工具库                      | 用于创建自定义工具                     |
| LiteLLM         | 最新版       | LLM统一接口                       | 支持接入多种大语言模型                 |
| python-dotenv   | 最新版       | 环境变量管理                      | 用于加载API密钥和配置                  |
| 大语言模型API   | 阿里云DashScope | AI代理的智能核心                  | 使用Qwen3-max或Qwen2.5-max模型         |

## 配置与部署

### 环境变量配置

项目使用`.env`文件存储API密钥和配置：

```env
# .env 文件示例
DASHSCOPE_API_KEY=your_api_key_here
```

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行项目

```bash
python main.py
```

## 系统工作原理

### 模拟循环机制

系统通过多层循环实现永恒轮回的概念：

1. **外层循环**：默认执行50次，每次循环代表一个完整的创世纪到毁灭的过程
2. **内层循环**：当成功收集12个核心火焰时触发，执行3次内层循环，表示新纪元的开始
3. **记忆累积**：每个黄金继承者在每次循环中积累记忆，增强后续表现

### 核心算法

#### 1. 熵与毁灭计算

毁灭因子计算公式：
```
destruction = min(entropy_level * 0.1 + random.uniform(0.1, 0.5), 1.0)
```
- entropy_level：基于文明分数计算的熵水平
- 随模拟进行，基础毁灭因子会逐渐降低

#### 2. 核心火焰收集概率

收集概率计算公式：
```
probability = min(current_civilization_score / 100, 1.0) * 0.083
```
- 随着文明分数提高，收集概率增加，但上限为8.3%
- 这意味着即使在理想条件下，收集全部12个火焰的概率也很低

#### 3. 创世纪细胞自动机

系统使用简化版的康威生命游戏规则：
- 任何活细胞周围有2-3个活细胞，继续存活
- 任何死细胞周围有3个活细胞，成为活细胞
- 其他情况，细胞死亡或保持死亡状态

## 关键模块与典型用例

### 模拟配置

```python
# 调整模拟参数示例
run_amphoreus_simulation(num_outer_loops=100)  # 增加外层循环次数
```

### 代理定制

```python
# 自定义代理示例（在实际应用中可能需要扩展现有类）
from agents.scepter_agent import create_scepter_agent

def create_custom_scepter():
    agent = create_scepter_agent()
    agent.temperature = 0.9  # 增加随机性
    return agent
```

### 工具扩展

可以通过继承`BaseTool`类创建新工具：

```python
from crewai_tools import BaseTool

class CustomTool(BaseTool):
    name: str = "自定义工具名称"
    description: str = "工具描述，说明输入输出格式"

    def _run(self, input_param: type) -> type:
        # 工具实现逻辑
        return result
```

## 性能与限制

1. **性能考量**：
   - 每次循环都会调用多个大语言模型API，可能产生较高费用
   - 模拟50次循环可能需要数分钟到数小时，取决于API响应速度

2. **系统限制**：
   - 当前版本使用随机值模拟某些结果，实际系统应从LLM响应中提取
   - 记忆继承机制相对简单，可以进一步优化为更复杂的累积系统

3. **依赖限制**：
   - 依赖阿里云DashScope服务，需要有效的API密钥
   - 网络连接质量会影响模拟速度

## 扩展方向

1. **模型支持扩展**：
   - 通过LiteLLM集成更多大语言模型
   - 实现模型自动切换机制，根据任务复杂度选择合适模型

2. **代理系统增强**：
   - 增加更多类型的代理
   - 实现代理间更复杂的交互机制
   - 引入代理进化系统，让代理随时间改进

3. **模拟机制优化**：
   - 增强记忆继承系统，使其更加复杂和有效
   - 实现更精细的熵计算模型
   - 添加可视化界面展示模拟过程

4. **应用场景拓展**：
   - 将系统应用于复杂问题解决
   - 探索在游戏开发、决策支持等领域的应用

## 贡献指南

1. **代码风格**：
   - 遵循PEP 8编码规范
   - 保持代码简洁明了
   - 为关键函数和类添加文档字符串

2. **提交流程**：
   - Fork仓库
   - 创建功能分支
   - 提交更改
   - 创建Pull Request

3. **报告问题**：
   - 使用GitHub Issues跟踪bug和功能请求
   - 提供详细的问题描述和复现步骤

## 许可证

[MIT License](https://opensource.org/licenses/MIT)

## 免责声明

本项目仅用于研究和教育目的，模拟结果不应被视为对现实世界的准确预测。使用本项目可能产生API调用费用，请用户自行承担。

---

*AMPHOREUS - 探索AI代理协作与复杂系统模拟的边界*