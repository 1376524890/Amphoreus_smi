# 📜 翁法罗斯模拟仿真系统 README  
*Amphoreus_smi – 基于《崩坏：星穹铁道》世界观的多智能体对抗学习沙盒*

---

## 1. 项目定位
**一句话简介**  
在隐匿星域「翁法罗斯」的模拟宇宙中，让 11 条命途化身智能体，在 6 段世界史的宏大剧本里自行演化，观察秩序、毁灭、欢愉……等意志如何交织、冲突、收敛，直至「再创世」。

---

## 2. 系统特色
| 维度 | 亮点 |
|---|---|
| 世界观 | 官方设定 + 本文补完，6 时代递进，可无限轮回 |
| Agent | 11 条命途 = 11 个可学习策略，性格由系统 prompt 固化 |
| 技术 | LangChain 调度 + 阿里云百炼 qwen-plus，低代码接入 |
| 数据 | 每轮日志 JSON 化，自动更新知识库，形成“活的史书” |
| 终止 | 行为相似度 + 奖励稳定性双重收敛判定 |

---

## 3. 运行总览（10 步闭环）
1. 管理员读取世界观（含 6 时代目标）。  
2. 初始化 11 个 Agent（固定系统提示词）。  
3. 当前时代 → 生成阶段目标。  
4. 各 Agent 调用 LLM 输出行动 & 预期影响。  
5. 记录完整日志 `run_xxx.json`。  
6. 管理员计算奖励（命途契合度 + 时代目标达成度）。  
7. 评估收敛（余弦相似度 + 奖励方差）。  
8. 若未收敛 → 继续迭代；若已收敛 → 触发「再创世」。  
9. 将高影响力事件写入 `summary.json`，更新知识库。  
10. 下一轮从「启蒙世」或指定断点重启。

---

## 4. 快速开始
### 4.1 安装
```bash
git clone https://github.com/your-org/bringstorm.git
cd bringstorm
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt   # LangChain / openai / sklearn / numpy
```

### 4.2 配置
```bash
export ALI_API_KEY="sk-xxxxxxxxxxxxxxxx"
```
如需修改迭代次数、相似度阈值，编辑 `config.py`。

### 4.3 启动
```bash
python main.py
```
首次运行会在 `log/` 生成：
- `run_001.json` … 每轮原始日志  
- `summary.json`      管理员筛选后的“正史”  
- `world_kb.json`     动态知识库（自动合并）

---

## 5. 目录结构
```
bringstorm/
├─ main.py                 # 主循环
├─ config.py               # 全局参数
├─ requirements.txt
├─ world/
│  ├─ world.json           # 官方世界观
│  └─ world_supplement.json# 6 时代目标
├─ agents/
│  ├─ agent.py             # Agent 类
│  └─ agent_templates.py   # 11 条命途系统提示
├─ llm/
│  └─ qwen_plus.py         # 阿里云百炼封装
├─ utils/
│  ├─ similarity.py        # 余弦收敛
│  └─ reward.py            # 多维奖励
└─ log/                    # 运行期自动生成
```

---

## 6. 二次开发指南
| 需求 | 入口文件 | 关键函数 |
|---|---|---|
| 新增命途 | `agent_templates.py` | 追加 key/value |
| 替换 LLM | `llm/qwen_plus.py` | 重写 `chat()` |
| 调整奖励 | `utils/reward.py` | 自定义 `calc_reward()` |
| 阶段切换 | `main.py` 阶段判定 | 修改触发条件 |
| 可视化 | 额外脚本 | 解析 `log/*.json` 绘图 |

---

## 7. 示例输出（节选）
```json5
[
  {
    "agent": "毁灭",
    "action": "引爆黑潮核心，令奥赫玛城邦半数区域化为灰烬。",
    "impact": -0.82,
    "reward": 0.68,
    "timestamp": "2025-07-14T12:34:56"
  },
  {
    "agent": "存护",
    "action": "张开七重壁垒，庇护剩余幸存者。",
    "impact": 0.75,
    "reward": 0.9,
    "timestamp": "2025-07-14T12:34:57"
  }
]
```

---

## 8. 常见问题 FAQ
| 问题 | 解答 |
|---|---|
| 收敛过快/过慢？ | 调 `SIM_THRESHOLD` 与 `MAX_ROUND` |
| LLM 返回格式错误？ | `agent.py` 已兜底解析，可再加强 JSON 校验 |
| 如何多人协作？ | 所有数据文件均为 JSON，可直接 Git 版本控制 |
| 想接入其他星神？ | 在 `world.json` 增补命途后，仿照模板新增 Agent |

---

## 9. 许可证
MIT License – 仅禁止用于商业抽卡概率模拟器 :)

---

> **愿你在无尽的轮回里，见证黄金、黑潮与火种的再一次相遇。**