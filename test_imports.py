#!/usr/bin/env python3
"""
测试脚本：验证AMPHOREUS项目的导入结构是否正确
"""

def test_imports():
    """测试所有主要模块的导入"""
    print("开始测试导入...")
    
    # 测试配置导入
    try:
        from config import get_llm
        print("✓ 成功导入配置模块")
    except ImportError as e:
        print(f"✗ 配置模块导入失败: {e}")
    
    # 测试工具导入
    try:
        from tools.genesis_tool import GenesisTool
        from tools.black_tide_tool import BlackTideTool
        from tools.coreflame_collector_tool import CoreflameCollectorTool
        from tools.memory_inheritance_tool import MemoryInheritanceTool
        print("✓ 成功导入所有工具模块")
    except ImportError as e:
        print(f"✗ 工具模块导入失败: {e}")
    
    # 测试代理导入
    try:
        from agents.scepter_agent import create_scepter_agent
        from agents.titan_agent import create_titan_agent
        from agents.destruction_agent import create_destruction_agent
        from agents.chrysos_agents import create_chrysos_agents
        print("✓ 成功导入所有代理模块")
    except ImportError as e:
        print(f"✗ 代理模块导入失败: {e}")
    
    # 测试任务导入
    try:
        from tasks.simulation_tasks import create_simulation_tasks
        print("✓ 成功导入任务模块")
    except ImportError as e:
        print(f"✗ 任务模块导入失败: {e}")
    
    # 测试完整导入链
    try:
        from main import run_amphoreus_simulation
        print("✓ 成功导入主模块")
        print("导入测试完成，所有模块导入正常！")
    except ImportError as e:
        print(f"✗ 主模块导入失败: {e}")

if __name__ == "__main__":
    test_imports()