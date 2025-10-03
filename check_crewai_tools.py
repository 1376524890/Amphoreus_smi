#!/usr/bin/env python3
"""
检查crewai_tools包的结构，找出正确的BaseTool导入路径
"""
import sys
import pkgutil
import crewai_tools

print(f"Python版本: {sys.version}")
print(f"crewai_tools模块路径: {crewai_tools.__file__}")
print(f"crewai_tools模块版本: {getattr(crewai_tools, '__version__', '未知')}")

# 列出crewai_tools包的所有子模块
print("\ncrewai_tools包的子模块:")
for _, name, is_pkg in pkgutil.iter_modules(crewai_tools.__path__):
    print(f"- {name} {'(包)' if is_pkg else ''}")

# 尝试导入可能包含BaseTool的模块
print("\n尝试查找BaseTool类...")
try:
    # 尝试从crewai_tools的核心模块导入
    import crewai_tools.tools.base as base_module
    if hasattr(base_module, 'BaseTool'):
        print("✓ 在crewai_tools.tools.base中找到BaseTool类")
        print(f"类路径: crewai_tools.tools.base.BaseTool")
    else:
        print("✗ 在crewai_tools.tools.base中未找到BaseTool类")
        
    # 尝试其他可能的路径
    try:
        from crewai_tools.base import BaseTool
        print("✓ 在crewai_tools.base中找到BaseTool类")
        print("类路径: crewai_tools.base.BaseTool")
    except ImportError:
        print("✗ 在crewai_tools.base中未找到BaseTool类")
        
    # 列出crewai_tools模块的所有属性
    print("\ncrewai_tools模块的属性:")
    for attr in dir(crewai_tools):
        if not attr.startswith('__'):
            print(f"- {attr}")
            
except Exception as e:
    print(f"检查过程中出错: {e}")