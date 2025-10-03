#!/usr/bin/env python3
"""
进一步检查crewai_tools包的tools子模块结构
"""
import sys
import pkgutil
import importlib

# 尝试导入crewai_tools.tools模块
try:
    import crewai_tools.tools
    print(f"crewai_tools.tools模块路径: {crewai_tools.tools.__file__}")
    
    # 列出tools子模块的所有子模块
    print("\ncrewai_tools.tools包的子模块:")
    for _, name, is_pkg in pkgutil.iter_modules(crewai_tools.tools.__path__):
        print(f"- {name} {'(包)' if is_pkg else ''}")
        
    # 尝试导入可能包含BaseTool的特定模块
    print("\n尝试导入可能包含BaseTool的模块...")
    
    # 检查common模块
    try:
        common_module = importlib.import_module('crewai_tools.tools.common')
        if hasattr(common_module, 'BaseTool'):
            print("✓ 在crewai_tools.tools.common中找到BaseTool类")
    except ImportError:
        print("✗ crewai_tools.tools.common模块不存在")
    
    # 检查base模块的其他可能路径
    try:
        base_module = importlib.import_module('crewai_tools.tools.base_tool')
        if hasattr(base_module, 'BaseTool'):
            print("✓ 在crewai_tools.tools.base_tool中找到BaseTool类")
    except ImportError:
        print("✗ crewai_tools.tools.base_tool模块不存在")
    
    # 检查tools模块的所有属性
    print("\ncrewai_tools.tools模块的属性:")
    for attr in dir(crewai_tools.tools):
        if not attr.startswith('__'):
            print(f"- {attr}")
            
    # 尝试使用dir函数查找所有可能的类
    print("\n尝试使用dir函数查找crewai_tools包中的所有类...")
    for root_module in ['crewai_tools', 'crewai_tools.tools']:
        try:
            module = importlib.import_module(root_module)
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and 'base' in attr_name.lower():
                    print(f"在{root_module}中找到类: {attr_name}")
        except ImportError:
            continue
    
    print("\n如果以上都没有找到BaseTool，可能需要查看crewai_tools的实际安装情况。")
    print("或者尝试使用以下方法创建自定义工具：")
    print("1. 从crewai直接继承Tool类（如果可用）")
    print("2. 查看crewai的文档了解正确的工具创建方法")
    print("3. 检查crewai_tools的版本是否兼容")
    
except Exception as e:
    print(f"检查过程中出错: {e}")
    print("\n建议的解决方案：")
    print("1. 检查crewai和crewai_tools的版本兼容性")
    print("2. 尝试重新安装这两个包")
    print("3. 如果问题依然存在，可以考虑修改代码以使用自定义工具实现")