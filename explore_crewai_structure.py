#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
探索CrewAI包结构的脚本
用于确定正确的Tool类导入方式
"""
import crewai
import inspect
import sys
from pprint import pprint

def explore_module(module, max_depth=1, current_depth=0, visited=None):
    if visited is None:
        visited = set()
    
    module_name = module.__name__
    if module_name in visited or current_depth > max_depth:
        return
    
    visited.add(module_name)
    print(f"\n探索模块: {module_name}")
    print(f"模块路径: {getattr(module, '__file__', 'N/A')}")
    
    # 列出模块中的所有公共属性
    public_attrs = [attr for attr in dir(module) if not attr.startswith('_')]
    print(f"公共属性 ({len(public_attrs)}):")
    pprint(public_attrs)
    
    # 检查是否有tools属性
    if hasattr(module, 'tools'):
        print("\n找到了tools属性，探索它...")
        tools_attr = getattr(module, 'tools')
        if inspect.ismodule(tools_attr):
            explore_module(tools_attr, max_depth, current_depth + 1, visited)
        else:
            print(f"tools不是一个模块，类型: {type(tools_attr)}")
            if hasattr(tools_attr, '__class__'):
                print(f"tools的类: {tools_attr.__class__.__name__}")
    
    # 尝试查找与Tool相关的类
    tool_classes = []
    for attr_name in public_attrs:
        attr = getattr(module, attr_name)
        if inspect.isclass(attr) and 'tool' in attr_name.lower():
            tool_classes.append((attr_name, attr))
    
    if tool_classes:
        print("\n找到与Tool相关的类:")
        for name, cls in tool_classes:
            print(f"- {name}: {cls}")
            print(f"  模块: {cls.__module__}")
            print(f"  基类: {[base.__name__ for base in cls.__bases__]}")

# 首先检查crewai的版本
print(f"CrewAI版本: {getattr(crewai, '__version__', '未知')}")

# 探索crewai模块
print("\n=== 探索crewai包结构 ===")
explore_module(crewai, max_depth=2)

# 尝试导入crewai.tools
print("\n=== 尝试直接导入crewai.tools ===")
try:
    from crewai import tools
    print(f"成功导入crewai.tools，类型: {type(tools)}")
    if hasattr(tools, '__all__'):
        print(f"tools的__all__: {tools.__all__}")
except ImportError as e:
    print(f"导入crewai.tools失败: {e}")

# 尝试从crewai.tools导入Tool类
print("\n=== 尝试从crewai.tools导入Tool ===")
try:
    from crewai.tools import Tool
    print(f"成功导入Tool类: {Tool}")
except ImportError as e:
    print(f"导入Tool失败: {e}")

# 查看crewai的__init__.py内容（如果可能）
print("\n=== 查看crewai的__init__.py内容 ===")
init_file = getattr(crewai, '__file__', None)
if init_file:
    try:
        with open(init_file, 'r') as f:
            content = f.read()
            print(f"init.py前1000字符:\n{content[:1000]}...")
    except Exception as e:
        print(f"无法读取init.py: {e}")
else:
    print("无法获取init.py路径")

print("\n=== 探索结束 ===")