import json
import os

def convert_json_to_knowledge(json_path, output_dir):
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        worldview = json.load(f)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 清空现有txt文件
    for filename in os.listdir(output_dir):
        if filename.endswith('.txt'):
            os.remove(os.path.join(output_dir, filename))
    
    # 生成文本内容
    content = []
    content.append(f"标题：{worldview['标题']}\n\n")
    content.append(f"世界背景：{worldview['世界背景']}\n\n")
    content.append(f"运行设定：{worldview['运行设定']}\n\n")
    
    content.append("迭代发展阶段：\n")
    for stage in worldview['迭代发展阶段']:
        content.append(f"- {stage['阶段名称']}：{stage['描述']}\n")
    content.append("\n")
    
    content.append("泰坦介绍：\n")
    for category in ['命运三泰坦', '支柱三泰坦', '其他泰坦']:
        content.append(f"  {category}：\n")
        for titan in worldview['泰坦介绍'][category]:
            content.append(f"  - {titan['名称']}：{titan['描述']}\n")
    content.append("\n")
    
    content.append("命途介绍：\n")
    for path in worldview['命途介绍']:
        content.append(f"- {path['命途名称']}：{path['描述']}\n")
    content.append("\n")
    
    content.append("角色设定：\n")
    content.append(f"说明：{worldview['角色设定']['说明']}\n")
    for role in worldview['角色设定']['角色列表']:
        content.append(f"- {role['角色名']}：{role['描述']}\n")
    
    # 写入文件
    output_path = os.path.join(output_dir, '翁法罗斯世界观.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(''.join(content))

if __name__ == "__main__":
    json_path = '/home/codeserver/AMPHOREUS/世界观.json'
    output_dir = '/home/codeserver/AMPHOREUS/world_knowledge'
    convert_json_to_knowledge(json_path, output_dir)