#!/usr/bin/env python3
"""
从Youcook2 JSON文件中提取数据并生成三个txt文件（train, validation, test）
格式：video_id#enc#sentence_id sentence_content
"""

import json
import os
from collections import defaultdict

def extract_youcook2_data(json_path, output_dir):
    """
    从Youcook2 JSON文件中提取数据并生成三个txt文件
    
    Args:
        json_path: JSON文件路径
        output_dir: 输出目录
    """
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 按subset分组存储数据
    subset_data = defaultdict(list)
    
    database = data.get('database', {})
    
    for video_id, video_info in database.items():
        subset = video_info.get('subset', '')
        annotations = video_info.get('annotations', [])
        
        if not subset or not annotations:
            continue
            
        # 为每个sentence生成一行数据
        for annotation in annotations:
            sentence_id = annotation.get('id', 0)
            sentence = annotation.get('sentence', '').strip()
            
            if sentence:
                # 格式：video_id#enc#sentence_id sentence_content
                line = f"{video_id}#enc#{sentence_id} {sentence}"
                subset_data[subset].append(line)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成三个txt文件
    subsets = ['training', 'validation', 'testing']
    
    for subset in subsets:
        if subset in subset_data:
            output_file = os.path.join(output_dir, f"{subset}.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                for line in subset_data[subset]:
                    f.write(line + '\n')
            print(f"生成 {output_file}，包含 {len(subset_data[subset])} 行数据")
        else:
            print(f"警告：没有找到 {subset} 数据")

def main():
    # 设置路径
    json_path = "/chenyaofo/chenchuanshen/datasets/Youcook2/youcookii_annotations_trainval.json"
    output_dir = "/chenyaofo/chenchuanshen/datasets/Youcook2/extracted_data"
    
    # 检查JSON文件是否存在
    if not os.path.exists(json_path):
        print(f"错误：JSON文件不存在 {json_path}")
        return
    
    print(f"开始处理JSON文件：{json_path}")
    print(f"输出目录：{output_dir}")
    
    # 提取数据
    extract_youcook2_data(json_path, output_dir)
    
    print("数据提取完成！")

if __name__ == "__main__":
    main()
