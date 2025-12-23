#!/bin/bash

# 选择是否使用GPT（True/False=Qwen）
USE_GPT=False

# Qwen模型推理设备（仅在USE_GPT=False时有效）
DEVICE="cuda:0"

frames_folder="/chenyaofo/chenchuanshen/datasets/Youcook2/raw_videos"
file_path="/chenyaofo/chenchuanshen/datasets/Youcook2/splits/train_list.txt"
output_json="/chenyaofo/chenchuanshen/workspace/generate_caption/youcook2/youcook_train_global.json"
python youcook_global.py --use_gpt $USE_GPT --device $DEVICE --file_path $file_path --output_json $output_json --frames_folder $frames_folder
