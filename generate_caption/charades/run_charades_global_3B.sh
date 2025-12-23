#!/bin/bash

# 选择是否使用GPT（True/False=Qwen）
USE_GPT=False

# Qwen模型推理设备（仅在USE_GPT=False时有效）
DEVICE="cuda:0"

frames_folder="/chenyaofo/chenchuanshen/datasets/charades/Charades_v1_rgb"
file_path="/chenyaofo/chenchuanshen/datasets/charades/charadestrain.caption.txt"
output_json="/chenyaofo/chenchuanshen/workspace/generate_caption/charades/charades_train_global_3B.json"
python charades_global_3B.py --use_gpt $USE_GPT --device $DEVICE --file_path $file_path --output_json $output_json --frames_folder $frames_folder
