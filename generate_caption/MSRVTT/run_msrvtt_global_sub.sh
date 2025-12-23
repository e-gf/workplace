#!/bin/bash

# 选择是否使用GPT（True/False=Qwen）
USE_GPT=False

# Qwen模型推理设备（仅在USE_GPT=False时有效）
DEVICE="cuda:0"

frames_folder="/chenyaofo/chenchuanshen/TVRdatasets/MSRVTT/videos/all"
file_path="/chenyaofo/chenchuanshen/TVRdatasets/MSRVTT/videos/test_list_new.txt"
output_json="/chenyaofo/chenchuanshen/workspace/generate_caption/msrvtt_test_global.json"
python msrvtt_global.py --use_gpt $USE_GPT --device $DEVICE --file_path $file_path --output_json $output_json --frames_folder $frames_folder
