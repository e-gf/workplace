file_path="/chenyaofo/chenchuanshen/datasets/charades/charadestest.caption.txt"
frames_folder="/chenyaofo/chenchuanshen/datasets/charades/Charades_v1_rgb"
segment_rate=0.2
output_file="/chenyaofo/chenchuanshen/workspace/generate_caption/charades/charades_test_020.json"

CUDA_VISIBLE_DEVICES=3 python charades_local.py --segment_rate $segment_rate  --output_file $output_file --file_path $file_path --frames_folder $frames_folder