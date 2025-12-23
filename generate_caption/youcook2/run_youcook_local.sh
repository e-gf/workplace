file_path="/chenyaofo/chenchuanshen/datasets/Youcook2/splits/train_list.txt"
frames_folder="/chenyaofo/chenchuanshen/datasets/Youcook2/raw_videos"
segment_rate=0.2
output_file="/chenyaofo/chenchuanshen/workspace/generate_caption/youcook2/youcook_train_020.json"

python youcook_local.py --segment_rate $segment_rate  --output_file $output_file --file_path $file_path --frames_folder $frames_folder