file_path="/chenyaofo/chenchuanshen/TVRdatasets/MSRVTT/videos/train_list_new.txt"
frames_folder="/chenyaofo/chenchuanshen/TVRdatasets/MSRVTT/videos/all"
segment_rate=0.34
output_file="/chenyaofo/chenchuanshen/workspace/generate_caption/msrvtt_train_034.json"

python msrvtt_local.py --segment_rate $segment_rate  --output_file $output_file --file_path $file_path --frames_folder $frames_folder