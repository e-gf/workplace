collection=Youcook2
visual_feature=i3d
clip_scale_w=0.5
frame_scale_w=0.5

root_path=/chenyaofo/chenchuanshen/workspace/mssl
device_ids=0
dataset_name=Youcook2
text_feat_path=/chenyaofo/chenchuanshen/workspace/generate_caption/youcook2/extracted_data/youcook2_query_roberta.hdf5
caption_train_txt=/chenyaofo/chenchuanshen/workspace/generate_caption/youcook2/extracted_data/training.txt
caption_test_txt=/chenyaofo/chenchuanshen/workspace/generate_caption/youcook2/extracted_data/validation.txt

# training
max_desc_len=120
caption_rate=1.0


local_hinge_weight=0.0
local_margin=0.2
global_soft_pos_weight=0.0
soft_pos_margin=0.1
soft_neg_margin=0.1


query_or_caption=3

plot_losses=local_hinge_loss
# training

query_or_caption=3

seed=2018


local_hinge_weight=0.0
global_soft_pos_weight=0.0
window_size=2
max_ctx_l=256
map_size=64
exp_id="windowsize${window_size}_seed${seed}_train_branch_ada_local-w${local_hinge_weight}-m${local_margin}_global-w${global_soft_pos_weight}-m${soft_pos_margin}_query_or_caption${query_or_caption}-max_ctx_l${max_ctx_l}-map_size${map_size}"
CUDA_VISIBLE_DEVICES=7 python method/train.py  --collection $collection --visual_feature $visual_feature \
                    --root_path $root_path  --dset_name $collection --exp_id $exp_id \
                    --clip_scale_w $clip_scale_w --frame_scale_w $frame_scale_w \
                    --device_ids $device_ids --dataset_name $dataset_name \
                    --text_feat_path $text_feat_path --caption_train_txt $caption_train_txt --caption_test_txt $caption_test_txt \
                    --caption_rate $caption_rate --max_desc_l $max_desc_len \
                    --local_margin $local_margin --local_hinge_weight $local_hinge_weight  \
                    --global_soft_pos_weight $global_soft_pos_weight --soft_pos_margin $soft_pos_margin \
                    --query_or_caption $query_or_caption --plot_losses $plot_losses \
                    --seed $seed --window_size $window_size --global_caption