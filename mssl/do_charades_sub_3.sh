collection=charades
visual_feature=i3d
clip_scale_w=0.5
frame_scale_w=0.5

root_path=/chenyaofo/chenchuanshen/workspace/mssl
device_ids=0
dataset_name=charades
text_feat_path=/chenyaofo/chenchuanshen/workspace/generate_caption/charades/charades_query_roberta.hdf5
caption_train_txt=/chenyaofo/chenchuanshen/datasets/charades/charadestrain.caption.txt
caption_test_txt=/chenyaofo/chenchuanshen/datasets/charades/charadestest.caption.txt
num_workers=12
# training
max_desc_len=120


local_margin=0.2
soft_pos_margin=0.1
soft_neg_margin=0.1



plot_losses=local_hinge_loss
# training
caption_rate=1.0
query_or_caption=3

seed=789
window_size=5

local_hinge_weight=0.9
global_soft_pos_weight=1.0


exp_id="add_all_ours_in_qwen3B_seed${seed}_local${local_hinge_weight}_global${global_soft_pos_weight}"
CUDA_VISIBLE_DEVICES=2 python method/train.py  --collection $collection --visual_feature $visual_feature \
                    --root_path $root_path  --dset_name $collection --exp_id $exp_id \
                    --clip_scale_w $clip_scale_w --frame_scale_w $frame_scale_w \
                    --device_ids $device_ids --dataset_name $dataset_name \
                    --text_feat_path $text_feat_path --caption_train_txt $caption_train_txt --caption_test_txt $caption_test_txt \
                    --caption_rate $caption_rate --max_desc_l $max_desc_len \
                    --local_margin $local_margin --local_hinge_weight $local_hinge_weight  \
                    --global_soft_pos_weight $global_soft_pos_weight --soft_pos_margin $soft_pos_margin \
                    --query_or_caption $query_or_caption --plot_losses $plot_losses --num_workers $num_workers \
                    --seed $seed --window_size $window_size  --global_caption
                    #  --max_es_cnt 100 --lr $learing_rate --hard_negative_start_epoch $hard_negative_start_epoch   --global_caption
                    #max_ctx_l拉大，是因为视频长度很长，过度压缩不好。尝试换到256看看

