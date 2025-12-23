#!/bin/bash
# 计算模型参数量的脚本
# 使用方法：bash count_params.sh

collection=charades
visual_feature=i3d

root_path=/chenyaofo/chenchuanshen/workspace/mssl
device_ids=0
dataset_name=charades

# 模型配置参数（与训练脚本保持一致）
max_desc_len=120
local_margin=0.2
soft_pos_margin=0.1
query_or_caption=3
window_size=5
local_hinge_weight=0.009
global_soft_pos_weight=0.009
hca_loss_type=infonce
plot_losses=local_hinge_loss

# 运行参数量计算脚本
python method/count_params.py \
    --collection $collection \
    --visual_feature $visual_feature \
    --root_path $root_path \
    --dset_name $collection \
    --device_ids $device_ids \
    --dataset_name $dataset_name \
    --max_desc_l $max_desc_len \
    --local_margin $local_margin \
    --local_hinge_weight $local_hinge_weight \
    --global_soft_pos_weight $global_soft_pos_weight \
    --soft_pos_margin $soft_pos_margin \
    --query_or_caption $query_or_caption \
    --plot_losses $plot_losses \
    --window_size $window_size \
    --hca_loss_type $hca_loss_type \
    --exp_id "count_params"  # 这个参数是必需的，但不会影响参数量计算


