#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算模型参数量的脚本
使用方法：
    python method/count_params.py --collection charades --visual_feature i3d --max_desc_l 120 ...
    或者直接使用与训练脚本相同的参数
"""

import sys
from pathlib import Path
from easydict import EasyDict as EDict
from collections import OrderedDict

# 获取当前文件的父目录
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

from method.config import BaseOptions
from method.model import MS_SL_Net
from utils.model_utils import count_parameters


def count_model_parameters(opt):
    """计算模型参数量"""
    
    # 创建模型配置（与 train.py 中的配置保持一致）
    # 使用 getattr 来安全获取属性，如果不存在则使用默认值
    model_config = EDict(
        visual_input_size=1024,  # i3d特征维度
        query_input_size=getattr(opt, 'q_feat_size', 1024),
        hidden_size=opt.hidden_size,  # hidden dimension
        max_ctx_l=opt.max_ctx_l,
        max_desc_l=opt.max_desc_l,
        map_size=opt.map_size,
        input_drop=opt.input_drop,
        device=getattr(opt, 'device_ids', [0]),
        drop=opt.drop,
        n_heads=opt.n_heads,  # self-att heads
        initializer_range=opt.initializer_range,  # for linear layer
        margin=opt.margin,  # margin for ranking loss
        use_hard_negative=False,  # reset at each epoch
        hard_pool_size=getattr(opt, 'hard_pool_size', 20),
        local_hinge_weight=getattr(opt, 'local_hinge_weight', 0.0),
        local_margin=getattr(opt, 'local_margin', 0.2),
        soft_pos_margin=getattr(opt, 'soft_pos_margin', 0.1),
        global_soft_pos_weight=getattr(opt, 'global_soft_pos_weight', 0.0),
        query_or_caption=getattr(opt, 'query_or_caption', 0),
        plot_losses=getattr(opt, 'plot_losses', ["loss_overall"]),
        window_size=getattr(opt, 'window_size', 5),
        hca_loss_type=getattr(opt, 'hca_loss_type', 'margin')
    )
    
    print("=" * 60)
    print("模型配置:")
    print("=" * 60)
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    print()
    
    # 实例化模型
    print("正在创建模型...")
    model = MS_SL_Net(model_config)
    
    # 计算参数量
    print()
    print("=" * 80)
    print("Model Size (Number of Parameters)")
    print("=" * 80)
    
    # 总体参数量
    n_all = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 格式化参数量显示
    def format_params(num):
        """格式化参数量显示"""
        if num >= 1e9:
            return f"{num / 1e9:.2f}B"  # Billion
        elif num >= 1e6:
            return f"{num / 1e6:.2f}M"  # Million
        elif num >= 1e3:
            return f"{num / 1e3:.2f}K"  # Thousand
        else:
            return str(num)
    
    print(f"\nTotal Parameters:        {n_all:,} ({format_params(n_all)})")
    print(f"Trainable Parameters:    {n_trainable:,} ({format_params(n_trainable)})")
    print(f"Non-trainable Parameters: {n_all - n_trainable:,} ({format_params(n_all - n_trainable)})")
    
    # 按模块统计参数量
    print("\n" + "-" * 80)
    print("Parameters by Module:")
    print("-" * 80)
    
    module_params = OrderedDict()
    for name, param in model.named_parameters():
        module_name = name.split('.')[0] if '.' in name else name
        if module_name not in module_params:
            module_params[module_name] = {'total': 0, 'trainable': 0}
        module_params[module_name]['total'] += param.numel()
        if param.requires_grad:
            module_params[module_name]['trainable'] += param.numel()
    
    # 按参数量排序
    sorted_modules = sorted(module_params.items(), key=lambda x: x[1]['total'], reverse=True)
    
    print(f"{'Module':<30} {'Total':<20} {'Trainable':<20}")
    print("-" * 80)
    for module_name, counts in sorted_modules:
        total = counts['total']
        trainable = counts['trainable']
        print(f"{module_name:<30} {total:>15,} ({format_params(total):>6})  {trainable:>15,} ({format_params(trainable):>6})")
    
    print("=" * 80)
    
    return n_all, n_trainable


if __name__ == '__main__':
    print("开始计算模型参数量...")
    print()
    
    # 解析配置参数（可以使用与训练脚本相同的参数）
    opt = BaseOptions().parse()
    
    # 计算参数量
    count_model_parameters(opt)

