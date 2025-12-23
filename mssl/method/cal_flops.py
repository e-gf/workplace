import os
import sys
import time
import json
import pprint
import random
import numpy as np
import pickle
from easydict import EasyDict as EDict
from tqdm import tqdm, trange
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import h5py

import sys
from pathlib import Path

# 获取当前文件的父目录
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

from method.config import BaseOptions
from method.model import MS_SL_Net
from method.data_provider import Dataset4MS_SL,VisDataSet4MS_SL,\
    TxtDataSet4MS_SL,collate_train,read_video_ids

from method.eval import eval_for_cal_flops,start_inference,compute_context_info
from method.optimization import BertAdam
from utils.basic_utils import AverageMeter, BigFile, read_dict, log_config
from utils.model_utils import count_parameters

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

try:
    from calflops import calculate_flops
    CALFLOPS_AVAILABLE = True
except ImportError:
    CALFLOPS_AVAILABLE = False
    logger.warning("calflops not available. Please install with: pip install calflops")

def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)

def initialize_eval_datasets(opt):
    """初始化评估数据集，与train.py中的初始化方式保持一致"""
    logger.info("正在初始化eval dataset")
    
    rootpath = opt.root_path
    dataset_files = opt.dataset_name+"_i3d"
    
    # 使用与train.py相同的路径
    text_feat_path = opt.text_feat_path
    
    # caption文件路径
    caption_files = {'train':f"/share/home/chenyaofo/project/chenchuanshen/ms-sl_gt-main/dataset/{dataset_files}/TextData/{opt.dataset_name}train.caption.txt",
                     'val':f"/share/home/chenyaofo/project/chenchuanshen/ms-sl_gt-main/dataset/{dataset_files}/TextData/{opt.dataset_name}val.caption.txt",}
    
    # 视觉特征路径
    visual_feat_path = os.path.join('/share/home/chenyaofo/project/chenchuanshen/ms-sl_gt-main', 'dataset', dataset_files, 'FeatureData', opt.visual_feature)
    video2frames = read_dict(os.path.join('/share/home/chenyaofo/project/chenchuanshen/ms-sl_gt-main', 'dataset', dataset_files, 'FeatureData', opt.visual_feature, 'video2frames.txt'))

    # 初始化验证数据集
    val_text_dataset = TxtDataSet4MS_SL(opt.caption_test_txt, text_feat_path, opt)
    val_video_ids_list = read_video_ids(opt.caption_test_txt)
    val_video_dataset = VisDataSet4MS_SL(None, video2frames, opt, video_ids=val_video_ids_list)
    
    logger.info("eval dataset初始化完成")
    return val_video_dataset, val_text_dataset

def create_model_config(opt):
    """创建模型配置，与train.py中的配置保持一致"""
    model_config = EDict(
        visual_input_size=1024,
        query_input_size=opt.q_feat_size,
        hidden_size=opt.hidden_size,  # hidden dimension
        max_ctx_l=opt.max_ctx_l,
        max_desc_l=opt.max_desc_l,
        map_size=opt.map_size,
        input_drop=opt.input_drop,
        device=opt.device_ids,
        drop=opt.drop,
        n_heads=opt.n_heads,  # self-att heads
        initializer_range=opt.initializer_range,  # for linear layer
        margin=opt.margin,  # margin for ranking loss
        use_hard_negative=False,  # reset at each epoch
        hard_pool_size=opt.hard_pool_size,
        local_hinge_weight=opt.local_hinge_weight,
        local_margin=opt.local_margin,
        soft_pos_margin=opt.soft_pos_margin,
        global_soft_pos_weight=opt.global_soft_pos_weight,
        query_or_caption=opt.query_or_caption,
        plot_losses=opt.plot_losses
    )
    return model_config

class EvalForwardWrapper(nn.Module):
    """包装eval_epoch为一个forward函数，用于计算FLOPs"""
    def __init__(self, model, val_video_dataset, val_text_dataset, opt):
        super(EvalForwardWrapper, self).__init__()
        self.model = model
        self.val_video_dataset = val_video_dataset
        self.val_text_dataset = val_text_dataset
        self.opt = opt
        # 设置模型为评估模式
        self.model.eval()
        with torch.no_grad():
            self.context_info = compute_context_info(model, val_video_dataset, opt)
            self.frame_key = self.model.mapping_linear[0](self.context_info['video_feat'])
            self.frame_value = self.model.mapping_linear[1](self.context_info['video_feat'])
        
            

        
    def forward(self, dummy_input=None):
        """执行eval_epoch并返回结果"""
        
        # 执行eval_epoch
        with torch.no_grad():
            eval_for_cal_flops(self.model, self.val_video_dataset, self.val_text_dataset, self.opt, self.context_info,self.frame_key,self.frame_value, self.opt.topk_x)
        
        # # 返回一个虚拟的输出用于FLOPs计算
        # # 这里我们返回一个标量，因为eval_epoch返回的是rsum
        # return torch.tensor(rsum, dtype=torch.float32, device=self.opt.device)

def calculate_eval_flops(opt):
    """计算eval过程中的FLOPs"""
    if not CALFLOPS_AVAILABLE:
        logger.error("calflops not available. Please install with: pip install calflops")
        return None
    
    logger.info("开始计算eval FLOPs...")
    
    # 初始化数据集
    val_video_dataset, val_text_dataset = initialize_eval_datasets(opt)
    
    # 创建模型配置
    model_config = create_model_config(opt)
    logger.info("model_config {}".format(model_config))
    
    # 创建模型
    NAME_TO_MODELS = {'MS_SL_Net': MS_SL_Net}
    model = NAME_TO_MODELS[opt.model_name](model_config)
    
    # # 如果有检查点，加载模型权重
    # if 0:
    #     checkpoint = torch.load("/share/home/chenyaofo/project/chenchuanshen/wfq/WorkSpace/ms-sl/charades/results/charades-train_branch_ada_local-w0.9-m0.2_global-w0.9-m0.1_query_or_caption3-2025_07_15_00_10_33/model.ckpt")
    #     model.load_state_dict(checkpoint["model"], strict=False)
    #     logger.info("Loaded model from checkpoint: {}".format(opt.ckpt_filepath))
    
    # 移动模型到设备
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        if len(opt.device_ids) > 1:
            logger.info("Use multi GPU", opt.device_ids)
            model = torch.nn.DataParallel(model, device_ids=opt.device_ids)
    
    for i in range(0,11):
        topk = i * 0.1
        opt.topk_x = topk
        # 创建包装器
        eval_wrapper = EvalForwardWrapper(model, val_video_dataset, val_text_dataset, opt)
        eval_wrapper.to(opt.device)

        # 计算FLOPs
        logger.info("开始计算FLOPs...")
        try:
            # 创建一个虚拟输入（实际上不会被使用，因为我们的forward函数不依赖输入）
            dummy_input = torch.randn(1, 1, device=opt.device)
            
            # 计算FLOPs - 使用官方示例的格式
            flops, macs, params = calculate_flops(model=eval_wrapper, 
                                                    input_shape=(1, 1),
                                                    output_as_string=True,
                                                    output_precision=4)
            
            logger.info("MS_SL_Net FLOPs:%s   MACs:%s   Params:%s    Top_k:%s" %(flops, macs, params,opt.topk_x))
            
        
        except Exception as e:
            import traceback
            logger.error(f"计算FLOPs时出错: {e}")
            logger.error(f"详细traceback如下:\n{traceback.format_exc()}")
            return None, None, None
        
    return flops, macs, params

def main():
    """主函数"""
    logger.info("Setup config, data and model for FLOPs calculation...")
    opt = BaseOptions().parse()
    set_seed(opt.seed)
    
    # 计算FLOPs
    flops, macs, params = calculate_eval_flops(opt)
    
    if flops is not None:
        logger.info("FLOPs计算完成!")
    else:
        logger.error("FLOPs计算失败!")

if __name__ == '__main__':
    main() 