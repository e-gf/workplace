import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools as it
from easydict import EasyDict as edict
from Models.gmmformer.model_components_new import clip_nce
from Losses.model_loss import LocalHingeLoss


class mask_div(nn.Module):
    def __init__(self, lambda_):
        torch.nn.Module.__init__(self)
        self.lambda_ = lambda_

    def forward(self, gauss_weight):
        num_props = gauss_weight.size(1)
        gauss_weight = gauss_weight / gauss_weight.sum(dim=-1, keepdim=True)
        target = torch.eye(num_props).unsqueeze(0).cuda() * self.lambda_
        source = torch.matmul(gauss_weight, gauss_weight.transpose(1, 2))
        div_loss = torch.norm(target - source, dim=(1, 2))**2
        return div_loss.mean()




class loss(nn.Module):
    def __init__(self, cfg):
        super(loss, self).__init__()
        self.cfg = cfg
        self.clip_nce_criterion = clip_nce(reduction='mean')
        self.div_criterion = mask_div(lambda_=0.15)
        self.local_hinge_loss = LocalHingeLoss(margin=self.cfg['local_margin'], reduction='mean')

    def forward(self, input_list, batch):
        '''
        param: query_labels: List[int]
        param: clip_scale_scores.shape = [5*bs,bs]
        param: frame_scale_scores.shape = [5*bs,5*bs]
        param: clip_scale_scores_.shape = [5*bs,bs]
        param: frame_scale_scores_.shape = [5*bs,5*bs]
        param: label_dict: Dict[List]
        '''

        query_labels = batch['text_labels']
        
        [rank_loss, clip_scale_scores, clip_scale_scores_, label_dict, query, gauss_weight, clip_scores,mapped_raw_clip_scale_scores,global_caption_mask, local_start_end_tensor] = input_list

        clip_nce_loss = self.cfg['loss_factor'][0] * self.clip_nce_criterion(query_labels, label_dict, clip_scale_scores_) 
        mask_div_loss = self.cfg['loss_factor'][1] * self.div_criterion(gauss_weight)
        match_loss = rank_loss * 25

        if rank_loss is None or str(rank_loss) == "nan":
            raise ValueError("rank_loss is None or nan")
        
        local_hinge_loss = self.cfg['local_hinge_weight'] * self.local_hinge_loss(global_caption_mask,mapped_raw_clip_scale_scores,local_start_end_tensor,label_dict)

        global_soft_pos_loss = self.cfg['global_soft_pos_weight'] * self.global_soft_loss(mapped_raw_clip_scale_scores, query_labels, global_caption_mask)

        loss =  clip_nce_loss + mask_div_loss + match_loss + local_hinge_loss + global_soft_pos_loss

        return loss
    def global_soft_loss(self, raw_clip_scale_scores, query_labels, global_caption_mask):
        device = raw_clip_scale_scores.device
        global_caption_mask_tensor = torch.tensor(global_caption_mask, device=device)
        global_indices = (global_caption_mask_tensor == 1).nonzero(as_tuple=True)[0]
        if len(global_indices) == 0:
            return torch.tensor(0.0, device=device)
        # 提取 global_clip_scores
        global_clip_scores = raw_clip_scale_scores[global_indices]  # [num_global, 528, V]
        global_labels = torch.tensor(query_labels, device=device)[global_indices] # [num_global]
        # 正样本分数
        pos_scores = global_clip_scores[:, 527, :]  # [num_global, V]
        pos_scores = pos_scores[torch.arange(len(global_labels)), global_labels]  # [num_global]
        # 150个弱负样本分数
        weak_scores = global_clip_scores[:, :150, :]  # [num_global, 150, V]
        weak_scores = weak_scores[torch.arange(len(global_labels)).unsqueeze(1), torch.arange(150), global_labels.unsqueeze(1)]  # [num_global, 150]
        # loss
        loss_batch = torch.clamp(self.cfg['soft_pos_margin'] + weak_scores - pos_scores.unsqueeze(1), min=0)
        pos_loss = loss_batch.mean() if loss_batch.numel() > 0 else torch.tensor(0.0, device=device)

        # 新增：local caption与global embedding的hinge loss v2t
        local_indices_all = (global_caption_mask_tensor == 2).nonzero(as_tuple=True)[0]
        total_local_neg_loss = torch.tensor(0.0, device=device)
        num_local_neg = 0
        for i, global_idx in enumerate(global_indices):
            target_video_idx = global_labels[i]
            # global caption的global embedding分数
            pos_score = raw_clip_scale_scores[global_idx, 527, target_video_idx]  # scalar
            # 找到同video的local caption索引
            local_indices = local_indices_all[(torch.tensor(query_labels, device=device)[local_indices_all] == target_video_idx)]
            if len(local_indices) > 0:
                # local caption的global embedding分数
                neg_scores = raw_clip_scale_scores[local_indices, 527, target_video_idx]  # [num_local]
                loss_local = torch.clamp(self.cfg['soft_pos_margin'] + neg_scores - pos_score, min=0)
                total_local_neg_loss += torch.sum(loss_local)
                num_local_neg += len(local_indices)
        if num_local_neg > 0:
            local_neg_loss = total_local_neg_loss / num_local_neg
        else:
            local_neg_loss = torch.tensor(0.0, device=device)
        # 返回两个loss之和
        return pos_loss + local_neg_loss