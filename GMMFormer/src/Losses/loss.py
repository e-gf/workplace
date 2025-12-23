import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict as edict
from Models.gmmformer.model_components import clip_nce, frame_nce
from Losses.model_loss import LocalHingeLoss

import ipdb


class query_diverse_loss(nn.Module):
    def __init__(self, config):
        torch.nn.Module.__init__(self)
        self.mrg = config['neg_factor'][0]
        self.alpha = config['neg_factor'][1]
        
    def forward(self, x, label_dict):

        bs = x.shape[0]
        x = F.normalize(x, dim=-1)
        cos = torch.matmul(x, x.t())

        N_one_hot = torch.zeros((bs, bs))
        for i, label in label_dict.items():
            N_one_hot[label[0]:(label[-1]+1), label[0]:(label[-1]+1)] = torch.ones((len(label), len(label)))
        N_one_hot = N_one_hot - torch.eye(bs)
        N_one_hot = N_one_hot.cuda()
    
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))
        
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
    
        neg_term = torch.log(1 + N_sim_sum).sum() / bs
        
        return neg_term


class loss(nn.Module):
    def __init__(self, cfg):
        super(loss, self).__init__()
        self.cfg = cfg
        self.clip_nce_criterion = clip_nce(reduction='mean')
        self.video_nce_criterion = clip_nce(reduction='mean')

        self.qdl = query_diverse_loss(cfg)
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
        
        clip_scale_scores = input_list[0]
        clip_scale_scores_ = input_list[1]
        label_dict = input_list[2]
        frame_scale_scores = input_list[3]
        frame_scale_scores_ = input_list[4]

        query = input_list[5]
        
        global_caption_mask = input_list[6]
        mapped_raw_clip_scale_scores = input_list[7]
        local_start_end_tensor = input_list[8]
        raw_label_dict = input_list[9]
        
        query_labels_ = input_list[10]
        

        clip_nce_loss = self.cfg['loss_factor'][0] * self.clip_nce_criterion(query_labels_, label_dict, clip_scale_scores_)
        clip_trip_loss = self.get_clip_triplet_loss(clip_scale_scores, query_labels_)

        frame_nce_loss = self.cfg['loss_factor'][1] * self.video_nce_criterion(query_labels_, label_dict, frame_scale_scores_)
        frame_trip_loss = self.get_clip_triplet_loss(frame_scale_scores, query_labels_)

        qdl_loss = self.cfg['loss_factor'][2] * self.qdl(query, label_dict)
        
        local_hinge_loss = self.cfg['local_hinge_weight'] * self.local_hinge_loss(global_caption_mask,mapped_raw_clip_scale_scores,local_start_end_tensor,raw_label_dict)

        global_soft_pos_loss = self.cfg['global_soft_pos_weight'] * self.global_soft_loss(mapped_raw_clip_scale_scores, query_labels, global_caption_mask)
        
        loss = clip_nce_loss + clip_trip_loss + frame_nce_loss + frame_trip_loss + qdl_loss + local_hinge_loss + global_soft_pos_loss

        return loss

    def get_clip_triplet_loss(self, query_context_scores, labels):
        v2t_scores = query_context_scores.t()
        t2v_scores = query_context_scores
        labels = np.array(labels)

        # cal_v2t_loss
        v2t_loss = 0
        for i in range(v2t_scores.shape[0]):
            pos_pair_scores = torch.mean(v2t_scores[i][np.where(labels == i)])


            neg_pair_scores, _ = torch.sort(v2t_scores[i][np.where(labels != i)[0]], descending=True)
            if self.cfg['use_hard_negative']:
                sample_neg_pair_scores = neg_pair_scores[0]
            else:
                v2t_sample_max_idx = neg_pair_scores.shape[0]
                sample_neg_pair_scores = neg_pair_scores[
                    torch.randint(0, v2t_sample_max_idx, size=(1,)).to(v2t_scores.device)]

            v2t_loss += (self.cfg['margin'] + sample_neg_pair_scores - pos_pair_scores).clamp(min=0).sum()

        # cal_t2v_loss
        text_indices = torch.arange(t2v_scores.shape[0]).to(t2v_scores.device)
        t2v_pos_scores = t2v_scores[text_indices, labels]
        mask_score = copy.deepcopy(t2v_scores.data)
        mask_score[text_indices, labels] = 999
        _, sorted_scores_indices = torch.sort(mask_score, descending=True, dim=1)
        t2v_sample_max_idx = min(1 + self.cfg['hard_pool_size'],
                                 t2v_scores.shape[1]) if self.cfg['use_hard_negative'] else t2v_scores.shape[1]
        sample_indices = sorted_scores_indices[
            text_indices, torch.randint(1, t2v_sample_max_idx, size=(t2v_scores.shape[0],)).to(t2v_scores.device)]

        t2v_neg_scores = t2v_scores[text_indices, sample_indices]

        t2v_loss = (self.cfg['margin'] + t2v_neg_scores - t2v_pos_scores).clamp(min=0)

        return t2v_loss.sum() / len(t2v_scores) + v2t_loss / len(v2t_scores)

    def get_frame_trip_loss(self, query_context_scores):
        """ ranking loss between (pos. query + pos. video) and (pos. query + neg. video) or (neg. query + pos. video)
        Args:
            query_context_scores: (N, N), cosine similarity [-1, 1],
                Each row contains the scores between the query to each of the videos inside the batch.
        """

        bsz = len(query_context_scores)

        diagonal_indices = torch.arange(bsz).to(query_context_scores.device)
        pos_scores = query_context_scores[diagonal_indices, diagonal_indices]  # (N, )
        query_context_scores_masked = copy.deepcopy(query_context_scores.data)
        # impossibly large for cosine similarity, the copy is created as modifying the original will cause error
        query_context_scores_masked[diagonal_indices, diagonal_indices] = 999
        pos_query_neg_context_scores = self.get_neg_scores(query_context_scores, query_context_scores_masked)
        neg_query_pos_context_scores = self.get_neg_scores(query_context_scores.transpose(0, 1),
                                                           query_context_scores_masked.transpose(0, 1))
        loss_neg_ctx = self.get_ranking_loss(pos_scores, pos_query_neg_context_scores)
        loss_neg_q = self.get_ranking_loss(pos_scores, neg_query_pos_context_scores)
        return loss_neg_ctx + loss_neg_q

    def get_neg_scores(self, scores, scores_masked):
        """
        scores: (N, N), cosine similarity [-1, 1],
            Each row are scores: query --> all videos. Transposed version: video --> all queries.
        scores_masked: (N, N) the same as scores, except that the diagonal (positive) positions
            are masked with a large value.
        """

        bsz = len(scores)
        batch_indices = torch.arange(bsz).to(scores.device)

        _, sorted_scores_indices = torch.sort(scores_masked, descending=True, dim=1)

        sample_min_idx = 1  # skip the masked positive

        sample_max_idx = min(sample_min_idx + self.cfg['hard_pool_size'], bsz) if self.cfg['use_hard_negative'] else bsz

        # sample_max_idx = 2

        # (N, )
        sampled_neg_score_indices = sorted_scores_indices[batch_indices, torch.randint(sample_min_idx, sample_max_idx,
                                                                                       size=(bsz,)).to(scores.device)]

        sampled_neg_scores = scores[batch_indices, sampled_neg_score_indices]  # (N, )
        return sampled_neg_scores

    def get_ranking_loss(self, pos_score, neg_score):
        """ Note here we encourage positive scores to be larger than negative scores.
        Args:
            pos_score: (N, ), torch.float32
            neg_score: (N, ), torch.float32
        """
        return torch.clamp(self.cfg['margin'] + neg_score - pos_score, min=0).sum() / len(pos_score)
    
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
    