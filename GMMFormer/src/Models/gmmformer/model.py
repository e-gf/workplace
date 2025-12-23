import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict as edict
from Models.gmmformer.model_components import BertAttention, LinearLayer, \
                                            TrainablePositionalEncoding, GMMBlock

import ipdb



class GMMFormer_Net(nn.Module):
    def __init__(self, config):
        super(GMMFormer_Net, self).__init__()
        self.config = config

        self.query_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_desc_l,
                                                           hidden_size=config.hidden_size, dropout=config.input_drop)
        self.clip_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                         hidden_size=config.hidden_size, dropout=config.input_drop)
        self.frame_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                          hidden_size=config.hidden_size, dropout=config.input_drop)

        self.query_input_proj = LinearLayer(config.query_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        self.query_encoder = BertAttention(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                 attention_probs_dropout_prob=config.drop))


        self.clip_input_proj = LinearLayer(config.visual_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        self.clip_encoder = GMMBlock(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                 attention_probs_dropout_prob=config.drop))
        self.clip_encoder_2 = GMMBlock(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                 attention_probs_dropout_prob=config.drop))

        self.frame_input_proj = LinearLayer(config.visual_input_size, config.hidden_size, layer_norm=True,
                                             dropout=config.input_drop, relu=True)
        self.frame_encoder_1 = GMMBlock(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                 attention_probs_dropout_prob=config.drop))
        self.frame_encoder_2 = GMMBlock(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                 attention_probs_dropout_prob=config.drop))
                    
        self.modular_vector_mapping = nn.Linear(config.hidden_size, out_features=1, bias=False)
        self.modular_vector_mapping_2 = nn.Linear(config.hidden_size, out_features=1, bias=False)

        self.pool_layers = nn.ModuleList([nn.Identity()] + [nn.AvgPool1d(i, stride=1) for i in range(2, config.map_size + 1)] )

        self.reset_parameters()

    def reset_parameters(self):
        """ Initialize the weights."""

        def re_init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Conv1d):
                module.reset_parameters()
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.apply(re_init)

    def set_hard_negative(self, use_hard_negative, hard_pool_size):
        """use_hard_negative: bool; hard_pool_size: int, """
        self.config.use_hard_negative = use_hard_negative
        self.config.hard_pool_size = hard_pool_size


    def forward(self, batch):

        clip_video_feat = batch['clip_video_features']
        query_feat = batch['text_feat']
        query_mask = batch['text_mask']
        query_labels = batch['text_labels']

        frame_video_feat = batch['frame_video_features']
        frame_video_mask = batch['videos_mask']
        
        global_caption_mask = batch['global_caption_mask']
        local_start_end_tensor = batch['local_start_end_tensor']

        encoded_frame_feat, vid_proposal_feat, mapped_clip_feat = self.encode_context(
            clip_video_feat, frame_video_feat, frame_video_mask)

        
        
        clip_scale_scores, clip_scale_scores_, frame_scale_scores, frame_scale_scores_,query_labels_,mapped_raw_clip_scale_scores,query \
            = self.get_pred_from_raw_query(
            query_feat, query_mask, query_labels, vid_proposal_feat, encoded_frame_feat, return_query_feats=True,global_caption_mask=global_caption_mask, mapped_clip_feat=mapped_clip_feat)

        raw_label_dict = {}
        for index, label in enumerate(query_labels):
            if label in raw_label_dict:
                raw_label_dict[label].append(index)
            else:
                raw_label_dict[label] = []
                raw_label_dict[label].append(index)
        
        # 正常训练阶段,使用所有loss
        label_dict = {}
        for index, label in enumerate(query_labels_):
            if label in label_dict:
                label_dict[label].append(index)
            else:
                label_dict[label] = []
                label_dict[label].append(index)


        return [clip_scale_scores, clip_scale_scores_, label_dict, frame_scale_scores, frame_scale_scores_, query, global_caption_mask, mapped_raw_clip_scale_scores, local_start_end_tensor, raw_label_dict, query_labels_]


    def encode_query(self, query_feat, query_mask):
        encoded_query = self.encode_input(query_feat, query_mask, self.query_input_proj, self.query_encoder,
                                          self.query_pos_embed)  # (N, Lq, D)
        if query_mask is not None:
            mask = query_mask.unsqueeze(1)
        

        video_query = self.get_modularized_queries(encoded_query, query_mask)  # (N, D) * 1

        return video_query


    def encode_context(self, clip_video_feat, frame_video_feat, video_mask=None):

        encoded_clip_feat = self.encode_input(clip_video_feat, None, self.clip_input_proj, self.clip_encoder,
                                               self.clip_pos_embed)
        encoded_clip_feat = self.clip_encoder_2(encoded_clip_feat, None)                # [bs, 32, 384]

        encoded_frame_feat = self.encode_input(frame_video_feat, video_mask, self.frame_input_proj,
                                                self.frame_encoder_1,
                                                self.frame_pos_embed)                   # [bs, N, 384]
        encoded_frame_feat = self.frame_encoder_2(encoded_frame_feat, video_mask.unsqueeze(1))

        encoded_frame_feat = self.get_modularized_frames(encoded_frame_feat, video_mask)
        
        vid_proposal_feat_map = self.encode_feat_map(encoded_clip_feat)


        return encoded_frame_feat, encoded_clip_feat, vid_proposal_feat_map

    def encode_feat_map(self, x_feat):
        batch_size, seq_len, feat_dim = x_feat.shape
        pool_in = x_feat.permute(0, 2, 1)

        proposal_feat_map = []
        # index_ranges = []
        for idx, pool in enumerate(self.pool_layers):
            x = pool(pool_in).permute(0, 2, 1)
            proposal_feat_map.append(x)
            # 计算当前池化操作对应的原始索引范围
            # if idx == 0:  # nn.Identity() 直接对应原始索引
            #     ranges = [(i, i) for i in range(seq_len)]
            # else:
            #     kernel_size = idx + 1  # 从2开始的池化窗口大小
            #     ranges = [(i, i + kernel_size - 1) for i in range(seq_len - kernel_size + 1)]
            
            # index_ranges.append(ranges)
        proposal_feat_map = torch.cat(proposal_feat_map, dim=1)


        return proposal_feat_map
    
    @staticmethod
    def encode_input(feat, mask, input_proj_layer, encoder_layer, pos_embed_layer):
        """
        Args:
            feat: (N, L, D_input), torch.float32
            mask: (N, L), torch.float32, with 1 indicates valid query, 0 indicates mask
            input_proj_layer: down project input
            encoder_layer: encoder layer
            pos_embed_layer: positional embedding layer
        """

        feat = input_proj_layer(feat)
        feat = pos_embed_layer(feat)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (N, 1, L), torch.FloatTensor
        return encoder_layer(feat, mask)  # (N, L, D_hidden)


    def get_modularized_queries(self, encoded_query, query_mask):
        """
        Args:
            encoded_query: (N, L, D)
            query_mask: (N, L)
            return_modular_att: bool
        """
        modular_attention_scores = self.modular_vector_mapping(encoded_query)  # (N, L, 2 or 1)
        modular_attention_scores = F.softmax(mask_logits(modular_attention_scores, query_mask.unsqueeze(2)), dim=1)
        modular_queries = torch.einsum("blm,bld->bmd", modular_attention_scores, encoded_query)  # (N, 2 or 1, D)
        return modular_queries.squeeze()

    def get_modularized_frames(self, encoded_query, query_mask):
        """
        Args:
            encoded_query: (N, L, D)
            query_mask: (N, L)
            return_modular_att: bool
        """
        modular_attention_scores = self.modular_vector_mapping_2(encoded_query)  # (N, L, 2 or 1)
        modular_attention_scores = F.softmax(mask_logits(modular_attention_scores, query_mask.unsqueeze(2)), dim=1)
        modular_queries = torch.einsum("blm,bld->bmd", modular_attention_scores, encoded_query)  # (N, 2 or 1, D)
        return modular_queries.squeeze()


    @staticmethod
    def get_clip_scale_scores(modularied_query, context_feat):

        modularied_query = F.normalize(modularied_query, dim=-1)
        context_feat = F.normalize(context_feat, dim=-1)

        clip_level_query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(2, 1, 0)

        query_context_scores, indices = torch.max(clip_level_query_context_scores,
                                                  dim=1)
        
        return query_context_scores, clip_level_query_context_scores


    @staticmethod
    def get_unnormalized_clip_scale_scores(modularied_query, context_feat):

        query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(2, 1, 0)

        output_query_context_scores, indices = torch.max(query_context_scores, dim=1)

        return output_query_context_scores


    def get_pred_from_raw_query(self, query_feat, query_mask, query_labels=None,
                                video_proposal_feat=None, encoded_frame_feat=None,
                                return_query_feats=False, 
                                global_caption_mask=None, mapped_clip_feat=None):

        video_query = self.encode_query(query_feat, query_mask)

        # get clip-level retrieval scores
        clip_scale_scores,_ = self.get_clip_scale_scores(       # [cap_num,128], [cap_num,128]
            video_query, video_proposal_feat)

        
        frame_scale_scores = torch.matmul(F.normalize(encoded_frame_feat, dim=-1), F.normalize(video_query, dim=-1).t()).permute(1, 0)

        if return_query_feats:
            clip_scale_scores_ = self.get_unnormalized_clip_scale_scores(video_query, video_proposal_feat)
            
            # our method
            _, mapped_raw_clip_scale_scores = self.get_clip_scale_scores(video_query, mapped_clip_feat)
            
            if global_caption_mask is not None:
                global_caption_mask_tensor = torch.tensor(global_caption_mask, device=video_query.device)

                if(self.config.query_or_caption == 0):
                    keep_indices = (global_caption_mask_tensor == 0).nonzero(as_tuple=True)[0] # 0是query 1是global 2是local 3是all
                elif(self.config.query_or_caption == 1):
                    keep_indices = (global_caption_mask_tensor != 1).nonzero(as_tuple=True)[0]
                elif(self.config.query_or_caption == 2):
                    keep_indices = (global_caption_mask_tensor != 2).nonzero(as_tuple=True)[0]
                elif(self.config.query_or_caption == 3):
                    keep_indices = (global_caption_mask_tensor != 3).nonzero(as_tuple=True)[0]
                else:
                    print("query_or_caption 参数无效")
                    exit()

                video_query = video_query[keep_indices]
                clip_scale_scores = clip_scale_scores[keep_indices]
                clip_scale_scores_ = clip_scale_scores_[keep_indices]
                frame_scale_scores = frame_scale_scores[keep_indices]
                query_labels_ = [query_labels[i] for i in keep_indices.tolist()]
                query = video_query
            

            frame_scale_scores_ = torch.matmul(encoded_frame_feat, video_query.t()).permute(1, 0)
            
            
            return clip_scale_scores, clip_scale_scores_, frame_scale_scores, frame_scale_scores_,query_labels_,mapped_raw_clip_scale_scores,query
        else:

            return clip_scale_scores, frame_scale_scores


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)
