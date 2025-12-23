import h5py
import numpy as np

caption_h5 = h5py.File('caption.hdf5', 'r')
raw_h5 = h5py.File('raw.hdf5', 'r')

# 收集所有 vid
vids = set()
for key in caption_h5.keys():
    if key.endswith('_global'):
        vids.add(key[:-7])  # 去掉"_global"
vids = sorted(list(vids))  # 可选排序保证索引稳定

emb_dim = caption_h5[list(caption_h5.keys())[0]].shape[0]  # 一般为1024

# 分组构建
caption_tensor = np.zeros((len(vids), 6, emb_dim), dtype='float32')
mask = np.zeros((len(vids), 6), dtype=np.int32)

# mapping: key是vid, value是index
vid_idx_map = {vid:i for i,vid in enumerate(vids)}

for vid in vids:
    # global
    gemb = caption_h5[f'{vid}_global'][:]
    caption_tensor[vid_idx_map[vid], 0] = gemb
    mask[vid_idx_map[vid], 0] = 1
    # local
    idx = 1
    for key in caption_h5.keys():
        if key.startswith(vid+'_') and not key.endswith('_global'):
            caption_tensor[vid_idx_map[vid], idx] = caption_h5[key][:]
            mask[vid_idx_map[vid], idx] = 1
            idx += 1
        if idx > 5:
            break  # 只取前5个local

print('caption feat读取完毕')

raw_keys = list(raw_h5.keys())
raw_tensor = []
query_gt_vid = []  # 保留每个query的gt_vid
for k in raw_keys:
    emb = raw_h5[k][:]
    raw_tensor.append(emb)
    # 按照你的新规范：vid_start_end_index
    parts = k.split('_')
    # 右边三个分别是 start, end, index
    vid = '_'.join(parts[:-3])
    query_gt_vid.append(vid)
raw_tensor = np.stack(raw_tensor, axis=0)  # [query_num, emb_dim]


print('raw feat读取完毕')


import torch
caption_tensor_torch = torch.tensor(caption_tensor)  # [vid_num, 6, dim]
raw_tensor_torch = torch.tensor(raw_tensor)          # [query_num, dim]
# Normalize tensors (如果上面没做过)
caption_tensor_torch = torch.nn.functional.normalize(caption_tensor_torch, dim=-1, p=2)
raw_tensor_torch = torch.nn.functional.normalize(raw_tensor_torch, dim=-1, p=2)
# 为了高效批处理
raw_tensor_exp = raw_tensor_torch.unsqueeze(1).unsqueeze(1) # [q,1,1,dim]
caption_tensor_exp = caption_tensor_torch.unsqueeze(0)      # [1,vid,6,dim]
# [q,vid,6]
print('raw tensor shape is:', raw_tensor_exp.shape)
print('caption tensor shape is:', caption_tensor_exp.shape)
sims = torch.matmul(raw_tensor_exp, caption_tensor_exp.transpose(-1,-2)).squeeze(-2)  # [q,vid,6]
# 这里每个query对所有vid的global/local都算出得分了
print(f'sim计算完毕,维度是{sims.shape}')

weight1, weight2 = 1.0, 0.0  # 举例，可调
global_scores = sims[:,:,0]         # [q,vid]
local_scores_max = sims[:,:,1:].max(dim=2).values   # [q,vid]
final_scores_global = weight1 * global_scores + weight2 * local_scores_max  # [q,vid]

weight1, weight2 = 0.7, 0.3  # 举例，可调
final_scores_ours = weight1 * global_scores + weight2 * local_scores_max  # [q,vid]



def get_ranks(scores, gt_vids, vids):
    ranks = []
    for i in range(scores.shape[0]):  # query loop
        gt_vid = gt_vids[i]
        gt_idx = vids.index(gt_vid)
        rank_pos = (-scores[i]).argsort().tolist().index(gt_idx) + 1
        ranks.append(rank_pos)
    ranks = np.array(ranks)
    R1 = np.mean(ranks <= 1) * 100
    R5 = np.mean(ranks <= 5) * 100
    R10 = np.mean(ranks <= 10) * 100
    R100 = np.mean(ranks <= 100) * 100
    sumR = (R1+R5+R10+R100)
    print(f'R@1: {R1:.4f}, R@5: {R5:.4f}, R@10: {R10:.4f}, R@100: {R100:.4f}, sumR: {sumR}')
    return

ranks = get_ranks(final_scores_global, query_gt_vid, vids)
ranks2 = get_ranks(final_scores_ours, query_gt_vid, vids)


