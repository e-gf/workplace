import os
import numpy as np
import h5py
from tqdm import tqdm
npy_folder = '/chenyaofo/chenchuanshen/workspace/charades/npy_files_from_ours'
output_h5 = '/chenyaofo/chenchuanshen/workspace/charades/video_i3d_feats.h5'
# 所有npy文件
npy_files = [f for f in os.listdir(npy_folder) if f.endswith('.npy')]
with h5py.File(output_h5, 'w') as h5f:
    for fname in tqdm(npy_files):
        video_id = os.path.splitext(fname)[0]
        npy_path = os.path.join(npy_folder, fname)
        feat = np.load(npy_path)
        grp = h5f.require_group(video_id)
        # 使用 'i3d_feat' 作为数据集名称
        grp.create_dataset('i3d_feat', data=feat)