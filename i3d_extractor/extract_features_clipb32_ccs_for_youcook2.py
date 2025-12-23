import os
import sys
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse

# from open_clip import create_model_and_transforms, tokenize  # 如果你用openai版clip则换from clip import load ....
import clip
# 超参数配置
VIDEO_DIR = '/chenyaofo/chenchuanshen/datasets/Youcook2/raw_videos'
SAVE_DIR = '/chenyaofo/chenchuanshen/datasets/Youcook2/video_clip_feats'
CLIP_MODEL = 'ViT-B-32'
# CLIP_PRETRAIN = 'laion2b_e16'

def get_all_videos(video_dir):
    video_exts = ('.mp4', '.webm', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.mpg', '.mpeg', '.m4v', '.3gp', '.mts', '.ts')
    video_files = []
    for root, _, files in os.walk(video_dir):
        for f in files:
            if f.lower().endswith(video_exts):
                video_files.append(os.path.join(root, f))
    return video_files

def extract_frames_per_sec(video_path):
    """
    抽取视频中每秒1帧，并做中心裁剪和resize到224。
    返回的shape为 (num_frames, 3, 224, 224)
    """
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / orig_fps if orig_fps > 0 else 0
    if duration == 0 or total_frames < 1:
        cap.release()
        return []
    frames = []
    for sec in range(int(duration)):
        frame_idx = int(sec * orig_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        # center crop
        if h < 224 or w < 224:
            pad_h = max(224 - h, 0)
            pad_w = max(224 - w, 0)
            frame = cv2.copyMakeBorder(frame, pad_h//2, pad_h-(pad_h//2), pad_w//2, pad_w-(pad_w//2), cv2.BORDER_CONSTANT, value=0)
            h, w, _ = frame.shape
        start_x = (w - 224) // 2
        start_y = (h - 224) // 2
        frame = frame[start_y:start_y+224, start_x:start_x+224, :]
        frames.append(frame)
    cap.release()
    return frames

def extract_feats_clip(clip_model, clip_preprocess, frames, device='cuda'):
    """
    用clip的视觉encoder提取所有帧特征，直接整个batch一次forward
    """
    # 1. 先把所有frames转成预处理后的tensor
    img_tensor_list = []
    transform = clip_preprocess
    for img in tqdm(frames, desc="Preprocess frames"):
        img_pil = transforms.ToPILImage()(img)
        img_tensor = transform(img_pil)
        img_tensor_list.append(img_tensor)
    # 2. 堆叠成batch
    batch_tensor = torch.stack(img_tensor_list, dim=0).to(device)      # shape: (N, 3, 224, 224)
    batch_tensor = batch_tensor.type(clip_model.dtype)                 # dtype对齐
    # 3. 一次性forward提特征
    with torch.no_grad():
        feat = clip_model.encode_image(batch_tensor)                   # shape: (N, 512)
        feats = feat.cpu().numpy()
    return feats   # shape: (num_frames, feature_dim)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, required=True, help='GPU device id')
    parser.add_argument('--videos_file', type=str, required=True, help='txt containing video paths')
    parser.add_argument('--save_dir', type=str, default=SAVE_DIR)
    parser.add_argument('--clip_model', type=str, default=CLIP_MODEL)
    # parser.add_argument('--clip_pretrain', type=str, default=CLIP_PRETRAIN)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = 'cuda'

    # 加载CLIP模型
    model, preprocess = clip.load('ViT-B/32', device=device)
    model = model.to(device)
    model.eval()

    # 读取视频列表
    with open(args.videos_file, 'r') as f:
        video_files = [x.strip() for x in f.readlines() if x.strip()]

    print(f'[{args.gpu}] Found {len(video_files)} videos in {args.videos_file}')
    os.makedirs(args.save_dir, exist_ok=True)
    for video_path in tqdm(video_files):
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        save_path = os.path.join(args.save_dir, f'{video_id}.npy')
        if os.path.exists(save_path):
            continue
        frames = extract_frames_per_sec(video_path)
        print('抽帧结束')
        if len(frames) == 0:
            print('No valid frame:', video_path)
            continue
        feats = extract_feats_clip(model, preprocess, frames, device=device)
        print(f'{feats.shape}, 存入了{save_path}')
        np.save(save_path, feats)

if __name__ == '__main__':
    main()
