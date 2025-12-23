import os
import sys
import cv2
import torch
import numpy as np
from pytorch_i3d import InceptionI3d
import torchvision.transforms as transforms
import videotransforms  # 你如果自定义了可以替换
from tqdm import tqdm
import argparse

VIDEO_DIR = '/chenyaofo/chenchuanshen/datasets/Youcook2/raw_videos'
SAVE_DIR = '/chenyaofo/chenchuanshen/datasets/Youcook2/video_i3d_feats'
MODEL_PATH = 'models/rgb_imagenet.pt'

def get_all_videos(video_dir):
    video_exts = ('.mp4', '.webm', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.mpg', '.mpeg', '.m4v', '.3gp', '.mts', '.ts')
    video_files = []
    for root, _, files in os.walk(video_dir):
        for f in files:
            if f.lower().endswith(video_exts):
                video_files.append(os.path.join(root, f))
    return video_files

def clip_center_crop_cv(img, size=224):
    """
    用OpenCV高效完成center crop
    img: numpy array (H, W, C)
    size: crop后的高度和宽度（正方形）
    """
    h, w, _ = img.shape
    start_x = (w - size) // 2
    start_y = (h - size) // 2
    end_x = start_x + size
    end_y = start_y + size
    # 防止视频小于size
    if start_x < 0 or start_y < 0:
        pad_h = max(size - h, 0)
        pad_w = max(size - w, 0)
        img = cv2.copyMakeBorder(img, pad_h//2, pad_h-(pad_h//2), pad_w//2, pad_w-(pad_w//2), cv2.BORDER_CONSTANT, value=0)
        h, w, _ = img.shape
        start_x = (w - size) // 2
        start_y = (h - size) // 2
        end_x = start_x + size
        end_y = start_y + size
    crop_img = img[start_y:end_y, start_x:end_x, :]
    return crop_img
def video_to_segments(video_path, fps_segments=16):
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / orig_fps if orig_fps > 0 else 0
    if duration == 0 or total_frames < 1:
        cap.release()
        return []
    # 1. 一次性读入全部帧，全部center crop
    frames = []
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = clip_center_crop_cv(frame, size=224)
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        return []
    frames = np.array(frames)  # (num_frames, 224, 224, 3)
    # 2. 预计算每个segment（每秒一个segment，每个segment fps_segments帧）
    segment_list = []
    for sec in range(int(duration)):
        idxs = [int((sec + i * (1.0 / fps_segments)) * orig_fps) for i in range(fps_segments)]
        if all(idx < frames.shape[0] for idx in idxs):
            segment = frames[idxs]  # shape: (fps_segments, 224, 224, 3)
            segment_list.append(segment)
        else:
            break
    return segment_list
# 其它运行提取I3D部分建议不变
def extract_feats_i3d(i3d, segments, device='cuda', batch_size=8):
    feats = []
    batch = []
    for seg in segments:
        seg = seg.transpose(3, 0, 1, 2)  # (3, T, H, W)
        batch.append(seg)
        if len(batch) == batch_size:
            batch_tensor = torch.from_numpy(np.stack(batch)).float() / 255.0
            batch_tensor = batch_tensor.to(device)
            with torch.no_grad():
                features = i3d.extract_features(batch_tensor)
                save = features.squeeze()
                if len(save.shape) == 1:
                    save = save.unsqueeze(0)
                feats.append(save.cpu().numpy())
            batch = []
    if batch:
        batch_tensor = torch.from_numpy(np.stack(batch)).float() / 255.0
        batch_tensor = batch_tensor.to(device)
        with torch.no_grad():
            features = i3d.extract_features(batch_tensor)
            save = features.squeeze()
            if len(save.shape) == 1:
                save = save.unsqueeze(0)
            feats.append(save.cpu().numpy())
    return np.concatenate(feats, axis=0)

# def clip_center_crop(img, size=224):
#     img = transforms.ToPILImage()(img)
#     img = transforms.CenterCrop(size)(img)
#     img = np.array(img)
#     return img

# def video_to_segments(video_path, fps_segments=16):
#     cap = cv2.VideoCapture(video_path)
#     orig_fps = cap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     duration = total_frames / orig_fps if orig_fps > 0 else 0
#     segment_list = []
#     if duration == 0:
#         cap.release()
#         return segment_list
#     for sec in tqdm(range(int(duration)), desc='Extracting video segments'):
#         seg_frames = []
#         for i in range(fps_segments):
#             target_time = sec + i * (1.0 / fps_segments)
#             target_frame = int(target_time * orig_fps)
#             cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame = clip_center_crop(frame)
#             seg_frames.append(frame)
#         if len(seg_frames) == fps_segments:
#             segment_list.append(np.stack(seg_frames))
#         else:
#             break
#     cap.release()
#     return segment_list

def extract_feats_i3d(i3d, segments, device='cuda', batch_size=12):
    feats = []
    batch = []
    for seg in tqdm(segments, desc='Extracting segments'):
        seg = seg.transpose(3, 0, 1, 2)
        batch.append(seg)
        if len(batch) == batch_size:
            batch_tensor = torch.from_numpy(np.stack(batch)).float() / 255.0
            batch_tensor = batch_tensor.to(device)
            with torch.no_grad():
                features = i3d.extract_features(batch_tensor)
                save = features.squeeze()
                if len(save.shape) == 1:
                    save = save.unsqueeze(0)
                feats.append(save.cpu().numpy())
            batch = []
    if batch:
        batch_tensor = torch.from_numpy(np.stack(batch)).float() / 255.0
        batch_tensor = batch_tensor.to(device)
        with torch.no_grad():
            features = i3d.extract_features(batch_tensor)
            save = features.squeeze()
            if len(save.shape) == 1:
                save = save.unsqueeze(0)
            feats.append(save.cpu().numpy())
    return np.concatenate(feats, axis=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, required=True, help='GPU device id')
    parser.add_argument('--videos_file', type=str, required=True, help='txt containing video paths')
    parser.add_argument('--save_dir', type=str, default=SAVE_DIR)
    parser.add_argument('--model_path', type=str, default=MODEL_PATH)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = 'cuda'

    # 加载模型
    i3d = InceptionI3d(400, in_channels=3)
    i3d.load_state_dict(torch.load(args.model_path))
    i3d = i3d.to(device)
    i3d.eval()

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
        segments = video_to_segments(video_path, fps_segments=16)
        print('抽帧结束')
        if len(segments) == 0:
            print('No valid segment:', video_path)
            continue
        feats = extract_feats_i3d(i3d, segments, device=device)
        print(f'{feats.shape}, 存入了{save_path}')
        np.save(save_path, feats)

if __name__ == '__main__':
    main()
