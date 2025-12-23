import h5py
import numpy as np
from tqdm import tqdm
import torch
import open_clip

# ===== 加载 CLIP tokenizer 和模型 =====
model_name = "ViT-B-32"
pretrained = "openai"
device = "cuda"

model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
tokenizer = open_clip.get_tokenizer(model_name)
model = model.to(device)
model.eval()

# ===== 读取训练与验证文本文件 =====
train_txt_path = "/chenyaofo/chenchuanshen/workspace/generate_caption/youcook2/extracted_data/training.txt"
val_txt_path = "/chenyaofo/chenchuanshen/workspace/generate_caption/youcook2/extracted_data/validation.txt"

lines = []
for path in [train_txt_path, val_txt_path]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines.extend(f.readlines())
    except FileNotFoundError:
        # 某个文件可能不存在，忽略即可
        pass

# ===== 输出文件路径 =====
output_h5_path = "/chenyaofo/chenchuanshen/workspace/generate_caption/youcook2/extracted_data/youcook2_query_clipb32.hdf5"
h5f = h5py.File(output_h5_path, 'w')

# ===== 设置 batch size =====
BATCH_SIZE = 32

# ===== 收集所有 (key, caption) 对 =====
items = []
for line in lines:
    line = line.strip()
    if line:
        parts = line.split(' ', 1)
        if len(parts) == 2:
            key = parts[0]
            caption = parts[1]
            items.append((key, caption))

# ===== 按 batch 编码并写入 HDF5 =====
for i in tqdm(range(0, len(items), BATCH_SIZE)):
    batch = items[i:i + BATCH_SIZE]
    keys = [x[0] for x in batch]
    captions = [x[1] for x in batch]

    # 编码 batch captions
    tokens = tokenizer(captions, context_length=77)
    tokens = torch.tensor(tokens).to(device)

    with torch.no_grad():
        text_features = model.encode_text(tokens)  # [batch, 512]
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # 可选：归一化
        # print(f"text_features shape: {text_features.shape}")  # 打印特征shape

    # 写入每个句子的特征（如重复键则覆盖）
    for j, key in enumerate(keys):
        vec = text_features[j].cpu().numpy().astype(np.float32)  # [512]
        if key in h5f:
            del h5f[key]
        h5f.create_dataset(key, data=vec)

h5f.close()

print(f"全部特征成功写入：{output_h5_path}")
print(f"总共写入的caption数量: {len(items)}")

# 统计唯一的video数量
unique_videos = set()
for key, _ in items:
    video_id = key.split('#')[0]
    unique_videos.add(video_id)
print(f"总共写入的video数量: {len(unique_videos)}")

# ===== 展示HDF5文件内容结构 =====
print("\n" + "="*50)
print("HDF5文件内容结构展示:")
print("="*50)

with h5py.File(output_h5_path, 'r') as h5f_read:
    print(f"文件路径: {output_h5_path}")
    print(f"文件大小: {h5f_read.id.get_filesize() / (1024*1024):.2f} MB")
    print(f"数据集总数: {len(h5f_read.keys())}")

    # 展示前5个数据集的信息
    print("\n前5个数据集的信息:")
    print("-" * 80)
    print(f"{'数据集名称':<30} {'形状':<15} {'数据类型':<10} {'大小(MB)':<10}")
    print("-" * 80)

    count = 0
    for key in h5f_read.keys():
        if count >= 5:
            break
        dataset = h5f_read[key]
        shape = dataset.shape
        dtype = dataset.dtype
        size_mb = dataset.nbytes / (1024*1024)
        print(f"{key:<30} {str(shape):<15} {str(dtype):<10} {size_mb:<10.3f}")
        count += 1

    if len(h5f_read.keys()) > 5:
        print(f"... 还有 {len(h5f_read.keys()) - 5} 个数据集")

    # 展示一个具体数据集的详细内容（CLIP文本向量为一维向量）
    print("\n第一个数据集的详细内容:")
    print("-" * 50)
    first_key = list(h5f_read.keys())[0]
    first_dataset = h5f_read[first_key]
    print(f"数据集名称: {first_key}")
    print(f"形状: {first_dataset.shape}")
    print(f"数据类型: {first_dataset.dtype}")
    print(f"维度信息: {first_dataset.ndim} 维")
    print(f"总元素数: {first_dataset.size}")
    print(f"内存大小: {first_dataset.nbytes / 1024:.2f} KB")

    # 展示前几个维度的值（前10个维度）
    print(f"\n前10个维度的值:")
    print("-" * 50)
    vec = first_dataset[:]
    print(vec[:10])

    # 展示向量统计信息
    print(f"\n向量统计信息:")
    print("-" * 50)
    print(f"最小值: {vec.min():.6f}")
    print(f"最大值: {vec.max():.6f}")
    print(f"均值: {vec.mean():.6f}")
    print(f"标准差: {vec.std():.6f}")

print("\n" + "="*50)
print("HDF5文件结构展示完成!")
print("="*50)