import torch
import h5py
import numpy as np
import os
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel

# ===== 加载 tokenizer 和模型 =====
# 设置镜像源
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
roberta = RobertaModel.from_pretrained('roberta-large').to('cuda')
roberta.eval()

# ===== 读取训练与验证文本文件 =====
train_txt_path = '/chenyaofo/chenchuanshen/datasets/charades/charadestrain.caption.txt'
val_txt_path = '/chenyaofo/chenchuanshen/datasets/charades/charadesval.caption.txt'
test_txt_path = '/chenyaofo/chenchuanshen/datasets/charades/charadestest.caption.txt'

lines = []
for path in [train_txt_path, val_txt_path, test_txt_path]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines.extend(f.readlines())
    except FileNotFoundError:
        # 某个文件可能不存在，忽略即可
        pass

# ===== 输出文件路径 =====
output_h5_path = "/chenyaofo/chenchuanshen/workspace/generate_caption/charades/charades_query_roberta.hdf5"
h5f = h5py.File(output_h5_path, 'w')

# ===== 设置 batch size =====
BATCH_SIZE = 32

# ===== 收集所有 (key, caption) 对 =====
items = []
for line in lines:
    line = line.strip()
    if line:  # 跳过空行
        # 解析格式: video_id#enc#segment_id caption
        parts = line.split(' ', 1)  # 只分割第一个空格
        if len(parts) == 2:
            key = parts[0]  # video_id#enc#segment_id
            caption = parts[1]  # caption文本
            items.append((key, caption))

print(f"总共读取到 {len(items)} 条caption数据")

# ===== 按 batch 编码并写入 HDF5 =====
for i in tqdm(range(0, len(items), BATCH_SIZE)):
    batch = items[i:i + BATCH_SIZE]
    keys = [x[0] for x in batch]
    captions = [x[1] for x in batch]

    # 编码 batch captions
    encoded = tokenizer(captions, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoded["input_ids"].cuda()
    attention_mask = encoded["attention_mask"].cuda()

    with torch.no_grad():
        outputs = roberta(input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state

    # 写入每个句子的特征（如重复键则覆盖）
    for j, key in enumerate(keys):
        # 每一句的 token 数量可能不同，所以按 j 分别写入
        length = attention_mask[j].sum().item()  # 真实 token 数
        token_vecs = token_embeddings[j, :length, :].cpu().numpy().astype(np.float32)
        if key in h5f:
            del h5f[key]
        h5f.create_dataset(key, data=token_vecs)

h5f.close()

print(f"全部特征成功写入：{output_h5_path}")
print(f"总共写入的caption数量: {len(items)}")

# 统计唯一的video数量
unique_videos = set()
for key, _ in items:
    video_id = key.split('#')[0]  # 提取video_id部分
    unique_videos.add(video_id)
print(f"总共写入的video数量: {len(unique_videos)}")

# ===== 展示HDF5文件内容结构 =====
print("\n" + "="*50)
print("HDF5文件内容结构展示:")
print("="*50)

# 重新打开HDF5文件进行读取展示
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
    
    # 展示一个具体数据集的详细内容
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
    
    # 展示前几个token的embedding值（前3个token，前5个维度）
    print(f"\n前3个token的embedding值（前5个维度）:")
    print("-" * 50)
    embeddings = first_dataset[:]
    for i in range(min(3, embeddings.shape[0])):
        token_embedding = embeddings[i, :5]  # 前5个维度
        print(f"Token {i+1}: {token_embedding}")
    
    # 展示embedding的统计信息
    print(f"\nEmbedding统计信息:")
    print("-" * 50)
    print(f"最小值: {embeddings.min():.6f}")
    print(f"最大值: {embeddings.max():.6f}")
    print(f"均值: {embeddings.mean():.6f}")
    print(f"标准差: {embeddings.std():.6f}")

print("\n" + "="*50)
print("HDF5文件结构展示完成!")
print("="*50)
