from transformers import AutoTokenizer, AutoModel
import torch
import json
import h5py
import os
# 设定设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

local_dir = "/chenyaofo/chenchuanshen/workspace/sentence-bert/model/bge-large-zh-v1.5"  # 与上一步保持一致
# 从本地目录加载，不会联网
tokenizer = AutoTokenizer.from_pretrained(local_dir)
model = AutoModel.from_pretrained(local_dir).to(device)
model.eval()

global_json = '/chenyaofo/chenchuanshen/workspace/generate_caption/youcook2/youcook_val_global.json'
with open(global_json, 'r', encoding='utf-8') as f:
    global_caps = json.load(f)


local_json = '/chenyaofo/chenchuanshen/workspace/generate_caption/youcook2/youcook_val_020_merged.json'
with open(local_json, 'r', encoding='utf-8') as f:
    local_caps = json.load(f)


def get_embeddings(text_list, tokenizer, model, device):
    encoded_input = tokenizer(text_list, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
        sentence_embeddings = model_output[0][:, 0]
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings.cpu()  # 转为cpu方便存储


# hdf5_file = 'caption.hdf5'
# with h5py.File(hdf5_file, 'w') as h5f:
#     # Global captions
#     for vid, items in global_caps.items():
#         if isinstance(items, list):
#             cap = items[0]['caption']
#         else:
#             cap = items['caption']
#         emb = get_embeddings([cap], tokenizer, model, device)[0].numpy()
#         h5f.create_dataset(f'{vid}_global', data=emb)
#         print(f'{vid}_global')

#     # Local captions
#     for vid, items in local_caps.items():
#         for seg in items:
#             start, end = seg["start_time"], seg["end_time"]
#             cap = seg["caption"]
#             key = f'{vid}_{start}_{end}'
#             emb = get_embeddings([cap], tokenizer, model, device)[0].numpy()
#             h5f.create_dataset(key, data=emb)

def get_unique_key(h5f, base_key):
    """返回在 h5f 中不重复的 key"""
    key = base_key
    idx = 0
    key = f"{base_key}_{idx}" #如果0是存在了的，就进入下面的while
    while key in h5f:
        idx += 1
        key = f"{base_key}_{idx}"
    return key

raw_json = '/chenyaofo/chenchuanshen/workspace/sentence-bert/youcookii_annotations_trainval.json'
with open(raw_json, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

with h5py.File('raw.hdf5', 'w') as h5f:
    for vid, vinfo in raw_data['database'].items():
        if vinfo['subset'] != 'validation':
            continue
        for ann in vinfo['annotations']:
            seg_start, seg_end = ann["segment"]
            sentence = ann["sentence"]
            base_key = f'{vid}_{seg_start}_{seg_end}'
            key = get_unique_key(h5f, base_key)
            emb = get_embeddings([sentence], tokenizer, model, device)[0].numpy()
            h5f.create_dataset(key, data=emb)