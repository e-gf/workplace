import os
import json
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from calflops import calculate_flops
from transformers import AutoModel, AutoTokenizer

# ===== 配置 =====
max_seq_length = 128
model_name = "roberta-large"

# ===== 加载模型与分词器 =====
model = AutoModel.from_pretrained(model_name).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

# ===== 读取数据源 =====
# 目前支持 JSON（与 json2hdf5.py 相同结构）与 caption 文本（形如 'video#enc#idx caption'）
data_path = "/chenyaofo/chenchuanshen/datasets/charades/charadestest.caption.txt"

captions: List[Tuple[str, str]] = []
if data_path.lower().endswith(".json"):
    with open(data_path, "r", encoding="utf-8") as f:
        data: Dict[str, List[Dict]] = json.load(f)
    for video_id, segments in data.items():
        for seg in segments:
            start = seg["start_time"]
            end = seg["end_time"]
            caption = seg["caption"]
            key = f"{video_id}_{start:.2f}-{end:.2f}"
            captions.append((key, caption))
else:
    with open(data_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if " " in line:
                key, caption = line.split(" ", 1)
            else:
                key, caption = line, ""
            captions.append((key, caption))

# ===== 逐条计算 FLOPs，并统计 =====
flops_list: List[float] = []

def _to_float_from_unit_string(val):
    """
    将诸如 '35.65 GFLOPS' 转为基础 FLOPs 数值（单位：FLOPs）。
    支持的单位：FLOPS, KFLOPS, MFLOPS, GFLOPS, TFLOPS, PFLOPS（不区分大小写，允许前后空格）。
    """
    if not isinstance(val, str):
        return float(val)
    s = val.strip().upper()
    parts = s.split()
    if not parts:
        return 0.0
    number = float(parts[0])
    unit = parts[1] if len(parts) > 1 else "FLOPS"
    multipliers = {
        "FLOPS": 1.0,
        "KFLOPS": 1e3,
        "MFLOPS": 1e6,
        "GFLOPS": 1e9,
        "TFLOPS": 1e12,
        "PFLOPS": 1e15,
    }
    if unit not in multipliers:
        for k, m in multipliers.items():
            if s.endswith(k):
                num_str = s[: -len(k)].strip()
                number = float(num_str)
                return number * m
        return number
    return number * multipliers[unit]

def _to_float_generic(val):
    """
    通用字符串到浮点数的转换：
    - 支持纯数字
    - 支持带空格单位：'355.36 M', '12.3 K', '1.2 B'
    - 支持紧凑形式：'355.36M', '12.3k', '1.2b'
    - 若包含 'FLOPS'，回退到 _to_float_from_unit_string
    单位：K=1e3, M=1e6, B/G=1e9, T=1e12, P=1e15
    """
    if not isinstance(val, str):
        return float(val)
    s = val.strip().upper().replace(",", "")
    if "FLOPS" in s:
        return _to_float_from_unit_string(s)
    parts = s.split()
    if len(parts) == 2:
        num_str, unit = parts[0], parts[1]
    else:
        if s and s[-1].isalpha():
            num_str, unit = s[:-1], s[-1]
        else:
            num_str, unit = s, ""
    try:
        number = float(num_str)
    except Exception:
        return float(s)
    unit_multipliers = {
        "": 1.0,
        "K": 1e3,
        "M": 1e6,
        "B": 1e9,
        "G": 1e9,
        "T": 1e12,
        "P": 1e15,
    }
    mul = unit_multipliers.get(unit, 1.0)
    return number * mul

def _format_with_units(value: float, unit_base: str = "FLOPs") -> str:
    """
    将大数值格式化成人类可读形式，自动选择 K/M/G/T/P 单位。
    例如：2.59e10 -> '25.95 GFLOPs'
    """
    value = _to_float_generic(value)
    units = ["", "K", "M", "G", "T", "P"]
    abs_val = float(abs(value))
    idx = 0
    while abs_val >= 1000.0 and idx < len(units) - 1:
        abs_val /= 1000.0
        idx += 1
    sign = "-" if value < 0 else ""
    return f"{sign}{abs_val:.2f} {units[idx]}{unit_base}"

for _, text in tqdm(captions, desc="Calculating FLOPs per caption"):
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        return_attention_mask=True,
        padding=False,
        truncation="longest_first",
        max_length=max_seq_length
    )

    # Roberta 可能无 token_type_ids，统一补齐
    if "token_type_ids" not in encoded:
        encoded["token_type_ids"] = [0] * len(encoded["input_ids"])

    # 将当前样本 pad 到其自身长度（无需统一到全局 max），calflops 需要张量
    inputs = {
        "input_ids": torch.tensor([encoded["input_ids"]], dtype=torch.long),
        "token_type_ids": torch.tensor([encoded["token_type_ids"]], dtype=torch.long),
        "attention_mask": torch.tensor([encoded["attention_mask"]], dtype=torch.long),
    }

    flops, macs, params = calculate_flops(model=model, kwargs=inputs, print_results=False)
    flops_list.append(_to_float_from_unit_string(flops))

# ===== 统计指标 =====
flops_np = np.array(flops_list, dtype=np.float64)

stats = {
    "model_name": model_name,
    "num_samples": int(flops_np.size),
    "params": int(_to_float_generic(params)),
    "flops_min": float(np.min(flops_np)) if flops_np.size > 0 else 0.0,
    "flops_max": float(np.max(flops_np)) if flops_np.size > 0 else 0.0,
    "flops_mean": float(np.mean(flops_np)) if flops_np.size > 0 else 0.0,
    "flops_median": float(np.median(flops_np)) if flops_np.size > 0 else 0.0,
    "flops_std": float(np.std(flops_np)) if flops_np.size > 0 else 0.0,
    "flops_p10": float(np.percentile(flops_np, 10)) if flops_np.size > 0 else 0.0,
    "flops_p90": float(np.percentile(flops_np, 90)) if flops_np.size > 0 else 0.0,
    "flops_p95": float(np.percentile(flops_np, 95)) if flops_np.size > 0 else 0.0,
}

print("FLOPs 统计：")
for k, v in stats.items():
    if k.startswith("flops_"):
        print(f"{k}: {v} ({_format_with_units(v, 'FLOPs')})")
    elif k == "params":
        # 参数量用更通用的单位，不加 FLOPs 后缀
        print(f"{k}: {v} ({_format_with_units(v, '')}params)".replace("  ", " "))
    else:
        print(f"{k}: {v}")

# 可选：保存结果到同目录 json
out_dir = os.path.dirname(data_path)
out_path = os.path.join(out_dir, f"{model_name}_flops_stats_maxlen{max_seq_length}.json")
try:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"统计结果已保存：{out_path}")
except Exception as e:
    print(f"保存统计结果失败：{e}")