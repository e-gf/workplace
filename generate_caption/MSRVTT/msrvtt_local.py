import os
import json
import torch
import random
from datetime import timedelta
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from typing import List, Dict, Optional
import base64
import cv2
from typing import List, Dict
# import openai
import threading
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from time import sleep
import argparse
from queue import Queue
from tqdm import tqdm
from qwen_vl_utils import process_vision_info


# api = 'x'
# openai_client = openai.AzureOpenAI(
#     azure_endpoint="x",
#     api_version="2023-07-01-preview",
#     api_key=api, 
#     timeout=30.0
# )


# 创建共享队列和写入锁
result_queue = Queue()
file_lock = threading.Lock()
WRITE_BATCH_SIZE = 20  # 每处理20个视频批量写入一次

from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import math

def optimized_processing(filtered_video_folders, frames_folder, examples, output_json, segment_rate, max_workers=12, model=None, processor=None, device=None):
    """最终严格批次控制版本"""
    # 初始化写入线程
    writer_thread = threading.Thread(target=batch_writer, args=(output_json,))
    writer_thread.start()
    # 断点重启时不清空结果文件，保留已有数据
    if not os.path.exists(output_json):
        open(output_json, 'w').close()  # 只在文件不存在时创建空文件

    # 计算总批次数
    batch_size = max_workers  # 每批任务数=最大并发数
    total_batches = math.ceil(len(filtered_video_folders) / batch_size)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(filtered_video_folders), desc="Processing videos") as pbar:
            for batch_idx in range(total_batches):
                # 获取当前批次数据
                start = batch_idx * batch_size
                end = start + batch_size
                current_batch = filtered_video_folders[start:end]

                # 提交当前批次任务
                futures = []
                for folder in current_batch:
                    future = executor.submit(
                        process_single_video,
                        folder,
                        frames_folder,
                        examples,
                        segment_rate,
                        model,
                        processor,
                        device
                    )
                    future.add_done_callback(
                        lambda f: result_queue.put(f.result()) if f.result() else None
                    )
                    futures.append(future)

                # 严格等待当前批次全部完成
                wait(futures, return_when=ALL_COMPLETED)
                
                # 强制刷新缓冲区（可选）
                if result_queue.qsize() > 0:
                    result_queue.put("FLUSH_BUFFER")  # 需要与batch_writer配合

                # 更新进度
                pbar.update(len(current_batch))

    # 终止写入线程
    result_queue.put(None)
    writer_thread.join()
    print(f"Processing completed. Results saved to {output_json}")

# 修改批量写入函数
def _write_buffer(buffer, output_json):
    """线程安全的缓冲区写入实现"""
    with file_lock:  # 使用全局定义的 file_lock 保证线程安全
        # 读取现有数据（如果文件存在）
        existing_data = {}
        try:
            if os.path.exists(output_json):
                with open(output_json, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
        except json.JSONDecodeError:
            print(f"警告：{output_json} 文件损坏，将覆盖写入")
        
        # 合并数据，避免重复写入
        for key, value in buffer.items():
            if key not in existing_data or existing_data[key] is None:
                existing_data[key] = value
        
        # 写入更新后的数据
        try:
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"写入文件失败: {str(e)}")

def batch_writer(output_json):
    buffer = {}
    # 启动时读取已有数据，确保断点重启时数据不丢失
    existing_data = {}
    if os.path.exists(output_json):
        try:
            with open(output_json, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            print(f"断点重启：已读取 {len(existing_data)} 个已完成的视频数据")
        except json.JSONDecodeError:
            print(f"警告：{output_json} 文件损坏，将重新开始")
            existing_data = {}
    
    while True:
        item = result_queue.get()
        if item == "FLUSH_BUFFER":  # 强制刷新信号
            _write_buffer(buffer, output_json)
            buffer = {}
        elif item is None:  # 终止信号
            if buffer:
                _write_buffer(buffer, output_json)
            break
        else:
            video_folder, result = item
            buffer[video_folder] = result
            if len(buffer) >= WRITE_BATCH_SIZE:
                _write_buffer(buffer, output_json)
                buffer = {}

def process_single_video(video_folder, frames_folder, examples, segment_rate, model=None, processor=None, device=None):
    """优化后的单视频处理函数，支持GPT和Qwen选择"""
    video_path = os.path.join(frames_folder, video_folder)
    if not os.path.isfile(video_path):
        return None

    try:
        print(f"start processing {video_folder}")
        frames_data = load_frames_from_folder(video_path, segment_rate)
        result = process_with_qwen(frames_data, examples, model, processor, device)
        # 返回video_path去除扩展名后的文件名
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        return (video_id, result)
    except Exception as e:
        print(f"Error processing {video_folder}: {str(e)}")
        return (os.path.splitext(os.path.basename(video_path))[0], None)

def load_frames_from_folder(video_path: str, segment_rate) -> List[Dict]:
    """
    从视频文件抽帧并预处理（等分区间采样）
    返回格式: [{
        "frames": [base64_str1, base64_str2,...], 
        "start": start_ratio,  # 起始比例(0-1)
        "end": end_ratio       # 结束比例(0-1)
    }]
    """
    def encode_frame(frame):
        h, w = frame.shape[:2]
        if max(h, w) > 1024:
            scale = 1024 / max(h, w)
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            raise ValueError("Frame encoding failed")
        return base64.b64encode(buffer.tobytes()).decode('utf-8')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    print(f"video info: {fps}fps, {total_frames}frames, {duration}s")

    segments = []
    start = 0.0
    while start < 1.0:
        end = min(start + segment_rate, 1.0)
        start_idx = int(start * total_frames)
        end_idx = int(end * total_frames)
        available_frames = end_idx - start_idx
        if available_frames < 6:
            start = end
            continue
        sample_size = min(max(6, available_frames), 12)
        step = max(1, available_frames // sample_size)
        sampled_indices = list(range(start_idx, end_idx, step))[:sample_size]
        current_segment = {
            "frames": [],
            "start": round(start, 2),
            "end": round(end, 2)
        }
        for idx in sampled_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                encoded = encode_frame(frame)
                current_segment["frames"].append(encoded)
            except Exception as e:
                print(f"Skipped frame {idx}: {str(e)}")
        if len(current_segment["frames"]) >= 6:
            segments.append(current_segment)
        start = end
    cap.release()
    if not segments:
        raise ValueError("No valid segments found after sampling")
    return segments

def load_examples(json_path: str) -> List[str]:
    """从JSON文件中加载示例，从annotations键的列表中提取caption值"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 从annotations键中提取所有caption值
    examples = []
    for annotation in data.get('annotations', []):
        caption = annotation.get('caption', '').strip()
        if caption:
            examples.append(caption)
    
    return examples

def get_random_examples(examples: List[str], num_examples: int = 5) -> str:
    """随机选择指定数量的示例"""
    if len(examples) <= num_examples:
        return "\n".join(examples)
    return "\n".join(random.sample(examples, num_examples))

prompt = """Your task is to generate a concise caption that describes the key content of these continuous video frames as a whole. Avoid overly detailed descriptions and focus on the core actions, subjects, and events. Each caption should be a short, straightforward sentence, similar to how users would search for a video.
Here are examples of user search queries for video segments, which reflect the desired style and length of your output:
{examples}

Based on these examples, generate a caption for the provided video frames that adheres to the same simple, direct language and length. Do not include additional explanatory details or elaborate on the background elements."""

def process_with_qwen(frames_data: List[Dict], examples: List[str], model, processor, device: str):
    """处理单个视频片段"""
    video_data = []
    for seg in tqdm(frames_data, desc="Processing segment"):
        # 为每个消息随机选择示例
        random_examples = get_random_examples(examples)
        
        # 构建消息
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": [f"data:image/jpeg;base64,{frame}" for frame in seg["frames"]],
                },
                {"type": "text", "text": prompt.format(examples=random_examples)}
            ]
        }]
        # # print
        # print(prompt.format(examples=random_examples))
        
        # 预处理输入
        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(device)
            
            # 生成描述
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            caption = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
            
            temp = {
                "start_time": seg.get("start", "Unknown"),
                "end_time": seg.get("end", "Unknown"),
                "caption": caption
            }
            print(temp)
            video_data.append(temp)
        except Exception as e:
            print(f"Error processing segment: {str(e)}")
            video_data.append({
                "start_time": seg.get("start", "Unknown"),
                "end_time": seg.get("end", "Unknown"),
                "caption": "[Processing Error]"
            })
    
    return video_data

if __name__ == "__main__":
    # 配置参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='Qwen推理设备，仅在use_gpt=False时有效')
    parser.add_argument('--frames_folder', type=str, help='视频文件夹路径')
    parser.add_argument('--file_path', type=str, help='ID和示例文件路径')
    parser.add_argument('--segment_rate', type=float, help='片段占是视频总长的比例')
    parser.add_argument('--output_file', type=str, help='输出json的文件路径')
    args = parser.parse_args()
    device = args.device
    frames_folder = args.frames_folder
    file_path = args.file_path
    output_json = args.output_file
    segment_rate = args.segment_rate

    model_path = "/chenyaofo/chenchuanshen/model/Qwen/Qwen2.5-VL-7B-Instruct"
    
    MAX_WORKERS = 12
    
    ids = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 每行就是一个video id，排除空行
            video_id = line.strip()
            if video_id and video_id not in ids:
                ids.append(video_id)
                
    # 加载示例
    json_path = "/chenyaofo/chenchuanshen/TVRdatasets/MSRVTT/annotation/MSR_VTT.json"
    examples = load_examples(json_path)
    if not examples:
        raise ValueError("No examples found in JSON file")
    
    # 初始化模型
    model = None
    processor = None

   
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto"
    )#     attn_implementation="flash_attention_2",
    processor = AutoProcessor.from_pretrained(model_path)
    
    MAX_WORKERS = 1
    # 如果用户未指定output_json，则用Qwen默认路径，否则用用户指定
        
    
    print("开始处理")
    # 断点重启：读取已完成的video_id，且片段数>=5的才算完成
    finished_ids = set()
    if output_json and os.path.exists(output_json):
        try:
            with open(output_json, 'r', encoding='utf-8') as f:
                finished_data = json.load(f)
                for vid, segs in finished_data.items():
                    if isinstance(segs, list) and len(segs) >= 3:
                        finished_ids.add(vid)
        except Exception as e:
            print(f"读取 output_json 失败: {e}")

    filtered_video_folders = []
    # 所有视频都在一个文件夹中，不需要多级目录
    video_folder = frames_folder
    if not os.path.exists(video_folder):
        print(f"视频文件夹不存在: {video_folder}")
        exit(1)
    
    for fname in os.listdir(video_folder):
        # 检查是否为视频文件
        if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')):
            vid = os.path.splitext(fname)[0]
            if vid in ids and vid not in finished_ids:
                filtered_video_folders.append(fname)
    
    optimized_processing(filtered_video_folders, frames_folder, examples, output_json, segment_rate=segment_rate, max_workers=MAX_WORKERS,  model=model, processor=processor, device=device)
