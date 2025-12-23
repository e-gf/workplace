import h5py
import numpy as np
import os
from tqdm import tqdm

def merge_hdf5_files(file1_path, file2_path, output_path):
    """
    合并两个HDF5文件
    
    Args:
        file1_path: 第一个HDF5文件路径
        file2_path: 第二个HDF5文件路径  
        output_path: 输出合并后的HDF5文件路径
    """
    
    print("="*60)
    print("HDF5文件合并工具")
    print("="*60)
    
    # 检查输入文件是否存在
    if not os.path.exists(file1_path):
        raise FileNotFoundError(f"文件不存在: {file1_path}")
    if not os.path.exists(file2_path):
        raise FileNotFoundError(f"文件不存在: {file2_path}")
    
    # 检查输出目录是否存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"文件1: {file1_path}")
    print(f"文件2: {file2_path}")
    print(f"输出文件: {output_path}")
    print()
    
    # 统计信息
    total_datasets = 0
    total_size_mb = 0
    duplicate_keys = []
    
    # 创建输出HDF5文件
    with h5py.File(output_path, 'w') as output_file:
        
        # 处理第一个文件
        print("正在处理第一个文件...")
        with h5py.File(file1_path, 'r') as file1:
            print(f"  文件1包含 {len(file1.keys())} 个数据集")
            
            for key in tqdm(file1.keys(), desc="复制文件1数据"):
                dataset = file1[key]
                output_file.create_dataset(key, data=dataset[:])
                total_datasets += 1
                total_size_mb += dataset.nbytes / (1024*1024)
        
        print()
        
        # 处理第二个文件
        print("正在处理第二个文件...")
        with h5py.File(file2_path, 'r') as file2:
            print(f"  文件2包含 {len(file2.keys())} 个数据集")
            
            for key in tqdm(file2.keys(), desc="复制文件2数据"):
                dataset = file2[key]
                
                # 检查是否有重复的key
                if key in output_file.keys():
                    duplicate_keys.append(key)
                    print(f"  警告: 发现重复的key '{key}'，跳过文件2中的该数据集")
                    continue
                
                output_file.create_dataset(key, data=dataset[:])
                total_datasets += 1
                total_size_mb += dataset.nbytes / (1024*1024)
    
    print()
    print("="*60)
    print("合并完成!")
    print("="*60)
    print(f"输出文件: {output_path}")
    print(f"总数据集数量: {total_datasets}")
    print(f"总文件大小: {total_size_mb:.2f} MB")
    
    if duplicate_keys:
        print(f"发现 {len(duplicate_keys)} 个重复的key:")
        for key in duplicate_keys[:5]:  # 只显示前5个
            print(f"  - {key}")
        if len(duplicate_keys) > 5:
            print(f"  ... 还有 {len(duplicate_keys) - 5} 个重复key")
    else:
        print("没有发现重复的key")
    
    # 验证合并结果
    print()
    print("验证合并结果...")
    verify_merge_result(file1_path, file2_path, output_path, duplicate_keys)

def verify_merge_result(file1_path, file2_path, output_path, duplicate_keys):
    """
    验证合并结果的正确性
    """
    print("-" * 40)
    print("验证结果:")
    print("-" * 40)
    
    with h5py.File(output_path, 'r') as output_file:
        with h5py.File(file1_path, 'r') as file1:
            with h5py.File(file2_path, 'r') as file2:
                
                # 统计各文件的数据集数量
                file1_count = len(file1.keys())
                file2_count = len(file2.keys())
                output_count = len(output_file.keys())
                expected_count = file1_count + file2_count - len(duplicate_keys)
                
                print(f"文件1数据集数量: {file1_count}")
                print(f"文件2数据集数量: {file2_count}")
                print(f"输出文件数据集数量: {output_count}")
                print(f"期望数据集数量: {expected_count}")
                
                if output_count == expected_count:
                    print("✓ 数据集数量验证通过")
                else:
                    print("✗ 数据集数量验证失败")
                
                # 验证文件1的所有数据都存在于输出文件中
                file1_verified = True
                for key in file1.keys():
                    if key not in output_file.keys():
                        print(f"✗ 文件1的key '{key}' 在输出文件中缺失")
                        file1_verified = False
                    else:
                        # 验证数据内容是否一致
                        if not np.array_equal(file1[key][:], output_file[key][:]):
                            print(f"✗ 文件1的key '{key}' 数据内容不一致")
                            file1_verified = False
                
                if file1_verified:
                    print("✓ 文件1数据验证通过")
                
                # 验证文件2的非重复数据都存在于输出文件中
                file2_verified = True
                for key in file2.keys():
                    if key in duplicate_keys:
                        continue  # 跳过重复的key
                    
                    if key not in output_file.keys():
                        print(f"✗ 文件2的key '{key}' 在输出文件中缺失")
                        file2_verified = False
                    else:
                        # 验证数据内容是否一致
                        if not np.array_equal(file2[key][:], output_file[key][:]):
                            print(f"✗ 文件2的key '{key}' 数据内容不一致")
                            file2_verified = False
                
                if file2_verified:
                    print("✓ 文件2数据验证通过")
                
                # 显示输出文件的基本信息
                print()
                print("输出文件信息:")
                print(f"  文件大小: {output_file.id.get_filesize() / (1024*1024):.2f} MB")
                print(f"  数据集总数: {len(output_file.keys())}")
                
                # 统计唯一的video数量（基于txt2hdf5.py的key格式）
                unique_videos = set()
                for key in output_file.keys():
                    video_id = key.split('#')[0]  # 提取video_id部分
                    unique_videos.add(video_id)
                print(f"  唯一video数量: {len(unique_videos)}")

def main():
    """
    主函数 - 示例用法
    """
    # 示例文件路径 - 请根据实际情况修改
    file1_path = "/chenyaofo/chenchuanshen/workspace/generate_caption/youcook2/extracted_data/youcook2_train_roberta.hdf5"
    file2_path = "/chenyaofo/chenchuanshen/workspace/generate_caption/youcook2/extracted_data/youcook2_val_roberta.hdf5"
    output_path = "/chenyaofo/chenchuanshen/workspace/generate_caption/youcook2/extracted_data/youcook2_query_roberta.hdf5"
    
    try:
        merge_hdf5_files(file1_path, file2_path, output_path)
    except Exception as e:
        print(f"合并过程中出现错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
