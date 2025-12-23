from huggingface_hub import snapshot_download
# 指定模型名称和目标路径
model_name = "BAAI/bge-large-zh-v1.5"
local_dir = "/chenyaofo/chenchuanshen/workspace/sentence-bert/model/bge-large-zh-v1.5"  # 修改为你自己的本地路径
# 下载模型（只会下载一次，后续不会重复下载）
snapshot_download(repo_id=model_name, local_dir=local_dir, local_dir_use_symlinks=False)