import os
import torch
import sys
sys.setrecursionlimit(100000)
root_path: str = os.path.abspath(os.path.dirname(__file__))
static_cache_path = os.path.join(root_path,"static")
video_cache_path: str = os.path.join(static_cache_path, "videos")
splits_cache_path = os.path.join(static_cache_path, "splits")

frame_cell: int = 12  # frame cell
resized_size: int = 240  # 整理成一个size
device = "cuda" if torch.cuda.is_available() else "cpu"


video_features = {}  # video向量存储字典
