from calvin.close_loop_eval_openvla_calvin_full import close_loop_eval_calvin
import torch
from finetune_droid import FinetuneConfig
import subprocess
import os
import torch.multiprocessing as mp

# cfg = FinetuneConfig
model_path = "/mnt/petrelfs/share_data/fangzhirui/full_finetune/calvin_task_ABC_D_mask--image_aug/checkpoint-57000/model.safetensors"
cfg = FinetuneConfig
root_folder = '/mnt/petrelfs/share_data/fangzhirui/full_finetune/57000'

result = close_loop_eval_calvin(model=model_path, root_folder = root_folder, cfg = cfg)



