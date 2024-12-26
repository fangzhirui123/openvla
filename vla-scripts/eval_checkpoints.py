from calvin.close_loop_eval_openvla_calvin import close_loop_eval_calvin
import torch
import os
from finetune_droid import FinetuneConfig

cfg = FinetuneConfig




result = close_loop_eval_calvin(model="/mnt/petrelfs/share_data/fangzhirui/tensorboard/embodied_test_2/vla_calvin_t1_o1_lab_907_1_920_euler5_calvin_20241223195720/openvla-7b+calvin_task_ABC_D+b16+lr-0.0002+lora-r256+dropout-0.0_10000", root_folder = "/mnt/petrelfs/share_data/fangzhirui/tensorboard/embodied_test_2/vla_calvin_t1_o1_lab_907_1_920_euler5_calvin_20241223195720/40000", cfg = cfg)