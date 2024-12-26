import argparse
import logging
import os
from pathlib import Path
from dataclasses import dataclass

from Datasets.utils import euler2rotm, rotm2euler
from calvin.evaluation.evaluate_policy import evaluate_policy
from utils.data_utils import get_pose_cam
#os.environ["MS2_ASSET_DIR"] = "/mnt/petrelfs/share_data/zhaochengyang/maniskill2/assets"
os.environ["MASTER_PORT"] = '12345'
import pickle
import sys
import time

import gymnasium as gym
import numpy as np
import sapien.core as sapien
import torch
import torch.nn as nn
import torchvision

from moviepy.editor import ImageSequenceClip
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor, AutoModelForVision2Seq, AutoProcessor
from transforms3d.quaternions import mat2quat, quat2mat
from moviepy.editor import ImageSequenceClip
from petrel_client.client import Client
from pytorch3d.transforms import (
                    Transform3d,
                    matrix_to_euler_angles,
                    matrix_to_quaternion,
                    matrix_to_rotation_6d,
                    quaternion_to_matrix,
                    euler_angles_to_matrix
                )
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from transformers.modeling_outputs import CausalLMOutputWithPast
from collections import defaultdict
import random
from utils.ddp_utils import init_distributed_mode

# param
#MAX_EPISODE_STEPS = 300
#TARGET_CONTROL_MODE = "pd_ee_delta_pose"  # param can be one of ['pd_ee_delta_pose', 'pd_ee_target_delta_pose']
#CAL_DELTA_METHOD = 2  # 0:direct 1:tf 2:model
#CAMERA_NAMES = ["hand_camera", "camera_1", "camera_2", "camera_3", "camera_4", "camera_5"]

#CAMERA_W = 224
#CAMERA_H = 224

# NATURAL_INSTRUCTIONS = {
#     "PickCube-v0": "pick up the red cube and move it to the green point",
#     "StackCube-v0": "stack the red cube on the green cube",
#     "PickSingleYCB-v0": "pick up the ",
#     # "PickSingleEGAD-v0": "Pick up an EGAD object and move it to a goal position",
#     "PegInsertionSide-v0": "insert the peg into the horizontal hole in the box",
#     # "PlugCharger-v0": "Plug a charger into a wall receptacle",
#     "AssemblingKits-v0": "insert the objects into the corresponding holes on the plate",
#     # "TurnFaucet-v0": "Turn on a faucet by rotating its handle",
#     # "PandaAvoidObstacles-v0": "Navigate the (Panda) robot arm through a region of dense obstacles and move the end-effector to a goal pose",
#     # "PickClutterYCB-v0": "Pick up an object from a clutter of 4-8 YCB objects",
# }
# CAMERA_POOL_FILE = "/mnt/petrelfs/share_data/zhangtianyi1/maniskill2/camera_pool_300k.npz"
# camera_pool = np.load(CAMERA_POOL_FILE)["cameras"]

from calvin_agent.models.calvin_base_model import CalvinBaseModel


SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_openvla_prompt(instruction: str) -> str:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    
# def get_openvla_prompt(instruction: str) -> str:
#         return f"What action should the robot take to {instruction.lower()}?"


class CustomModel1(CalvinBaseModel):
    def __init__(self, model, cfg):
        self.openvla_path = model
        self.cfg = cfg  #config
        # raise NotImplementedError
        
        if cfg.single_gpu_running:
            local_rank = rank = 0
            os.environ["LOCAL_RANK"]= '0'
        else:
            init_distributed_mode(args, cfg)
            local_rank = int(os.environ["LOCAL_RANK"])
            rank = int(os.environ["RANK"])  
        #print("rank, local_rank",rank, local_rank)
        DEVICE = "cuda:" + str(os.environ["LOCAL_RANK"]) if torch.cuda.is_available() else "cpu"  
        #print("DEVICE",DEVICE)
        # [Validate] Ensure GPU Available & Set Device / Distributed Context
        assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
        # distributed_state = PartialState()
        torch.cuda.set_device(DEVICE)
        torch.cuda.empty_cache()
        device_id = DEVICE
        #print("device_id",device_id)
        
        
        self.device = torch.device(device_id) if torch.cuda.is_available() else torch.device("cpu")

        # Load VLA Model using HF AutoClasses
        self.processor = AutoProcessor.from_pretrained(self.openvla_path, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.openvla_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)
        # if os.path.isdir(self.openvla_path):
        #     with open(Path(self.openvla_path) / "dataset_statistics.json", "r") as f:
        #         self.vla.norm_stats = json.load(f)
        
        
        
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
        self.prompt_builder_fn = PurePromptBuilder
        self.prompt_builder = self.prompt_builder_fn("openvla")
        self.base_tokenizer = self.processor.tokenizer
        self.image_transform=self.processor.image_processor.apply_transform
        
        
        
        

    def reset(self):
        """
        This is called
        """
        #self.vla.reset_observation()
        pass
        # raise NotImplementedError

    def step(self, obs, goal):
        """
        Args:
            obs: environment observations
            goal: embedded language goal
        Returns:
            action: predicted action
        """
        #print("cfg",self.cfg)
        # with torch.no_grad():
        #     prompt = get_openvla_prompt(instruction=goal)
        #     inputs = self.processor(prompt, Image.fromarray(obs['rgb_obs']['rgb_static']).convert("RGB")).to(self.device, dtype=torch.bfloat16)
        #     print("prompt",prompt,"obs",obs)
        #     action = self.vla.predict_action(**inputs, unnorm_key="calvin", do_sample=False)
        #     #print("action",action)
        #     #action [0.0105425 0.00201696 -0.01463591 -0.00842118 -0.01122888 -0.05814414 0.]
            
        with torch.no_grad():
            prompt = get_openvla_prompt(instruction=goal)
            input_ids = self.base_tokenizer(prompt, add_special_tokens=True).input_ids
            #print("input_ids",input_ids)
            img = Image.fromarray(obs['rgb_obs']['rgb_static'])#.convert("RGB")
            input_ids = torch.tensor(input_ids).to(self.device)#.to(torch.bfloat16).to(self.device)
            input_ids = input_ids.unsqueeze(0)
            #print("input_ids1",input_ids)
            pixel_values = self.image_transform(img).to(torch.bfloat16).to(self.device)
            pixel_values = pixel_values.unsqueeze(0)
            action = self.vla.predict_action(input_ids=input_ids, pixel_values=pixel_values, unnorm_key="calvin", non_autoregressive=True)
            #print("action",action)
        
        # Action mode: ee_rel_pose_local
        state = obs['robot_obs'] # (15,)
        xyz_state = state[:3]
        rpy_state = state[3:6]
        rotm_state = euler2rotm(rpy_state)
        rel_action = action
        xyz_action = rel_action[:3] / 50 # scale down by 50  
        rpy_action = rel_action[3:6] / 20 # scale down by 20
        gripper_action = rel_action[6]
        gripper_action = 1 if gripper_action>0 else -1 
        #print("gripper_action", gripper_action)
        rotm_action = euler2rotm(rpy_action)
        xyz_next_state = xyz_state + rotm_state @ xyz_action
        rotm_next_state = rotm_state @ rotm_action
        rpy_next_state = rotm2euler(rotm_next_state)
        action = np.zeros(7)
        action[:3] = (xyz_next_state - xyz_state) * 50  
        action[3:6] = (rpy_next_state - rpy_state) * 20
        action[-1] = gripper_action
        action = torch.from_numpy(action)[None,...].cpu().detach().numpy()
        # self.rollout_step_counter += 1
        #print("action1",action)
    
        return action






# class CustomModel1(CalvinBaseModel):
#     def __init__(self, model, cfg):
#         self.openvla_path = model
#         self.cfg = cfg  #config
#         # raise NotImplementedError
        
#         if cfg.single_gpu_running:
#             local_rank = rank = 0
#             os.environ["LOCAL_RANK"]= '0'
#         else:
#             init_distributed_mode(args, cfg)
#             local_rank = int(os.environ["LOCAL_RANK"])
#             rank = int(os.environ["RANK"])  
#         #print("rank, local_rank",rank, local_rank)
#         DEVICE = "cuda:" + str(os.environ["LOCAL_RANK"]) if torch.cuda.is_available() else "cpu"  
#         #print("DEVICE",DEVICE)
#         # [Validate] Ensure GPU Available & Set Device / Distributed Context
#         assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
#         # distributed_state = PartialState()
#         torch.cuda.set_device(DEVICE)
#         torch.cuda.empty_cache()
#         device_id = DEVICE
#         #print("device_id",device_id)
        
        
#         self.device = torch.device(device_id) if torch.cuda.is_available() else torch.device("cpu")

#         # Load VLA Model using HF AutoClasses
#         self.processor = AutoProcessor.from_pretrained(self.openvla_path, trust_remote_code=True)
#         # self.vla = AutoModelForVision2Seq.from_pretrained(
#         #     self.openvla_path,
#         #     attn_implementation="flash_attention_2",
#         #     torch_dtype=torch.bfloat16,
#         #     low_cpu_mem_usage=True,
#         #     trust_remote_code=True,
#         # ).to(self.device)
#         self.vla = AutoModelForVision2Seq.from_pretrained(
#             self.openvla_path,
#             torch_dtype=torch.bfloat16,
#             quantization_config=None,
#             low_cpu_mem_usage=True,
#             trust_remote_code=True,
#             local_files_only=True
#         ).to(self.device)
#         #self.vla = torch.nn.parallel.DistributedDataParallel(self.vla.cuda(local_rank), device_ids=[local_rank], find_unused_parameters=True)
#         # if os.path.isdir(self.openvla_path):
#         #     with open(Path(self.openvla_path) / "dataset_statistics.json", "r") as f:
#         #         self.vla.norm_stats = json.load(f)
        
        
#         self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
#         self.prompt_builder_fn = PurePromptBuilder
#         self.prompt_builder = self.prompt_builder_fn("openvla")
#         self.base_tokenizer = self.processor.tokenizer
#         self.image_transform=self.processor.image_processor.apply_transform
        

#     def reset(self):
#         """
#         This is called
#         """
#         #self.vla.reset_observation()
#         pass
#         # raise NotImplementedError

#     def step(self, obs, goal):
#         """
#         Args:
#             obs: environment observations
#             goal: embedded language goal
#         Returns:
#             action: predicted action
#         """
#         #print("cfg",self.cfg)
#         with torch.no_grad():
#             prompt = get_openvla_prompt(instruction=goal)
#             input_ids = self.base_tokenizer(prompt, add_special_tokens=True).input_ids
#             #print("input_ids",input_ids)
#             img = Image.fromarray(obs['rgb_obs']['rgb_static'])#.convert("RGB")
#             input_ids = torch.tensor(input_ids).to(self.device)#.to(torch.bfloat16).to(self.device)
#             input_ids = input_ids.unsqueeze(0)
#             #print("input_ids1",input_ids)
#             pixel_values = self.image_transform(img).to(torch.bfloat16).to(self.device)
#             pixel_values = pixel_values.unsqueeze(0)
#             #print("pixel_values",pixel_values.shape)
#             with torch.autocast("cuda", dtype=torch.bfloat16):
#                 output: CausalLMOutputWithPast = self.vla(input_ids=input_ids, pixel_values=pixel_values)
#             #print("output",output)
#             #print("self.vla.modules",self.vla.modules)
#             action_logits = output.logits[:, self.vla.vision_backbone.featurizer.patch_embed.num_patches : -1]
#             #print("self.vla.modules",self.vla.modules)
#                 #print("action_logits.shape",action_logits.shape)
#                 # action_logits.shape torch.Size([16, 33, 32064])
#             action_preds = action_logits.argmax(dim=2)
#             #inputs = self.processor(prompt, Image.fromarray(obs['rgb_obs']['rgb_static']).convert("RGB")).to(self.device, dtype=torch.bfloat16)
#             #print("prompt",prompt,"obs",obs)
#             #action = self.vla.predict_action(**inputs, unnorm_key="calvin", do_sample=False)
#             #print("action_preds",action_preds)
#             action_preds = action_preds[:,-8:-1].view(-1)#.cpu().numpy()
#             action = torch.tensor(
#                     self.action_tokenizer.decode_token_ids_to_actions(action_preds.cpu().numpy())
#                 ).cpu().numpy()
#             #print("action",action)
#             # index_of_2 = torch.nonzero(action_preds == 2).squeeze(1).item()
#             # start_index = max(0, index_of_2 - 7)
#             # generated_ids = action_preds.logits.argmax(-1)[start_index:min(index_of_2,7)]
#             # print("generated_ids",generated_ids)
#             #action [0.0105425 0.00201696 -0.01463591 -0.00842118 -0.01122888 -0.05814414 0.]
        
#         # Action mode: ee_rel_pose_local
#         state = obs['robot_obs'] # (15,)
#         xyz_state = state[:3]
#         rpy_state = state[3:6]
#         rotm_state = euler2rotm(rpy_state)
#         rel_action = action
#         xyz_action = rel_action[:3] / 50 # scale down by 50  
#         rpy_action = rel_action[3:6] / 20 # scale down by 20
#         gripper_action = rel_action[6]
#         gripper_action = 1 if gripper_action>0 else -1 
#         #print("gripper_action", gripper_action)
#         rotm_action = euler2rotm(rpy_action)
#         xyz_next_state = xyz_state + rotm_state @ xyz_action
#         rotm_next_state = rotm_state @ rotm_action
#         rpy_next_state = rotm2euler(rotm_next_state)
#         action = np.zeros(7)
#         action[:3] = (xyz_next_state - xyz_state) * 50  
#         action[3:6] = (rpy_next_state - rpy_state) * 20
#         action[-1] = gripper_action
#         action = torch.from_numpy(action)[None,...].cpu().detach().numpy()
#         # self.rollout_step_counter += 1
#         #print("action1",action)
    
#         return action


@dataclass
class Args:
    # fmt: off
    rank: int = 8                            
args = Args
def close_loop_eval_calvin(
    #="rgbd",
    #reward_mode=None,
    #control_mode=TARGET_CONTROL_MODE,
    #render_mode="cameras",
    #record_dir=None,
    #render_goal_point=True,
    test_episodes_num=100,
    model=None,
    #eval_data_list=None,
    args=args,
    #rand_seed=0,
    #json_repo="/mnt/petrelfs/share_data/zhaochengyang/maniskill2/demos/v0/rigid_body/",
    #camera_coord=True,
    #stride=1,
    root_folder = None,
    #data_root_path = None,
    cfg=None,
    #eval_dataset=None,
):

    print('begin ....', root_folder)


    from calvin_env.envs.play_table_env import get_env
    from calvin_env.utils.utils import EglDeviceNotFoundError, get_egl_device_id
    def make_env(dataset_path):
        val_folder = Path(dataset_path) / "validation"
        env = get_env(val_folder, show_gui=False)

        # if "EGL_VISIBLE_DEVICES" in os.environ:
        #     logger.warning("Environment variable EGL_VISIBLE_DEVICES is already set. Is this intended?")
        device = torch.device('cuda', int(os.environ["LOCAL_RANK"]))
        cuda_id = device.index if device.type == "cuda" else 0
        try:
            egl_id = get_egl_device_id(cuda_id)
        except EglDeviceNotFoundError:
            # logger.warning(
            #     "Couldn't find correct EGL device. Setting EGL_VISIBLE_DEVICE=0. "
            #     "When using DDP with many GPUs this can lead to OOM errors. "
            #     "Did you install PyBullet correctly? Please refer to calvin env README"
            # )
            egl_id = 0
        os.environ["EGL_VISIBLE_DEVICES"] = str(egl_id)
        # logger.info(f"EGL_DEVICE_ID {egl_id} <==> CUDA_DEVICE_ID {cuda_id}")
        # insert your own env wrapper
        # env = Wrapper(env)
        return env
    model = CustomModel1(model,cfg)
    #model = CustomModel1(model)
    
    env = make_env('/mnt/petrelfs/share_data/zhangtianyi1/task_ABC_D/')
    
    os.path.join(root_folder, 'tmp_'+str(args.rank))
    
    
    f1 = open('debug_calvin'+str(os.environ['RANK'])+'.out', 'a')
    f1.write(str(os.environ['RANK']) +','+str(args.rank)+'check\n')
    f1.write(os.path.join(root_folder, 'tmp_'+str(args.rank))+'\n')
    f1.flush()

    f1.close()
    
    
    results = evaluate_policy(model, env, epoch=0., eval_log_dir=os.path.join(root_folder, 'tmp_'+str(args.rank)), debug=0, rank=args.rank, each_length=test_episodes_num)

    print("eval_result", results)
    
    return results  #0, None, 0






