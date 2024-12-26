"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""


import os
import time
os.environ['LD_PRELOAD']='/mnt/petrelfs/houzhi/lib/libtcmalloc_minimal.so.4.5.8'
import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union
import sys
current_path = os.getcwd()
sys.path.append(os.path.join(current_path, "../"))
sys.path.append(os.path.join(current_path))

from PIL import Image
import draccus
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from dataclasses import dataclass, field
import wandb
from openvla.prismatic.conf.vla import VLAConfig, VLARegistry
from openvla.prismatic.models.load import load_vla
from openvla.prismatic.vla.materialize import get_vla_dataset_and_collator
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset, DummyDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from utils.ddp_utils import init_distributed_mode, reduce_and_average
from prismatic.util import set_global_seed
# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["AWS_ACCESS_KEY_ID"] = "YBP12CWY1JKLDUPGKHMO"
os.environ["AWS_SECRET_ACCESS_KEY"] = "TGRNeq25hm3WA4X6DvHKbw4JGVGc7ZXpCwicvpiC"
# os.environ["AWS_REGION"] = "us-east-1"
os.environ["S3_ENDPOINT"] = "http://10.135.7.249:80"
os.environ["S3_USE_HTTPS"] = "0"
os.environ["S3_VERIFY_SSL"] = "0"

os.environ["AWS_ACCESS_KEY_ID"] = "H9HBOW256ACSZ0G62JGG"
os.environ["AWS_SECRET_ACCESS_KEY"] = "o3fiSkvVaNRsDiLMhqA1unUNYKzWfxnyGTErZLrW"
# os.environ["AWS_REGION"] = "us-east-1"
os.environ["S3_ENDPOINT"] = "http://p-ceph-norm-inside.pjlab.org.cn"
os.environ["S3_USE_HTTPS"] = "0"
os.environ["S3_VERIFY_SSL"] = "0"

import tensorflow_io as tfio
# # === Utilities ===
# # fmt: off
# def create_vision_transform(vla: nn.Module, input_size: int) -> Callable[[Image.Image], torch.Tensor]:
#     """Gets image transform for the vision encoder."""
#     data_cfg = timm.data.resolve_model_data_config(vla.vision_backbone)
#     data_cfg["input_size"] = (3, input_size, input_size)
#     return timm.data.create_transform(
#         input_size=data_cfg["input_size"],
#         interpolation=data_cfg["interpolation"],
#         mean=data_cfg["mean"],
#         std=data_cfg["std"],
#         crop_pct=1.0,           # Set to 1.0 to disable cropping
#         crop_mode="center",     # Default crop mode --> no-op when `crop_pct == 1.0`
#         is_training=False,      # Disable image_aug when loading transform; handled by RLDS dataloader
#     )
#
# # fmt: on
def _load_malloc_lib():
  if 'LD_PRELOAD' in os.environ:
    #malloc_lib = os.environ['LD_PRELOAD'].split('/')[-1]
    malloc_lib = os.environ['LD_PRELOAD']
  else:
    # Else find the standard libc library and return
    malloc_lib = ctypes.util.find_library('c')
  return ctypes.CDLL(malloc_lib)

_libmalloc = _load_malloc_lib()

def print_malloc_stats():
  if _libmalloc is not None:
    _libmalloc.malloc_stats()

@dataclass
class TrainConfig:
    # fmt: off

    # VLAConfig (`prismatic/conf/vla.py`); override with --vla.type `VLARegistry.<VLA>.vla_id`
    vla: VLAConfig = field(
        default_factory=VLAConfig.get_choice_class(VLARegistry.DINOSIGLIP_224PX_MX_OXE_MAGIC_SOUP_PLUS.vla_id)
    )

    # Directory Paths
    data_root_dir: Path = Path(                                     # Path to Open-X dataset directory
        "s3://openx"
    )
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints

    # Resume Run Parameters
    pretrained_checkpoint: Optional[Path] = None                    # Absolute Path to Checkpoint
    is_resume: bool = True                                          # Whether we are continuing a prior training run
                                                                    #   (only applicable given pretrained checkpoint)
    resume_step: Optional[int] = None                               # Global Step to Resume (should match checkpoint)
    resume_epoch: Optional[int] = None                              # Epoch to Resume (should match checkpoint)

    # Run Arguments
    run_id: Optional[str] = None                                    # Run ID for logging, Weights & Biases
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases
    save_interval: int = 2500                                       # Interval for saving checkpoints (in steps)
    image_aug: bool = False                                         # Whether to enable image augmentations
    seed: int = 7                                                   # Random seed (for reproducibility)

    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = Path(".hf_token")                  # Environment variable or Path to HF Token

    # Tracking Parameters
    trackers: Tuple[str, ...] = ("jsonl", "wandb")                  # Trackers to initialize (if W&B, add config!)
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"                          # Name of entity to log under
    data_name : str = 'debug'

    def __post_init__(self) -> None:
        """Lift optimization parameters from `self.vla` for ease of use =>> validate on `expected_world_size`"""
        self.epochs = self.vla.epochs
        self.max_steps = self.vla.max_steps
        self.global_batch_size = self.vla.global_batch_size
        self.per_device_batch_size = self.vla.per_device_batch_size

        self.learning_rate = self.vla.learning_rate
        self.weight_decay = self.vla.weight_decay
        self.max_grad_norm = self.vla.max_grad_norm
        self.lr_scheduler_type = self.vla.lr_scheduler_type
        self.warmup_ratio = self.vla.warmup_ratio

        self.train_strategy = self.vla.train_strategy

        # [Validate] Assert on `expected_world_size`

    # fmt: on


@draccus.wrap()
def train(cfg: TrainConfig) -> None:
    import argparse
    args = argparse.Namespace()
    print(args)
   # init_distributed_mode(args, cfg)
    # local_rank = int(os.environ["LOCAL_RANK"])
 
    # Configure Unique Run Name & Save Directory
    vla_id = cfg.vla.vla_id
    cfg.run_id = (
        f"{vla_id}+n{cfg.vla.expected_world_size // 8}+b{cfg.per_device_batch_size}+x{cfg.seed}"
        if cfg.run_id is None
        else cfg.run_id
    )
    if cfg.run_id_note is not None:
        cfg.run_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        cfg.run_id += "--image_aug"
    # hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    # Load VLA checkpoint (if resuming from training) or Base VLM otherwise (from `cfg.vla.base_vlm` ID or Path)
    #   =>> Note :: Verifies that all parameters are loaded in FP32 on load!

    # [Validate] Model should be in Full Precision!
    # for param in vlm.parameters():


    # [Explicit] Call to `freeze_backbones` here for clarity =>> will log exactly what is/is not frozen
    # vlm.freeze_backbones(stage)
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    @dataclass
    class DataAdapterForOpenx:        
        def __call__(self, rlds_batch ):
            loss_weight = torch.logical_not(torch.tensor(rlds_batch['action_past_goal']))
            dataset_name, action = rlds_batch["dataset_name"], torch.tensor(rlds_batch["action"])
            from PIL import Image

            lang = rlds_batch["task"]["language_instruction"].decode().strip()
            dataset_name = rlds_batch["dataset_name"].decode()
            print(rlds_batch["observation"]["image_primary"].shape) # 540 640
            pixel_values = torch.tensor(rlds_batch["observation"]["image_primary"])
            
            # Normalize 
            pixel_values = (pixel_values / 255. - torch.tensor(IMAGENET_DEFAULT_MEAN)) / torch.tensor(IMAGENET_DEFAULT_STD)
            
            pixel_values = pixel_values.permute(0, 3, 1, 2)
            return dict(pixel_values=pixel_values, action=action, dataset_name=dataset_name, language_instruction= lang, loss_weight=loss_weight)

    
    @dataclass
    class TestTrans:
        
        def __call__(self, rlds_batch ):
            return rlds_batch
    # import ipdb;ipdb.set_trace()
    # cls = RLDSDataset if not episodic else EpisodicRLDSDataset
    #dataset = RLDSDataset(cfg.data_root_dir, 'up_to_date_mixture', TestTrans(), resize_resolution=(224, 224), shuffle_buffer_size=100,
    #                        train=True, image_aug=cfg.image_aug, window_size= 2,future_action_window_size= 30,)
    #import ipdb;ipdb.set_trace()
    # 17720024


    sum_l = 0
    # for data_name in [ 'berkeley_fanuc_manipulation', 'utaustin_mutex', 'iamlab_cmu_pickup_insert_converted_externally_to_rlds', 
    #                   'dlr_edan_shared_control_converted_externally_to_rlds', 'austin_sirius_dataset_converted_externally_to_rlds', 'ucsd_kitchen_dataset_converted_externally_to_rlds',
    #                   'furniture_bench_dataset_converted_externally_to_rlds',
    #                    'cmu_stretch', 'robo_set', 'fmb','dobbe', 'fractal20220817_data', 'kuka','taco_play','jaco_play','berkeley_cable_routing','roboturk','viola','berkeley_autolab_ur5','toto',
    #                    'language_table','stanford_hydra_dataset_converted_externally_to_rlds',
    #                    'austin_buds_dataset_converted_externally_to_rlds','nyu_franka_play_dataset_converted_externally_to_rlds',]:

    for data_name in [cfg.data_name]:
        # 'cmu_stretch','fractal20220817_data', 'dobbe', 'droid', 'robo_set', 'viola',  

        # data_name = 'up_to_date_mixture'
        # data_name = 'fractal20220817_data'
        # data_name = 'kuka'
        # 224 224, 128 128
        # iamlab_cmu_pickup_insert_converted_externally_to_rlds : 360 640 
        # dlr_edan_shared_control_converted_externally_to_rlds 
        # austin_sirius_dataset_converted_externally_to_rlds 84 84
        # ucsd_kitchen_dataset_converted_externally_to_rlds 480 640 
        # furniture_bench_dataset_converted_externally_to_rlds 224 224
        # cmu_stretch 128 128
        # robo_set 240 424
        # fmb  256
        # dobbe 256
        # fractal20220817_data 256 320
        # kuka 512 640
        # taco_play 150 200
        # jaco 224 224
        # berkeley_cable_routing 128 128
        # roboturk 480 640
        # viola
        # berkeley_autolab_ur5 480 640
        # toto 480 640
        # language_table 360 640
        # stanford_hydra_dataset_converted_externally_to_rlds 240 320
        # austin_buds_dataset_converted_externally_to_rlds 128 128 
        # nyu_franka_play_dataset_converted_externally_to_rlds 128 128
        # dataset = RLDSDataset(cfg.data_root_dir, data_name, DataAdapterForOpenx(), resize_resolution=(224, 224), shuffle_buffer_size=2560,

        #                     train=True, image_aug=cfg.image_aug, window_size= 2, 
        #     future_action_window_size= 3, batch_size=32, batchfy=False)

        # Create a DataLoader with the initialized sampler, per-device-bsz, and collator
#         worker_init_fn = set_global_seed(7, get_worker_init_fn=True)
#         collator = PaddedCollatorForActionPrediction(
#             2048, 32000, padding_side='right',
#         )
#         def my_collactor_func(instances):
            
#             pixel_values = []
#             actions = []
#             dataset_name = []
#             language_instruct = []
#             loss_weight = []
#             for item in instances:
#                 pixel_values.append(item['pixel_values'])
#                 actions.append(item['action'])
#                 dataset_name.append(item['dataset_name'])
#                 language_instruct.append(item['language_instruction'])
#                 loss_weight.append(item['loss_weight'])
#                 # dict(pixel_values=pixel_values, action=action, dataset_name=dataset_name, language_instruction= lang, loss_weight=loss_weight)
#             return dict(pixel_values=torch.stack(pixel_values), action=torch.stack(actions), dataset_name=dataset_name, language_instruction= language_instruct, loss_weight=torch.stack(loss_weight))
#             pass
# #2048    32000
#         dataloader = DataLoader(
#             dataset,
#             batch_size=64,
#             sampler=None,
#             collate_fn=lambda x : x,
#             num_workers=0,
#             worker_init_fn=worker_init_fn,
#         )

        dataset = RLDSDataset(cfg.data_root_dir, data_name, TestTrans(), resize_resolution=(224, 224), shuffle_buffer_size=12800,
                            train=True, image_aug=cfg.image_aug, window_size= 2, 
                            future_action_window_size= 30, batch_size=64, batchfy=False, center_crop=True,)
        
        data_length = len(dataset)
        print('length:', len(dataset))
        e_time = time.time()
        # for i, item in enumerate(dataset):
        #     print(i, time.time() - e_time)
        #     e_time = time.time()
        #     pass
        # exit()
        for iii in range(1):
            #del dataset
            print(data_name)
            i = 0
            e_time = time.time()
            for i, item in enumerate(dataset):
                # import ipdb;ipdb.set_trace()
                # Image.fromarray(item['observation']['image_primary'][0][0]).save('./temp1/{}_{}_{}.png'.format(data_name, item['task']['language_instruction'][0], i))
                # if i % 100 == 0:
                #    print(i, item['action'][:1, 0, 0], item['action'][:1].max(0), item['action'][:1].min(0))
                # continue
                # for iii in range(len(item['observation']['image_primary'][0])):
                #    Image.fromarray(item['observation']['image_primary'][0][iii]).save('tmp_{}.png'.format(iii))
                
                # import ipdb;ipdb.set_trace()
                # loss_weight = torch.logical_not(torch.tensor(rlds_batch['action_past_goal']))
                # dict_keys(['observation', 'task', 'action', 'dataset_name', 'absolute_action_mask', 'action_past_goal'])
                # break
                #import ipdb;ipdb.set_trace()
                # i += 1
                # exit()
                #print(i, time.time() - e_time)
                if i % 100 == 0:
                    # import gc
                    # gc.collect()
                    #print_malloc_stats()
                    import psutil
                    used_mem = psutil.virtual_memory().used
                    print(used_mem / 1024 / 1024, 'mb', data_name, i, time.time() - e_time)
                e_time = time.time()
                # if i > (data_length // 128):
                #    break
                # continue
                
                # if item['action_past_goal'].sum() >= 1:
                # import ipdb;ipdb.set_trace()
                # print(item.keys(), item['observation'].keys())
                if i == 0 or i % 50 == 0:
                    # import ipdb;ipdb.set_trace()
                    # print(data_name)
                    # print(data_name, item['absolute_action_mask'])
                    # print(data_name, i, 'action_past_goal', item['action_past_goal'], item['task']['pad_mask_dict'], item['task']['language_instruction'])
                    # print(item['observation']['image_primary'].shape)
                    pass
                    # Image.fromarray(item['observation']['image_primary'][0][0]).save('./temp_center/{}_{}_{}.png'.format(item["dataset_name"][0].decode(), item['task']['language_instruction'][0][:20], i))
                # if i > 2000:
                #     break
            break
    # dataset = RLDSDataset(cfg.data_root_dir, cfg.vla.data_mix, TestTrans(), resize_resolution=(224, 224), shuffle_buffer_size=cfg.vla.shuffle_buffer_size, train=True, image_aug=cfg.image_aug,)
    print(sum_l, 'total')
    print('init dataset')
    # import ipdb;ipdb.set_trace()
    # Get VLA Dataset & Collato


if __name__ == "__main__":

    # SLURM_STEP_NODELIST = os.environ['SLURM_STEP_NODELIST']
    # import subprocess
    # output = subprocess.check_output("scontrol show hostname {} | head -n1".format(SLURM_STEP_NODELIST), shell=True)
    # os.environ['MASTER_ADDR'] = output.strip().decode('ascii')
#     node_list=$SLURM_STEP_NODELIST
# master_addr=$(scontrol show hostname ${node_list} | head -n1)
# export MASTER_ADDR=$master_addr
    train()
