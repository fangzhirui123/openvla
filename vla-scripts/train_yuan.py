"""
train.py

Training script for Vision-Language-Action (VLA) Policies, built on top of pretrained VLMs, trained using mixtures of
the Open-X Embodiment dataset. Performs training in native PyTorch, using Fully-Sharded Data Parallel (FSDP) to run
distributed across GPUs (and nodes). By default, assumes that CUDA toolkit is >= 11.0 (to support BF16 mixed precision).

Notes & Prerequisites:
    - If you want to set a custom location for all HF / TIMM artifacts --> `export HF_HOME="<PATH>"` *before* running!
        => For example (add to end of .bashrc): `export HF_HOME="/mnt/fsx/skaramcheti/cache"`
    - If you want to suppress random Tensorflow logs --> `export TF_CPP_MIN_LOG_LEVEL=3`

Run with:
    - [Single Node One-GPU (Debug)] : torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train.py
    - [Single Node Multi-GPU (= $K)]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/train.py
"""

import json
import os
import sys
from torch.utils.tensorboard import SummaryWriter


current_path = os.getcwd()
sys.path.append(os.path.join(current_path, "../"))
sys.path.append('/mnt/petrelfs/houzhi/Code/embodied_foundation/openvla')
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union
from utils.ddp_utils import init_distributed_mode, reduce_and_average
os.environ["AWS_ACCESS_KEY_ID"] = "H9HBOW256ACSZ0G62JGG"
os.environ["AWS_SECRET_ACCESS_KEY"] = "o3fiSkvVaNRsDiLMhqA1unUNYKzWfxnyGTErZLrW"
os.environ["S3_ENDPOINT"] = "http://p-ceph-norm-inside.pjlab.org.cn"
os.environ["S3_USE_HTTPS"] = "0"
os.environ["S3_VERIFY_SSL"] = "0"

import tensorflow_io as tfio

import draccus
import torch
import torch.distributed as dist
import yaml

from prismatic.conf import VLAConfig, VLARegistry
from prismatic.models import load, load_vla
from prismatic.overwatch import initialize_overwatch
from prismatic.training import VLAMetrics, get_train_strategy
from prismatic.util import set_global_seed
from prismatic.vla import get_vla_dataset_and_collator
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


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
    trackers: Tuple[str, ...] = ("jsonl",)                  # Trackers to initialize (if W&B, add config!)
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"                          # Name of entity to log under

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
        # assert (
        #     self.vla.expected_world_size == overwatch.world_size()
        # ), f"Expected World Size = {self.vla.expected_world_size} but Found {overwatch.world_size()} GPUs!"

    # fmt: on


@draccus.wrap()
def train(cfg: TrainConfig) -> None:
    overwatch.info("OpenVLA Training :: Warming Up")
    import argparse
    args = argparse.Namespace()
    init_distributed_mode(args, cfg)
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])  

    # Note => Under `torchrun` initializing `overwatch` will automatically set up `torch.distributed`
    torch.cuda.set_device(device_id := local_rank)
    torch.cuda.empty_cache()
    print(rank, local_rank)

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

    cfg.vla.data_mix = 'robo_set'
    # Start =>> Build Directories and Set Randomness
    overwatch.info('"Do or do not; there is no try."', ctx_level=1)
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    os.makedirs(run_dir := (cfg.run_root_dir / cfg.run_id), exist_ok=True)
    os.makedirs(cfg.run_root_dir / cfg.run_id / "checkpoints", exist_ok=True)
    #writer = SummaryWriter(os.path.join(cfg.run_root_dir, cfg.run_id, 'tensorboard'))

    # Save Configuration =>> additionally save a JSON version for later HF Integration
    if overwatch.is_rank_zero():
        draccus.dump(cfg, open(run_dir / "config.yaml", "w"))
        with open(run_dir / "config.yaml", "r") as f_yaml, open(run_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    # Load VLA checkpoint (if resuming from training) or Base VLM otherwise (from `cfg.vla.base_vlm` ID or Path)
    #   =>> Note :: Verifies that all parameters are loaded in FP32 on load!
    overwatch.info(f"Loading Base VLM `{cfg.vla.base_vlm}` from ID/Path")
    if cfg.pretrained_checkpoint is not None and os.path.isdir(cfg.pretrained_checkpoint) and len(os.listdir(cfg.pretrained_checkpoint)) > 0:
        if os.path.isdir(cfg.pretrained_checkpoint) and len(os.listdir(cfg.pretrained_checkpoint)) > 0:
            ckpt_path = cfg.pretrained_checkpoint
            cfg.pretrained_checkpoint = os.path.join(ckpt_path, sorted(os.listdir(ckpt_path), key=lambda x: os.path.getmtime(os.path.join(ckpt_path, x)))[-1])
            cfg.pretrained_checkpoint = Path(cfg.pretrained_checkpoint)
            print('load', cfg.pretrained_checkpoint)
            cfg.resume_step = int(re.search("step-(.+?)-", cfg.pretrained_checkpoint.name).group(1))
            cfg.resume_epoch = int(re.search("epoch-(.+?)-", cfg.pretrained_checkpoint.name).group(1))
            pass
        # [Validate] Pretrained Checkpoint `step` and `epoch` should match `resume_step` and `resume_epoch`
        #   =>> Note :: We make developers pass in `resume_*` arguments as an extra sanity check!
        if cfg.is_resume:
            assert int(re.search("step-(.+?)-", cfg.pretrained_checkpoint.name).group(1)) == cfg.resume_step
            assert int(re.search("epoch-(.+?)-", cfg.pretrained_checkpoint.name).group(1)) == cfg.resume_epoch

        vlm = load_vla(cfg.pretrained_checkpoint, hf_token=hf_token, load_for_training=True)

    else:
        vlm = load(cfg.vla.base_vlm, hf_token=hf_token, load_for_training=True)

    print(vlm.vision_backbone.default_image_resolution, 'aaaa',  flush=True)
    # [Validate] Model should be in Full Precision!
    for param in vlm.parameters():
        assert param.dtype == torch.float32, f"Loaded VLM parameter not in full precision: {param}"

    # Determine training "stage" based on frozen vs unfrozen parameters --> supports different fine-tuning schemes!
    if not cfg.vla.freeze_vision_backbone and not cfg.vla.freeze_llm_backbone:
        stage = "vla-full-train"  # Full fine-tuning
    elif cfg.vla.freeze_vision_backbone and not cfg.vla.freeze_llm_backbone:
        stage = "vla-train"  # Frozen vision encoder
    elif not cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone:
        assert cfg.vla.unfreeze_last_llm_layer, "You should unfreeze at least the last layer of your LLM!"
        stage = "vla-sandwich-train"  # Fine-tuning vision encoder, projector, and LLM last layer
    elif cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone:
        assert cfg.vla.unfreeze_last_llm_layer, "Need to unfreeze at least last LLM layer to train!"
        stage = "vla-last-layer-train"  # Fine-tuning LLM last layer only
    else:
        raise ValueError(
            "Weight freezing configuration not supported. VLA config has the following parameters: "
            f"freeze_vision_backbone: {cfg.vla.freeze_vision_backbone}"
            f"freeze_llm_backbone: {cfg.vla.freeze_llm_backbone}"
            f"unfreeze_last_llm_layer: {cfg.vla.unfreeze_last_llm_layer}"
        )

    # [Explicit] Call to `freeze_backbones` here for clarity =>> will log exactly what is/is not frozen
    overwatch.info(f"Invoking `VLM.freeze_backbones()` for `{vla_id}` => Stage: `{stage}`")
    vlm.freeze_backbones(stage)
    # import ipdb;ipdb.set_trace()
#     ipdb> LlamaTokenizerFast(name_or_path='/mnt/petrelfs/share_data/zhangqinglong/Husky/work_dirs/Llama-2-7b-hf', vocab_size=32000, model_max_length=2048, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '<PAD>'}, clean_up_tokenization_spaces=False),  added_tokens_decoder={
#         0: AddedToken("<unk>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
#         1: AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
#         2: AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
#         32000: AddedToken("<PAD>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
# }

    # Print number of total/trainable model parameters
    num_params = sum(p.numel() for p in vlm.parameters())
    num_trainable_params = sum(p.numel() for p in vlm.parameters() if p.requires_grad)
    overwatch.info(
        f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
    )
    os.environ['http_proxy']=''
    os.environ['https_proxy']=''

    # Get VLA Dataset & Collator
    # cfg.per_device_batch_size = 2
    overwatch.info(f"Creating VLA Open-X Dataset with Mixture `{cfg.vla.data_mix}`")
    vla_dataset, action_tokenizer, collator = get_vla_dataset_and_collator(
        cfg.data_root_dir,
        cfg.vla.data_mix,
        image_transform=vlm.vision_backbone.get_image_transform(),
        tokenizer=vlm.llm_backbone.get_tokenizer(),
        prompt_builder_fn=vlm.llm_backbone.prompt_builder_fn,
        default_image_resolution=vlm.vision_backbone.default_image_resolution,
        shuffle_buffer_size=cfg.vla.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        window_size=1,
        future_action_window_size=0,
        batch_size = cfg.per_device_batch_size,
        gen_prompt_transform=True
    )
    print(len(vla_dataset))
    # Save dataset statistics for de-normalization at inference time
    if overwatch.is_rank_zero():
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Train Strategy
    overwatch.info(f"Initializing Train Strategy `{cfg.train_strategy}`")
    print('aaaa:====================', cfg.vla.enable_gradient_checkpointing, cfg.vla.enable_mixed_precision_training, flush=True)
    train_strategy = get_train_strategy(
        train_strategy=cfg.train_strategy,
        vlm=vlm,
        device_id=device_id,
        stage=stage,
        epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        global_batch_size=cfg.global_batch_size,
        per_device_batch_size=cfg.per_device_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        enable_gradient_checkpointing=cfg.vla.enable_gradient_checkpointing,
        enable_mixed_precision_training=cfg.vla.enable_mixed_precision_training,
        reduce_in_full_precision=cfg.vla.reduce_in_full_precision,
        worker_init_fn=worker_init_fn,
    )
    train_strategy.run_setup(run_dir=run_dir, n_train_examples=len(vla_dataset))

    # Create Metrics =>> Handles on the fly tracking, logging to specified trackers (e.g., JSONL, Weights & Biases)
    overwatch.info(f"Creating Metrics with Active Trackers => `{cfg.trackers}`")
    metrics = VLAMetrics(
        cfg.trackers,
        cfg.run_id,
        run_dir,
        draccus.encode(cfg),
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
        tf_writer=None,
        window_size=1,
        resume_step=cfg.resume_step,
        resume_epoch=cfg.resume_epoch,
    )

    # Run VLA Training
    overwatch.info("Starting VLA Training Loop")
    train_strategy.run_vla_training(
        vla_dataset,
        collator,
        action_tokenizer,
        metrics,
        save_interval=cfg.save_interval,
    )

    # Finalize
    overwatch.info("Done with Training =>> Finalizing Metrics")
    metrics.finalize()

    # And... we're done!
    overwatch.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    SLURM_STEP_NODELIST = os.environ['SLURM_STEP_NODELIST']
    import subprocess
    output = subprocess.check_output("scontrol show hostname {} | head -n1".format(SLURM_STEP_NODELIST), shell=True)
    os.environ['MASTER_ADDR'] = output.strip().decode('ascii')
    train()
