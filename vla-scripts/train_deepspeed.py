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





current_path = os.getcwd()
sys.path.append(os.path.join(current_path, "../"))
sys.path.append('/mnt/petrelfs/houzhi/Code/embodied_foundation/openvla')
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

os.environ["AWS_ACCESS_KEY_ID"] = "H9HBOW256ACSZ0G62JGG"
os.environ["AWS_SECRET_ACCESS_KEY"] = "o3fiSkvVaNRsDiLMhqA1unUNYKzWfxnyGTErZLrW"
os.environ["S3_ENDPOINT"] = "http://p-ceph-norm-inside.pjlab.org.cn"
os.environ["S3_USE_HTTPS"] = "0"
os.environ["S3_VERIFY_SSL"] = "0"

sys.path.append('/mnt/petrelfs/houzhi/Code/embodied_foundation/openvla/InternVL/internvl_chat/')
sys.path.append('/mnt/petrelfs/houzhi/Code/embodied_foundation/openvla/')
sys.path.append('/mnt/petrelfs/houzhi/Code/embodied_foundation/')

os.environ["AWS_ACCESS_KEY_ID"] = "H9HBOW256ACSZ0G62JGG"
os.environ["AWS_SECRET_ACCESS_KEY"] = "o3fiSkvVaNRsDiLMhqA1unUNYKzWfxnyGTErZLrW"
os.environ["S3_ENDPOINT"] = "http://p-ceph-norm-inside.pjlab.org.cn"
os.environ["S3_USE_HTTPS"] = "0"
os.environ["S3_VERIFY_SSL"] = "0"

import torch.distributed as dist

import tensorflow_io as tfio
from dist_utils import init_dist

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
from prismatic.util.data_utils import IGNORE_INDEX
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
        # default_factory=VLAConfig.get_choice_class(VLARegistry.DINOSIGLIP_224PX_MX_OXE_MAGIC_SOUP_PLUS.vla_id)
        default_factory=VLAConfig.get_choice_class(VLARegistry.Exp_FreezeVIT_DinoSigLIP_224px_Bridge.vla_id)

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
    gen_prompt_transform: int = 1

    # Run Arguments
    run_id: Optional[str] = None                                    # Run ID for logging, Weights & Biases
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases
    save_interval: int = 20000                                       # Interval for saving checkpoints (in steps)
    image_aug: bool = False                                         # Whether to enable image augmentations
    seed: int = 7                                                   # Random seed (for reproducibility)
    two_inps: bool = True
    freeze_mast3r: bool = False
    single_camera: int = 1

    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = Path(".hf_token")                  # Environment variable or Path to HF Token

    # Tracking Parameters
    trackers: Tuple[str, ...] = ("jsonl",)                  # Trackers to initialize (if W&B, add config!)
    # wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    # wandb_entity: str = "stanford-voltron"                          # Name of entity to log under

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
def main(cfg: TrainConfig) -> None:

# def main():

    # overwatch.info("OpenVLA Training :: Warming Up")
    # init_distributed_mode(args, cfg)
    # deepspeed.init_distributed()
    launcher = 'slurm'
    # import argparse
    # args = argparse.Namespace()
    print('begin init', flush=True)
    init_dist(launcher=launcher, backend='nccl')
    print('init end', flush=True)
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])  

    print('WANDB_API_KEY' in os.environ, 'WANDB_PROJECT' in os.environ)

    # Note => Under `torchrun` initializing `overwatch` will automatically set up `torch.distributed`
    # torch.cuda.set_device(device_id := local_rank)
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

    # for n,p in vlm.named_parameters():
    #     print(n, p.mean())

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
    vlm.freeze_backbones(stage, freeze_mast3r=cfg.freeze_mast3r)
    # for n,p in vlm.named_parameters():
    #     if p.requires_grad:
    #         print(n, 'no freeze')
    #     else:
    #         print(n, 'freeze')
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
    # | >> [*] # Parameters (in millions):      train.py:236
                        #   8236.561 Total, 8236.561 Trainable
    # import ipdb;ipdb.set_trace()
    # os.environ['http_proxy']=''
    # os.environ['https_proxy']=''

    # Get VLA Dataset & Collator
    # cfg.per_device_batch_size = 2
    cfg.per_device_batch_size = 64
    overwatch.info(f"Creating VLA Open-X Dataset with Mixture `{cfg.vla.data_mix}`")
    tokenizer = vlm.llm_backbone.get_tokenizer()
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
        gen_prompt_transform=cfg.gen_prompt_transform,
        two_inps = cfg.two_inps,
        load_camera_views=("primary", ) if cfg.single_camera else ("primary", "secondary",)
    )
    print(len(vla_dataset))
    # Save dataset statistics for de-normalization at inference time
    if overwatch.is_rank_zero():
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    from transformers import (
        Trainer,
        TrainingArguments,
    )

    # print(training_args)
    metrics = VLAMetrics(
        cfg.trackers,
        cfg.run_id,
        run_dir,
        draccus.encode(cfg),
        tf_writer=None,
        window_size=1,
        resume_step=cfg.resume_step,
        resume_epoch=cfg.resume_epoch,
    )    

    class MyTrainer(Trainer):

        def get_train_dataloader(self):
            return self.train_dataset    


        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs['labels']
            dataset_names = inputs.pop('dataset_names')
            # print(model.vision_backbone.dino_featurizer.blocks[0].norm1.weight.dtype)
            # print(model.vision_backbone.dino_featurizer.blocks[1].norm1.weight.dtype)
            # print(model.vision_backbone.dino_featurizer.blocks[2].norm1.weight.dtype)
            # import ipdb;ipdb.set_trace()
            # model.llm_backbone.to(torch.float32)
            if 'DEBUG' in os.environ:
                print(model.vision_model)
            # for layer in model.modules():
            #     if isinstance(layer, torch.nn.LayerNorm) or isinstance(layer, LlamaRMSNorm):
            #         print(layer.weight.dtype, layer, 'fp32')
            #         layer.float()  # 将 LayerNorm 转为 FP32
            # for n, p in model.named_parameters():
            #     print(n, p.dtype)
            # print(model.vision_backbone.dino_featurizer.blocks[0].norm1.weight)
            # import ipdb;ipdb.set_trace()
            # print(dist.get_rank(), dataset_names, self.state.global_step)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=True
                    ):
                loss, output = super().compute_loss(model, inputs, return_outputs=True)
            # import ipdb;ipdb.set_trace()
            logits = output.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            from prismatic.util.data_utils import IGNORE_INDEX
            mask = torch.logical_and(shift_labels != IGNORE_INDEX, shift_labels != tokenizer.pad_token_id)


            action_preds = output.logits[:,  : -1].argmax(dim=2)
            action_gt = labels[:, 1:].to(action_preds.device)
            action_preds = action_preds[:, -action_gt.shape[1]:]
            if cfg.gen_prompt_transform != 5:
                # import ipdb;ipdb.set_trace()
                mask = torch.logical_and(action_gt < action_tokenizer.action_token_begin_idx + (action_tokenizer.n_bins + 1) + 1, action_gt > action_tokenizer.action_token_begin_idx)

            if 'DEBUG' in os.environ:
                import ipdb;ipdb.set_trace()
            # Compute Accuracy

            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()
            action_accuracy_a = action_accuracy



            if cfg.gen_prompt_transform == 5 and self.state.global_step % self.state.logging_steps == 0:
                self.log({'action_accuracy': action_accuracy.item(), 'loss': loss.item()})
            if dist.get_rank() == 0:
                metrics.commit(loss=loss, action_accuracy=action_accuracy)
                # Compute L1 Loss on Predicted (Continuous) Actions
                try:
                    continuous_actions_pred = torch.tensor(
                        action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                    )
                    # import ipdb;ipdb.set_trace()
                    continuous_actions_gt = torch.tensor(
                        action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                    )
                    action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

                    # mask = labels > action_tokenizer.action_token_begin_idx
                    if self.state.global_step % self.state.logging_steps == 0:
                        self.log({'action_accuracy': action_accuracy.item(), 'action_accuracy_a': action_accuracy_a.item(), 'action_l1_loss': action_l1_loss.item()})
                        # VLAMetrics

                    metrics.commit(action_accuracy=action_accuracy, action_accuracy_a = action_accuracy_a,  l1_loss=action_l1_loss, update_step_time=True)

                    # Compute metrics per dataset --> only on rank_zero since we don't log them on other workers anyways
                    if dist.get_rank() == 0:
                        datasets = set(dataset_names)
                        if len(datasets) > 1:
                            for ds in datasets:
                                ds_mask = torch.tensor([elem == ds for elem in dataset_names])
                                action_accuracy_ds = correct_preds[ds_mask].sum().float() / mask[ds_mask].sum().float()
                                continuous_actions_pred_ds = torch.tensor(
                                    action_tokenizer.decode_token_ids_to_actions(
                                        action_preds[ds_mask][mask[ds_mask]].cpu().numpy()
                                    )
                                )
                                continuous_actions_gt_ds = torch.tensor(
                                    action_tokenizer.decode_token_ids_to_actions(
                                        action_gt[ds_mask][mask[ds_mask]].cpu().numpy()
                                    )
                                )
                                action_l1_loss_ds = torch.nn.functional.l1_loss(
                                    continuous_actions_pred_ds, continuous_actions_gt_ds
                                )
                                metrics.commit_for_dataset(
                                    dataset_name=ds.decode(), action_accuracy=action_accuracy_ds, l1_loss=action_l1_loss_ds
                                )
                    # Compute epoch value using number of completed gradient steps
                    epoch = (metrics.global_step + 1) // (len(vla_dataset) // (training_args.n_gpu*training_args.per_device_train_batch_size))

                    # Push Metrics
                    metrics.commit(global_step=metrics.global_step + 1, epoch=epoch, lr=self.lr_scheduler.get_last_lr()[0])
                    status = metrics.push()

                except:
                    import traceback
                    traceback.print_exc()
                    try:
                        print('detokenized str', self.tokenizer.decode(list(action_gt[mask].cpu().numpy())))
                    except:
                        pass



            return loss

    from transformers import TrainerCallback
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    training_args1 = TrainingArguments(
        output_dir=run_dir,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        logging_dir=run_dir,
        logging_steps=10,
        save_total_limit=1,
        save_steps=200000,
        bf16=True,
        weight_decay=0.0,
        warmup_ratio=0.0,
        learning_rate=2e-5,
        lr_scheduler_type = "constant",
        gradient_checkpointing=False,
        ignore_data_skip=True,
        save_only_model=True,
        deepspeed="zero_stage3_config.json"  # Specify your DeepSpeed config file
    )

    # Initialize the Trainer
    trainer = MyTrainer(
        model=vlm,
        args=training_args1,
        train_dataset=vla_dataset,
        eval_dataset=None, 
        # callbacks=[PreTrainCallback()],
    )
    # vlm.vision_backbone.to(dtype=torch.bfloat16)
    # vlm.llm_backbone.to(dtype=torch.float32)
    # model.vision_backbone.dino_featurizer.blocks[0].norm1.weight

    # Start training
    # import ipdb;ipdb.set_trace()
    print(vlm.vision_backbone.dino_featurizer.blocks[0].norm1.weight)
    trainer.train()


if __name__ == "__main__":
    main()