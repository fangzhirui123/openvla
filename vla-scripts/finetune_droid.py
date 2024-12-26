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

import datetime
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
os.environ["MASTER_PORT"] = '22223'

import sys
import time

current_path = os.getcwd()
sys.path.append(os.path.join(current_path, "../"))
from openvla.prismatic.models.load import load_vla
from openvla.prismatic.vla.datasets.datasets import RLDSBatchTransform_lab#, RLDSBatchTransform_lab1
#from openvla.openvla_warp.datasets_finetune import DroidDataset_warp, LabDataset_warp

from torch.utils.tensorboard import SummaryWriter

from openvla_warp.datasets_finetune import CalvinDataset_warp



import draccus
import numpy as np
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

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from openvla.utils.ddp_utils import init_distributed_mode, reduce_and_average
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
#from openvla.calvin.close_loop_eval_openvla_calvin import close_loop_eval_calvin
from typing import Optional, Tuple, Union
# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")        # Path to Open-X dataset directory
    dataset_name: str = "calvin_task_ABC_D"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 16                                            # Fine-tuning batch size
    max_steps: int = 2000                                        # Max number of fine-tuning steps
    save_steps: int = 500                                          # Interval for checkpoint saving
    eval_steps: int = 5000
    learning_rate: float = 2e-5                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 256 #32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # Tracking Parameters
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"                          # Name of entity to log under

    warp_dataset_name: str = 'calvin'
    single_gpu_running: bool = False
    data_list: str = None
    tag: str = '' # exp_id
    euler_delta: int = 3
    remove_small_diff: int = 0
    pretrained_path: str = ''
    local_rank: int = 4 
    hf_token: Union[str, Path] = Path(".hf_token")  
    # fmt: on


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:

    if cfg.warp_dataset_name == 'calvin':
        print('use calvin data')
        vla_dataset = CalvinDataset_warp(data_dir='/mnt/petrelfs/share_data/zhangtianyi1/',
                 seq_len=1,
                 act_len=1, 
                 forward_n_max=25, 
                 mode='train',
                 subfolder='task_ABC_D',
                 use_data_augmentation=False,
                 task_num=10000,
                 use_play=False,
                 use_labeled=True,
                 wrap_grmg_data=1)
        
        #vla_dataset_eval = torch.utils.data.ConcatDataset([vla_dataset_eval, vla_dataset_eval, vla_dataset_eval, vla_dataset_eval])

    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    exp_root_dir = '/mnt/petrelfs/share_data/fangzhirui/tensorboard/embodied_test_2/vla_{}_{}_{}'.format(cfg.warp_dataset_name, cfg.tag, current_time)
    writer = SummaryWriter(os.path.join(exp_root_dir, 'tensorboard'))


    cfg.run_root_dir = Path(exp_root_dir)
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")
    import argparse
    args = argparse.Namespace()
    #print("os.environ",os.environ)
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
    # import ipdb;ipdb.set_trace()
    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    adapter_dir = run_dir
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained("openvla-7b", trust_remote_code=True)
    #print(type(processor))
    #<class 'transformers_modules.openvla-7b.processing_prismatic.PrismaticProcessor'>
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True
    )
    if len(cfg.pretrained_path) > 1:
        # use pretrained
        hf_token = Path(".hf_token").read_text().strip()
        vla = load_vla(cfg.pretrained_path, hf_token='', load_for_training=True)
        # vla.load()
        print('load model successfully')
    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)
    #for n, p in vla.named_parameters():
        #print(n, p.requires_grad)
    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if len(cfg.pretrained_path) > 1:
        vla_old = vla
        target_modules = [n for n, m in vla.named_modules() if isinstance(m, torch.nn.Linear)]
        lora_config = LoraConfig(r=cfg.lora_rank,lora_alpha=min(cfg.lora_rank, 16),lora_dropout=cfg.lora_dropout,target_modules=target_modules,init_lora_weights="gaussian",)
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()    
    elif cfg.use_lora:

        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # for n, p in vla.named_parameters():
    #     if p.requires_grad:
    #         print(n, )
    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    # vla = DDP(vla, device_ids=[DEVICE], find_unused_parameters=True, gradient_as_bucket_view=True)
    if cfg.single_gpu_running:
        vla = vla.cuda(local_rank)
    else:
        vla = torch.nn.parallel.DistributedDataParallel(vla.cuda(local_rank), device_ids=[local_rank], find_unused_parameters=True)

    # import ipdb;ipdb.set_trace()
    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    
        # if not p.requires_grad:
        #     print('not train:', n)
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # vla_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    # )
    # ---
    batch_transform = RLDSBatchTransform_lab(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )

    # return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name)
    #         dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
            # img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
            # lang = rlds_batch["task"]["language_instruction"].decode().lower()

    # vla_dataset_eval.set_batch_transform(batch_transform)
    vla_dataset.set_batch_transform(batch_transform)
    ii = 0
    # while True:
    #     for item in vla_dataset:
    #         item['img_orig'].save('img_{}.png'.format(ii))
    #         ii += 1
    #print("vla_dataset.__getitem__(0)11111",vla_dataset.__getitem__(0))
    if cfg.single_gpu_running:
        vla_dataset.__getitem__(0)
    # vla_dataset = RLDSDataset(
    #     cfg.data_root_dir,
    #     cfg.dataset_name,
    #     batch_transform,
    #     resize_resolution=tuple(vla.module.config.image_sizes),
    #     shuffle_buffer_size=cfg.shuffle_buffer_size,
    #     image_aug=cfg.image_aug,
    # )

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    # if local_rank == 0:
    #     save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(vla_dataset)
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        collate_fn=collator,
        num_workers=8,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

#    eval_sampler = torch.utils.data.distributed.DistributedSampler(vla_dataset_eval)
#    dataloader_eval = DataLoader(
#        vla_dataset_eval,
#        batch_size=cfg.batch_size,
#        sampler=eval_sampler,
#        collate_fn=collator,
#        num_workers=8,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
#    )
#
    # Initialize Logging =>> W&B
    # if local_rank == 0:
    #     wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    # Train!
    length_data = len(dataloader)
    epochs = 0

    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)


    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        step_i = 0
        while True:
            train_sampler.set_epoch(epochs)
            for step_idx, batch in enumerate(dataloader):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if type(batch["pixel_values"]) == dict:
                        batch["pixel_values"]['dino'] = batch["pixel_values"]['dino'].to(torch.bfloat16).to(device_id)
                        batch["pixel_values"]['siglip'] = batch["pixel_values"]['siglip'].to(torch.bfloat16).to(device_id)
                    else:
                        batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16).to(device_id)
                    #print("pixel_values",batch["pixel_values"].shape)
                    #print("vla1111111111",type(vla))
                    #print("input_ids",batch["input_ids"])
                    #print("labels",batch["labels"])
                    output: CausalLMOutputWithPast = vla(input_ids=batch["input_ids"].to(device_id),attention_mask=batch["attention_mask"].to(device_id), pixel_values=batch["pixel_values"],labels=batch["labels"],)
                    loss = output.loss


                normalized_loss = loss
                # Backward!
                normalized_loss.backward()

                # Compute Accuracy and L1 Loss for Logging
                #print("model.vision_backbone",vla.module.vision_backbone)
                #print("vla.module.vision_backbone.featurizer.patch_embed.num_patches",vla.module.vision_backbone.featurizer.patch_embed.num_patches)
                #   256
                action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
                #print("action_logits.shape",action_logits.shape)
                # action_logits.shape torch.Size([16, 33, 32064])
                action_preds = action_logits.argmax(dim=2)
                #print("action_preds",action_preds.shape)
                #print("action_preds",action_preds)
                #action_preds torch.Size([16, 35])
    #      action_preds tensor([[31872, 31862, 31930, 31865, 31883, 31861, 31880, 31888, 31844, 31824,
    #      31847, 31853, 31878, 31881, 31878, 31884, 31918, 31976, 31749, 31779,
    #      31859, 31920, 31883, 31920, 31897, 31866, 31880, 31902, 31744,     2],
    #     [31872, 31837, 31875, 31877, 31865, 31889, 31875, 31852, 31852, 31857,
    #      31865, 31845, 31849, 31845, 31857, 31872, 31884, 31837, 31876, 31779,
    #      31897, 31848, 31856, 31858, 31893, 31887, 31839, 31896, 31872,     2]],
    #    device='cuda:0')
                action_gt = batch["labels"][:, 1:].to(action_preds.device)
                #print("action_gt",action_gt.shape)
                #print("action_gt",action_gt)
                
    #     action_gt tensor([[ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
    #       -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
    #       -100,  -100, 31868, 31871, 31888, 31845, 31849, 31871, 31744,     2],
    #     [ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
    #       -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
    #       -100,  -100, 31871, 31931, 31908, 31807, 31847, 31886, 31744,     2]],
    #    device='cuda:0')
    #   [2,30]
                mask = action_gt > action_tokenizer.action_token_begin_idx
                # action_tokenizer.action_token_begin_idx 31743
                # Compute Accuracy
                correct_preds = (action_preds == action_gt) & mask
                action_accuracy = correct_preds.sum().float() / mask.sum().float()

                # Compute L1 Loss on Predicted (Continuous) Actions
                continuous_actions_pred = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                )
    #   continuous_actions_pred tensor([-0.0863, -0.3765, -0.1961,  0.0471, -0.0627, -0.2353,  0.9961,  0.1255,
    #      0.1098, -0.1647, -0.1176,  0.2588, -0.1882,  0.0000],
    #    dtype=torch.float64)
                if step_i > 199 and 'DEBUG' in os.environ:
                    import ipdb;ipdb.set_trace()
                
                continuous_actions_gt = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                )
    #   continuous_actions_gt tensor([ 0.0314,  0.0078, -0.1255,  0.2118,  0.1804,  0.0078,  0.9961,  0.0078,
    #     -0.4627, -0.2824,  0.5098,  0.1961, -0.1098,  0.9961],
    #    dtype=torch.float64)
                action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

                # Store recent train metrics
                recent_losses.append(loss.item())
                recent_action_accuracies.append(action_accuracy.item())
                recent_l1_losses.append(action_l1_loss.item())
                gradient_step_idx = step_idx // cfg.grad_accumulation_steps

                smoothened_loss = sum(recent_losses) / len(recent_losses)
                smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
                smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)

                # Push Metrics to W&B (every 10 steps)
                if rank == 0 and step_i % 10 == 0:
                    
                    if writer is not None:
                        writer.add_scalar("action_l1_loss", action_l1_loss, step_i + epochs* length_data)
                        writer.add_scalar("train_loss", loss, step_i + epochs* length_data)
                        writer.add_scalar("action_accuracy", action_accuracy, step_i + epochs* length_data)


                # Optimizer Step
                if (step_i + 1) % cfg.grad_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    progress.update()
                    progress.set_postfix({'action_l1_loss': action_l1_loss.item()})
                    progress.set_postfix({'loss': loss.item()})
                    progress.set_postfix({'action_accuracy': action_accuracy.item()})


                # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
                # print(step_i, step_i % cfg.save_steps, rank)
                if step_i > 0 and step_i % cfg.save_steps == 0:
                    if rank == 0:
                        print(f"Saving Model Checkpoint for Step {step_i}", flush=True)

                        # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
                        save_dir = str(adapter_dir if cfg.use_lora else run_dir)+f"_{step_i}"
                        print(save_dir, flush=True)
                        # Save Processor & Weights
                        processor.save_pretrained(run_dir)
                        vla.module.save_pretrained(save_dir)
                        
                        if cfg.use_lora:
                            base_vla = AutoModelForVision2Seq.from_pretrained(
                                cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
                            )
                            merged_vla = PeftModel.from_pretrained(base_vla, save_dir)
                            merged_vla = merged_vla.merge_and_unload()
                            merged_vla.save_pretrained(str(save_dir))
                        
                    # Block on Main Process Checkpointing
                    dist.barrier()
                    # Merge LoRA weights into model backbone for faster inference
                    #   =>> Note that merging is slow and can be done post-hoc to speed up training
                    if step_i == cfg.max_steps and cfg.use_lora:
                        base_vla = AutoModelForVision2Seq.from_pretrained(
                            cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
                        )
                        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir+f"_{step_i}")
                        merged_vla = merged_vla.merge_and_unload()
                        if rank == 0:
                            merged_vla.save_pretrained(str(run_dir))

                    # Block on Main Process Checkpointing
                    dist.barrier()
                
                # if step_i > 0 and step_i % cfg.eval_steps == 0:
                #     #try:
                #     result = close_loop_eval_calvin(model=cfg.vla_path, root_folder = "/mnt/petrelfs/share_data/fangzhirui/tensorboard/embodied1/", cfg=cfg)
                #     #except:
                #         #import traceback
                #         #traceback.print_exc()
                #     #progress.set_postfix({'eval_result': result.item()})
                #     if writer is not None:
                #         writer.add_scalar("eval_result", result, step_i)
                step_i += 1
            if progress.last_print_n >= cfg.max_steps:
                break
            epochs += 1

@torch.no_grad
def evaluate(dataloader_eval, vla, device_id, action_tokenizer, writer, total_step_id, rank):
    action_l1_loss_total = 0
    loss_total = 0
    acc_total = 0
    
    for step_idx, batch in enumerate(dataloader_eval):
        print('eval', step_idx)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                )
                loss = output.loss

            # Compute Accuracy and L1 Loss for Logging
            action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx

            # Compute Accuracy
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            # Compute L1 Loss on Predicted (Continuous) Actions
            continuous_actions_pred = torch.tensor(action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())).to(device_id)
            continuous_actions_gt = torch.tensor(action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())).to(device_id)
            action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

            action_l1_loss_total = action_l1_loss_total + reduce_and_average(action_l1_loss).item()
            acc_total += reduce_and_average(action_accuracy).item()
            loss_total += reduce_and_average(loss).item()

    loss_total = loss_total / (step_idx + 1)
    action_l1_loss_total = action_l1_loss_total/ (step_idx + 1)
    acc_total = acc_total / (step_idx + 1)

    print('eval', step_idx, rank, writer is not None)
    if rank == 0:
        
        if writer is not None:
            writer.add_scalar("eval_action_l1_loss", action_l1_loss_total, total_step_id)
            writer.add_scalar("eval_loss", loss_total, total_step_id)
            writer.add_scalar("eval_action_accuracy", acc_total, total_step_id)

if __name__ == "__main__":

    SLURM_STEP_NODELIST = os.environ['SLURM_STEP_NODELIST']
    import subprocess
    output = subprocess.check_output("scontrol show hostname {} | head -n1".format(SLURM_STEP_NODELIST), shell=True)
    os.environ['MASTER_ADDR'] = output.strip().decode('ascii')
#     node_list=$SLURM_STEP_NODELIST
# master_addr=$(scontrol show hostname ${node_list} | head -n1)
# export MASTER_ADDR=$master_addr
    finetune()
