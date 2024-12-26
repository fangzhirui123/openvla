/mnt/petrelfs/fangzhirui/anaconda3/envs/openvla_xin/bin/deepspeed --num_nodes 1 --num_gpus 8 vla-scripts/train.py \
  --pretrained_checkpoint /mnt/petrelfs/fangzhirui/openvla/openvla_7b_full/checkpoints/ \
  --data_root_dir /mnt/petrelfs/share_data/zhangtianyi1/ \
  --run_root_dir /mnt/petrelfs/share_data/fangzhirui/full_finetune/ \
  --run_id calvin_task_ABC_D_mask_print \
  --image_aug True \
  --save_interval 3000 \
  --is_resume False