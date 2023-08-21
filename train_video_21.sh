python train_net_video.py \
  --dist-url auto \
  --num-gpus 4 \
  --config-file configs/youtubevis_2021/video_maskformer2_R50_bs16_8k_proj.yaml \
  INPUT.SAMPLING_FRAME_NUM 3 \
  OUTPUT_DIR 21_Proj_loss_1st \
  SEED 54911559 \
  WANDB.ENABLED True \
  WANDB.ENTITY garvinxxx \
  WANDB.NAME 21_Proj_loss_1st

python train_net_video.py \
  --dist-url auto \
  --num-gpus 4 \
  --config-file configs/youtubevis_2021/video_maskformer2_R50_bs16_8k_proj.yaml \
  INPUT.SAMPLING_FRAME_NUM 3 \
  OUTPUT_DIR 21_Proj_loss_2nd \
  SEED 54911559 \
  WANDB.ENABLED True \
  WANDB.ENTITY garvinxxx \
  WANDB.NAME 21_Proj_loss_2nd

python train_net_video.py \
  --dist-url auto \
  --num-gpus 4 \
  --config-file configs/youtubevis_2021/video_maskformer2_R50_bs16_8k_proj.yaml \
  INPUT.SAMPLING_FRAME_NUM 3 \
  OUTPUT_DIR 21_Proj_loss_3rd \
  SEED 54911559 \
  WANDB.ENABLED True \
  WANDB.ENTITY garvinxxx \
  WANDB.NAME 21_Proj_loss_3rd