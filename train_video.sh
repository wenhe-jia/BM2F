python train_net_video.py \
  --dist-url auto \
  --num-gpus 4 \
  --config-file configs/youtubevis_2021_mini/video_maskformer2_R50_bs16_8k_projSTpair.yaml \
  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.TEMPORAL_PAIRWISE_WEIGHT 0.5 \
  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.WARM_UP False \
  OUTPUT_DIR TempPair_Top2_FixIntervalFrame2_Weight05_NoWarmUp_2nd \
  SOLVER.CHECKPOINT_PERIOD 16000 \
  WANDB.ENABLED True \
  WANDB.ENTITY garvinxxx \
  WANDB.NAME TempPair_Top2_FixIntervalFrame2_Weight05_NoWarmUp_2nd

python train_net_video.py \
  --dist-url auto \
  --num-gpus 4 \
  --config-file configs/youtubevis_2021_mini/video_maskformer2_R50_bs16_8k_projSTpair.yaml \
  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.TEMPORAL_PAIRWISE_WEIGHT 0.5 \
  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.WARM_UP False \
  OUTPUT_DIR TempPair_Top2_FixIntervalFrame2_Weight05_NoWarmUp_3rd \
  SOLVER.CHECKPOINT_PERIOD 16000 \
  WANDB.ENABLED True \
  WANDB.ENTITY garvinxxx \
  WANDB.NAME TempPair_Top2_FixIntervalFrame2_Weight05_NoWarmUp_3rd

python train_net_video.py \
  --dist-url auto \
  --num-gpus 4 \
  --config-file configs/youtubevis_2021_mini/video_maskformer2_R50_bs16_8k_projSTpair.yaml \
  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.TEMPORAL_PAIRWISE_WEIGHT 0.5 \
  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.WARM_UP True \
  OUTPUT_DIR TempPair_Top2_FixIntervalFrame2_Weight05_WarmUp2k_1st \
  SOLVER.CHECKPOINT_PERIOD 16000 \
  WANDB.ENABLED True \
  WANDB.ENTITY garvinxxx \
  WANDB.NAME TempPair_Top2_FixIntervalFrame2_Weight05_WarmUp2k_1st

python train_net_video.py \
  --dist-url auto \
  --num-gpus 4 \
  --config-file configs/youtubevis_2021_mini/video_maskformer2_R50_bs16_8k_projSTpair.yaml \
  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.TEMPORAL_PAIRWISE_WEIGHT 0.5 \
  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.WARM_UP True \
  OUTPUT_DIR TempPair_Top2_FixIntervalFrame2_Weight05_WarmUp2k_2nd \
  SOLVER.CHECKPOINT_PERIOD 16000 \
  WANDB.ENABLED True \
  WANDB.ENTITY garvinxxx \
  WANDB.NAME TempPair_Top2_FixIntervalFrame2_Weight05_WarmUp2k_2nd

python train_net_video.py \
  --dist-url auto \
  --num-gpus 4 \
  --config-file configs/youtubevis_2021_mini/video_maskformer2_R50_bs16_8k_projSTpair.yaml \
  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.TEMPORAL_PAIRWISE_WEIGHT 0.5 \
  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.WARM_UP True \
  OUTPUT_DIR TempPair_Top2_FixIntervalFrame2_Weight05_WarmUp2k_3rd \
  SOLVER.CHECKPOINT_PERIOD 16000 \
  WANDB.ENABLED True \
  WANDB.ENTITY garvinxxx \
  WANDB.NAME TempPair_Top2_FixIntervalFrame2_Weight05_WarmUp2k_3rd