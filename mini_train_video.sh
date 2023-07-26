# baseline, frame 3
#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2021_mini/video_maskformer2_R50_bs16_8k_projpair.yaml \
#  INPUT.SAMPLING_FRAME_NUM 3 \
#  OUTPUT_DIR Baseline_FixIntervalFrame2ShortRange_1st \
#  SOLVER.CHECKPOINT_PERIOD 16000 \
#  WANDB.ENABLED True \
#  WANDB.ENTITY garvinxxx \
#  WANDB.NAME Mini_Baseline_FixIntervalFrame2ShortRange_1st

# MATA, frame 2
#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2021_mini/video_maskformer2_R50_bs16_8k_projSTpair.yaml \
#  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.TEMPORAL_PAIRWISE_WEIGHT 0.5 \
#  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.WARM_UP True \
#  OUTPUT_DIR TempPair_Top2_FixIntervalFrame2_Weight05_WarmUp2k_1st \
#  SOLVER.CHECKPOINT_PERIOD 16000 \
#  WANDB.ENABLED True \
#  WANDB.ENTITY garvinxxx \
#  WANDB.NAME Mini_TempPair_Top2_FixIntervalFrame2_Weight05_WarmUp2k_1st

# MATA, frame 3
#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2021_mini/video_maskformer2_R50_bs16_8k_projSTpair.yaml \
#  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.TEMPORAL_TOPK 1 \
#  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.TEMPORAL_PAIRWISE_WEIGHT 0.5 \
#  INPUT.SAMPLING_FRAME_NUM 3 \
#  OUTPUT_DIR Mini_TempPair_Top1_Frame3_Weight05_NoWarmUp_2nd \
#  SOLVER.CHECKPOINT_PERIOD 16000 \
#  WANDB.ENABLED True \
#  WANDB.ENTITY garvinxxx \
#  WANDB.NAME Mini_TempPair_Top1_Frame3_Weight05_NoWarmUp_2nd

python train_net_video.py \
  --dist-url auto \
  --num-gpus 4 \
  --config-file configs/youtubevis_2021_mini/video_maskformer2_R50_bs16_8k_projSTpair.yaml \
  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.TEMPORAL_TOPK 10 \
  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.TEMPORAL_COLOR_THRESH 0.05 \
  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.TEMPORAL_PAIRWISE_WEIGHT 1.0 \
  INPUT.SAMPLING_FRAME_NUM 3 \
  OUTPUT_DIR TempPair_Top10Color005_Frame3_Weight10_NoWarmUp \
  SOLVER.CHECKPOINT_PERIOD 16000 \
  WANDB.ENABLED True \
  WANDB.ENTITY garvinxxx \
  WANDB.NAME TempPair_Top10Color005_Frame3_Weight10_NoWarmUp

#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2021_mini/video_maskformer2_R50_bs16_8k_projSTpair.yaml \
#  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.TEMPORAL_TOPK 5 \
#  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.TEMPORAL_PAIRWISE_WEIGHT 0.5 \
#  INPUT.SAMPLING_FRAME_NUM 3 \
#  OUTPUT_DIR TempPair_Top5_Frame3_Weight05_NoWarmUp_1st \
#  SOLVER.CHECKPOINT_PERIOD 16000 \
#  WANDB.ENABLED True \
#  WANDB.ENTITY garvinxxx \
#  WANDB.NAME Mini_TempPair_Top5_Frame3_Weight05_NoWarmUp_1st

