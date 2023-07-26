python train_net_video.py \
  --dist-url auto \
  --num-gpus 4 \
  --config-file configs/youtubevis_2021/video_maskformer2_R50_bs16_8k_projSTpair.yaml \
  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.TEMPORAL_TOPK 10 \
  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.TEMPORAL_DIST_THRESH 2.0 \
  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.TEMPORAL_PAIRWISE_WEIGHT 1.0 \
  INPUT.SAMPLING_FRAME_NUM 3 \
  INPUT.FIXED_SAMPLING_INTERVAL True \
  OUTPUT_DIR NotMini_TempPair_Dist20Top10_Frame3ShortRange_Weight10_NoWarmUp_1st \
  SOLVER.CHECKPOINT_PERIOD 16000 \
  TEST.EVAL_PERIOD 16000 \
  WANDB.ENABLED True \
  WANDB.ENTITY garvinxxx \
  WANDB.NAME NotMini_TempPair_Dist20Top10_Frame3ShortRange_Weight10_NoWarmUp_1st

python train_net_video.py \
  --dist-url auto \
  --num-gpus 4 \
  --config-file configs/youtubevis_2021/video_maskformer2_R50_bs16_8k_projSTpair.yaml \
  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.TEMPORAL_TOPK 10 \
  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.TEMPORAL_DIST_THRESH 2.0 \
  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.TEMPORAL_PAIRWISE_WEIGHT 1.0 \
  INPUT.SAMPLING_FRAME_NUM 3 \
  INPUT.FIXED_SAMPLING_INTERVAL True \
  OUTPUT_DIR NotMini_TempPair_Dist20Top10_Frame3ShortRange_Weight10_NoWarmUp_2nd \
  SOLVER.CHECKPOINT_PERIOD 16000 \
  TEST.EVAL_PERIOD 16000 \
  WANDB.ENABLED True \
  WANDB.ENTITY garvinxxx \
  WANDB.NAME NotMini_TempPair_Dist20Top10_Frame3ShortRange_Weight10_NoWarmUp_2nd

python train_net_video.py \
  --dist-url auto \
  --num-gpus 4 \
  --config-file configs/youtubevis_2021/video_maskformer2_R50_bs16_8k_projSTpair.yaml \
  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.TEMPORAL_TOPK 10 \
  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.TEMPORAL_DIST_THRESH 2.0 \
  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.TEMPORAL_PAIRWISE_WEIGHT 1.0 \
  INPUT.SAMPLING_FRAME_NUM 3 \
  INPUT.FIXED_SAMPLING_INTERVAL True \
  OUTPUT_DIR NotMini_TempPair_Dist20Top10_Frame3ShortRange_Weight10_NoWarmUp_3nd \
  SOLVER.CHECKPOINT_PERIOD 16000 \
  TEST.EVAL_PERIOD 16000 \
  WANDB.ENABLED True \
  WANDB.ENTITY garvinxxx \
  WANDB.NAME NotMini_TempPair_Dist20Top10_Frame3ShortRange_Weight10_NoWarmUp_3nd




#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 2 \
#  --config-file configs/youtubevis_2021/video_maskformer2_R50_bs16_8k_projpair_batch4.yaml \
#  INPUT.SAMPLING_FRAME_NUM 3 \
#  INPUT.FIXED_SAMPLING_INTERVAL True \
#  OUTPUT_DIR Baseline_Fix_6th \
#  SOLVER.CHECKPOINT_PERIOD 16000 \
#  WANDB.ENABLED False \
#  WANDB.ENTITY garvinxxx \
#  WANDB.NAME Baseline_Fix_6th
