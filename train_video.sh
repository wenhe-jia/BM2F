#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2021_mini/video_maskformer2_R50_bs16_8k_proj_spatpair_temppair.yaml \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.TEMPORAL_PAIRWISE_WEIGHT 0.5 \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE.TOPK 10 \
#  OUTPUT_DIR DEBUG_ViTs_TempPair-weight05-FeatAlign-ThrbyColor-top10 \
#  WANDB.ENABLED False \
#  WANDB.ENTITY garvinxxx \
#  WANDB.NAME ViTs_TempPair-weight05-FeatAlign-ThrbyColor-top10

python train_net_video.py \
  --dist-url auto \
  --num-gpus 4 \
  --config-file configs/youtubevis_2021_mini/video_maskformer2_R50_bs16_8k_proj_spatpair_batch4.yaml \
  OUTPUT_DIR Baseline_batch4_FixInterval \
  SEED 41344227 \
  WANDB.ENABLED False \
  WANDB.ENTITY garvinxxx \
  WANDB.NAME Baseline_batch4_FixInterval
