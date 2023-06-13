#CUDA_VISIBLE_DEVICES=2,3 python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 2 \
#  --config-file configs/youtubevis_2021_mini/video_maskformer2_R50_bs16_8k_proj_spatpair_temppair.yaml \
#  --resume \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.TEMPORAL_PAIRWISE_WEIGHT 0.5 \
#  OUTPUT_DIR DEBUG_ViT-g \
#  WANDB.ENABLED True WANDB.ENTITY garvinxxx WANDB.NAME ytvis21mini_Proj_SpatPair_TempPair-weight05


#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2021_mini/video_maskformer2_R50_bs16_8k_proj_spatpair_temppair.yaml \
#  --resume \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.TEMPORAL_PAIRWISE_WEIGHT 0.5 \
#  OUTPUT_DIR DEBUG_ViT-s_2 \
#  WANDB.ENABLED True \
#  WANDB.ENTITY garvinxxx \
#  WANDB.RESUME m3f8xwn0 \
#  WANDB.NAME ytvis21mini_Proj_SpatPair_TempPair-weight05_ViTs_2



python train_net_video.py \
  --dist-url auto \
  --num-gpus 4 \
  --config-file configs/youtubevis_2021_mini/video_maskformer2_R50_bs16_8k_proj_spatpair_temppair.yaml \
  --resume \
  MODEL.MASK_FORMER.WEAK_SUPERVISION.TEMPORAL_PAIRWISE_WEIGHT 0.5 \
  MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE.TOPK 10 \
  OUTPUT_DIR DEBUG_ViTs_TempPair-weight05-FeatAlign-ThrbyColor-top10 \
  WANDB.ENABLED False \
  WANDB.ENTITY garvinxxx \
  WANDB.NAME ViTs_TempPair-weight05-FeatAlign-ThrbyColor-top10
