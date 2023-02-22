python train_net_video.py \
  --eval-only \
  --num-gpus 4 \
  --config-file configs/youtubevis_2021/video_maskformer2_R50_bs16_8k_projection.yaml \
  MODEL.WEIGHTS YTVIS21_Proj2D_fgbg_debug_1st/model_final.pth \
  OUTPUT_DIR YTVIS21_Proj2D_fgbg_debug_1st
#  DATALOADER.NUM_WORKERS 0 \
