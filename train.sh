python train_net_video.py \
  --dist-url auto \
  --num-gpus 4 \
  --config-file configs/youtubevis_2021/video_maskformer2_R50_bs16_8k_projection.yaml \
  OUTPUT_DIR YTVIS21_Proj2D
#  OUTPUT_DIR YTVIS21_Proj2D_withinBox_CritOnly
