python train_net_video.py \
  --dist-url auto \
  --num-gpus 1 \
  --config-file configs/youtubevis_2021/video_maskformer2_R50_bs16_8k_projection.yaml \
  SOLVER.IMS_PER_BATCH 1 \
  SOLVER.MAX_ITER 10000 \
  OUTPUT_DIR DEBUG