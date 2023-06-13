#CUDA_VISIBLE_DEVICES=2,3
python train_net_video.py \
  --dist-url "tcp://127.0.0.1:60205" \
  --num-gpus 1 \
  --config-file configs/youtubevis_2021_mini/video_maskformer2_R50_bs16_8k_proj_spatpair_temppair.yaml \
  MODEL.MASK_FORMER.WEAK_SUPERVISION.TEMPORAL_PAIRWISE_WEIGHT 0.5 \
  MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE.TOPK 10 \
  OUTPUT_DIR DEBUG \
  SOLVER.MAX_ITER 10000 \
  SOLVER.IMS_PER_BATCH 1 \
  DATALOADER.NUM_WORKERS 1