# single gpu debug
#CUDA_LAUNCH_BLOCKING=1 python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 1 \
#  --config-file configs/youtubevis_2021_mini/video_maskformer2_R50_bs16_8k_projSTpair.yaml \
#  SOLVER.IMS_PER_BATCH 1 \
#  SOLVER.MAX_ITER 10000 \
#  OUTPUT_DIR DEBUG \
#  DATALOADER.NUM_WORKERS 0

# multi gpu debug
python train_net_video.py \
  --dist-url auto \
  --num-gpus 4 \
  --config-file configs/youtubevis_2021_mini/video_maskformer2_R50_bs16_8k_projSTpair.yaml \
  OUTPUT_DIR DEBUG \

# baseline compare
#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2021_mini/video_maskformer2_R50_bs16_8k_projpair.yaml \
#  SOLVER.CHECKPOINT_PERIOD 16000 \
#  OUTPUT_DIR DEBUG