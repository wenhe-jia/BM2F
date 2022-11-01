CUDA_VISIBLE_DEVICES=0,1 \
  python train_net_video.py \
  --config-file configs/youtubevis_2021/video_maskformer2_R50_bs4_8ep.yaml \
  --num-gpus 2
