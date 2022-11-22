CUDA_VISIBLE_DEVICES=0 python train_net.py \
  --dist-url tcp://127.0.0.1:$(( RANDOM % 1000 + 20100 )) \
  --num-gpus 1 \
  --config-file configs/coco_wo_lsj/debug-boxmask_matcher-projection.yaml \
