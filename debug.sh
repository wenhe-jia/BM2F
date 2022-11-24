CUDA_VISIBLE_DEVICES=0 python train_net.py \
  --dist-url tcp://127.0.0.1:$(( RANDOM % 1000 + 20100 )) \
  --config-file configs/coco_wo_lsj/maskformer2_R50_bs16_12ep_projpair.yaml \
  --num-gpus 1 \
  SOLVER.MAX_ITER 10 \
  SOLVER.IMS_PER_BATCH 2 \
  DATALOADER.NUM_WORKERS 0 \
  OUTPUT_DIR debug-ProjPair