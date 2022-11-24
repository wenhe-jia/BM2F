python train_net.py \
  --dist-url tcp://127.0.0.1:$(( RANDOM % 1000 + 20100 )) \
  --num-gpus 4 \
  --config-file configs/coco_wo_lsj/maskformer2_R50_bs16_12ep_ProjPair.yaml \
  --resume \
  OUTPUT_DIR coco-ins-ProjPair \
  WANDB.ENABLED True WANDB.ENTITY garvinxxx WANDB.NAME coco_projection-pairwise \
