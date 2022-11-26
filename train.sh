python train_net.py \
  --dist-url tcp://127.0.0.1:$(( RANDOM % 1000 + 20100 )) \
  --num-gpus 4 \
  --config-file configs/coco_wo_lsj/maskformer2_R50_bs16_12ep_projpair.yaml \
  TEST.EVAL_PERIOD 5000 \
  OUTPUT_DIR coco-ins-projpair-wo-MaskAttn \
  WANDB.ENABLED True WANDB.ENTITY garvinxxx WANDB.NAME coco_projpair_wo-MaskAttn \
