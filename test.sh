python train_net.py \
  --config-file configs/coco_wo_lsj/maskformer2_R50_bs16_12ep_projection.yaml \
  --eval-only \
  --num-gpus 4 \
  MODEL.WEIGHTS coco-ins-projection-weightup-12ep/model_0119999.pth