#######
#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2019/video_maskformer2_R50_bs16_8k_proj.yaml \
#  INPUT.SAMPLING_FRAME_NUM 3 \
#  INPUT.FIXED_SAMPLING_INTERVAL False \
#  SOLVER.CHECKPOINT_PERIOD 6000 \
#  TEST.EVAL_PERIOD 6000 \
#  OUTPUT_DIR YTVIS19_ProjLL_Frame3_1st
#
#mv YTVIS19_ProjLL_Frame3_1st/inference YTVIS19_ProjLL_Frame3_1st/Offline_inference
#
#python train_net_video.py \
#  --eval-only \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2019/video_maskformer2_R50_bs16_8k_proj.yaml \
#  MODEL.MASK_FORMER_VIDEO.TEST.TRACKER_TYPE minvis \
#  TEST.DETECTIONS_PER_IMAGE 10 \
#  MODEL.WEIGHTS YTVIS19_ProjLL_Frame3_1st/model_final.pth \
#  OUTPUT_DIR YTVIS19_ProjLL_Frame3_1st
#
#mv YTVIS19_ProjLL_Frame3_1st/inference YTVIS19_ProjLL_Frame3_1st/MinVIS_inference
#######
#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2019/video_maskformer2_R50_bs16_8k_proj.yaml \
#  INPUT.SAMPLING_FRAME_NUM 3 \
#  INPUT.FIXED_SAMPLING_INTERVAL False \
#  SOLVER.CHECKPOINT_PERIOD 6000 \
#  TEST.EVAL_PERIOD 6000 \
#  OUTPUT_DIR YTVIS19_ProjLL_Frame3_2nd \
#
#mv YTVIS19_ProjLL_Frame3_2nd/inference YTVIS19_ProjLL_Frame3_2nd/Offline_inference
#
#python train_net_video.py \
#  --eval-only \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2019/video_maskformer2_R50_bs16_8k_proj.yaml \
#  MODEL.MASK_FORMER_VIDEO.TEST.TRACKER_TYPE minvis \
#  TEST.DETECTIONS_PER_IMAGE 10 \
#  MODEL.WEIGHTS YTVIS19_ProjLL_Frame3_2nd/model_final.pth \
#  OUTPUT_DIR YTVIS19_ProjLL_Frame3_2nd
#
#mv YTVIS19_ProjLL_Frame3_2nd/inference YTVIS19_ProjLL_Frame3_2nd/MinVIS_inference
######
#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2019/video_maskformer2_R50_bs16_8k_proj.yaml \
#  INPUT.SAMPLING_FRAME_NUM 3 \
#  INPUT.FIXED_SAMPLING_INTERVAL False \
#  SOLVER.CHECKPOINT_PERIOD 6000 \
#  TEST.EVAL_PERIOD 6000 \
#  OUTPUT_DIR YTVIS19_ProjLL_Frame3_3rd \
#
#mv YTVIS19_ProjLL_Frame3_3rd/inference YTVIS19_ProjLL_Frame3_3rd/Offline_inference
#
#python train_net_video.py \
#  --eval-only \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2019/video_maskformer2_R50_bs16_8k_proj.yaml \
#  MODEL.MASK_FORMER_VIDEO.TEST.TRACKER_TYPE minvis \
#  TEST.DETECTIONS_PER_IMAGE 10 \
#  MODEL.WEIGHTS YTVIS19_ProjLL_Frame3_3rd/model_final.pth \
#  OUTPUT_DIR YTVIS19_ProjLL_Frame3_3rd
#
#mv YTVIS19_ProjLL_Frame3_3rd/inference YTVIS19_ProjLL_Frame3_3rd/MinVIS_inference
#######
#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2019/video_maskformer2_R50_bs16_8k_proj.yaml \
#  INPUT.SAMPLING_FRAME_NUM 3 \
#  INPUT.FIXED_SAMPLING_INTERVAL False \
#  SOLVER.CHECKPOINT_PERIOD 6000 \
#  TEST.EVAL_PERIOD 6000 \
#  OUTPUT_DIR YTVIS19_ProjLL_Frame3_4th \
#
#mv YTVIS19_ProjLL_Frame3_4th/inference YTVIS19_ProjLL_Frame3_4th/Offline_inference
#
#python train_net_video.py \
#  --eval-only \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2019/video_maskformer2_R50_bs16_8k_proj.yaml \
#  MODEL.MASK_FORMER_VIDEO.TEST.TRACKER_TYPE minvis \
#  TEST.DETECTIONS_PER_IMAGE 10 \
#  MODEL.WEIGHTS YTVIS19_ProjLL_Frame3_4th/model_final.pth \
#  OUTPUT_DIR YTVIS19_ProjLL_Frame3_4th
#
#mv YTVIS19_ProjLL_Frame3_4th/inference YTVIS19_ProjLL_Frame3_4th/MinVIS_inference
#######
#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2019/video_maskformer2_R50_bs16_8k_proj.yaml \
#  INPUT.SAMPLING_FRAME_NUM 3 \
#  INPUT.FIXED_SAMPLING_INTERVAL False \
#  SOLVER.CHECKPOINT_PERIOD 6000 \
#  TEST.EVAL_PERIOD 6000 \
#  OUTPUT_DIR YTVIS19_ProjLL_Frame3_5th \
#
#mv YTVIS19_ProjLL_Frame3_5th/inference YTVIS19_ProjLL_Frame3_5th/Offline_inference
#
#python train_net_video.py \
#  --eval-only \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2019/video_maskformer2_R50_bs16_8k_proj.yaml \
#  MODEL.MASK_FORMER_VIDEO.TEST.TRACKER_TYPE minvis \
#  TEST.DETECTIONS_PER_IMAGE 10 \
#  MODEL.WEIGHTS YTVIS19_ProjLL_Frame3_5th/model_final.pth \
#  OUTPUT_DIR YTVIS19_ProjLL_Frame3_5th
#
#mv YTVIS19_ProjLL_Frame3_5th/inference YTVIS19_ProjLL_Frame3_5th/MinVIS_inference
#
##########################################
#######
#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2019/video_maskformer2_R50_bs16_8k_projpair.yaml \
#  INPUT.SAMPLING_FRAME_NUM 3 \
#  INPUT.FIXED_SAMPLING_INTERVAL False \
#  SOLVER.CHECKPOINT_PERIOD 6000 \
#  TEST.EVAL_PERIOD 6000 \
#  OUTPUT_DIR YTVIS19_ProjLLPair_Frame3_1st \
#
#mv YTVIS19_ProjLLPair_Frame3_1st/inference YTVIS19_ProjLLPair_Frame3_1st/Offline_inference
#
#python train_net_video.py \
#  --eval-only \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2019/video_maskformer2_R50_bs16_8k_proj.yaml \
#  MODEL.MASK_FORMER_VIDEO.TEST.TRACKER_TYPE minvis \
#  TEST.DETECTIONS_PER_IMAGE 10 \
#  MODEL.WEIGHTS YTVIS19_ProjLLPair_Frame3_1st/model_final.pth \
#  OUTPUT_DIR YTVIS19_ProjLLPair_Frame3_1st
#
#mv YTVIS19_ProjLLPair_Frame3_1st/inference YTVIS19_ProjLLPair_Frame3_1st/MinVIS_inference
#######
#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2019/video_maskformer2_R50_bs16_8k_projpair.yaml \
#  INPUT.SAMPLING_FRAME_NUM 3 \
#  INPUT.FIXED_SAMPLING_INTERVAL False \
#  SOLVER.CHECKPOINT_PERIOD 6000 \
#  TEST.EVAL_PERIOD 6000 \
#  OUTPUT_DIR YTVIS19_ProjLLPair_Frame3_2nd \
#
#mv YTVIS19_ProjLLPair_Frame3_2nd/inference YTVIS19_ProjLLPair_Frame3_2nd/Offline_inference
#
#python train_net_video.py \
#  --eval-only \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2019/video_maskformer2_R50_bs16_8k_proj.yaml \
#  MODEL.MASK_FORMER_VIDEO.TEST.TRACKER_TYPE minvis \
#  TEST.DETECTIONS_PER_IMAGE 10 \
#  MODEL.WEIGHTS YTVIS19_ProjLLPair_Frame3_2nd/model_final.pth \
#  OUTPUT_DIR YTVIS19_ProjLLPair_Frame3_2nd
#
#mv YTVIS19_ProjLLPair_Frame3_2nd/inference YTVIS19_ProjLLPair_Frame3_2nd/MinVIS_inference
#######
#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2019/video_maskformer2_R50_bs16_8k_projpair.yaml \
#  INPUT.SAMPLING_FRAME_NUM 3 \
#  INPUT.FIXED_SAMPLING_INTERVAL False \
#  SOLVER.CHECKPOINT_PERIOD 6000 \
#  TEST.EVAL_PERIOD 6000 \
#  OUTPUT_DIR YTVIS19_ProjLLPair_Frame3_3rd \
#
#mv YTVIS19_ProjLLPair_Frame3_3rd/inference YTVIS19_ProjLLPair_Frame3_3rd/Offline_inference
#
#python train_net_video.py \
#  --eval-only \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2019/video_maskformer2_R50_bs16_8k_proj.yaml \
#  MODEL.MASK_FORMER_VIDEO.TEST.TRACKER_TYPE minvis \
#  TEST.DETECTIONS_PER_IMAGE 10 \
#  MODEL.WEIGHTS YTVIS19_ProjLLPair_Frame3_3rd/model_final.pth \
#  OUTPUT_DIR YTVIS19_ProjLLPair_Frame3_3rd
#
#mv YTVIS19_ProjLLPair_Frame3_3rd/inference YTVIS19_ProjLLPair_Frame3_3rd/MinVIS_inference
#######
#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2019/video_maskformer2_R50_bs16_8k_projpair.yaml \
#  INPUT.SAMPLING_FRAME_NUM 3 \
#  INPUT.FIXED_SAMPLING_INTERVAL False \
#  SOLVER.CHECKPOINT_PERIOD 6000 \
#  TEST.EVAL_PERIOD 6000 \
#  OUTPUT_DIR YTVIS19_ProjLLPair_Frame3_4th \
#
#mv YTVIS19_ProjLLPair_Frame3_4th/inference YTVIS19_ProjLLPair_Frame3_4th/Offline_inference
#
#python train_net_video.py \
#  --eval-only \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2019/video_maskformer2_R50_bs16_8k_proj.yaml \
#  MODEL.MASK_FORMER_VIDEO.TEST.TRACKER_TYPE minvis \
#  TEST.DETECTIONS_PER_IMAGE 10 \
#  MODEL.WEIGHTS YTVIS19_ProjLLPair_Frame3_4th/model_final.pth \
#  OUTPUT_DIR YTVIS19_ProjLLPair_Frame3_4th
#
#mv YTVIS19_ProjLLPair_Frame3_4th/inference YTVIS19_ProjLLPair_Frame3_4th/MinVIS_inference
#######
#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2019/video_maskformer2_R50_bs16_8k_projpair.yaml \
#  INPUT.SAMPLING_FRAME_NUM 3 \
#  INPUT.FIXED_SAMPLING_INTERVAL False \
#  SOLVER.CHECKPOINT_PERIOD 6000 \
#  TEST.EVAL_PERIOD 6000 \
#  OUTPUT_DIR YTVIS19_ProjLLPair_Frame3_5th \
#
#mv YTVIS19_ProjLLPair_Frame3_5th/inference YTVIS19_ProjLLPair_Frame3_5th/Offline_inference
#
#python train_net_video.py \
#  --eval-only \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2019/video_maskformer2_R50_bs16_8k_proj.yaml \
#  MODEL.MASK_FORMER_VIDEO.TEST.TRACKER_TYPE minvis \
#  TEST.DETECTIONS_PER_IMAGE 10 \
#  MODEL.WEIGHTS YTVIS19_ProjLLPair_Frame3_5th/model_final.pth \
#  OUTPUT_DIR YTVIS19_ProjLLPair_Frame3_5th
#
#mv YTVIS19_ProjLLPair_Frame3_5th/inference YTVIS19_ProjLLPair_Frame3_5th/MinVIS_inference



# --------- MATA ---------
########## top 5 ##########
python train_net_video.py \
  --dist-url auto \
  --num-gpus 4 \
  --config-file configs/youtubevis_2019/video_maskformer2_R101_bs16_8k_projSTpair.yaml \
  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.TEMPORAL_TOPK 5 \
  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.TEMPORAL_PAIRWISE_WEIGHT 1.0 \
  INPUT.SAMPLING_FRAME_NUM 3 \
  INPUT.FIXED_SAMPLING_INTERVAL True \
  OUTPUT_DIR 19_Top5_1st \
  SOLVER.CHECKPOINT_PERIOD 6000 \
  TEST.EVAL_PERIOD 6000 \

cd 19_Top5_1st/inference
rm instances_predictions.pth
zip top5_Offline_1st.zip results.json
cd ../..
mv 19_Top5_1st/inference 19_Top5_1st/Offline_inference

python train_net_video.py \
  --eval-only \
  --num-gpus 4 \
  --config-file configs/youtubevis_2019/video_maskformer2_R101_bs16_8k_projSTpair.yaml \
  MODEL.MASK_FORMER_VIDEO.TEST.TRACKER_TYPE minvis \
  TEST.DETECTIONS_PER_IMAGE 10 \
  MODEL.WEIGHTS 19_Top5_1st/model_final.pth \
  OUTPUT_DIR 19_Top5_1st

cd 19_Top5_1st/inference
rm instances_predictions.pth
zip top5_MinVIS_1st.zip results.json
cd ../..
mv 19_Top5_1st/inference 19_Top5_1st/MinVIS_inference
#####

#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2019/video_maskformer2_R101_bs16_8k_projSTpair.yaml \
#  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.TEMPORAL_TOPK 5 \
#  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.TEMPORAL_PAIRWISE_WEIGHT 1.0 \
#  INPUT.SAMPLING_FRAME_NUM 3 \
#  INPUT.FIXED_SAMPLING_INTERVAL True \
#  OUTPUT_DIR 19_Top5_2nd \
#  SOLVER.CHECKPOINT_PERIOD 6000 \
#  TEST.EVAL_PERIOD 6000 \
#
#cd 19_Top5_2nd/inference
#rm instances_predictions.pth
#zip top5_Offline_2nd.zip results.json
#cd ../..
#mv 19_Top5_2nd/inference 19_Top5_2nd/Offline_inference
#
#python train_net_video.py \
#  --eval-only \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2019/video_maskformer2_R101_bs16_8k_projSTpair.yaml \
#  MODEL.MASK_FORMER_VIDEO.TEST.TRACKER_TYPE minvis \
#  TEST.DETECTIONS_PER_IMAGE 10 \
#  MODEL.WEIGHTS 19_Top5_2nd/model_final.pth \
#  OUTPUT_DIR 19_Top5_2nd
#
#cd 19_Top5_2nd/inference
#rm instances_predictions.pth
#zip top5_MinVIS_2nd.zip results.json
#cd ../..
#mv 19_Top5_2nd/inference 19_Top5_2nd/MinVIS_inference
######
#
#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2019/video_maskformer2_R101_bs16_8k_projSTpair.yaml \
#  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.TEMPORAL_TOPK 5 \
#  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.TEMPORAL_PAIRWISE_WEIGHT 1.0 \
#  INPUT.SAMPLING_FRAME_NUM 3 \
#  INPUT.FIXED_SAMPLING_INTERVAL True \
#  OUTPUT_DIR 19_Top5_3rd \
#  SOLVER.CHECKPOINT_PERIOD 6000 \
#  TEST.EVAL_PERIOD 6000 \
#
#cd 19_Top5_3rd/inference
#rm instances_predictions.pth
#zip top5_Offline_3rd.zip results.json
#cd ../..
#mv 19_Top5_3rd/inference 19_Top5_3rd/Offline_inference
#
#python train_net_video.py \
#  --eval-only \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2019/video_maskformer2_R101_bs16_8k_projSTpair.yaml \
#  MODEL.MASK_FORMER_VIDEO.TEST.TRACKER_TYPE minvis \
#  TEST.DETECTIONS_PER_IMAGE 10 \
#  MODEL.WEIGHTS 19_Top5_3rd/model_final.pth \
#  OUTPUT_DIR 19_Top5_3rd
#
#cd 19_Top5_3rd/inference
#rm instances_predictions.pth
#zip top5_MinVIS_3rd.zip results.json
#cd ../..
#mv 19_Top5_3rd/inference 19_Top5_3rd/MinVIS_inference
######
#
#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2019/video_maskformer2_R101_bs16_8k_projSTpair.yaml \
#  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.TEMPORAL_TOPK 1 \
#  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.TEMPORAL_PAIRWISE_WEIGHT 1.0 \
#  INPUT.SAMPLING_FRAME_NUM 3 \
#  INPUT.FIXED_SAMPLING_INTERVAL True \
#  OUTPUT_DIR 19_Top5_4th \
#  SOLVER.CHECKPOINT_PERIOD 6000 \
#  TEST.EVAL_PERIOD 6000 \
#
#cd 19_Top5_4th/inference
#rm instances_predictions.pth
#zip top5_Offline_4th.zip results.json
#cd ../..
#mv 19_Top5_4th/inference 19_Top5_4th/Offline_inference
#
#python train_net_video.py \
#  --eval-only \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2019/video_maskformer2_R101_bs16_8k_projSTpair.yaml \
#  MODEL.MASK_FORMER_VIDEO.TEST.TRACKER_TYPE minvis \
#  TEST.DETECTIONS_PER_IMAGE 10 \
#  MODEL.WEIGHTS 19_Top5_4th/model_final.pth \
#  OUTPUT_DIR 19_Top5_4th
#
#cd 19_Top5_4th/inference
#rm instances_predictions.pth
#zip top5_MinVIS_4th.zip results.json
#cd ../..
#mv 19_Top5_4th/inference 19_Top5_4th/MinVIS_inference
######
#
#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2019/video_maskformer2_R101_bs16_8k_projSTpair.yaml \
#  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.TEMPORAL_TOPK 5 \
#  MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.TEMPORAL_PAIRWISE_WEIGHT 1.0 \
#  INPUT.SAMPLING_FRAME_NUM 3 \
#  INPUT.FIXED_SAMPLING_INTERVAL True \
#  OUTPUT_DIR 19_Top5_5th \
#  SOLVER.CHECKPOINT_PERIOD 6000 \
#  TEST.EVAL_PERIOD 6000 \
#
#cd 19_Top5_5th/inference
#rm instances_predictions.pth
#zip top5_Offline_5th.zip results.json
#cd ../..
#mv 19_Top5_5th/inference 19_Top5_5th/Offline_inference
#
#python train_net_video.py \
#  --eval-only \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2019/video_maskformer2_R101_bs16_8k_projSTpair.yaml \
#  MODEL.MASK_FORMER_VIDEO.TEST.TRACKER_TYPE minvis \
#  TEST.DETECTIONS_PER_IMAGE 10 \
#  MODEL.WEIGHTS 19_Top5_5th/model_final.pth \
#  OUTPUT_DIR 19_Top5_5th
#
#cd 19_Top5_5th/inference
#rm instances_predictions.pth
#zip top5_MinVIS_5th.zip results.json
#cd ../..
#mv 19_Top5_5th/inference 19_Top5_5th/MinVIS_inference
######