# --- RectfProjPair ---
#######
#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/ovis/video_maskformer2_R50_bs16_8k_projpair.yaml \
#  MODEL.MASK_FORMER_VIDEO.TEST.TRACKER_TYPE minvis \
#  MODEL.MASK_FORMER_VIDEO.TEST.MERGE_ON_CPU True \
#  INPUT.FIXED_SAMPLING_INTERVAL True \
#  TEST.DETECTIONS_PER_IMAGE 10 \
#  OUTPUT_DIR Rectf_Seed37150921_1st \
#
#cd Rectf_Seed37150921_1st/inference
#rm instances_predictions.pth
#zip m2f_Frame3_FixInterval_RectfProjPair_Seed37150921_1st.zip results.json
#cd ../..
#mv Rectf_Seed37150921_1st/inference Rectf_Seed37150921_1st/MinVIS_inference

######
python train_net_video.py \
  --dist-url auto \
  --num-gpus 4 \
  --config-file configs/ovis/video_maskformer2_R50_bs16_8k_projpair.yaml \
  MODEL.MASK_FORMER_VIDEO.TEST.TRACKER_TYPE minvis \
  MODEL.MASK_FORMER_VIDEO.TEST.MERGE_ON_CPU True \
  INPUT.FIXED_SAMPLING_INTERVAL True \
  TEST.DETECTIONS_PER_IMAGE 10 \
  OUTPUT_DIR Rectf_Seed37150921_2nd \

cd Rectf_Seed37150921_2nd/inference
rm instances_predictions.pth
zip m2f_Frame3_FixInterval_RectfProjPair_Seed37150921_2nd.zip results.json
cd ../..
mv Rectf_Seed37150921_2nd/inference Rectf_Seed37150921_2nd/MinVIS_inference

######
python train_net_video.py \
  --dist-url auto \
  --num-gpus 4 \
  --config-file configs/ovis/video_maskformer2_R50_bs16_8k_projpair.yaml \
  MODEL.MASK_FORMER_VIDEO.TEST.TRACKER_TYPE minvis \
  MODEL.MASK_FORMER_VIDEO.TEST.MERGE_ON_CPU True \
  INPUT.FIXED_SAMPLING_INTERVAL True \
  TEST.DETECTIONS_PER_IMAGE 10 \
  OUTPUT_DIR Rectf_Seed37150921_3rd \

cd Rectf_Seed37150921_3rd/inference
rm instances_predictions.pth
zip m2f_Frame3_FixInterval_RectfProjPair_Seed37150921_3rd.zip results.json
cd ../..
mv Rectf_Seed37150921_3rd/inference Rectf_Seed37150921_3rd/MinVIS_inference

