#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2021/video_maskformer2_R50_bs16_8k_projpair_longiter.yaml \
#  OUTPUT_DIR YTVIS21_Proj2DPair_32k  # training
#
#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2021/video_maskformer2_R50_bs16_8k_projpair_cascade_pseudo_static_longiter.yaml \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.UPDATE_TYPE tube \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.INTERSACTION True \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.TUBE_UPDATE.CLS_ENABLED True \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.TUBE_UPDATE.QUALITY_TYPE mul_and_pow \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.STATIC_QUALITY_THR 0.8 \
#  OUTPUT_DIR YTVIS21_Proj2DPair_CasPseSta08_Intersact_Tube_Cls_AlwaysBoxmask_32k  # training


# frame, cls
############################################################################################################
##########
#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2021/video_maskformer2_R50_bs16_8k_projpair_cascade_pseudo_static.yaml \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.UPDATE_TYPE frame \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.INTERSACTION True \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.FRAME_UPDATE.CLS_ENABLED True \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.FRAME_UPDATE.PROJ_DICE_TYPE mul \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.FRAME_UPDATE.QUALITY_TYPE mul \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.STATIC_QUALITY_THR 0.8 \
#  OUTPUT_DIR YTVIS21_Proj2DPair_CasPseSta08_Intersact_Frame_Cls_ProjDiceMul_QualMul  # Done

#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --resume \
#  --config-file configs/youtubevis_2021/video_maskformer2_R50_bs16_8k_projpair_cascade_pseudo_static.yaml \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.UPDATE_TYPE frame \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.INTERSACTION True \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.FRAME_UPDATE.CLS_ENABLED True \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.FRAME_UPDATE.PROJ_DICE_TYPE mul \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.FRAME_UPDATE.QUALITY_TYPE mul \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.STATIC_QUALITY_THR 0.5 \
#  OUTPUT_DIR YTVIS21_Proj2DPair_CasPseSta05_Intersact_Frame_Cls_ProjDiceMul_QualMul  # Done

############################################################################################################
##########
#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2021/video_maskformer2_R50_bs16_8k_projpair_cascade_pseudo_static.yaml \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.UPDATE_TYPE frame \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.INTERSACTION True \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.FRAME_UPDATE.CLS_ENABLED True \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.FRAME_UPDATE.PROJ_DICE_TYPE mul_and_pow \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.FRAME_UPDATE.QUALITY_TYPE mul \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.STATIC_QUALITY_THR 0.8 \
#  OUTPUT_DIR YTVIS21_Proj2DPair_CasPseSta08_Intersact_Frame_Cls_ProjDiceMulAndPow_QualMul  # no need any more

#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2021/video_maskformer2_R50_bs16_8k_projpair_cascade_pseudo_static.yaml \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.UPDATE_TYPE frame \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.INTERSACTION True \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.FRAME_UPDATE.CLS_ENABLED True \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.FRAME_UPDATE.PROJ_DICE_TYPE mul_and_pow \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.FRAME_UPDATE.QUALITY_TYPE mul \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.STATIC_QUALITY_THR 0.5 \
#  OUTPUT_DIR YTVIS21_Proj2DPair_CasPseSta05_Intersact_Frame_Cls_ProjDiceMulAndPow_QualMul  # no need any more

############################################################################################################
#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2021/video_maskformer2_R50_bs16_8k_projpair_cascade_pseudo_static.yaml \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.UPDATE_TYPE frame \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.INTERSACTION True \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.FRAME_UPDATE.CLS_ENABLED True \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.FRAME_UPDATE.PROJ_DICE_TYPE mean \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.FRAME_UPDATE.QUALITY_TYPE mul_and_pow \
#  OUTPUT_DIR YTVIS21_Proj2DPair_CasPseSta08_Intersact_Frame_Cls_ProjDiceMean_QualMulAndPow

############################################################################################################
##########
#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2021/video_maskformer2_R50_bs16_8k_projpair_cascade_pseudo_static.yaml \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.UPDATE_TYPE frame \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.INTERSACTION True \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.FRAME_UPDATE.CLS_ENABLED True \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.FRAME_UPDATE.PROJ_DICE_TYPE mul \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.FRAME_UPDATE.QUALITY_TYPE mul_and_pow \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.STATIC_QUALITY_THR 0.8 \
#  OUTPUT_DIR YTVIS21_Proj2DPair_CasPseSta08_Intersact_Frame_Cls_ProjDiceMul_QualMulAndPow  # Done
#
#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2021/video_maskformer2_R50_bs16_8k_projpair_cascade_pseudo_static.yaml \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.UPDATE_TYPE frame \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.INTERSACTION True \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.FRAME_UPDATE.CLS_ENABLED True \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.FRAME_UPDATE.PROJ_DICE_TYPE mul \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.FRAME_UPDATE.QUALITY_TYPE mul_and_pow \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.STATIC_QUALITY_THR 0.5 \
#  OUTPUT_DIR YTVIS21_Proj2DPair_CasPseSta05_Intersact_Frame_Cls_ProjDiceMul_QualMulAndPow  # Done

############################################################################################################
#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2021/video_maskformer2_R50_bs16_8k_projpair_cascade_pseudo_static.yaml \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.UPDATE_TYPE frame \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.INTERSACTION True \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.FRAME_UPDATE.CLS_ENABLED True \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.FRAME_UPDATE.PROJ_DICE_TYPE mul_and_pow \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.FRAME_UPDATE.QUALITY_TYPE mul_and_pow \
#  OUTPUT_DIR YTVIS21_Proj2DPair_CasPseSta08_Intersact_Frame_Cls_ProjDiceMulAndPow_QualMulAndPow

# frame, no cls
############################################################################################################
#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2021/video_maskformer2_R50_bs16_8k_projpair_cascade_pseudo_static.yaml \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.UPDATE_TYPE frame \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.INTERSACTION True \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.FRAME_UPDATE.CLS_ENABLED False \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.FRAME_UPDATE.PROJ_DICE_TYPE mean \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.FRAME_UPDATE.QUALITY_TYPE mul_and_pow \
#  OUTPUT_DIR YTVIS21_Proj2DPair_CasPseSta08_Intersact_Frame_ProjDiceMean_QualMulAndPow

############################################################################################################
##########
#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2021/video_maskformer2_R50_bs16_8k_projpair_cascade_pseudo_static.yaml \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.UPDATE_TYPE frame \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.INTERSACTION True \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.FRAME_UPDATE.CLS_ENABLED False \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.FRAME_UPDATE.PROJ_DICE_TYPE mul \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.FRAME_UPDATE.QUALITY_TYPE mul_and_pow \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.STATIC_QUALITY_THR 0.8 \
#  OUTPUT_DIR YTVIS21_Proj2DPair_CasPseSta08_Intersact_Frame_ProjDiceMul_QualMulAndPow  # TODO
#
#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2021/video_maskformer2_R50_bs16_8k_projpair_cascade_pseudo_static.yaml \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.UPDATE_TYPE frame \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.INTERSACTION True \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.FRAME_UPDATE.CLS_ENABLED False \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.FRAME_UPDATE.PROJ_DICE_TYPE mul \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.FRAME_UPDATE.QUALITY_TYPE mul_and_pow \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.STATIC_QUALITY_THR 0.5 \
#  OUTPUT_DIR YTVIS21_Proj2DPair_CasPseSta05_Intersact_Frame_ProjDiceMul_QualMulAndPow  # TODO

############################################################################################################
#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2021/video_maskformer2_R50_bs16_8k_projpair_cascade_pseudo_static.yaml \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.UPDATE_TYPE frame \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.INTERSACTION True \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.FRAME_UPDATE.CLS_ENABLED False \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.FRAME_UPDATE.PROJ_DICE_TYPE mul_and_pow \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.FRAME_UPDATE.QUALITY_TYPE mul_and_pow \
#  OUTPUT_DIR YTVIS21_Proj2DPair_CasPseSta08_Intersact_Frame_ProjDiceMulAndPow_QualMulAndPow

# tube, cls
############################################################################################################
#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2021/video_maskformer2_R50_bs16_8k_projpair_cascade_pseudo_static.yaml \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.UPDATE_TYPE frame \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.INTERSACTION True \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.TUBE_UPDATE.CLS_ENABLED True \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.TUBE_UPDATE.QUALITY_TYPE mul_and_pow \
#  OUTPUT_DIR YTVIS21_Proj2DPair_CasPseSta08_Intersact_Tube_Cls_QualMul

############################################################################################################
##########
#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2021/video_maskformer2_R50_bs16_8k_projpair_cascade_pseudo_static.yaml \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.UPDATE_TYPE tube \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.INTERSACTION True \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.TUBE_UPDATE.CLS_ENABLED True \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.TUBE_UPDATE.QUALITY_TYPE mul_and_pow \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.STATIC_QUALITY_THR 0.8 \
#  OUTPUT_DIR YTVIS21_Proj2DPair_CasPseSta08_Intersact_Tube_Cls_QualMulAndPow  # Done

#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2021/video_maskformer2_R50_bs16_8k_projpair_cascade_pseudo_static.yaml \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.UPDATE_TYPE tube \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.INTERSACTION True \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.TUBE_UPDATE.CLS_ENABLED True \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.TUBE_UPDATE.QUALITY_TYPE mul_and_pow \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.STATIC_QUALITY_THR 0.5 \
#  OUTPUT_DIR YTVIS21_Proj2DPair_CasPseSta05_Intersact_Tube_Cls_QualMulAndPow  # TODO

# tube, no cls
############################################################################################################
#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2021/video_maskformer2_R50_bs16_8k_projpair_cascade_pseudo_static.yaml \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.UPDATE_TYPE tube \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.INTERSACTION True \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.TUBE_UPDATE.CLS_ENABLED False \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.TUBE_UPDATE.QUALITY_TYPE mul_and_pow \
#  OUTPUT_DIR YTVIS21_Proj2DPair_CasPseSta08_Intersact_Tube_QualMul

############################################################################################################
#python train_net_video.py \
#  --dist-url auto \
#  --num-gpus 4 \
#  --config-file configs/youtubevis_2021/video_maskformer2_R50_bs16_8k_projpair_cascade_pseudo_static.yaml \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.UPDATE_TYPE tube \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.INTERSACTION True \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.TUBE_UPDATE.CLS_ENABLED False \
#  MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.TUBE_UPDATE.QUALITY_TYPE mul_and_pow \
#  OUTPUT_DIR YTVIS21_Proj2DPair_CasPseSta08_Intersact_Tube_QualMulAndPow
