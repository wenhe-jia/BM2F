# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import math
from typing import Tuple

from scipy.optimize import linear_sum_assignment


import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks

from .modeling.criterion import VideoSetCriterionProj, VideoSetCriterionProjPair, VideoSetCriterionProjSTPair
from .modeling.matcher import VideoHungarianMatcherProj, VideoHungarianMatcherProjPair
from .utils.memory import retry_if_cuda_oom

from .utils.weaksup_utils import unfold_wo_center, get_images_color_similarity, filter_temporal_pairs_by_color_similarity

import os
import cv2
import numpy as np
from skimage import color
import copy
import random


logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class VideoMaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        weak_supervision: bool,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # video
        num_frames,
        output_dir,
        pairwise_size,
        pairwise_dilation,
        temporal_pairwise,
        use_input_resolution,
        filter_temp_by_color,
        temporal_color_threshold,
        # inference
        tracker_type: str,
        window_inference: bool,
        test_topk_per_image: int,
        is_multi_cls: bool,
        apply_cls_thres: float,
        merge_on_cpu: bool,
        # tracking
        num_max_inst_test: int,
        num_frames_test: int,
        num_frames_window_test: int,
        clip_stride: int,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.num_frames = num_frames

        self.weak_supervision = weak_supervision
        self.mask_out_stride = 4
        self.pairwise_size = pairwise_size
        self.pairwise_dilation = pairwise_dilation

        self.temporal_pairwise = temporal_pairwise
        self.use_input_resolution = use_input_resolution
        self.filter_temp_by_color = filter_temp_by_color
        self.temporal_color_threshold = temporal_color_threshold

        self.output_dir = output_dir

        self.tracker_type = tracker_type  # if 'ovis' in data_name and use swin large backbone => "mdqe"

        # additional args reference
        self.is_multi_cls = is_multi_cls
        self.apply_cls_thres = apply_cls_thres
        self.window_inference = window_inference
        self.test_topk_per_image = test_topk_per_image
        self.merge_on_cpu = merge_on_cpu

        # clip-by-clip tracking
        self.num_max_inst_test = num_max_inst_test
        self.num_frames_test = num_frames_test
        self.num_frames_window_test = num_frames_window_test
        self.clip_stride = clip_stride

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT
        weak_supervision = False
        temporal_pairwise = False

        # define supervision type
        supervision_type = cfg.MODEL.MASK_FORMER.SUP_TYPE
        if supervision_type != "mask":
            weak_supervision = True

        # classification loss weight
        if supervision_type == "mask":
            weight_dict = {
                "loss_ce": cfg.MODEL.MASK_FORMER.CLASS_WEIGHT,
                "loss_mask": cfg.MODEL.MASK_FORMER.MASK_WEIGHT,
                "loss_dice": cfg.MODEL.MASK_FORMER.DICE_WEIGHT
            }
        elif supervision_type == "mask_projection":
            weight_dict = {
                "loss_ce": cfg.MODEL.MASK_FORMER.CLASS_WEIGHT,
                "loss_mask_projection": cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PROJECTION_WEIGHT
            }
        elif supervision_type == "mask_projection_and_pairwise":
            weight_dict = {
                "loss_ce": cfg.MODEL.MASK_FORMER.CLASS_WEIGHT,
                "loss_mask_projection": cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PROJECTION_WEIGHT,
                "loss_mask_pairwise": cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE_WEIGHT
            }
        elif supervision_type == "mask_projection_and_STpairwise":
            temporal_pairwise = True
            weight_dict = {
                "loss_ce": cfg.MODEL.MASK_FORMER.CLASS_WEIGHT,
                "loss_mask_projection": cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PROJECTION_WEIGHT,
                "loss_mask_pairwise": cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE_WEIGHT,
                "loss_mask_temporal_pairwise": cfg.MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.TEMPORAL_PAIRWISE_WEIGHT,
            }
        else:
            raise Exception("Unknown mask_target_type type !!!")

        # set loss weight dict
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        # building criterion
        if supervision_type == "mask":
            matcher = VideoHungarianMatcher(
                cost_class=cfg.MODEL.MASK_FORMER.CLASS_WEIGHT,
                cost_mask=cfg.MODEL.MASK_FORMER.MASK_WEIGHT,
                cost_dice=cfg.MODEL.MASK_FORMER.DICE_WEIGHT,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            )
            losses = ["labels", "masks"]
            criterion = VideoSetCriterion(
                sem_seg_head.num_classes,
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=no_object_weight,
                losses=losses,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
                oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
                importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            )
        elif supervision_type == "mask_projection":
            matcher = VideoHungarianMatcherProj(
                cost_class=cfg.MODEL.MASK_FORMER.CLASS_WEIGHT,
                cost_projection=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PROJECTION_WEIGHT,
            )
            losses = ["labels", "projection_masks"]
            criterion = VideoSetCriterionProj(
                sem_seg_head.num_classes,
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=no_object_weight,
                losses=losses,
            )
        elif supervision_type == "mask_projection_and_pairwise":
            matcher = VideoHungarianMatcherProjPair(
                cost_class=cfg.MODEL.MASK_FORMER.CLASS_WEIGHT,
                cost_projection=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PROJECTION_WEIGHT,
                cost_pairwise=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE_WEIGHT,
                pairwise_size=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE.SIZE,
                pairwise_dilation=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE.DILATION,
                pairwise_color_thresh=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE.COLOR_THRESH,
                pairwise_warmup_iters=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE.WARMUP_ITERS,
            )
            losses = ["labels", "projection_masks", "pairwise"]
            criterion = VideoSetCriterionProjPair(
                sem_seg_head.num_classes,
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=no_object_weight,
                pairwise_size=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE.SIZE,
                pairwise_dilation=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE.DILATION,
                pairwise_color_thresh=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE.COLOR_THRESH,
                pairwise_warmup_iters=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE.WARMUP_ITERS,
                losses=losses,
            )
        elif supervision_type == "mask_projection_and_STpairwise":
            matcher = VideoHungarianMatcherProjPair(
                cost_class=cfg.MODEL.MASK_FORMER.CLASS_WEIGHT,
                cost_projection=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PROJECTION_WEIGHT,
                cost_pairwise=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE_WEIGHT,
                pairwise_size=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE.SIZE,
                pairwise_dilation=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE.DILATION,
                pairwise_color_thresh=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE.COLOR_THRESH,
                pairwise_warmup_iters=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE.WARMUP_ITERS,
            )
            losses = ["labels", "projection_masks", "pairwise", "temporal_pairwise"]
            criterion = VideoSetCriterionProjSTPair(
                sem_seg_head.num_classes,
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=no_object_weight,
                pairwise_size=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE.SIZE,
                pairwise_dilation=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE.DILATION,
                pairwise_color_thresh=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE.COLOR_THRESH,
                pairwise_warmup_iters=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE.WARMUP_ITERS,
                temporal_warmup=cfg.MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.WARM_UP,
                use_input_resolution_for_temp=cfg.MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.USE_INPUT_RESOLUTION,
                losses=losses,
            )
        else:
            raise Exception("Unknown supervision type !!!")

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "weak_supervision": weak_supervision,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": True,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # video
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "output_dir": cfg.OUTPUT_DIR,
            "pairwise_size": cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE.SIZE,
            "pairwise_dilation": cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE.DILATION,
            "temporal_pairwise": temporal_pairwise,
            "use_input_resolution": cfg.MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.USE_INPUT_RESOLUTION,
            "filter_temp_by_color": cfg.MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.FILTER_BY_COLOR,
            "temporal_color_threshold": cfg.MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.TEMPORAL_COLOR_THRESH,
            # inference
            "tracker_type": cfg.MODEL.MASK_FORMER_VIDEO.TEST.TRACKER_TYPE,
            "window_inference": cfg.MODEL.MASK_FORMER_VIDEO.TEST.WINDOW_INFERENCE,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "is_multi_cls": cfg.MODEL.MASK_FORMER_VIDEO.TEST.MULTI_CLS_ON,
            "apply_cls_thres": cfg.MODEL.MASK_FORMER_VIDEO.TEST.APPLY_CLS_THRES,
            "merge_on_cpu": cfg.MODEL.MASK_FORMER_VIDEO.TEST.MERGE_ON_CPU,
            # tracking
            "num_max_inst_test": cfg.MODEL.MASK_FORMER_VIDEO.TEST.NUM_MAX_INST,
            "num_frames_test": cfg.MODEL.MASK_FORMER_VIDEO.TEST.NUM_FRAMES,
            "num_frames_window_test": cfg.MODEL.MASK_FORMER_VIDEO.TEST.NUM_FRAMES_WINDOW,
            "clip_stride": cfg.MODEL.MASK_FORMER_VIDEO.TEST.CLIP_STRIDE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": list[Tensor], frame in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        if not self.training:
            print('\n--- video: {} | name: {} | length: {}--- \n'.format(
                    batched_inputs[0]['video_id'],
                    batched_inputs[0]['file_names'][0].split('/')[-2],
                    len(batched_inputs[0]['file_names'])
                )
            )

            vid_path = self.output_dir + '/vis/vid_' + \
                       batched_inputs[0]['file_names'][0].split('/')[-2] + '/'
            # os.makedirs(vid_path, exist_ok=True)

        org_images = []
        for video in batched_inputs:
            for frame in video["image"]:
                org_images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in org_images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)

        if self.training:
            # mask classification target
            if self.weak_supervision:
                targets = self.prepare_weaksup_targets(batched_inputs, org_images)
            else:
                targets = self.prepare_targets(batched_inputs, images)

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)

            ##### DEBUG #####
            if self.temporal_pairwise:
                temppair_vid = targets[0]["total_temp_pair"]
                pos_temppair_vid = targets[0]["pos_temp_pair"]
                for i in range(1, len(targets)):
                    temppair_vid += targets[i]["total_temp_pair"]
                    pos_temppair_vid += targets[i]["pos_temp_pair"]
                losses.update(
                    {
                        "loss_pos_temp_pair_prop": (pos_temppair_vid / torch.clamp(temppair_vid, min=1.0)).detach(),
                        "loss_temp_pair_per_batch": temppair_vid.detach(),

                    }
                )

            return losses
        else:
            # NOTE consider only B=1 case.
            if self.tracker_type == 'org_m2f_offline':
                features = self.backbone(images.tensor)
                outputs = self.sem_seg_head(features)

                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]

                mask_cls_result = mask_cls_results[0]
                # upsample masks
                mask_pred_result = retry_if_cuda_oom(F.interpolate)(
                    mask_pred_results[0],
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )

                del outputs

                input_per_image = batched_inputs[0]
                image_size = images.image_sizes[0]  # image size without padding after data augmentation

                height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
                width = input_per_image.get("width", image_size[1])

                return retry_if_cuda_oom(self.inference_video_offline)(
                    mask_cls_result, mask_pred_result, image_size, height, width
                )
            elif self.tracker_type == 'minvis':
                outputs_s = self.run_window_inference(images.tensor)
                outputs_s = self.post_processing(outputs_s)
                return retry_if_cuda_oom(self.inference_video)(batched_inputs, images, outputs_s)
            else:
                raise ValueError('the type of tracker only supports {minvis, mdqe}.')
            
            # mask_cls_results = outputs["pred_logits"]
            # mask_pred_results = outputs["pred_masks"]
            # 
            # mask_cls_result = mask_cls_results[0]
            # # upsample masks
            # mask_pred_result = retry_if_cuda_oom(F.interpolate)(
            #     mask_pred_results[0],
            #     size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            #     mode="bilinear",
            #     align_corners=False,
            # )
            # 
            # del outputs
            # 
            # input_per_image = batched_inputs[0]
            # image_size = images.image_sizes[0]  # image size without padding after data augmentation
            # 
            # height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
            # width = input_per_image.get("width", image_size[1])
            # 
            # return retry_if_cuda_oom(self.inference_video)(mask_cls_result, mask_pred_result, image_size, height, width, vid_path)

    def prepare_weaksup_targets(self, targets, org_images):
        """
        :param targets: {
            vid1: {
                'height'      : int,
                'width'       : int,
                'length'      : int,
                'video_id'    : int,
                'image'       : list(Tensor),,
                'instances'   : list(Instances),
                'file_names'  : list(str),
                'match_coords': dict{
                    ins_id(int): {
                        [
                            (frame_1&2){"curr_pts": tensor(K, 2), "next_pts": tensor(K, 2)},
                            ...
                        ]
                    },
                    ...
                }
            },
            vid2: ...,
            ...
        }
        :param org_images: [frame in (3, H, W)] * BT,
        :return:
        """
        B = len(targets)
        T = self.num_frames

        org_image_masks = [torch.ones_like(x[0], dtype=torch.float32) for x in org_images]
        org_images = ImageList.from_tensors(org_images, self.size_divisibility).tensor  # (B*T, 3, H, W)
        org_image_masks = ImageList.from_tensors(
            org_image_masks, self.size_divisibility, pad_value=0.0
        ).tensor  # (BT, H, W)

        h_pad, w_pad = org_images.shape[-2:]
        stride = self.mask_out_stride  # 4
        start = int(stride // 2)
        assert org_images.size(2) % stride == 0
        assert org_images.size(3) % stride == 0
        # down sample org image and masks(of torch.ones)
        # (B*T, 3, H, W) --> (B*T, 3, H/4, W/4) --> (B, T, W/4, H/4, 3)
        downsampled_images = F.avg_pool2d(
            org_images.float(), kernel_size=stride,
            stride=stride, padding=0
        )
        _h, _w = downsampled_images.shape[-2:]
        # (B * T, 3, H/4, W/4) --> (B, T, 3, W/4, H/4) --> (B, T, W/4, H/4, 3)
        downsampled_images = downsampled_images.reshape(B, T, 3, _h, _w)
        # (B*T, H, W) -> (B*T, h/4, W/4) -> (B, T, H/4, W/4)
        downsampled_image_masks = org_image_masks[:, start::stride, start::stride].reshape(B, T, _h, _w)

        _org_images = org_images.reshape(B, T, 3, h_pad, w_pad) if self.use_input_resolution else None

        gt_instances = []
        for vid_ind, targets_per_video in enumerate(targets):
            _num_instance = len(targets_per_video["instances"][0])

            mask_shape = [_num_instance, T, h_pad, w_pad]
            gt_boxmasks_full_per_video = torch.zeros(mask_shape, dtype=torch.float32, device=self.device)
            gt_masks_full_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)
            
            x_bound_shape = [_num_instance, T, h_pad]
            left_bounds_full_per_video = torch.zeros(x_bound_shape, dtype=torch.float32, device=self.device)
            right_bounds_full_per_video = torch.zeros(x_bound_shape, dtype=torch.float32, device=self.device)

            y_bound_shape = [_num_instance, T, w_pad]
            top_bounds_full_per_video = torch.zeros(y_bound_shape, dtype=torch.float32, device=self.device)
            bottom_bounds_full_per_video = torch.zeros(y_bound_shape, dtype=torch.float32, device=self.device)

            color_similarity_shape = [_num_instance, T, self.pairwise_size * self.pairwise_size - 1, _h, _w]
            color_similarity_per_video = torch.zeros(color_similarity_shape, dtype=torch.float32, device=self.device)

            frame_lab_vid = []
            gt_ids_per_video = []
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(self.device)
                gt_ids_per_video.append(targets_per_frame.gt_ids[:, None])  # [(num_ins, 1), (num_ins, 1)]

                h, w = targets_per_frame.image_size
                gt_masks_full_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks.tensor

                # color similarity
                # (H/4, W/4, 3)
                frame_lab = color.rgb2lab(downsampled_images[vid_ind, f_i].byte().permute(1, 2, 0).cpu().numpy())
                frame_lab = torch.as_tensor(frame_lab, device=downsampled_images.device, dtype=torch.float32)
                frame_lab = frame_lab.permute(2, 0, 1)[None]  # (1, 3, H/4, W/4)
                frame_color_similarity = get_images_color_similarity(
                    frame_lab, downsampled_image_masks[vid_ind, f_i],
                    self.pairwise_size, self.pairwise_dilation
                )  # (1, k*k-1, H/4, W/4)

                if self.temporal_pairwise:
                    frame_lab_vid.append(frame_lab[0])

                # generate rectangle gt masks from boxes of shape (N, 4) in abs coordinates
                if len(targets_per_frame) > 0:
                    gt_boxes = [gt_box.squeeze() for gt_box in targets_per_frame.gt_boxes.tensor.split(1)]
                    for ins_i, gt_box in enumerate(gt_boxes):
                        gt_boxmasks_full_per_video[
                            ins_i, f_i, int(gt_box[1]):int(gt_box[3] + 1), int(gt_box[0]):int(gt_box[2] + 1)
                        ] = 1.0

                        gt_mask = gt_boxmasks_full_per_video[ins_i, f_i].int()  # (H, W)
                        # bounds for y projection
                        left_bounds_full_per_video[ins_i, f_i] = torch.argmax(gt_mask, dim=1)
                        right_bounds_full_per_video[ins_i, f_i] = gt_mask.shape[1] - \
                                                                   torch.argmax(gt_mask.flip(1), dim=1)
                        # bounds for x projection
                        top_bounds_full_per_video[ins_i, f_i] = torch.argmax(gt_mask, dim=0)
                        bottom_bounds_full_per_video[ins_i, f_i] = gt_mask.shape[0] - \
                                                              torch.argmax(gt_mask.flip(0), dim=0)

                        # color similarity for individual instance at current frame
                        color_similarity_per_video[ins_i, f_i] = frame_color_similarity

            gt_ids_per_video = torch.cat(gt_ids_per_video, dim=1)  # (N, num_frame)
            valid_idx = (gt_ids_per_video != -1).any(dim=-1)  # (num_ins,), 别取到再所有帧上都是空的gt

            gt_classes_per_video = targets_per_frame.gt_classes[valid_idx]  # N,
            gt_ids_per_video = gt_ids_per_video[valid_idx]  # N, num_frames

            # (G, T, h_pad/4, w_pad/4)
            gt_boxmasks_per_video = gt_boxmasks_full_per_video[:, :, start::stride, start::stride][valid_idx].float()
            gt_masks_per_video = gt_masks_full_per_video[:, :, start::stride, start::stride][valid_idx].float()
            left_bounds_per_video = left_bounds_full_per_video[:, :, start::stride] / stride
            right_bounds_per_video = right_bounds_full_per_video[:, :, start::stride] / stride
            top_bounds_per_video = top_bounds_full_per_video[:, :, start::stride] / stride
            bottom_bounds_per_video = bottom_bounds_full_per_video[:, :, start::stride] / stride
            color_similarity_per_video = color_similarity_per_video[valid_idx].float()

            # prepare temporal pairs
            temp_pairs = []
            # DEBUG
            num_match_per_video = torch.tensor(0, dtype=torch.float32, device=self.device)
            num_pos_match_per_video = torch.tensor(0, dtype=torch.float32, device=self.device)
            if self.temporal_pairwise:
                gt_ids_per_video_unique = torch.max(gt_ids_per_video, dim=1)[0]
                for ins_i in range(gt_ids_per_video_unique.shape[0]):
                    gt_id = int(gt_ids_per_video_unique[ins_i])

                    temp_pairs_ins = []
                    for match_ind, curr_and_next in enumerate(targets_per_video["match_coords"][gt_id]):
                        curr_pts = curr_and_next["curr_pts"].to(self.device)
                        next_pts = curr_and_next["next_pts"].to(self.device)

                        if self.use_input_resolution:
                            curr_pts[:, 0] = curr_pts[:, 0].clamp(0, gt_boxmasks_full_per_video.shape[3] - 1)
                            curr_pts[:, 1] = curr_pts[:, 1].clamp(0, gt_boxmasks_full_per_video.shape[2] - 1)
                            next_pts[:, 0] = next_pts[:, 0].clamp(0, gt_boxmasks_full_per_video.shape[3] - 1)
                            next_pts[:, 1] = next_pts[:, 1].clamp(0, gt_boxmasks_full_per_video.shape[2] - 1)
                        else:
                            curr_pts[:, 0] = curr_pts[:, 0].clamp(0, gt_boxmasks_per_video.shape[3] - 1)
                            curr_pts[:, 1] = curr_pts[:, 1].clamp(0, gt_boxmasks_per_video.shape[2] - 1)
                            next_pts[:, 0] = next_pts[:, 0].clamp(0, gt_boxmasks_per_video.shape[3] - 1)
                            next_pts[:, 1] = next_pts[:, 1].clamp(0, gt_boxmasks_per_video.shape[2] - 1)

                        if self.filter_temp_by_color:
                            if self.use_input_resolution:
                                curr_pts, next_pts = filter_temporal_pairs_by_color_similarity(
                                    curr_pts, next_pts,
                                    _org_images[vid_ind, match_ind], _org_images[vid_ind, match_ind + 1],
                                    color_similarity_threshold=self.temporal_color_threshold,
                                    input_image=True
                                )
                            else:
                                curr_pts, next_pts = filter_temporal_pairs_by_color_similarity(
                                    curr_pts, next_pts,
                                    frame_lab_vid[match_ind], frame_lab_vid[match_ind + 1],
                                    color_similarity_threshold=self.temporal_color_threshold,
                                    input_image=False
                                )
                        temp_pairs_ins.append((curr_pts, next_pts))

                        ##### DEBUG #####
                        # ----- calculate positive pair -----
                        num_match_per_video += torch.tensor(
                            curr_pts.shape[0], dtype=torch.float32, device=self.device
                        )
                        num_pos_match_per_video += calculate_matching_pos(
                            curr_pts,
                            next_pts,
                            gt_masks_full_per_video[ins_i, match_ind].float(),
                            gt_masks_full_per_video[ins_i, match_ind + 1].float(),
                        ).clone().detach()

                        # ----- vis pairs -----
                        # save_path = "/home/jiawenhe/projects/weaksup-vis/Weakly-Sup-VIS/DEBUG/vis_train/{}".format(
                        #     targets_per_video["file_names"][0].split("/")[-2]
                        # )
                        # os.makedirs(save_path, exist_ok=True)
                        #
                        # _canvas = np.ascontiguousarray(torch.cat(
                        #     [_org_images[vid_ind, match_ind], _org_images[vid_ind, match_ind + 1]], dim=-1
                        # ).byte().permute(1, 2, 0).cpu().numpy())  # (H, 2*W, 3)
                        # _canvas = np.ascontiguousarray(cv2.cvtColor(_canvas, cv2.COLOR_BGR2RGB))
                        # mask = np.ascontiguousarray((torch.cat(
                        #     [gt_masks_full_per_video[ins_i, match_ind], gt_masks_full_per_video[ins_i, match_ind + 1]],
                        #     dim=-1
                        # ) * 255)[:, :, None].repeat(1, 1, 3).byte().cpu().numpy())  # (H, 2*W)
                        # canvas_prop = 0.75
                        # canvas = np.ascontiguousarray(_canvas * canvas_prop + mask * (1 - canvas_prop)).astype(np.uint8)
                        # canvas = np.ascontiguousarray(np.concatenate([canvas, _canvas], axis=0))
                        #
                        # next_pts_shifted = copy.deepcopy(next_pts)
                        # next_pts_shifted[:, 0] += int(canvas.shape[1] / 2)
                        #
                        # draw_num = 20
                        # if curr_pts.shape[0] < draw_num:
                        #     draw_num = curr_pts.shape[0]
                        # paint_inds = random.sample(range(curr_pts.shape[0]), draw_num)
                        #
                        # for pt_ind in range(len(paint_inds)):
                        #     curr_pt = curr_pts[paint_inds[pt_ind]]
                        #     next_pt = next_pts_shifted[paint_inds[pt_ind]]
                        #     # print(
                        #     #     '\n curr_pt: ({}, {}),  next_pt: ({}, {})'.format(
                        #     #         int(curr_pt[0]), int(curr_pt[1]), int(next_pt[0]), int(next_pt[1])
                        #     #     )
                        #     # )
                        #     line_color = COCO_CATEGORIES[np.random.randint(0, len(COCO_CATEGORIES))]
                        #     cv2.line(
                        #         canvas,
                        #         (int(curr_pt[0]), int(curr_pt[1])),
                        #         (int(next_pt[0]), int(next_pt[1])),
                        #         (line_color[0], line_color[1], line_color[2]),
                        #         2
                        #     )
                        #
                        # save_path = os.path.join(
                        #     save_path,
                        #     "ins_{}_frame_{}_{}.jpg".format(
                        #         gt_id,
                        #         targets_per_video["file_names"][match_ind].split("/")[-1].split(".")[0],
                        #         targets_per_video["file_names"][match_ind + 1].split("/")[-1].split(".")[0],
                        #     )
                        # )
                        # cv2.imwrite(save_path, canvas)


                    temp_pairs.append(temp_pairs_ins)

            gt_instances.append(
                {
                    "labels": gt_classes_per_video, "ids": gt_ids_per_video,
                    "box_masks": gt_boxmasks_per_video,
                    "masks": gt_masks_per_video,
                    "padded_input_height": h_pad,
                    "padded_input_width": w_pad,
                    "left_bounds": left_bounds_per_video[valid_idx].float(),
                    "right_bounds": right_bounds_per_video[valid_idx].float(),
                    "top_bounds": top_bounds_per_video[valid_idx].float(),
                    "bottom_bounds": bottom_bounds_per_video[valid_idx].float(),
                    "color_similarities": color_similarity_per_video,
                    "temporal_pairs": temp_pairs,
                    "total_temp_pair": num_match_per_video,
                    "pos_temp_pair": num_pos_match_per_video,
                }
            )
        return gt_instances

    def prepare_targets(self, targets, images):
        # images:
        h_pad, w_pad = images.tensor.shape[-2:]
        gt_instances = []
        for targets_per_video in targets:
            _num_instance = len(targets_per_video["instances"][0])
            mask_shape = [_num_instance, self.num_frames, h_pad, w_pad]

            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)

            gt_ids_per_video = []
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(self.device)
                h, w = targets_per_frame.image_size

                gt_ids_per_video.append(targets_per_frame.gt_ids[:, None])  # [(N, 1), (N, 1), ...]
                gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks.tensor

            gt_ids_per_video = torch.cat(gt_ids_per_video, dim=1)  # (N, num_frame)
            valid_idx = (gt_ids_per_video != -1).any(dim=-1)

            gt_classes_per_video = targets_per_frame.gt_classes[valid_idx]  # N,
            gt_ids_per_video = gt_ids_per_video[valid_idx]  # N, num_frames

            gt_instances.append({"labels": gt_classes_per_video, "ids": gt_ids_per_video})
            gt_masks_per_video = gt_masks_per_video[valid_idx].float()  # N, num_frames, H, W
            gt_instances[-1].update({"masks": gt_masks_per_video})
        return gt_instances
    
    def inference_video_offline(self, pred_cls, pred_masks, img_size, output_height, output_width):
        if len(pred_cls) > 0:
            scores = F.softmax(pred_cls, dim=-1)[:, :-1]
            labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
            # keep top-10 predictions
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(10, sorted=False)
            labels_per_image = labels[topk_indices]
            topk_indices = topk_indices // self.sem_seg_head.num_classes
            pred_masks = pred_masks[topk_indices]

            pred_masks = pred_masks[:, :, : img_size[0], : img_size[1]]
            pred_masks = F.interpolate(
                pred_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
            )

            masks = pred_masks > 0.

            out_scores = scores_per_image.tolist()  # [score] * topk
            out_labels = labels_per_image.tolist()  # [category] * topk
            out_masks = [m for m in masks.cpu()]  # [(T, H, W)] * topk

            # for i in range(len(out_scores)):
            #     score = out_scores[i]
            #     label = out_labels[i]
            #     seq_masks = out_masks[i]
            #
            #     pred_path = vid_path + 'pred_' + str(i) + '_score_' + str(score)[:4] + '/'
            #     os.makedirs(pred_path, exist_ok=True)
            #     for t in range(seq_masks.shape[0]):
            #         mask_frame = seq_masks[t, :, :].to(dtype=torch.uint8).numpy() * 255
            #         cv2.imwrite(pred_path + 'frame_' + str(t) + '.png', mask_frame)
        else:
            out_scores = []
            out_labels = []
            out_masks = []

        video_output = {
            "image_size": (output_height, output_width),
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
        }

        return video_output

    def run_window_inference(self, images_tensor):
        out_list = []
        start_idx_window, end_idx_window = 0, 0
        for i in range(len(images_tensor)):
            if i + self.num_frames_test > len(images_tensor):
                break

            if i + self.num_frames_test > end_idx_window:
                start_idx_window, end_idx_window = i, i + self.num_frames_window_test
                frame_idx_window = range(start_idx_window, end_idx_window)
                features_window = self.backbone(images_tensor[start_idx_window:end_idx_window])

            features = {k: v[frame_idx_window.index(i):frame_idx_window.index(i)+self.num_frames_test]
                        for k, v in features_window.items()}
            out = self.sem_seg_head(features)
            del out['aux_outputs']
            if self.merge_on_cpu:
                out = {k: v.cpu() for k, v in out.items()}
            out_list.append(out)

        outputs = {}
        # (vid_length - temp_stride + 1) x Q x Cls
        outputs['pred_logits'] = torch.cat([x['pred_logits'].float() for x in out_list]).detach()  # (V-t+1)xQxK
        # (vid_length - temp_stride + 1) x Q x T x H x W
        outputs['pred_masks'] = torch.cat([x['pred_masks'].float() for x in out_list]).detach()  # (V-t+1)xQxtxHxW
        # (vid_length - temp_stride + 1) x Q x D
        outputs['pred_embds'] = torch.cat([x['pred_embds'].float() for x in out_list]).detach()  # (V-t+1)xQxC
        return outputs

    def post_processing(self, outputs):
        n_clips, q, n_t, h, w = outputs['pred_masks'].shape
        pred_logits = list(torch.unbind(outputs['pred_logits']))  # (V-t+1) q c
        pred_masks = list(torch.unbind(outputs['pred_masks']))    # (V-t+1) q t h w
        pred_embds = list(torch.unbind(outputs['pred_embds']))    # (V-t+1) q d

        # 使用第一个 clip 的结果做 tracking 的初始化
        out_logits = [pred_logits[0]]
        out_masks = [pred_masks[0]]
        out_embds = [pred_embds[0]]
        for i in range(1, len(pred_logits)):
            mem_embds = torch.stack(out_embds[-2:]).mean(dim=0)  # [(q, c), (q, c)] -> (2, q, c) -> (q, c)
            indices = self.match_from_embds(mem_embds, pred_embds[i])
            out_logits.append(pred_logits[i][indices, :])
            out_masks.append(pred_masks[i][indices, :, :, :])
            out_embds.append(pred_embds[i][indices, :])

        out_logits = sum(out_logits) / len(out_logits)
        out_masks_mean = []
        for v in range(n_clips+n_t-1):
            n_t_valid = min(v+1, n_t)
            m = []
            for t in range(n_t_valid):
                if v-t < n_clips:
                    m.append(out_masks[v-t][:, t])  # q, h, w
            out_masks_mean.append(torch.stack(m).mean(dim=0))  # q, h, w

        outputs['pred_masks'] = torch.stack(out_masks_mean, dim=1)  # t * [q h w] -> q t h w
        outputs['pred_scores'] = F.softmax(out_logits, dim=-1)[:, :-1]  # q k+1
        return outputs

    def match_from_embds(self, tgt_embds, cur_embds):
        cur_embds = cur_embds / cur_embds.norm(dim=1)[:, None]
        tgt_embds = tgt_embds / tgt_embds.norm(dim=1)[:, None]
        cos_sim = torch.mm(cur_embds, tgt_embds.transpose(0, 1))

        cost_embd = 1 - cos_sim
        C = 1.0 * cost_embd
        C = C.cpu()

        indices = linear_sum_assignment(C.transpose(0, 1))  # target x current
        indices = indices[1]  # permutation that makes current aligns to target

        return indices
    
    def inference_video(self, batched_inputs, images, outputs):
        mask_scores = outputs["pred_scores"]  # cQ, K+1
        mask_pred = outputs["pred_masks"]  # cQ, V, H, W or [V/C * (cQ, C, H, W)]

        # upsample masks
        interim_size = images.tensor.shape[-2:]
        image_size = images.image_sizes[0]  # image size without padding after data augmentation
        out_height = batched_inputs[0].get("height", image_size[0])  # raw image size before data augmentation
        out_width = batched_inputs[0].get("width", image_size[1])

        num_topk = max(int(mask_scores.gt(0.05).sum()), self.test_topk_per_image)
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(
            self.num_queries, 1).flatten(0, 1)
        scores_per_video, topk_indices = mask_scores.flatten(0, 1).topk(num_topk, sorted=False)
        labels_per_video = labels[topk_indices]
        topk_indices = torch.div(topk_indices, self.sem_seg_head.num_classes, rounding_mode='floor')

        mask_pred = mask_pred[topk_indices]
        mask_pred = retry_if_cuda_oom(F.interpolate)(
            mask_pred,
            size=interim_size,
            mode="bilinear",
            align_corners=False,
        )  # cQ, t, H, W
        mask_pred = mask_pred[:, :, : image_size[0], : image_size[1]]

        mask_quality_scores = (mask_pred > 1).flatten(1).sum(1) / (mask_pred > -1).flatten(1).sum(1).clamp(min=1)
        scores_per_video = scores_per_video * mask_quality_scores

        masks_per_video = []
        for m in mask_pred:
            # slower speed but memory efficiently for long videos
            m = retry_if_cuda_oom(F.interpolate)(
                m.unsqueeze(0),
                size=(out_height, out_width),
                mode="bilinear",
                align_corners=False
            ).squeeze(0) > 0.
            masks_per_video.append(m.cpu())

        scores_per_video = scores_per_video.tolist()
        labels_per_video = labels_per_video.tolist()

        processed_results = {
            "image_size": (out_height, out_width),
            "pred_scores": scores_per_video,
            "pred_labels": labels_per_video,
            "pred_masks": masks_per_video,
        }

        return processed_results


def calculate_matching_pos(curr_kps, next_kps, curr_mask, next_mask):
    curr_kps = curr_kps.long()
    next_kps = next_kps.long()
    curr_label = curr_mask[curr_kps[:, 1], curr_kps[:, 0]]
    next_label = next_mask[next_kps[:, 1], next_kps[:, 0]]
    pos_match = ((curr_label - next_label) == 0).float().sum()
    return pos_match


COCO_CATEGORIES = [
    [220, 20, 60],
    [119, 11, 32],
    [0, 0, 142],
    [0, 0, 230],
    [106, 0, 228],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 70],
    [0, 0, 192],
    [250, 170, 30],
    [100, 170, 30],
    [220, 220, 0],
    [175, 116, 175],
    [250, 0, 30],
    [165, 42, 42],
    [255, 77, 255],
    [0, 226, 252],
    [182, 182, 255],
    [0, 82, 0],
    [120, 166, 157],
    [110, 76, 0],
    [174, 57, 255],
    [199, 100, 0],
    [72, 0, 118],
    [255, 179, 240],
    [0, 125, 92],
    [209, 0, 151],
    [188, 208, 182],
    [0, 220, 176],
    [255, 99, 164],
    [92, 0, 73],
    [133, 129, 255],
    [78, 180, 255],
    [0, 228, 0],
    [174, 255, 243],
    [45, 89, 255],
    [134, 134, 103],
    [145, 148, 174],
    [255, 208, 186],
    [197, 226, 255],
    [171, 134, 1],
    [109, 63, 54],
    [207, 138, 255],
    [151, 0, 95],
    [9, 80, 61],
    [84, 105, 51],
    [74, 65, 105],
    [166, 196, 102],
    [208, 195, 210],
    [255, 109, 65],
    [0, 143, 149],
    [179, 0, 194],
    [209, 99, 106],
    [5, 121, 0],
    [227, 255, 205],
    [147, 186, 208],
    [153, 69, 1],
    [3, 95, 161],
    [163, 255, 0],
    [119, 0, 170],
    [0, 182, 199],
    [0, 165, 120],
    [183, 130, 88],
    [95, 32, 0],
    [130, 114, 135],
    [110, 129, 133],
    [166, 74, 118],
    [219, 142, 185],
    [79, 210, 114],
    [178, 90, 62],
    [65, 70, 15],
    [127, 167, 115],
    [59, 105, 106],
    [142, 108, 45],
    [196, 172, 0],
    [95, 54, 80],
    [128, 76, 255],
    [201, 57, 1],
    [246, 0, 122],
    [191, 162, 208],
    [255, 255, 128],
    [147, 211, 203],
    [150, 100, 100],
    [168, 171, 172],
    [146, 112, 198],
    [210, 170, 100],
    [92, 136, 89],
    [218, 88, 184],
    [241, 129, 0],
    [217, 17, 255],
    [124, 74, 181],
    [70, 70, 70],
    [255, 228, 255],
    [154, 208, 0],
    [193, 0, 92],
    [76, 91, 113],
    [255, 180, 195],
    [106, 154, 176],
    [230, 150, 140],
    [60, 143, 255],
    [128, 64, 128],
    [92, 82, 55],
    [254, 212, 124],
    [73, 77, 174],
    [255, 160, 98],
    [255, 255, 255],
    [104, 84, 109],
    [169, 164, 131],
    [225, 199, 255],
    [137, 54, 74],
    [135, 158, 223],
    [7, 246, 231],
    [107, 255, 200],
    [58, 41, 149],
    [183, 121, 142],
    [255, 73, 97],
    [107, 142, 35],
    [190, 153, 153],
    [146, 139, 141],
    [70, 130, 180],
    [134, 199, 156],
    [209, 226, 140],
    [96, 36, 108],
    [96, 96, 96],
    [64, 170, 64],
    [152, 251, 152],
    [208, 229, 228],
    [206, 186, 171],
    [152, 161, 64],
    [116, 112, 0],
    [0, 114, 143],
    [102, 102, 156],
    [250, 141, 255],
]