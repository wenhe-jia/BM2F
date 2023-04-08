# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks

from .modeling.criterion import VideoSetCriterionProj, VideoSetCriterionProjPair
from .modeling.matcher import VideoHungarianMatcherProj, VideoHungarianMatcherProjPair
from .utils.memory import retry_if_cuda_oom

from .utils.weaksup_utils import unfold_wo_center, get_images_color_similarity

import os, cv2
import numpy as np
from skimage import color
import copy


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
        pairwise_dilation
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

        self.output_dir = output_dir

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT
        weak_supervision = False

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
                update_mask=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.ENABLED,
                mask_update_steps=[
                    int(x * cfg.SOLVER.MAX_ITER)
                    for x in cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.STEPS
                ],
                update_pix_thrs=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.PIX_THRS,
                conf_type=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.CONF_TYPE,
                static_conf_thr=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.STATIC_CONF_THR,
                max_iter=cfg.SOLVER.MAX_ITER
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
                update_mask=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.ENABLED,
                mask_update_steps=[
                    int(x * cfg.SOLVER.MAX_ITER)
                    for x in cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.STEPS
                ],
                update_pix_thrs=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.PIX_THRS,
                conf_type=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.CONF_TYPE,
                static_conf_thr=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_UPDATE.STATIC_CONF_THR,
                max_iter=cfg.SOLVER.MAX_ITER
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
            "pairwise_dilation": cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE.DILATION
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
            return losses
        else:
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

            return retry_if_cuda_oom(self.inference_video)(mask_cls_result, mask_pred_result, image_size, height, width, vid_path)

    def prepare_weaksup_targets(self, targets, org_images):
        """
        :param targets: [vid1[frame1, frame2], vid2[frame1, frame2]]
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

        gt_instances = []
        for vid_ind, targets_per_video in enumerate(targets):
            _num_instance = len(targets_per_video["instances"][0])

            mask_shape = [_num_instance, T, h_pad, w_pad]
            gt_boxmasks_full_per_video = torch.zeros(mask_shape, dtype=torch.float32, device=self.device)

            x_bound_shape = [_num_instance, T, h_pad]
            left_bounds_full_per_video = torch.zeros(x_bound_shape, dtype=torch.float32, device=self.device)
            right_bounds_full_per_video = torch.zeros(x_bound_shape, dtype=torch.float32, device=self.device)

            y_bound_shape = [_num_instance, T, w_pad]
            top_bounds_full_per_video = torch.zeros(y_bound_shape, dtype=torch.float32, device=self.device)
            bottom_bounds_full_per_video = torch.zeros(y_bound_shape, dtype=torch.float32, device=self.device)

            color_similarity_shape = [_num_instance, T, self.pairwise_size * self.pairwise_size - 1, _h, _w]
            color_similarity_per_video = torch.zeros(color_similarity_shape, dtype=torch.float32, device=self.device)

            # rectangle gt mask from boxes for mask projection loss
            # TODO: add images_color_similarity for pairwise loss
            gt_ids_per_video = []
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(self.device)
                gt_ids_per_video.append(targets_per_frame.gt_ids[:, None])  # [(num_ins, 1), (num_ins, 1)]

                # color similarity
                # (H/4, W/4, 3)
                frame_lab = color.rgb2lab(downsampled_images[vid_ind, f_i].byte().permute(1, 2, 0).cpu().numpy())
                frame_lab = torch.as_tensor(frame_lab, device=downsampled_images.device, dtype=torch.float32)
                frame_lab = frame_lab.permute(2, 0, 1)[None]  # (1, 3, H/4, W/4)
                frame_color_similarity = get_images_color_similarity(
                    frame_lab, downsampled_image_masks[vid_ind, f_i],
                    self.pairwise_size, self.pairwise_dilation
                )  # (1, k*k-1, H/4, W/4)

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

            # (G, T, h_pad/4, w_pad/4)
            gt_boxmasks_per_video = gt_boxmasks_full_per_video[:, :, start::stride, start::stride]
            left_bounds_per_video = left_bounds_full_per_video[:, :, start::stride] / stride
            right_bounds_per_video = right_bounds_full_per_video[:, :, start::stride] / stride
            top_bounds_per_video = top_bounds_full_per_video[:, :, start::stride] / stride
            bottom_bounds_per_video = bottom_bounds_full_per_video[:, :, start::stride] / stride

            gt_ids_per_video = torch.cat(gt_ids_per_video, dim=1)  # (N, num_frame)
            valid_idx = (gt_ids_per_video != -1).any(dim=-1)  # (num_ins,), 别取到再所有帧上都是空的gt

            gt_classes_per_video = targets_per_frame.gt_classes[valid_idx]  # N,
            gt_ids_per_video = gt_ids_per_video[valid_idx]  # N, num_frames

            gt_instances.append(
                {
                    "labels": gt_classes_per_video, "ids": gt_ids_per_video,
                    "box_masks": gt_boxmasks_per_video[valid_idx].float(),
                    "left_bounds": left_bounds_per_video[valid_idx].float(),
                    "right_bounds": right_bounds_per_video[valid_idx].float(),
                    "top_bounds": top_bounds_per_video[valid_idx].float(),
                    "bottom_bounds": bottom_bounds_per_video[valid_idx].float(),
                    "color_similarities": color_similarity_per_video[valid_idx].float(),
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

    def inference_video(self, pred_cls, pred_masks, img_size, output_height, output_width, vid_path):
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
