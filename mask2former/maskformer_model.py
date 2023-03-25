# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import cv2
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.structures.masks import BitMasks

from .modeling.criterion import SetCriterion, SetCriterionProjection, SetCriterionWeakSup
from .modeling.matcher import HungarianMatcherProjMask, HungarianMatcherWeakSup
from .utils.weaksup_utils import unfold_wo_center, get_images_color_similarity

from skimage import color
import copy


@META_ARCH_REGISTRY.register()
class MaskFormer(nn.Module):
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
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        # pairwise
        pairwise_size: int,
        pairwise_dilation: int
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

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        self.weak_supervision = weak_supervision
        # TODO: set follows configurable
        self.bottom_pixels_removed = 10
        self.pairwise_size = pairwise_size
        self.pairwise_dilation = pairwise_dilation
        self.mask_out_stride = 4

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # define supervision type
        weak_supervision = cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.ENABLED
        matcher_target_type = cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.MATCHER_TYPE
        mask_target_type = cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_TARGET_TYPE

        if weak_supervision:
            if mask_target_type == "projection_mask":
                weight_dict = {
                    "loss_ce": cfg.MODEL.MASK_FORMER.CLASS_WEIGHT,
                    "loss_mask_projection": cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_PROJECTION_WEIGHT
                }
            elif mask_target_type == "projection_and_pairwise":
                weight_dict = {
                    "loss_ce": cfg.MODEL.MASK_FORMER.CLASS_WEIGHT,
                    "loss_mask_projection": cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_PROJECTION_WEIGHT,
                    "loss_pairwise": cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE_WEIGHT
                }
            else:
                raise Exception("Unknown mask_target_type type !!!")
        else:
            weight_dict = {
                "loss_ce": cfg.MODEL.MASK_FORMER.CLASS_WEIGHT,
                "loss_mask": cfg.MODEL.MASK_FORMER.MASK_WEIGHT,
                "loss_dice": cfg.MODEL.MASK_FORMER.DICE_WEIGHT
            }

        # build matcher
        if matcher_target_type == "mask":
            matcher = HungarianMatcherMask(
                cost_class=cfg.MODEL.MASK_FORMER.CLASS_WEIGHT,
                cost_mask=cfg.MODEL.MASK_FORMER.MASK_WEIGHT,
                cost_dice=cfg.MODEL.MASK_FORMER.DICE_WEIGHT,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
                target_type=matcher_target_type
            )
        elif matcher_target_type == "projection_mask":
            matcher = HungarianMatcherProjMask(
                cost_class=cfg.MODEL.MASK_FORMER.CLASS_WEIGHT,
                cost_projection=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_PROJECTION_WEIGHT,
            )
        elif mask_target_type == "projection_and_pairwise":
            matcher = HungarianMatcherWeakSup(
                cost_class=cfg.MODEL.MASK_FORMER.CLASS_WEIGHT,
                cost_projection=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.MASK_PROJECTION_WEIGHT,
                cost_pairwise=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE_WEIGHT,
                pairwise_size=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE.SIZE,
                pairwise_dilation=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE.DILATION,
                pairwise_color_thresh=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE.COLOR_THRESH,
                pairwise_warmup_iters=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE.WARMUP_ITERS,
            )
        else:
            raise Exception("Unknown Matcher type !!!")

        # set loss weight dict
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        # build criterion
        if weak_supervision:
            if mask_target_type == "projection_mask":
                losses = ["labels", "projection_masks"]
                criterion = SetCriterionProjection(
                    sem_seg_head.num_classes,
                    matcher=matcher,
                    weight_dict=weight_dict,
                    eos_coef=no_object_weight,
                    losses=losses,
                )
            elif mask_target_type == "projection_and_pairwise":
                # losses = ["labels", "projection_masks", "pairwise"]
                losses = ["labels", "pairwise"]
                criterion = SetCriterionWeakSup(
                    sem_seg_head.num_classes,
                    matcher=matcher,
                    weight_dict=weight_dict,
                    eos_coef=no_object_weight,
                    pairwise_size=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE.SIZE,
                    pairwise_dilation=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE.DILATION,
                    pairwise_color_thresh=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE.COLOR_THRESH,
                    pairwise_warmup_iters=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE.WARMUP_ITERS,
                    losses=losses,
                    point_sample=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE.POINT_SAMPLE,
                    num_points=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE.TRAIN_NUM_POINTS,
                    oversample_ratio=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE.OVERSAMPLE_RATIO,
                    importance_sample_ratio=cfg.MODEL.MASK_FORMER.WEAK_SUPERVISION.PAIRWISE.IMPORTANCE_SAMPLE_RATIO,
                )
            else:
                raise Exception("Unknown mask_target_type type !!!")
        else:
            losses = ["labels", "masks"]
            criterion = SetCriterion(
                sem_seg_head.num_classes,
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=no_object_weight,
                losses=losses,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
                oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
                importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
                mask_target_type="mask",
            )

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
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
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
                   * "image": Tensor, image in (C, H, W) format.
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
        # print('='*10,'Start','='*10)
        # m1 = torch.cuda.memory_allocated()
        # print('MODEL start memory cost:', m1 / (1024 * 1024), 'MB')
        # mmax = torch.cuda.max_memory_allocated()
        # print('MODEL start max memory cost:', mmax / (1024 * 1024), 'MB')

        original_images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in original_images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        # cv2.imwrite('debug/model/img.png', batched_inputs[0]['image'].numpy().transpose(1, 2, 0))

        # m2 = torch.cuda.memory_allocated()
        features = self.backbone(images.tensor)
        # m3 = torch.cuda.memory_allocated()
        # print('features memory cost:', (m3-m2) / (1024 * 1024), 'MB')
        # mmax = torch.cuda.max_memory_allocated()
        # print('features max memory cost:', mmax / (1024 * 1024), 'MB')
        outputs = self.sem_seg_head(features)
        # m4 = torch.cuda.memory_allocated()
        # print('outputs memory cost:', (m4-m3) / (1024 * 1024), 'MB')
        # mmax = torch.cuda.max_memory_allocated()
        # print('outputs max memory cost:', mmax / (1024 * 1024), 'MB')

        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                if self.weak_supervision:
                    image_heights = [x["height"] for x in batched_inputs]
                    targets = self.prepare_weaksup_targets(gt_instances, original_images, image_heights)
                else:
                    targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            # mend = torch.cuda.memory_allocated()
            # print('MODEL end memory cost:', mend / (1024 * 1024), 'MB')
            # mmax = torch.cuda.max_memory_allocated()
            # print('MODEL end max memory cost:', mmax / (1024 * 1024), 'MB\n')
            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r
                
                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["instances"] = instance_r

            return processed_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device
            )
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks

            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )

        return new_targets

    def prepare_weaksup_targets(self, targets, org_images, img_heights):
        # 使用没有经过ImageList之前的图像是为了把每张mask底部的若干像素置为0
        # masks of org image shape
        # org_image_masks = [torch.ones_like(x[0], dtype=torch.float32) for x in org_images]
        # # mask out the bottom area where the COCO dataset probably has wrong annotations
        # for i in range(len(org_image_masks)):
        #     im_h = img_heights[i]
        #     pixels_removed = int(
        #         self.bottom_pixels_removed *
        #         float(org_images[i].size(1)) / float(im_h)
        #     )
        #     if pixels_removed > 0:
        #         org_image_masks[i][-pixels_removed:, :] = 0

        # calculate color similarity, pad org images which is not normalized by pix mean and std
        original_images = ImageList.from_tensors(org_images, self.size_divisibility).tensor
        # original_image_masks = ImageList.from_tensors(org_image_masks, self.size_divisibility, pad_value=0.0).tensor

        stride = self.mask_out_stride  # 4
        start = int(stride // 2)
        assert original_images.size(2) % stride == 0
        assert original_images.size(3) % stride == 0
        # down sample org image and masks(of torch.ones)
        # downsampled_images = F.avg_pool2d(
        #     original_images.float(), kernel_size=stride,
        #     stride=stride, padding=0
        # )[:, [2, 1, 0]]  # (N, C, H, W) --> (N, C, H/4, W/4) --> (N, W/4, H/4, C)
        # downsampled_image_masks = original_image_masks[:, start::stride, start::stride]  # (N, H/4, W/4), do not use interpolate to ensure org pixel

        h_pad, w_pad = original_images.shape[-2:]  ###
        new_targets = []  # store targets of each image
        for im_ind, targets_per_image in enumerate(targets):
            # images_lab = color.rgb2lab(downsampled_images[im_ind].byte().permute(1, 2, 0).cpu().numpy()[:, :, ::-1])  # (H/4, W/4, 3)
            # images_lab = torch.as_tensor(images_lab, device=downsampled_images.device, dtype=torch.float32)
            # images_lab = images_lab.permute(2, 0, 1)[None]  # (1, 3, H/4, W/4)
            # images_color_similarity = get_images_color_similarity(
            #     images_lab, downsampled_image_masks[im_ind],
            #     self.pairwise_size, self.pairwise_dilation
            # )  # (1, k*k-1, H/4, W/4)

            ###### for progressive projection upper bounds with gt_mask #####
            # gt_masks = targets_per_image.gt_masks
            # padded_masks = torch.zeros(
            #     (gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device
            # )
            # padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks

            # gt box mask
            gt_boxes = targets_per_image.gt_boxes.tensor  # (N, 4)
            box_masks_full_per_image = torch.zeros(
                (gt_boxes.shape[0], h_pad, w_pad), dtype=torch.float32, device=self.device
            )
            left_bounds_full_per_image = torch.zeros(
                (gt_boxes.shape[0], h_pad), dtype=torch.float32, device=self.device
            )
            right_bounds_full_per_image = torch.zeros(
                (gt_boxes.shape[0], h_pad), dtype=torch.float32, device=self.device
            )
            top_bounds_full_per_image = torch.zeros(
                (gt_boxes.shape[0], w_pad), dtype=torch.float32, device=self.device
            )
            bottom_bounds_full_per_image = torch.zeros(
                (gt_boxes.shape[0], w_pad), dtype=torch.float32, device=self.device
            )

            if gt_boxes.shape[0] > 0:
                for ins_i, gt_box in enumerate(gt_boxes.split(1)):
                    gt_box = gt_box.squeeze()
                    box_masks_full_per_image[
                        ins_i,
                        int(gt_box[1]):int(gt_box[3]) + 1,
                        int(gt_box[0]):int(gt_box[2]) + 1
                    ] = 1.0

                    # left_bounds_full_per_image[ins_i, :int(gt_box[1])] = 0.
                    # left_bounds_full_per_image[ins_i, int(gt_box[1]):int(gt_box[3] + 1)] = int(gt_box[0])
                    # left_bounds_full_per_image[ins_i, int(gt_box[3] + 1):] = 0.
                    # 
                    # right_bounds_full_per_image[ins_i, :int(gt_box[1])] = w_pad
                    # right_bounds_full_per_image[ins_i, int(gt_box[1]):int(gt_box[3] + 1)] = int(gt_box[2] + 1)
                    # right_bounds_full_per_image[ins_i, int(gt_box[3] + 1):] = w_pad
                    # 
                    # top_bounds_full_per_image[ins_i, :int(gt_box[0])] = 0.
                    # top_bounds_full_per_image[ins_i, int(gt_box[0]):int(gt_box[2] + 1)] = int(gt_box[1])
                    # top_bounds_full_per_image[ins_i, int(gt_box[2] + 1):] = 0.
                    # 
                    # bottom_bounds_full_per_image[ins_i, :int(gt_box[0])] = h_pad
                    # bottom_bounds_full_per_image[ins_i, int(gt_box[0]):int(gt_box[2] + 1)] = int(gt_box[3] + 1)
                    # bottom_bounds_full_per_image[ins_i, int(gt_box[2] + 1):] = h_pad

                    ###### for progressive projection upper bounds with gt_mask #####
                    gt_mask = box_masks_full_per_image[ins_i].int()  # (H, W)
                    # bounds for y projection
                    left_bounds_full_per_image[ins_i] = torch.argmax(gt_mask, dim=1)
                    right_bounds_full_per_image[ins_i] = gt_mask.shape[1] - \
                                                         torch.argmax(gt_mask.flip(1), dim=1)
                    # bounds for x projection
                    top_bounds_full_per_image[ins_i] = torch.argmax(gt_mask, dim=0)
                    bottom_bounds_full_per_image[ins_i] = gt_mask.shape[0] - \
                                                          torch.argmax(gt_mask.flip(0), dim=0)

            box_masks_per_image = box_masks_full_per_image[:, start::stride, start::stride]
            left_bounds_per_image = left_bounds_full_per_image[:, start::stride] / stride
            right_bounds_per_image = right_bounds_full_per_image[:, start::stride] / stride
            top_bounds_per_image = top_bounds_full_per_image[:, start::stride] / stride
            bottom_bounds_per_image = bottom_bounds_full_per_image[:, start::stride] / stride

            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    # "masks": padded_masks,
                    "box_masks": box_masks_per_image,
                    # "images_color_similarity": torch.cat(
                    #     [images_color_similarity for _ in range(len(targets_per_image))], dim=0
                    # )  # (image_gt_num, k*k-1, H/4, W/4)
                    "left_bounds": left_bounds_per_image.float(),
                    "right_bounds": right_bounds_per_image.float(),
                    "top_bounds": top_bounds_per_image.float(),
                    "bottom_bounds": bottom_bounds_per_image.float(),
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.sem_seg_head.num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # bbox_pred = BitMasks(mask_pred).get_bounding_boxes().tensor
        # for ins_idx in range(mask_pred.shape[0]):
        #     ins_mask = mask_pred[ins_idx, :, :]
        #     ins_bbox = bbox_pred[ins_idx, :]
        #
        #     save_mask = ((ins_mask > 0).int() * 255).cpu().numpy()
        #     cv2.rectangle(
        #         save_mask,
        #         (int(ins_bbox[0].cpu()), int(ins_bbox[1].cpu())),
        #         (int(ins_bbox[2].cpu()), int(ins_bbox[3].cpu())),
        #         (0, 0, 255)
        #     )
        #     cv2.imwrite('debug/infer-15k/pred_ins_{}.png'.format(ins_idx), save_mask)


        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result