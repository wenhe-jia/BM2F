# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from detectron2.structures.masks import BitMasks

from mask2former.utils.misc import is_dist_avail_and_initialized

from ..utils.weaksup_utils import unfold_wo_center

import os, cv2, copy
import numpy as np


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example. (num_ins*T, num_point)
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class). (num_ins*T, num_point)
        num_ins = num_pred = num_gt
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)  # (num_ins,)
    denominator = inputs.sum(-1) + targets.sum(-1)  # (num_ins,)
    loss = 1 - (numerator + 1) / (denominator + 1)  # (num_ins,)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def projection3D_dice_loss(
        inputs_x: torch.Tensor,
        targets_x: torch.Tensor,
        inputs_y: torch.Tensor,
        targets_y: torch.Tensor,
        inputs_t: torch.Tensor,
        targets_t: torch.Tensor,
        num_masks: float,
    ):
    """
    :param inputs_x: (num_ins, T*H_pad/4)
    :param targets_x: (num_ins, T*H_pad/4)
    :param inputs_y: (num_ins, T*W_pad/4)
    :param targets_y: (num_ins, T*W_pad/4)
    :param inputs_y: (num_ins, H_pad/4 * W_pad/4)
    :param targets_y: (num_ins, H_pad/4 * W_pad/4)
    :param num_masks: int
    :return: loss: float tensor
    """
    assert inputs_x.shape[0] == targets_x.shape[0]
    assert inputs_y.shape[0] == targets_y.shape[0]
    assert inputs_t.shape[0] == targets_t.shape[0]
    eps = 1e-5

    # calaulate x axis projection loss
    intersection_x = (inputs_x * targets_x).sum(dim=1)  # (num_ins*T,)
    union_x = (inputs_x ** 2.0).sum(dim=1) + (targets_x ** 2.0).sum(dim=1) + eps  # (num_ins*T,)
    loss_x = 1. - (2 * intersection_x / union_x)  # (num_ins*T,)

    # calaulate y axis projection loss
    intersection_y = (inputs_y * targets_y).sum(dim=1)  # (num_ins*T,)
    union_y = (inputs_y ** 2.0).sum(dim=1) + (targets_y ** 2.0).sum(dim=1) + eps  # (num_ins*T,)
    loss_y = 1. - (2 * intersection_y / union_y)  # (num_ins*T,)

    # # calaulate t axis projection loss
    intersection_t = (inputs_t * targets_t).sum(dim=1)  # (num_ins*T,)
    union_t = (inputs_t ** 2.0).sum(dim=1) + (targets_t ** 2.0).sum(dim=1) + eps  # (num_ins*T,)
    loss_t = 1. - (2 * intersection_t / union_t)  # (num_ins*T,)

    return (loss_x + loss_y + loss_t).sum() / num_masks


projection3D_dice_loss_jit = torch.jit.script(
    projection3D_dice_loss
)  # type: torch.jit.ScriptModule

def projection2D_dice_loss(
        inputs_x: torch.Tensor,
        targets_x: torch.Tensor,
        inputs_y: torch.Tensor,
        targets_y: torch.Tensor,
        num_masks: float,
):
    assert inputs_x.shape[0] == targets_x.shape[0]
    assert inputs_y.shape[0] == targets_y.shape[0]
    eps = 1e-5

    # calaulate x axis projection loss
    intersection_x = (inputs_x * targets_x).sum(dim=1)  # (num_ins*T,)
    union_x = (inputs_x ** 2.0).sum(dim=1) + (targets_x ** 2.0).sum(dim=1) + eps  # (num_ins*T,)
    loss_x = 1. - (2 * intersection_x / union_x)  # (num_ins*T,)

    # calaulate y axis projection loss
    intersection_y = (inputs_y * targets_y).sum(dim=1)  # (num_ins*T,)
    union_y = (inputs_y ** 2.0).sum(dim=1) + (targets_y ** 2.0).sum(dim=1) + eps  # (num_ins*T,)
    loss_y = 1. - (2 * intersection_y / union_y)  # (num_ins*T,)

    return (loss_x + loss_y).sum() / num_masks


projection2D_dice_loss_jit = torch.jit.script(
    projection2D_dice_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


def pairwise_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
):
    assert len(inputs.shape) == 5
    assert len(targets.shape) == 5

    # (N, T, k*k-1, H, W) --flatten--> (N, T, E) --sum--> (N, T)
    numerator = (inputs.flatten(2) * targets.flatten(2)).sum(dim=2)
    denominator = targets.flatten(2).sum(dim=2).clamp(min=1.0)

    # (N, T) --mean(dim=1)--> (N,) --sum--> ()
    loss_pairwise = (numerator / denominator).mean(1).sum()  # (N, T)

    # _loss_pairwise = (inputs * targets).sum() / targets.sum().clamp(min=1.0)
    return loss_pairwise / num_masks


pairwise_loss_jit = torch.jit.script(
    pairwise_loss
)  # type: torch.jit.ScriptModule


def calculate_pred_similaries_video(pred_mask, kernel_size, dilation):
    # pred_mask: (N, T, H, W)
    # Prepare data for pairwise loss
    log_fg_prob = F.logsigmoid(pred_mask)
    log_bg_prob = F.logsigmoid(-pred_mask)

    # 此处取出的值是每个点周围k*k个点的概率（前/背景）
    log_fg_prob_unfold = unfold_wo_center(
        log_fg_prob, kernel_size=kernel_size, dilation=dilation
    )  # (N, T, k*k-1, H/4, W/4)
    log_bg_prob_unfold = unfold_wo_center(
        log_bg_prob, kernel_size=kernel_size, dilation=dilation
    )  # (N, T, k*k-1, H/4, W/4)

    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    # (N, T, H, W) -> (N, T, 1, H, W), (N, T, 1, H, W) + (N, T, k*k-1, H, W) -> (N, T, k*k-1, H, W)
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold  # (N, T, k*k-1, H/4, W/4)
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold  # (N, T, k*k-1, H/4, W/4)

    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)  # (N, T, k*k-1, H, W)
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_  # (N, T, k*k-1, H, W)

    return -log_same_prob


def temporal_pairwise_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
):
    assert inputs.shape == targets.shape
    return inputs.sum() / targets.sum().clamp(min=1.0)


temporal_pairwise_loss_jit = torch.jit.script(
    temporal_pairwise_loss
)  # type: torch.jit.ScriptModule


def calculate_temp_similarities(mask_curr, mask_next, pts_curr, pts_next):
    """
        Args:
        mask_curr (Tensor): (H, W)
        mask_next (Tensor): (H, W)
        pts_curr (Tensor): (K, 2)
        pts_next (Tensor): (K, 2)

        pts in XY format
    Returns:
        temp_similarities (Tensor): (K)
    """
    assert pts_curr.shape == pts_next.shape

    preds_curr = mask_curr[pts_curr[:, 1], pts_curr[:, 0]]  # (K)
    preds_next = mask_next[pts_next[:, 1], pts_next[:, 0]]  # (K)

    # (K)
    log_fg_prob_curr = F.logsigmoid(preds_curr)
    log_fg_prob_next = F.logsigmoid(preds_next)
    log_bg_prob_curr = F.logsigmoid(-preds_curr)
    log_bg_prob_next = F.logsigmoid(-preds_next)

    log_same_fg_prob = log_fg_prob_curr + log_fg_prob_next
    log_same_bg_prob = log_bg_prob_curr + log_bg_prob_next

    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) + torch.exp(log_same_bg_prob - max_)
    ) + max_

    return -log_same_prob  # (K)


class VideoSetCriterionProjSTPair(nn.Module):
    """This class computes the loss for DETR.
            The process happens in two steps:
                1) we compute hungarian assignment between ground truth boxes and the outputs of the model
                2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
            """

    def __init__(
            self,
            num_classes,
            matcher,
            weight_dict,
            eos_coef,
            pairwise_size,
            pairwise_dilation,
            pairwise_color_thresh,
            pairwise_warmup_iters,
            temporal_warmup,
            use_input_resolution_for_temp,
            losses,
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.pairwise_size = pairwise_size
        self.pairwise_dilation = pairwise_dilation
        self.pairwise_color_thresh = pairwise_color_thresh
        self.pairwise_warmup_iters = pairwise_warmup_iters
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # for pairwise loss warmup
        self.register_buffer("_iter", torch.zeros([1]))

        self.temporal_warmup = temporal_warmup
        self.use_input_resolution_for_temp = use_input_resolution_for_temp

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_projection_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the 1D projection loss and the pairwise loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx].sigmoid()  # need to sigmoid

        # Modified to handle video
        target_boxmasks = torch.cat(
            [t['box_masks'][i] for t, (_, i) in zip(targets, indices)]
        ).to(src_masks)
        target_left_bounds = torch.cat(
            [t['left_bounds'][i] for t, (_, i) in zip(targets, indices)]
        ).to(src_masks)
        target_right_bounds = torch.cat(
            [t['right_bounds'][i] for t, (_, i) in zip(targets, indices)]
        ).to(src_masks)
        target_top_bounds = torch.cat(
            [t['top_bounds'][i] for t, (_, i) in zip(targets, indices)]
        ).to(src_masks)
        target_bottom_bounds = torch.cat(
            [t['bottom_bounds'][i] for t, (_, i) in zip(targets, indices)]
        ).to(src_masks)

        # (N, T, H_pad/4, W_pad/4) -> (NT, 1, H_pad/4, W_pad/4)
        src_masks = src_masks.flatten(0, 1)[:, None]
        target_boxmasks = target_boxmasks.flatten(0, 1)[:, None]
        target_left_bounds = target_left_bounds.flatten()  # (NTH)
        target_right_bounds = target_right_bounds.flatten()  # (NTH)
        target_top_bounds = target_top_bounds.flatten()  # (NTW)
        target_bottom_bounds = target_bottom_bounds.flatten()  # (NTW)

        if src_idx[0].shape[0] > 0:
            """
                bellow is for original projection loss
            """
            # src_masks_y = src_masks.max(dim=3, keepdim=True)[0].flatten(1)
            # src_masks_x = src_masks.max(dim=2, keepdim=True)[0].flatten(1)
            #
            # with torch.no_grad():
            #     target_boxmasks_y = target_boxmasks.max(dim=3, keepdim=True)[0].flatten(1)
            #     target_boxmasks_x = target_boxmasks.max(dim=2, keepdim=True)[0].flatten(1)

            """
                bellow is for projection limited label loss
            """
            NT = src_masks.shape[0]
            H = src_masks.shape[2]
            W = src_masks.shape[3]

            src_masks_y, max_inds_x = src_masks.max(dim=3, keepdim=True)  # (NT, 1, H, 1), (NT, 1, H, 1)
            src_masks_x, max_inds_y = src_masks.max(dim=2, keepdim=True)  # (NT, 1, 1, W), (NT, 1, 1, W)

            src_masks_y = src_masks_y.flatten(1)  # (NT, H)
            src_masks_x = src_masks_x.flatten(1)  # (NT, W)
            max_inds_x = max_inds_x.flatten()  # (NTW)
            max_inds_y = max_inds_y.flatten()  # (NTH)

            with torch.no_grad():
                flag_l = max_inds_x >= target_left_bounds
                flag_r = max_inds_x < target_right_bounds
                flag_y = (flag_l * flag_r).view(NT, H)

                flag_t = max_inds_y >= target_top_bounds
                flag_b = max_inds_y < target_bottom_bounds
                flag_x = (flag_t * flag_b).view(NT, W)

                target_boxmasks_y = target_boxmasks.max(dim=3, keepdim=True)[0].flatten(1) * flag_y  # (NT, W)
                target_boxmasks_x = target_boxmasks.max(dim=2, keepdim=True)[0].flatten(1) * flag_x  # (NT, H)

            losses = {
                "loss_mask_projection": projection2D_dice_loss_jit(
                    src_masks_x, target_boxmasks_x,
                    src_masks_y, target_boxmasks_y,
                    num_masks
                )
            }

            del src_masks_x, src_masks_y
            del target_boxmasks_x, target_boxmasks_y
        else:
            """
            bellow is for original projection loss and limited projection loss
            """
            losses = {"loss_mask_projection": torch.tensor([0], dtype=torch.float32, device=src_masks.device)}

        del src_masks
        del target_boxmasks
        return losses

    def loss_spatial_pairwise(self, outputs, targets, indices, num_masks):
        """
            Compute the losses related to the masks: the 1D projection loss.
            targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]

        # Modified to handle video
        target_boxmasks = torch.cat(
            [t['box_masks'][i] for t, (_, i) in zip(targets, indices)]
        ).to(src_masks)  # (N, T, H, W)

        target_similarities = torch.cat(
            [t['color_similarities'][i] for t, (_, i) in zip(targets, indices)]
        )  # (N, T, k*k-1, H, W)

        if src_idx[0].shape[0] > 0:
            with torch.no_grad():
                target_similarities = (target_similarities >= self.pairwise_color_thresh).float() * \
                                      target_boxmasks[:, :, None].float()

            src_similarities = calculate_pred_similaries_video(src_masks, self.pairwise_size, self.pairwise_dilation)

            warmup_factor = min(self._iter.item() / float(self.pairwise_warmup_iters), 1.0)
            losses = {
                "loss_mask_pairwise":
                    pairwise_loss_jit(src_similarities, target_similarities, num_masks) * warmup_factor
            }

            del src_similarities
        else:
            losses = {"loss_mask_pairwise": torch.tensor([0], dtype=torch.float32, device=src_masks.device)}

        del src_masks
        del target_boxmasks
        del target_similarities
        return losses

    def loss_temporal_pairwise(self, outputs, targets, indices, num_masks):
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]  # (N, T, H, W)

        if self.use_input_resolution_for_temp:
            padded_input_h = targets[0]["padded_input_height"]
            padded_input_w = targets[0]["padded_input_width"]
            src_masks = F.interpolate(
                src_masks,
                size=(padded_input_h, padded_input_w),
                mode='bilinear', align_corners=False
            )

        target_pairs = []
        for batch_ind, target in enumerate(targets):
            tgt_pairs_src = target['temporal_pairs']
            batch_tgt_indices = indices[batch_ind][1].tolist()
            for tgt_i in batch_tgt_indices:
                target_pairs.append(tgt_pairs_src[tgt_i])

        del targets

        if src_idx[0].shape[0] > 0:
            # _temppair_losses = []
            # for ins_i in range(src_masks.shape[0]):
            #     ins_pairs_cross_frames = target_pairs[ins_i]  # [((k, 2), (k, 2))] * T-1
            #
            #     ins_src_similarities = []
            #     for t_i, pairs in enumerate(ins_pairs_cross_frames):
            #         pts_curr = pairs[0]
            #         pts_next = pairs[1]
            #
            #         pts_curr[:, 0] = pts_curr[:, 0].clamp(0, src_masks.shape[3] - 1)
            #         pts_curr[:, 1] = pts_curr[:, 1].clamp(0, src_masks.shape[2] - 1)
            #         pts_next[:, 0] = pts_next[:, 0].clamp(0, src_masks.shape[3] - 1)
            #         pts_next[:, 1] = pts_next[:, 1].clamp(0, src_masks.shape[2] - 1)
            #
            #         ins_src_similarities.append(
            #             calculate_temp_similarities(
            #                 src_masks[ins_i, t_i, :, :],
            #                 src_masks[ins_i, t_i + 1, :, :],
            #                 pts_curr.long(),
            #                 pts_next.long(),
            #             )
            #         )
            #
            #     ins_src_similarities = torch.cat(ins_src_similarities, dim=0)
            #     if ins_src_similarities.shape[0] > 0:
            #         _temppair_losses.append(ins_src_similarities.mean())
            #
            # temppair_loss = sum(_temppair_losses) / max(len(_temppair_losses), 1) if len(_temppair_losses) > 0 \
            #     else src_masks.new_zero(1)
            #
            # warmup_factor = min(self._iter.item() / float(self.pairwise_warmup_iters), 1.0)
            # losses = {
            #     "loss_mask_temporal_pairwise":
            #         temppair_loss * warmup_factor if self.temporal_warmup else temppair_loss
            # }


            src_similarities = []
            for ins_i in range(src_masks.shape[0]):
                ins_pairs_cross_frames = target_pairs[ins_i]  # [((k, 2), (k, 2))] * T-1

                ins_src_similarities = []
                for t_i, pairs in enumerate(ins_pairs_cross_frames):
                    pts_curr = pairs[0]
                    pts_next = pairs[1]

                    pts_curr[:, 0] = pts_curr[:, 0].clamp(0, src_masks.shape[3] - 1)
                    pts_curr[:, 1] = pts_curr[:, 1].clamp(0, src_masks.shape[2] - 1)
                    pts_next[:, 0] = pts_next[:, 0].clamp(0, src_masks.shape[3] - 1)
                    pts_next[:, 1] = pts_next[:, 1].clamp(0, src_masks.shape[2] - 1)

                    ins_src_similarities.append(
                        calculate_temp_similarities(
                            src_masks[ins_i, t_i, :, :],
                            src_masks[ins_i, t_i + 1, :, :],
                            pts_curr.long(),
                            pts_next.long(),
                        )
                    )

                ins_src_similarities = torch.cat(ins_src_similarities, dim=0)
                src_similarities.append(ins_src_similarities)

            src_similarities = torch.cat(src_similarities, dim=0)

            warmup_factor = min(self._iter.item() / float(self.pairwise_warmup_iters), 1.0)
            if src_similarities.shape[0] > 0:
                temppair_loss = src_similarities.mean(dim=0)
            else:
                temppair_loss = torch.tensor([0], dtype=torch.float32, device=src_masks.device)

            losses = {
                "loss_mask_temporal_pairwise":
                    temppair_loss * warmup_factor if self.temporal_warmup else temppair_loss
            }

            del src_similarities
        else:
            losses = {"loss_mask_temporal_pairwise": torch.tensor([0], dtype=torch.float32, device=src_masks.device)}

        del src_masks
        del target_pairs
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'projection_masks': self.loss_projection_masks,
            'pairwise': self.loss_spatial_pairwise,
            'temporal_pairwise': self.loss_temporal_pairwise,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        self._iter += 1
        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class VideoSetCriterionProjPair(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
            self,
            num_classes,
            matcher,
            weight_dict,
            eos_coef,
            pairwise_size,
            pairwise_dilation,
            pairwise_color_thresh,
            pairwise_warmup_iters,
            losses,
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.pairwise_size = pairwise_size
        self.pairwise_dilation = pairwise_dilation
        self.pairwise_color_thresh = pairwise_color_thresh
        self.pairwise_warmup_iters = pairwise_warmup_iters
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # for pairwise loss warmup
        self.register_buffer("_iter", torch.zeros([1]))

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_pairwise(self, outputs, targets, indices, num_masks):
        """
            Compute the losses related to the masks: the 1D projection loss.
            targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]

        # Modified to handle video
        target_boxmasks = torch.cat(
            [t['box_masks'][i] for t, (_, i) in zip(targets, indices)]
        ).to(src_masks)  # (N, T, H, W)

        target_similarities = torch.cat(
            [t['color_similarities'][i] for t, (_, i) in zip(targets, indices)]
        )  # (N, T, k*k-1, H, W)

        if src_idx[0].shape[0] > 0:
            with torch.no_grad():
                target_similarities = (target_similarities >= self.pairwise_color_thresh).float() * \
                                      target_boxmasks[:, :, None].float()

            src_similarities = calculate_pred_similaries_video(src_masks, self.pairwise_size, self.pairwise_dilation)

            warmup_factor = min(self._iter.item() / float(self.pairwise_warmup_iters), 1.0)
            losses = {
                "loss_mask_pairwise":
                    pairwise_loss_jit(src_similarities, target_similarities, num_masks) * warmup_factor
            }
        else:
            losses = {"loss_mask_pairwise": torch.tensor([0], dtype=torch.float32, device=src_masks.device)}


        del src_masks
        del target_boxmasks
        del target_similarities
        return losses

    def loss_projection_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the 1D projection loss and the pairwise loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx].sigmoid()  # need to sigmoid

        # Modified to handle video
        target_boxmasks = torch.cat(
            [t['box_masks'][i] for t, (_, i) in zip(targets, indices)]
        ).to(src_masks)
        target_left_bounds = torch.cat(
            [t['left_bounds'][i] for t, (_, i) in zip(targets, indices)]
        ).to(src_masks)
        target_right_bounds = torch.cat(
            [t['right_bounds'][i] for t, (_, i) in zip(targets, indices)]
        ).to(src_masks)
        target_top_bounds = torch.cat(
            [t['top_bounds'][i] for t, (_, i) in zip(targets, indices)]
        ).to(src_masks)
        target_bottom_bounds = torch.cat(
            [t['bottom_bounds'][i] for t, (_, i) in zip(targets, indices)]
        ).to(src_masks)

        # (N, T, H_pad/4, W_pad/4) -> (NT, 1, H_pad/4, W_pad/4)
        src_masks = src_masks.flatten(0, 1)[:, None]
        target_boxmasks = target_boxmasks.flatten(0, 1)[:, None]
        target_left_bounds = target_left_bounds.flatten()  # (NTH)
        target_right_bounds = target_right_bounds.flatten()  # (NTH)
        target_top_bounds = target_top_bounds.flatten()  # (NTW)
        target_bottom_bounds = target_bottom_bounds.flatten()  # (NTW)

        if src_idx[0].shape[0] > 0:
            """
                bellow is for original projection loss
            """
            # src_masks_y = src_masks.max(dim=3, keepdim=True)[0].flatten(1)
            # src_masks_x = src_masks.max(dim=2, keepdim=True)[0].flatten(1)
            #
            # with torch.no_grad():
            #     target_boxmasks_y = target_boxmasks.max(dim=3, keepdim=True)[0].flatten(1)
            #     target_boxmasks_x = target_boxmasks.max(dim=2, keepdim=True)[0].flatten(1)

            """
                bellow is for projection limited label loss
            """
            NT = src_masks.shape[0]
            H = src_masks.shape[2]
            W = src_masks.shape[3]

            src_masks_y, max_inds_x = src_masks.max(dim=3, keepdim=True)  # (NT, 1, H, 1), (NT, 1, H, 1)
            src_masks_x, max_inds_y = src_masks.max(dim=2, keepdim=True)  # (NT, 1, 1, W), (NT, 1, 1, W)

            src_masks_y = src_masks_y.flatten(1)  # (NT, H)
            src_masks_x = src_masks_x.flatten(1)  # (NT, W)
            max_inds_x = max_inds_x.flatten()  # (NTW)
            max_inds_y = max_inds_y.flatten()  # (NTH)

            with torch.no_grad():
                flag_l = max_inds_x >= target_left_bounds
                flag_r = max_inds_x < target_right_bounds
                flag_y = (flag_l * flag_r).view(NT, H)

                flag_t = max_inds_y >= target_top_bounds
                flag_b = max_inds_y < target_bottom_bounds
                flag_x = (flag_t * flag_b).view(NT, W)

                target_boxmasks_y = target_boxmasks.max(dim=3, keepdim=True)[0].flatten(1) * flag_y  # (NT, W)
                target_boxmasks_x = target_boxmasks.max(dim=2, keepdim=True)[0].flatten(1) * flag_x  # (NT, H)

            losses = {
                "loss_mask_projection": projection2D_dice_loss_jit(
                    src_masks_x, target_boxmasks_x,
                    src_masks_y, target_boxmasks_y,
                    num_masks
                )
            }

            del src_masks_x, src_masks_y
            del target_boxmasks_x, target_boxmasks_y
        else:
            """
            bellow is for original projection loss and limited projection loss
            """
            losses = {"loss_mask_projection": torch.tensor([0], dtype=torch.float32, device=src_masks.device)}

        del src_masks
        del target_boxmasks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'projection_masks': self.loss_projection_masks,
            'pairwise': self.loss_pairwise,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        self._iter += 1
        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class VideoSetCriterionProj(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
            self,
            num_classes,
            matcher,
            weight_dict,
            eos_coef,
            losses,
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        self.register_buffer("_iter", torch.zeros([1]))

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_projection_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the 1D projection loss and the pairwise loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx].sigmoid()  # need to sigmoid
        # Modified to handle video
        target_boxmasks = torch.cat(
            [t['box_masks'][i] for t, (_, i) in zip(targets, indices)]
        ).to(src_masks)
        # target_left_bounds = torch.cat(
        #     [t['left_bounds'][i] for t, (_, i) in zip(targets, indices)]
        # ).to(src_masks)
        # target_right_bounds = torch.cat(
        #     [t['right_bounds'][i] for t, (_, i) in zip(targets, indices)]
        # ).to(src_masks)
        # target_top_bounds = torch.cat(
        #     [t['top_bounds'][i] for t, (_, i) in zip(targets, indices)]
        # ).to(src_masks)
        # target_bottom_bounds = torch.cat(
        #     [t['bottom_bounds'][i] for t, (_, i) in zip(targets, indices)]
        # ).to(src_masks)

        # (N, T, H_pad/4, W_pad/4) -> (NT, 1, H_pad/4, W_pad/4)
        src_masks = src_masks.flatten(0, 1)[:, None]
        target_boxmasks = target_boxmasks.flatten(0, 1)[:, None]
        # target_left_bounds = target_left_bounds.flatten()  # (NTH)
        # target_right_bounds = target_right_bounds.flatten()  # (NTH)
        # target_top_bounds = target_top_bounds.flatten()  # (NTW)
        # target_bottom_bounds = target_bottom_bounds.flatten()  # (NTW)

        if src_idx[0].shape[0] > 0:
            """
                bellow is for original projection loss
            """
            # project mask to x & y axis
            src_masks_y = src_masks.max(dim=3, keepdim=True)[0].flatten(1)
            src_masks_x = src_masks.max(dim=2, keepdim=True)[0].flatten(1)

            with torch.no_grad():
                target_boxmasks_y = target_boxmasks.max(dim=3, keepdim=True)[0].flatten(1)
                target_boxmasks_x = target_boxmasks.max(dim=2, keepdim=True)[0].flatten(1)

            """
                bellow is for projection limited label loss
            """
            # NT = src_masks.shape[0]
            # H = src_masks.shape[2]
            # W = src_masks.shape[3]
            #
            # src_masks_y, max_inds_x = src_masks.max(dim=3, keepdim=True)  # (NT, 1, H, 1), (NT, 1, H, 1)
            # src_masks_x, max_inds_y = src_masks.max(dim=2, keepdim=True)  # (NT, 1, 1, W), (NT, 1, 1, W)
            #
            # src_masks_y = src_masks_y.flatten(1)  # (NT, H)
            # src_masks_x = src_masks_x.flatten(1)  # (NT, W)
            # max_inds_x = max_inds_x.flatten()  # (NTW)
            # max_inds_y = max_inds_y.flatten()  # (NTH)
            #
            # with torch.no_grad():
            #     flag_l = max_inds_x >= target_left_bounds
            #     flag_r = max_inds_x < target_right_bounds
            #     flag_y = (flag_l * flag_r).view(NT, H)
            #
            #     flag_t = max_inds_y >= target_top_bounds
            #     flag_b = max_inds_y < target_bottom_bounds
            #     flag_x = (flag_t * flag_b).view(NT, W)
            #
            #     target_boxmasks_y = target_boxmasks.max(dim=3, keepdim=True)[0].flatten(1) * flag_y  # (NT, W)
            #     target_boxmasks_x = target_boxmasks.max(dim=2, keepdim=True)[0].flatten(1) * flag_x  # (NT, H)

            losses = {
                "loss_mask_projection": projection2D_dice_loss_jit(
                    src_masks_x, target_boxmasks_x,
                    src_masks_y, target_boxmasks_y,
                    num_masks
                )
            }

            del src_masks_x, src_masks_y
            del target_boxmasks_x, target_boxmasks_y
        else:
            """
            bellow is for original projection loss and limited projection loss
            """
            losses = {
                "loss_mask_projection": torch.tensor([0], dtype=torch.float32, device=src_masks.device),
            }

        del src_masks
        del target_boxmasks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'projection_masks': self.loss_projection_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        For BoxInst, dict in targets(list) contains:
            "labels": (num_gt,)
            "ids": (num_gt, T)
            "masks": (num_gt, T, H, W)
            "box_masks": (num_gt, T, H/4, W/4)
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        self._iter += 1
        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class VideoSetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        # Modified to handle video
        target_masks = torch.cat([t['masks'][i] for t, (_, i) in zip(targets, indices)]).to(src_masks)

        # No need to upsample predictions as we are using normalized coordinates :)
        # NT x 1 x H x W
        src_masks = src_masks.flatten(0, 1)[:, None]
        target_masks = target_masks.flatten(0, 1)[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)