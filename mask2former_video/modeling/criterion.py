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
    """
    :param inputs_x: (num_ins, T*H_pad/4)
    :param targets_x: (num_ins, T*H_pad/4)
    :param inputs_y: (num_ins, T*W_pad/4)
    :param targets_y: (num_ins, T*W_pad/4)
    :param num_masks: int
    :return: loss: float tensor
    """
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
    loss_pairwise = (inputs * targets).sum() / targets.sum().clamp(min=1.0)
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
    # (Q, T, H, W) -> (Q, T, 1, H, W), (Q, T, 1, H, W) + (Q, T, k*k-1, H, W) ->(Q, T, k*k-1, H, W)
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold  # (N, T, k*k-1, H/4, W/4)
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold  # (N, T, k*k-1, H/4, W/4)

    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)  # (N, T, k*k-1, H, W)
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_  # (N, T, k*k-1, H, W)

    return -log_same_prob


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
            # below for mask update
            update_mask,
            mask_update_steps,
            update_pix_thrs,
            conf_type,
            static_conf_thr,
            max_iter
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

        self.update_masks = update_mask
        self.mask_update_steps = mask_update_steps
        self.update_pix_thrs = update_pix_thrs
        self.conf_type = conf_type
        self.static_conf_thr = static_conf_thr
        self.max_iter = max_iter
        assert len(self.mask_update_steps) == len(self.update_pix_thrs) + 1

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
        src_masks = src_masks[src_idx].sigmoid()  # need to sigmoid, (N, T, H, W)
        # Modified to handle video
        target_boxmasks = torch.cat(
            [t['box_masks'][i] for t, (_, i) in zip(targets, indices)]
        ).to(src_masks)  # (N, T, H, W)
        target_similarities = torch.cat(
            [t['color_similarities'][i] for t, (_, i) in zip(targets, indices)]
        )  # (N, T, k*k-1, H, W)

        T = src_masks.shape[1]

        if src_idx[0].shape[0] > 0:
            with torch.no_grad():
                target_similarities = (target_similarities >= self.pairwise_color_thresh).float() * \
                                      target_boxmasks[:, :, None].float()

            src_similarities = calculate_pred_similaries_video(src_masks, self.pairwise_size, self.pairwise_dilation)

            warmup_factor = min(self._iter.item() / float(self.pairwise_warmup_iters), 1.0)
            losses = {
                "loss_mask_pairwise":
                    (pairwise_loss_jit(src_similarities, target_similarities, num_masks) / T) * warmup_factor
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
        # target_boxmasks_limited_x = torch.cat(
        #     [t['box_masks_limited_x'][i] for t, (_, i) in zip(targets, indices)]
        # ).to(src_masks)
        # target_boxmasks_limited_y = torch.cat(
        #     [t['box_masks_limited_y'][i] for t, (_, i) in zip(targets, indices)]
        # ).to(src_masks)
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
        # target_boxmasks_limited_x = target_boxmasks_limited_x.flatten(0, 1)[:, None]
        # target_boxmasks_limited_y = target_boxmasks_limited_y.flatten(0, 1)[:, None]
        target_left_bounds = target_left_bounds.flatten()  # (NTH)
        target_right_bounds = target_right_bounds.flatten()  # (NTH)
        target_top_bounds = target_top_bounds.flatten()  # (NTW)
        target_bottom_bounds = target_bottom_bounds.flatten()  # (NTW)

        if src_idx[0].shape[0] > 0:
            """
                bellow is for original projection loss
            """
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

    @torch.no_grad()
    def update_targets(
            self,
            src_per_batch,
            targets_per_batch,
            indices_per_batch,
            conf_thr=0.5,
            pix_thr=0.3,
    ):
        """
        indices_curr_lvl: [(tensor(G,), tensor(G,))] x num_img
        """
        src_scores_per_batch = F.softmax(src_per_batch["pred_logits"], dim=-1)[:, :-1]  # (B, Q, C)
        src_mask_probs_per_batch = src_per_batch["pred_masks"].sigmoid()  # (B, Q, T, H, W)
        B = src_mask_probs_per_batch.shape[0]
        T = src_mask_probs_per_batch.shape[2]
        src_scores_per_batch = src_scores_per_batch.split(B, 0)[0]  # [(Q, C)] x B
        src_mask_probs_per_batch = src_mask_probs_per_batch.split(B, 0)[0]  # [(Q, T, H, W)] x B

        new_targets = []
        for indice, src_scores, src_mask_probs, targets in zip(
            indices_per_batch, src_scores_per_batch, src_mask_probs_per_batch, targets_per_batch
        ):
            gt_classes = targets["labels"]  # (G,)
            box_masks = targets["box_masks"]  # (G, T, H, W)
            left_bounds = targets["left_bounds"]  # (G, T, H)
            right_bounds = targets["right_bounds"]  # (G, T, H)
            top_bounds = targets["top_bounds"]  # (G, T, W)
            bottom_bounds = targets["bottom_bounds"]  # (G, T, W)

            new_box_masks = torch.zeros_like(box_masks)
            new_left_bounds = torch.zeros_like(left_bounds)
            new_right_bounds = torch.zeros_like(right_bounds)
            new_top_bnounds = torch.zeros_like(top_bounds)
            new_bottom_bounds = torch.zeros_like(bottom_bounds)

            for src_idx, tgt_idx in zip(indice[0], indice[1]):
                cls_score = src_scores[src_idx, gt_classes[tgt_idx]]
                # new_box_masks = (src_mask_probs[src_idx] >= pix_thr) * box_masks[tgt_idx]  # (T, H, W)

                # if new_box_masks[tgt_idx].sum() >= overlap_thr:
                #     new_box_masks[tgt_idx] = new_box_mask

                for f_i in range(T):
                    mask_probs = src_mask_probs[src_idx, f_i]
                    # mask_probs = mask_probs * box_masks[tgt_idx, f_i]  # 框内算pix_score
                    pix_score = (mask_probs * (mask_probs > 0.5).float()).sum() / \
                                ((mask_probs > 0.5).float().sum() + 1e-6)

                    if torch.pow(cls_score * pix_score, 2) >= quality_thr:
                        new_box_masks[tgt_idx, f_i] = (mask_probs > 0.5) * box_masks[tgt_idx, f_i]
                        # bounds for y projection
                        new_left_bounds[tgt_idx, f_i] = torch.argmax(new_box_masks[tgt_idx, f_i], dim=1)
                        new_right_bounds[tgt_idx, f_i] = new_box_masks[tgt_idx, f_i].shape[1] - \
                                                    torch.argmax(new_box_masks[tgt_idx, f_i].flip(1), dim=1)
                        # bounds for x projection
                        new_top_bnounds[tgt_idx, f_i] = torch.argmax(new_box_masks[tgt_idx, f_i], dim=0)
                        new_bottom_bounds[tgt_idx, f_i] = new_box_masks[tgt_idx, f_i].shape[0] - \
                                                     torch.argmax(new_box_masks[tgt_idx, f_i].flip(0), dim=0)
                    else:
                        new_box_masks[tgt_idx, f_i] = box_masks[tgt_idx, f_i]
                        new_left_bounds[tgt_idx, f_i] = left_bounds[tgt_idx, f_i]
                        new_right_bounds[tgt_idx, f_i] = right_bounds[tgt_idx, f_i]
                        new_top_bnounds[tgt_idx, f_i] = top_bounds[tgt_idx, f_i]
                        new_bottom_bounds[tgt_idx, f_i] = bottom_bounds[tgt_idx, f_i]

            targets["box_masks"] = new_box_masks
            targets["left_bounds"] = new_left_bounds
            targets["right_bounds"] = new_right_bounds
            targets["top_bounds"] = new_top_bnounds
            targets["bottom_bounds"] = new_bottom_bounds

            new_targets.append(targets)
        return new_targets

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        if self.update_masks:
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

            # calculate mask update pix thr
            # thr_ind = 0
            # for ind in range(len(self.mask_update_steps) - 1):
            #     if self._iter >= self.mask_update_steps[ind] and self._iter < self.mask_update_steps[ind + 1]:
            #         thr_ind = ind
            # pix_thr = self.update_pix_thrs[thr_ind]

            # compute loss of auxiliary decoder layer
            # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
            if "aux_outputs" in outputs:
                for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                    indices = self.matcher(aux_outputs, targets)
                    for loss in self.losses:
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                        l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                        losses.update(l_dict)
                    # update targets after curr lvl loss calculation
                    if self.conf_type == "static":
                        conf_thr = self.static_conf_thr
                    elif self.conf_type == "dynamic":
                        conf_thr = 1 / (1 + np.exp(-2 * (1 - self._iter / self.max_iter)))
                    else:
                        raise Exception("Unknown update type !!!")
                    targets = self.update_targets(aux_outputs, targets, indices, conf_thr=conf_thr)

            # compute los of final decoder layer
            outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
            # Retrieve the matching between the outputs of the last layer and the targets
            indices = self.matcher(outputs_without_aux, targets)
            for loss in self.losses:
                losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))
        else:
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
            # below for mask update
            update_mask,
            mask_update_steps,
            update_pix_thrs,
            conf_type,
            static_conf_thr,
            max_iter
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

        self.update_masks = update_mask
        self.mask_update_steps = mask_update_steps
        self.update_pix_thrs = update_pix_thrs
        self.conf_type = conf_type
        self.static_conf_thr = static_conf_thr
        self.max_iter = max_iter
        assert len(self.mask_update_steps) == len(self.update_pix_thrs) + 1

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

    @torch.no_grad()
    def update_targets(
            self,
            src_per_batch,
            targets_per_batch,
            indices_per_batch,
            conf_thr=0.5,
            pix_thr=0.3,
    ):
        """
        indices_curr_lvl: [(tensor(G,), tensor(G,))] x num_img
        """
        src_scores_per_batch = F.softmax(src_per_batch["pred_logits"], dim=-1)  # (B, Q, C+1), C+1 i for dummy gt
        src_mask_probs_per_batch = src_per_batch["pred_masks"].sigmoid()  # (B, Q, T, H, W)
        B = src_mask_probs_per_batch.shape[0]
        T = src_mask_probs_per_batch.shape[2]
        src_scores_per_batch = src_scores_per_batch.split(B, 0)[0]  # [(Q, C)] x B
        src_mask_probs_per_batch = src_mask_probs_per_batch.split(B, 0)[0]  # [(Q, T, H, W)] x B

        new_targets = []
        for indice, src_scores, src_mask_probs, targets in zip(
            indices_per_batch, src_scores_per_batch, src_mask_probs_per_batch, targets_per_batch
        ):
            gt_classes = targets["labels"]  # (G,)
            box_masks = targets["box_masks"]  # (G, T, H, W)
            left_bounds = targets["left_bounds"]  # (G, T, H)
            right_bounds = targets["right_bounds"]  # (G, T, H)
            top_bounds = targets["top_bounds"]  # (G, T, W)
            bottom_bounds = targets["bottom_bounds"]  # (G, T, W)

            new_box_masks = torch.zeros_like(box_masks)
            new_left_bounds = torch.zeros_like(left_bounds)
            new_right_bounds = torch.zeros_like(right_bounds)
            new_top_bnounds = torch.zeros_like(top_bounds)
            new_bottom_bounds = torch.zeros_like(bottom_bounds)

            for src_idx, tgt_idx in zip(indice[0], indice[1]):
                cls_score = src_scores[src_idx, gt_classes[tgt_idx]]

                for f_i in range(T):
                    mask_probs = src_mask_probs[src_idx, f_i]
                    # mask_probs = mask_probs * box_masks[tgt_idx, f_i]  # 框内算pix_score
                    pix_score = (mask_probs * (mask_probs > 0.5).float()).sum() / \
                                ((mask_probs > 0.5).float().sum() + 1e-6)

                    if torch.pow(cls_score * pix_score, 0.5) >= conf_thr:
                        new_box_masks[tgt_idx, f_i] = (mask_probs > 0.5) * box_masks[tgt_idx, f_i]
                        # bounds for y projection
                        new_left_bounds[tgt_idx, f_i] = torch.argmax(new_box_masks[tgt_idx, f_i], dim=1)
                        new_right_bounds[tgt_idx, f_i] = new_box_masks[tgt_idx, f_i].shape[1] - \
                                                    torch.argmax(new_box_masks[tgt_idx, f_i].flip(1), dim=1)
                        # bounds for x projection
                        new_top_bnounds[tgt_idx, f_i] = torch.argmax(new_box_masks[tgt_idx, f_i], dim=0)
                        new_bottom_bounds[tgt_idx, f_i] = new_box_masks[tgt_idx, f_i].shape[0] - \
                                                     torch.argmax(new_box_masks[tgt_idx, f_i].flip(0), dim=0)
                    else:
                        new_box_masks[tgt_idx, f_i] = box_masks[tgt_idx, f_i]
                        new_left_bounds[tgt_idx, f_i] = left_bounds[tgt_idx, f_i]
                        new_right_bounds[tgt_idx, f_i] = right_bounds[tgt_idx, f_i]
                        new_top_bnounds[tgt_idx, f_i] = top_bounds[tgt_idx, f_i]
                        new_bottom_bounds[tgt_idx, f_i] = bottom_bounds[tgt_idx, f_i]

            targets["box_masks"] = new_box_masks
            targets["left_bounds"] = new_left_bounds
            targets["right_bounds"] = new_right_bounds
            targets["top_bounds"] = new_top_bnounds
            targets["bottom_bounds"] = new_bottom_bounds

            new_targets.append(targets)
        return new_targets

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
        if self.update_masks:
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

            # calculate mask update pix thr
            # thr_ind = 0
            # for ind in range(len(self.mask_update_steps) - 1):
            #     if self._iter >= self.mask_update_steps[ind] and self._iter < self.mask_update_steps[ind + 1]:
            #         thr_ind = ind
            # pix_thr = self.update_pix_thrs[thr_ind]

            # compute loss of auxiliary decoder layer
            # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
            if "aux_outputs" in outputs:
                for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                    indices = self.matcher(aux_outputs, targets)
                    for loss in self.losses:
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                        l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                        losses.update(l_dict)
                    # update targets after curr lvl loss calculation
                    if self.conf_type == "static":
                        conf_thr = self.static_conf_thr
                    elif self.conf_type == "dynamic":
                        conf_thr = 1 / (1 + torch.exp(-2 * (1 - self._iter / self.max_iter)))
                    else:
                        raise Exception("Unknown update type !!!")
                    targets = self.update_targets(aux_outputs, targets, indices, conf_thr=conf_thr)

            # compute los of final decoder layer
            outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
            # Retrieve the matching between the outputs of the last layer and the targets
            indices = self.matcher(outputs_without_aux, targets)
            for loss in self.losses:
                losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))
        else:
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