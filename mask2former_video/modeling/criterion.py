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

import os, cv2
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


class VideoSetCriterionProjMask(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
            self, num_classes, matcher, weight_dict, eos_coef, losses, update_mask, mask_update_steps, update_pix_thrs
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
            # project mask to x & y axis
            # masks_x: (num_ins, T, H_pad/4, W_pad/4)->(num_ins, T, H_pad, 1)->(num_ins, T*H_pad)
            # masks_y: (num_ins, T, H_pad/4, W_pad/4)->(num_ins, T, 1, W_pad)->(num_ins, T*W_pad)
            # max projection

            # src_masks_y = src_masks.max(dim=3, keepdim=True)[0].flatten(1)
            # src_masks_x = src_masks.max(dim=2, keepdim=True)[0].flatten(1)
            #
            # with torch.no_grad():
            #     # max projection
            #     target_boxmasks_y = target_boxmasks.max(dim=3, keepdim=True)[0].flatten(1)
            #     target_boxmasks_x = target_boxmasks.max(dim=2, keepdim=True)[0].flatten(1)
            #
            # losses = {
            #     "loss_mask_projection": projection2D_dice_loss_jit(
            #         src_masks_x, target_boxmasks_x,
            #         src_masks_y, target_boxmasks_y,
            #         num_masks
            #     )
            # }
            #
            # del src_masks_x, src_masks_y
            # del target_box_masks_x, target_box_masks_y
            """
                bellow is for projection limited sample loss
            """
            # src_masks_limited_x = src_masks * target_boxmasks_limited_x
            # src_masks_limited_y = src_masks * target_boxmasks_limited_y
            #
            # src_masks_y = src_masks_limited_x.max(dim=-1, keepdim=True)[0].flatten(1)
            # src_masks_x = src_masks_limited_y.max(dim=-2, keepdim=True)[0].flatten(1)
            #
            # with torch.no_grad():
            #     # max projection
            #     target_boxmasks_x = target_boxmasks.max(dim=2, keepdim=True)[0].flatten(1)
            #     target_boxmasks_y = target_boxmasks.max(dim=3, keepdim=True)[0].flatten(1)
            #
            # losses = {
            #     "loss_mask_projection": projection2D_dice_loss_jit(
            #         src_masks_x, target_boxmasks_x,
            #         src_masks_y, target_boxmasks_y,
            #         num_masks
            #     )
            # }

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
            # del src_masks_limited_x, src_masks_limited_y
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

    def _get_limited_projections_loop(self, src_masks, tgt_box_masks):
        tgt_boxes = BitMasks(tgt_box_masks.squeeze()).get_bounding_boxes().tensor.to(dtype=torch.int64)  # (N, 4) in BoxMode:XYXY
        src_masks_fg = src_masks * tgt_box_masks

        src_masks_y, src_masks_x = [], []
        for ind in range(src_masks.shape[0]):
            src_mask = src_masks[ind, 0]
            src_mask_fg = src_masks_fg[ind, 0]
            tgt_box = tgt_boxes[ind]
            # tgt_box = BitMasks(tgt_box_masks[ind]).get_bounding_boxes().tensor.to(dtype=torch.uint8)[0]

            # limited projection on y axis
            upper_region_proj_y = src_mask[:tgt_box[1], :].max(dim=1, keepdim=True)[0]
            tgt_region_proj_y = src_mask_fg[tgt_box[1]:tgt_box[3], :].max(dim=1, keepdim=True)[0]
            lower_region_proj_y = src_mask[tgt_box[3]:, :].max(dim=1, keepdim=True)[0]
            proj_y = torch.cat([upper_region_proj_y, tgt_region_proj_y, lower_region_proj_y], dim=0)[None, :]
            src_masks_y.append(proj_y)

            # limited projection on x axis
            left_region_proj_x = src_mask[:, :tgt_box[0]].max(dim=0, keepdim=True)[0]
            tgt_region_proj_x = src_mask_fg[:, tgt_box[0]:tgt_box[2]].max(dim=0, keepdim=True)[0]
            right_region_proj_x = src_mask[:, tgt_box[2]:].max(dim=0, keepdim=True)[0]
            proj_x = torch.cat([left_region_proj_x, tgt_region_proj_x, right_region_proj_x], dim=1)[:, None]
            src_masks_x.append(proj_x)

        _src_masks_y = torch.stack(src_masks_y, dim=0).flatten(1, 3)
        _src_masks_x = torch.stack(src_masks_x, dim=0).flatten(1, 3)
        return _src_masks_y, _src_masks_x

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
    def update_targets(self, src_probs_per_batch, targets_per_batch, indices_per_batch, pix_thr=0.3, overlap_thr=0.0):
        """
        indices_curr_lvl: [(tensor(G,), tensor(G,))] x num_img
        """
        src_probs_per_batch = src_probs_per_batch["pred_masks"]  # (B, Q, T, H, W)
        B = src_probs_per_batch.shape[0]
        T = src_probs_per_batch.shape[2]
        src_probs_per_batch = src_probs_per_batch.split(B, 0)[0]  # 【(Q, T, H, W)] x B

        new_targets = []
        for indice, src_probs, targets in zip(indices_per_batch, src_probs_per_batch, targets_per_batch):
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
                new_box_mask = (src_probs[src_idx].sigmoid() >= pix_thr) * box_masks[tgt_idx]  # (T, H, W)

                if new_box_masks[tgt_idx].sum() >= overlap_thr:
                    new_box_masks[tgt_idx] = new_box_mask

                    for f_i in range(T):
                        # bounds for y projection
                        new_left_bounds[tgt_idx, f_i] = torch.argmax(new_box_masks[tgt_idx, f_i], dim=1)
                        new_right_bounds[tgt_idx, f_i] = new_box_masks[tgt_idx, f_i].shape[1] - \
                                                    torch.argmax(new_box_masks[tgt_idx, f_i].flip(1), dim=1)
                        # bounds for x projection
                        new_top_bnounds[tgt_idx, f_i] = torch.argmax(new_box_masks[tgt_idx, f_i], dim=0)
                        new_bottom_bounds[tgt_idx, f_i] = new_box_masks[tgt_idx, f_i].shape[0] - \
                                                     torch.argmax(new_box_masks[tgt_idx, f_i].flip(0), dim=0)
                else:
                    new_box_masks[tgt_idx] = box_masks[tgt_idx]
                    new_left_bounds[tgt_idx] = left_bounds[tgt_idx]
                    new_right_bounds[tgt_idx] = right_bounds[tgt_idx]
                    new_top_bnounds[tgt_idx] = top_bounds[tgt_idx]
                    new_bottom_bounds[tgt_idx] = bottom_bounds[tgt_idx]

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

            # pix_thrs = [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]
            # pix_thrs = [0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30]
            # pix_thrs = [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50]
            # pix_thrs = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
            # calculate mask update pix thr
            thr_ind = 0
            for ind in range(len(self.mask_update_steps) - 1):
                if self._iter >= self.mask_update_steps[ind] and self._iter < self.mask_update_steps[ind + 1]:
                    thr_ind = ind
            pix_thr = self.update_pix_thrs[thr_ind]

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
                    targets = self.update_targets(aux_outputs, targets, indices, pix_thr=pix_thr)

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