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

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
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
        target_box_masks = torch.cat([t['box_masks'][i] for t, (_, i) in zip(targets, indices)]).to(src_masks)
        # src_masks: (N, T, H_pad/4, W_pad/4)
        # target_masks: (N, T, H_pad/4, W_pad/4)

        # (N, T, H_pad/4, W_pad/4)->(NT, 1, H_pad/4, W_pad/4)
        src_masks = src_masks.flatten(0, 1)[:, None]
        target_box_masks = target_box_masks.flatten(0, 1)[:, None]

        if src_idx[0].shape[0] > 0:
            """
            bellow is for original projection loss
            """
            # project mask to x & y axis
            # masks_x: (num_ins, T, H_pad/4, W_pad/4)->(num_ins, T, H_pad, 1)->(num_ins, T*H_pad)
            # masks_y: (num_ins, T, H_pad/4, W_pad/4)->(num_ins, T, 1, W_pad)->(num_ins, T*W_pad)
            # max projection

            # src_masks_x = src_masks.max(dim=3, keepdim=True)[0].flatten(1, 3)
            # src_masks_y = src_masks.max(dim=2, keepdim=True)[0].flatten(1, 3)
            #
            # with torch.no_grad():
            #     # max projection
            #     target_box_masks_x = target_box_masks.max(dim=3, keepdim=True)[0].flatten(1, 3)
            #     target_box_masks_y = target_box_masks.max(dim=2, keepdim=True)[0].flatten(1, 3)
            #
            # losses = {
            #     "loss_mask_projection": projection2D_dice_loss_jit(
            #         src_masks_x, target_box_masks_x,
            #         src_masks_y, target_box_masks_y,
            #         num_masks
            #     )
            # }
            #
            # del src_masks_x, src_masks_y
            # del target_box_masks_x, target_box_masks_y

            """
            bellow is for foreground / background separate loss
            """
            # src_masks_box_fg = src_masks * target_box_masks
            # src_masks_box_bg = ((1. - src_masks) * (1. - target_box_masks)) + target_box_masks
            #
            # src_masks_x_fg = src_masks_box_fg.max(dim=3, keepdim=True)[0].flatten(1, 3)
            # src_masks_y_fg = src_masks_box_fg.max(dim=2, keepdim=True)[0].flatten(1, 3)
            # src_masks_x_bg = src_masks_box_bg.min(dim=3, keepdim=True)[0].flatten(1, 3)
            # src_masks_y_bg = src_masks_box_bg.min(dim=2, keepdim=True)[0].flatten(1, 3)
            #
            # # # (NT, 1, H, 1) or (NT, 1, 1, W)
            # # src_masks_x_fg, src_y_inds_fg = src_masks_box_fg.max(dim=3, keepdim=True)
            # # src_masks_y_fg, src_x_inds_fg = src_masks_box_fg.max(dim=2, keepdim=True)
            # # src_masks_x_bg, src_y_inds_bg = src_masks_box_bg.min(dim=3, keepdim=True)
            # # src_masks_y_bg, src_x_inds_bg = src_masks_box_bg.min(dim=2, keepdim=True)
            # # src_masks_x_fg = src_masks_x_fg.flatten(1, 3)
            # # src_masks_y_fg = src_masks_y_fg.flatten(1, 3)
            # # src_masks_x_bg = src_masks_x_bg.flatten(1, 3)
            # # src_masks_y_bg = src_masks_y_bg.flatten(1, 3)
            # #
            # # # org mask
            # # src_masks_x, src_y_inds = src_masks.max(dim=3, keepdim=True)
            # # src_masks_y, src_x_inds = src_masks.max(dim=2, keepdim=True)
            # # for obj_id in range(src_masks.shape[0]):
            # #     obj_path = "/home/user/Program/jwh/Weakly-Sup-VIS/DEBUG/loss_debug/obj_{}/".format(obj_id)
            # #     os.makedirs(obj_path, exist_ok=True)
            # #
            # #     src = src_masks[obj_id, 0]
            # #     cv2.imwrite(obj_path + 'pred_{}.png'.format(obj_id), (src * 200).to(dtype=torch.uint8).cpu().numpy())
            # #
            # #     tgt = target_box_masks[obj_id, 0]
            # #     cv2.imwrite(obj_path + 'gt_{}.png'.format(obj_id), tgt.to(dtype=torch.uint8).cpu().numpy() * 255)
            # #     cv2.imwrite(obj_path + 'gt_inverse_{}.png'.format(obj_id), (1. - tgt).to(dtype=torch.uint8).cpu().numpy() * 255)
            # #
            # #
            # #     cv2.imwrite(obj_path + 'pred_{}_fg.png'.format(obj_id),
            # #                 (src_masks_box_fg[obj_id, 0] * 255).to(dtype=torch.uint8).cpu().numpy())
            # #     cv2.imwrite(obj_path + 'pred_{}_inverse.png'.format(obj_id), ((1. - src_masks[obj_id, 0]) * 255).to(dtype=torch.uint8).cpu().numpy())
            # #     cv2.imwrite(obj_path + 'pred_{}_bg.png'.format(obj_id),
            # #                 (src_masks_box_bg[obj_id, 0] * 255).to(dtype=torch.uint8).cpu().numpy())
            # #
            # #
            # #     org_sample_map = torch.zeros_like(src_masks[obj_id, 0])
            # #     x_inds = src_y_inds[obj_id, 0]
            # #     y_inds = src_x_inds[obj_id, 0]
            # #     for ind in range(org_sample_map.shape[0]):
            # #         org_sample_map[ind, x_inds.squeeze()[ind]] = 1
            # #     for ind in range(org_sample_map.shape[1]):
            # #         org_sample_map[y_inds.squeeze()[ind], ind] = 0.6
            # #     cv2.imwrite(obj_path + 'max_sample.png'.format(obj_id), (org_sample_map * 255).to(dtype=torch.uint8).cpu().numpy())
            # #
            # #     fg_sample_map = torch.zeros_like(org_sample_map)
            # #     fg_x_inds = src_y_inds_fg[obj_id, 0]
            # #     fg_y_inds = src_x_inds_fg[obj_id, 0]
            # #     for ind in range(fg_sample_map.shape[0]):
            # #         fg_sample_map[ind, fg_x_inds.squeeze()[ind]] = 1
            # #     for ind in range(fg_sample_map.shape[1]):
            # #         fg_sample_map[fg_y_inds.squeeze()[ind], ind] = 0.6
            # #     cv2.imwrite(obj_path + 'fg_max_sample.png'.format(obj_id), (fg_sample_map * 255).to(dtype=torch.uint8).cpu().numpy())
            # #
            # #     bg_sample_map = torch.zeros_like(org_sample_map)
            # #     bg_x_inds = src_y_inds_bg[obj_id, 0]
            # #     bg_y_inds = src_x_inds_bg[obj_id, 0]
            # #     for ind in range(bg_sample_map.shape[0]):
            # #         bg_sample_map[ind, bg_x_inds.squeeze()[ind]] = 1
            # #     for ind in range(bg_sample_map.shape[1]):
            # #         bg_sample_map[bg_y_inds.squeeze()[ind], ind] = 0.6
            # #     cv2.imwrite(obj_path + 'bg_min_sample.png'.format(obj_id), (bg_sample_map * 255).to(dtype=torch.uint8).cpu().numpy())
            #
            #
            # with torch.no_grad():
            #     target_box_masks_x_fg = target_box_masks.max(dim=3, keepdim=True)[0].flatten(1, 3)
            #     target_box_masks_y_fg = target_box_masks.max(dim=2, keepdim=True)[0].flatten(1, 3)
            #     target_box_masks_x_bg = (1. - target_box_masks).max(dim=3, keepdim=True)[0].flatten(1, 3)
            #     target_box_masks_y_bg = (1. - target_box_masks).max(dim=2, keepdim=True)[0].flatten(1, 3)
            #
            # losses = {
            #     "loss_mask_projection_fg": projection2D_dice_loss_jit(
            #         src_masks_x_fg, target_box_masks_x_fg,
            #         src_masks_y_fg, target_box_masks_y_fg,
            #         num_masks
            #     ),
            #     "loss_mask_projection_bg": projection2D_dice_loss_jit(
            #         src_masks_x_bg, target_box_masks_x_bg,
            #         src_masks_y_bg, target_box_masks_y_bg,
            #         num_masks
            #     ),
            # }
            #
            # del src_masks_box_fg, src_masks_box_bg
            # del src_masks_x_fg, src_masks_y_fg, src_masks_x_bg, src_masks_y_bg
            # del target_box_masks_x_fg, target_box_masks_y_fg, target_box_masks_x_bg, target_box_masks_y_bg

            """
            bellow is for limited projection loss
            """
            src_masks_y, src_masks_x = self._get_limited_projections(src_masks, target_box_masks)

            with torch.no_grad():
                # max projection
                target_box_masks_x = target_box_masks.max(dim=2, keepdim=True)[0].flatten(1, 3)
                target_box_masks_y = target_box_masks.max(dim=3, keepdim=True)[0].flatten(1, 3)

            losses = {
                "loss_mask_projection": projection2D_dice_loss_jit(
                    src_masks_x, target_box_masks_x,
                    src_masks_y, target_box_masks_y,
                    num_masks
                )
            }

            del tgt_boxes, src_masks_fg
            del src_masks_x, src_masks_y
            del target_box_masks_x, target_box_masks_y

        else:
            """
            bellow is for original projection loss and limited projection loss
            """
            losses = {
                "loss_mask_projection": torch.tensor([0], dtype=torch.float32, device=src_masks.device),
            }

            """
            bellow is for foreground / background separate loss
            """
            # losses = {
            #     "loss_mask_projection_fg": torch.tensor([0], dtype=torch.float32, device=src_masks.device),
            #     "loss_mask_projection_bg": torch.tensor([0], dtype=torch.float32, device=src_masks.device)
            # }


        del src_masks
        del target_box_masks
        return losses

    def _get_limited_projections(self, src_masks, tgt_box_masks):
        tgt_boxes = BitMasks(target_box_masks.squeeze()).get_bounding_boxes().tensor  # (N, 4) in BoxMode:XYXY
        src_masks_fg = src_masks * tgt_box_masks

        src_masks_y, src_masks_x = [], []
        for ind in range(src_masks.shape[0]):
            src_mask = src_masks[ind, 0]
            src_mask_fg = src_masks_fg[ind, 0]
            tgt_box = tgt_boxes[ind].to(dtype=torch.uint8)

            # limited projection on y axis
            upper_region = src_mask[:tgt_box[1], :].max(dim=1, keepdim=True)[0]
            tgt_region_y = src_mask_fg[tgt_box[1]:tgt_box[3], :].max(dim=1, keepdim=True)[0]
            lower_region = src_mask[tgt_box[3]:, :].max(dim=1, keepdim=True)[0]
            src_masks_y.append(torch.cat([upper_region, tgt_region_y, lower_region], dim=0)[None, :])

            # limited projection on x axis
            left_region = src_mask[:, :tgt_box[0]].max(dim=0, keepdim=True)[0]
            tgt_region_x = src_mask_fg[:, tgt_box[0]:tgt_box[2]].max(dim=0, keepdim=True)[0]
            right_region = src_mask[:, tgt_box[2]:].max(dim=0, keepdim=True)[0]
            src_masks_x.append(torch.cat([left_region, tgt_region_x, right_region], dim=1)[:, None])

        try:
            src_masks_y = torch.stack(src_masks_y, dim=0).flatten(1, 3)
            src_masks_x = torch.stack(src_masks_x, dim=0).flatten(1, 3)
        except:
            print("src_masks: ", src_masks.shape)
            print("tgt_boxes: ", tgt_boxes)
            for i in range(len(src_masks_x)):
                print('y:', src_masks_y[i].shape, 'x: ', src_masks_x[i])
        return src_masks_y, src_masks_x

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