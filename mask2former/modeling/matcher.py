# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import sys
import cv2
import numpy as np

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

from torchvision.ops.boxes import box_area

from detectron2.projects.point_rend.point_features import point_sample
from detectron2.structures.masks import BitMasks

from ..utils.weaksup_utils import unfold_wo_center


def batch_pairwise_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    :param inputs: (num_query, k*k-1, H/4, W/4)
    :param targets: (num_gt, k*k-1, H/4, W/4)
    :return: cost matrix: (num_query, num_gt)
    """
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)

    numerator = torch.einsum("nc,mc->nm", inputs, targets)
    denominator = targets.sum(dim=1)[None, :].clamp(min=1.0)  # (1, num_gt)
    loss = numerator / denominator
    return loss


batch_parirwise_loss_jit = torch.jit.script(
    batch_pairwise_loss
)  # type: torch.jit.ScriptModule


def batch_projection_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    inputs = inputs.sigmoid()
    eps = 1e-5
    numerator = torch.einsum("nc,mc->nm", inputs, targets)
    denominator = (inputs ** 2.0).sum(-1)[:, None] + (targets ** 2.0).sum(-1)[None, :]  # (num_pred, 1) + (1, num_gt)
    loss = 1. - (2 * numerator / denominator)
    return loss


batch_projection_dice_loss_jit = torch.jit.script(
    batch_projection_dice_loss
)  # type: torch.jit.ScriptModule


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


class HungarianMatcherProjMask(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
            self,
            cost_class: float = 1,
            cost_projection: float = 1,
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_projection = cost_projection

        assert cost_class != 0 or cost_projection != 0, "all costs cant be 0"

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):
            out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
            tgt_ids = targets[b]["labels"]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            out_mask = outputs["pred_masks"][b]  # (num_query, H_pred, W_pred)

            # gt masks are already padded when preparing target
            tgt_box_mask = targets[b]["box_masks"].to(out_mask)

            out_mask = out_mask[:, None]  # (num_query, 1, H_pred, W_pred)
            tgt_box_mask = tgt_box_mask[:, None]  # (num_gt, 1, H_pred, W_pred)

            # project mask to x & y axis
            # masks_x: (num_ins, 1, H_pad/4, 1)->(num_ins, H_pad/4)
            # masks_y: (num_ins, 1, 1, W_pad/4)->(num_ins, W_pad/4)
            src_mask_x = out_mask.max(dim=3, keepdim=True)[0].flatten(1, 3).float()
            src_mask_y = out_mask.max(dim=2, keepdim=True)[0].flatten(1, 3).float()

            with torch.no_grad():
                tgt_box_mask_x = tgt_box_mask.max(dim=3, keepdim=True)[0].flatten(1, 3).float()
                tgt_box_mask_y = tgt_box_mask.max(dim=2, keepdim=True)[0].flatten(1, 3).float()

            with autocast(enabled=False):
                src_mask_x = src_mask_x.float()
                tgt_mask_x = tgt_box_mask_x.float()

                src_mask_y = src_mask_y.float()
                tgt_mask_y = tgt_box_mask_y.float()

                # Compute the dice loss between masks
                cost_dice_x = batch_dice_loss_jit(src_mask_x, tgt_box_mask_x)
                cost_dice_y = batch_dice_loss_jit(src_mask_y, tgt_box_mask_y)
                cost_projection = cost_dice_x + cost_dice_y

            # Final cost matrix
            C = (self.cost_class * cost_class + self.cost_projection * cost_projection)

            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_projection: {}".format(self.cost_projection),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class HungarianMatcherWeakSup(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
            self,
            cost_class: float = 1,
            cost_projection: float = 1,
            cost_pairwise: float = 1,
            pairwise_size: int = 3,
            pairwise_dilation: int = 2,
            pairwise_color_thresh: float = 0.3,
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_projection = cost_projection
        self.cost_pairwise = cost_pairwise

        self.pairwise_size = pairwise_size
        self.pairwise_dilation = pairwise_dilation
        self.pairwose_color_thresh = pairwise_color_thresh

        assert cost_class != 0 or cost_projection != 0 or cost_pairwise != 0, "all costs cant be 0"

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):
            out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
            tgt_ids = targets[b]["labels"]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            out_mask = outputs["pred_masks"][b]  # (num_query, H_pred, W_pred)

            # gt masks are already padded when preparing target
            tgt_box_mask = targets[b]["box_masks"].to(out_mask)

            out_mask = out_mask[:, None]  # (num_query, 1, H_pred, W_pred)
            tgt_box_mask = tgt_box_mask[:, None]  # (num_gt, 1, H_pred, W_pred)

            # images_color_similarities = targets[b]["images_color_similarity"].to(out_mask)  # (num_gt, k*k-1, H/4, W/4)

            # project mask to x & y axis
            # masks_x: (num_ins, 1, H_pad/4, 1)->(num_ins, H_pad/4)
            # masks_y: (num_ins, 1, 1, W_pad/4)->(num_ins, W_pad/4)
            src_mask_x = out_mask.max(dim=3, keepdim=True)[0].flatten(1, 3).float()
            src_mask_y = out_mask.max(dim=2, keepdim=True)[0].flatten(1, 3).float()

            # # Prepare data for pairwise loss
            # log_fg_prob = F.logsigmoid(out_mask)
            # log_bg_prob = F.logsigmoid(-out_mask)
            # # 此处取出的值是每个点周围k*k个点的概率（前/背景）
            # log_fg_prob_unfold = unfold_wo_center(
            #     log_fg_prob, kernel_size=self.pairwise_size,
            #     dilation=self.pairwise_dilation
            # )  # (N, 1, k*k-1, H/4, W/4)
            # log_bg_prob_unfold = unfold_wo_center(
            #     log_bg_prob, kernel_size=self.pairwise_size,
            #     dilation=self.pairwise_dilation
            # )  # (N, 1, k*k-1, H/4, W/4)
            #
            # # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
            # # we compute the the probability in log space to avoid numerical instability
            # log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold  # (N, 1, k*k-1, H/4, W/4)
            # log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold  # (N, 1, k*k-1, H/4, W/4)
            #
            # max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
            # log_same_prob = torch.log(
            #     torch.exp(log_same_fg_prob - max_) +
            #     torch.exp(log_same_bg_prob - max_)
            # ) + max_
            # src_similarities = -log_same_prob[:, 0]  # (N, k*k-1, H/4, W/4)

            with torch.no_grad():
                tgt_mask_x = tgt_box_mask.max(dim=3, keepdim=True)[0].flatten(1, 3).float()
                tgt_mask_y = tgt_box_mask.max(dim=2, keepdim=True)[0].flatten(1, 3).float()
                # tgt_similarities = ((images_color_similarities >= self.pairwose_color_thresh).float() * \
                #                    tgt_box_mask.float())

            with autocast(enabled=False):
                src_mask_x = src_mask_x.float()
                tgt_mask_x = tgt_mask_x.float()

                src_mask_y = src_mask_y.float()
                tgt_mask_y = tgt_mask_y.float()

                # Compute the dice cost
                cost_dice_x = batch_dice_loss_jit(src_mask_x, tgt_mask_x)
                cost_dice_y = batch_dice_loss_jit(src_mask_y, tgt_mask_y)
                cost_projection = cost_dice_x + cost_dice_y

                # Compute the pairwise cost
                # cost_pairwise = batch_parirwise_loss_jit(src_similarities, tgt_similarities)


            # Final cost matrix
            C = (
                self.cost_class * cost_class +
                self.cost_projection * cost_projection
                # self.cost_pairwise * cost_pairwise
            )

            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_projection: {}".format(self.cost_projection),
            "cost_pairwise: {}".format(self.cost_pairwise),
            "pairwise_size: {}".format(self.pairwise_size),
            "pairwise_dilation: {}".format(self.pairwise_dilation),
            "pairwose_color_thresh: {}".format(self.pairwose_color_thresh),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class HungarianMatcherMask(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
            self,
            cost_class: float = 1,
            cost_mask: float = 1,
            cost_dice: float = 1,
            num_points: int = 0,
            target_type: str="mask"
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points
        self.target_type = target_type

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):
            out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
            tgt_ids = targets[b]["labels"]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            out_mask = outputs["pred_masks"][b]  # (num_query, H_pred, W_pred)

            # gt masks are already padded when preparing target
            if self.target_type == "mask":
                tgt_mask = targets[b]["masks"].to(out_mask)
            elif self.target_type == "box_mask":
                tgt_mask = targets[b]["box_masks_full"].to(out_mask)
            else:
                raise Exception("Unknown target type !!!")

            out_mask = out_mask[:, None]  # (num_query, 1, H_pred, W_pred)
            tgt_mask = tgt_mask[:, None]  # (num_gt, 1, H_pred, W_pred)
            
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)  # (num_gt, 1, num_points) -> (num_gt, num_points)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)  # (num_query, 1, num_points) -> (num_query, num_points)

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)

                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)

            # Final cost matrix
            C = (
                    self.cost_mask * cost_mask
                    + self.cost_class * cost_class
                    + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
