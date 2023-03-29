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
    :param inputs: (Q, k*k-1, H/4, W/4)
    :param targets: (G, k*k-1, H/4, W/4)
    :return: cost matrix: (Q, G)
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


def calculate_axis_projection(out_mask, tgt_box_mask, axis):
    with autocast(enabled=False):
        src_mask_axis = out_mask.max(dim=axis, keepdim=True)[0].flatten(1, 3).float()
        tgt_box_mask_axis = tgt_box_mask.max(dim=axis, keepdim=True)[0].flatten(1, 3).float()
    return batch_dice_loss_jit(src_mask_axis, tgt_box_mask_axis)

def calculate_similarity_cost(
    out_mask,
    tgt_box_mask,
    tgt_similarities,
    color_thr,
    kernel_size,
    dilation,
):
    with autocast(enabled=False):
        # only use pairwise color similarities inside the GT box
        tgt_similarities = (tgt_similarities >= color_thr).float() * tgt_box_mask.float()  # (G, k*k-1, H, W)

        # Prepare data for pairwise loss
        log_fg_prob = F.logsigmoid(out_mask)  # (Q, 1, H, W)
        log_bg_prob = F.logsigmoid(-out_mask)  # (Q, 1, H, W)
        # 此处取出的值是每个点周围k*k个点的概率（前/背景）
        log_fg_prob_unfold = unfold_wo_center(
            log_fg_prob, kernel_size=kernel_size,
            dilation=dilation
        )  # (N, 1, k*k-1, H/4, W/4)
        log_bg_prob_unfold = unfold_wo_center(
            log_bg_prob, kernel_size=kernel_size,
            dilation=dilation
        )  # (N, 1, k*k-1, H/4, W/4)

        # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
        # we compute the the probability in log space to avoid numerical instability
        # (Q, 1, H, W) -> (Q, 1, 1, H, W), (Q, 1, 1, H, W) + (Q, 1, k*k-1, H, W) ->(Q, 1, k*k-1, H, W)
        log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold  # (N, 1, k*k-1, H/4, W/4)
        log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold  # (N, 1, k*k-1, H/4, W/4)

        max_ = torch.max(log_same_fg_prob, log_same_bg_prob)  # (N, 1, k*k-1, H/4, W/4)
        # (N, 1, k*k-1, H/4, W/4) ->   # (N, k*k-1, H/4, W/4)
        src_similarities = -(torch.log(
            torch.exp(log_same_fg_prob - max_) +
            torch.exp(log_same_bg_prob - max_)
        ) + max_)[:, 0]

        return batch_parirwise_loss_jit(src_similarities.float(), tgt_similarities)


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


#############################################
######### projection limited label ##########
#############################################
def batch_limited_label_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    :param inputs: (Q*G, C)
    :param targets: (Q*G, C)
    :return: loss: (Q, G)
    """
    numerator = 2 * (inputs * targets).sum(-1)  # (Q*G)
    denominator = inputs.sum(-1) + targets.sum(-1)  # (Q*G)
    loss = 1 - (numerator + 1) / (denominator + 1)  # (Q*G)
    return loss


batch_limited_label_dice_loss_jit = torch.jit.script(
    batch_limited_label_dice_loss
)  # type: torch.jit.ScriptModule


def batch_axis_projection_limited_label(
    out_mask,
    tgt_boxmask,
    tgt_first_bounds,
    tgt_second_bounds,
    axis=-1
):
    out_mask = out_mask.sigmoid()

    Q = out_mask.shape[0]
    G = tgt_boxmask.shape[0]
    T = 1
    C = T * tgt_first_bounds.shape[-1]

    # (Q, 1, H, W) -> (Q, 1, H or W)
    out_mask_proj, proj_inds_axis = out_mask.max(dim=axis, keepdim=True)
    # (Q, 1, H or W) -> (Q, C) -> (Q, C*G) -> (Q*G, C)
    out_mask_proj = out_mask_proj.flatten(1).repeat(1, G).view(Q*G, C)
    proj_inds_axis = proj_inds_axis.flatten(1).repeat(1, G).view(Q*G, C)

    # (G, 1, H, W) -> (G, 1, H or W) -> (G, C) -> (Q*G, C)
    tgt_boxmask_proj = tgt_boxmask.max(dim=axis, keepdim=True)[0].flatten(1).repeat(Q, 1)

    # (G, 1, H or W) -> (G, C) -> (Q*G, C)
    tgt_first_bounds = tgt_first_bounds.flatten(1).repeat(Q, 1)
    tgt_second_bounds = tgt_second_bounds.flatten(1).repeat(Q, 1)

    # bool, (Q*G, C)
    flag_first = proj_inds_axis >= tgt_first_bounds
    flag_second = proj_inds_axis < tgt_second_bounds
    flag_proj = flag_first * flag_second  # (Q*G, C)

    tgt_boxmask_proj *= flag_proj

    cost = batch_limited_label_dice_loss_jit(out_mask_proj, tgt_boxmask_proj).view(Q, G)
    return cost


class HungarianMatcherProjPair(nn.Module):
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
            pairwise_warmup_iters: int = 10000,
            point_sample: bool = False,
            num_points: int = 12544

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
        self.pairwise_color_thresh = pairwise_color_thresh
        self.pairwise_warmup_iters = pairwise_warmup_iters

        self.point_sample = point_sample
        self.num_points = num_points

        assert cost_class != 0 or cost_projection != 0 or cost_pairwise != 0, "all costs cant be 0"

        # for pairwise loss warmup
        self.register_buffer("_iter", torch.zeros([1]))

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

            tgt_similarities = targets[b]["images_color_similarity"].to(out_mask)  # (num_gt, k*k-1, H/4, W/4)

            # # Compute the dice loss between masks
            cost_projection = calculate_axis_projection(out_mask, tgt_box_mask, 3) +  \
                              calculate_axis_projection(out_mask, tgt_box_mask, 2)

            warmup_factor = min(self._iter.item() / float(self.pairwise_warmup_iters), 1.0)
            cost_pairwise = calculate_similarity_cost(
                out_mask, tgt_box_mask, tgt_similarities,
                self.pairwise_color_thresh, self.pairwise_size, self.pairwise_dilation
            ) * warmup_factor

            # Final cost matrix
            C = (
                    self.cost_class * cost_class +
                    self.cost_projection * cost_projection +
                    self.cost_pairwise * cost_pairwise
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
        self._iter += 1
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_projection: {}".format(self.cost_projection),
            "cost_pairwise: {}".format(self.cost_pairwise),
            "pairwise_size: {}".format(self.pairwise_size),
            "pairwise_dilation: {}".format(self.pairwise_dilation),
            "pairwise_color_thresh: {}".format(self.pairwise_color_thresh),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class HungarianMatcherProj(nn.Module):
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
        # print('=' * 10, 'Matcher proj', '=' * 10)
        # m1 = torch.cuda.memory_allocated()
        # print('matcher start memory cost:', m1 / (1024 * 1024), 'MB')
        # mmax = torch.cuda.max_memory_allocated()
        # print('matcher start max memory cost:', mmax / (1024 * 1024), 'MB')
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []
        update_target = True

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
            tgt_left_bounds = targets[b]["left_bounds"].to(out_mask)
            tgt_right_bounds = targets[b]["right_bounds"].to(out_mask)
            tgt_top_bounds = targets[b]["top_bounds"].to(out_mask)
            tgt_bottom_bounds = targets[b]["bottom_bounds"].to(out_mask)

            out_mask = out_mask[:, None]  # (num_query, 1, H_pred, W_pred)
            tgt_boxmask = tgt_box_mask[:, None]  # (num_gt, 1, H_pred, W_pred)
            tgt_left_bounds = tgt_left_bounds[:, None]  # (num_gt, 1, H_pred)
            tgt_right_bounds = tgt_right_bounds[:, None]  # (num_gt, 1, H_pred)
            tgt_top_bounds = tgt_top_bounds[:, None]  # (num_gt, 1, W_pred)
            tgt_bottom_bounds = tgt_bottom_bounds[:, None]  # (num_gt, 1, W_pred)

            # Compute the dice loss between masks
            # cost_projection = calculate_axis_projection(out_mask, tgt_boxmask, 3) +  \
            #                   calculate_axis_projection(out_mask, tgt_boxmask, 2)

            # projection limited label
            cost_projection = \
                batch_axis_projection_limited_label(
                    out_mask, tgt_boxmask, tgt_left_bounds, tgt_right_bounds, axis=-1
                ) + \
                batch_axis_projection_limited_label(
                    out_mask, tgt_boxmask, tgt_top_bounds, tgt_bottom_bounds, axis=-2
                )

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


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 0):
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

            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)

            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

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
