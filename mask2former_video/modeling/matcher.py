    # Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast
from torchvision.ops.boxes import box_area

from detectron2.projects.point_rend.point_features import point_sample


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


def calculate_axis_projection(out_mask, tgt_box_mask, axis):
    """
    :param out_mask: (N, T, H, W)
    :param tgt_box_mask: (N, T, H, W)
    :param axis: axis to project
    :return:
    """
    src_mask_axis = out_mask.max(dim=axis, keepdim=True)[0].flatten(1, 3).float()  # (N, T*(H or W))
    tgt_box_mask_axis = tgt_box_mask.max(dim=axis, keepdim=True)[0].flatten(1, 3).float()  # (N, T*(H or W))

    # src_mask_axis_topk = out_mask.topk(3, dim=axis, sorted=False)[0].flatten(1, 3).float()
    # tgt_box_mask_axis_topk = tgt_box_mask.topk(3, dim=axis, sorted=False)[0].flatten(1, 3).float()
    return batch_dice_loss_jit(src_mask_axis, tgt_box_mask_axis)
    # return batch_dice_loss_jit(src_mask_axis_topk, tgt_box_mask_axis_topk)


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


def batch_projection_dice(inputs: torch.Tensor, inputs_MaskedByTgts: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss of projected mask, similar to generalized IOU for masks
    :param inputs: (Q, T*(W/H)))
    :param inputs_MaskedByTgts: (Q, G, T*(W/H))
    :param targets: (G, T*(W/H)))
    """

    def calculate_cost_mtrx(srcs, tgts):
        cost_mtx = []
        for src_ind in range(srcs.shape[0]):
            IdvSrc_MaskedByTgts = srcs[src_ind]  # (G, T*(W/H))

            cost_IdvSrc_to_Tgts = []
            for tgt_ind in range(tgts.shape[0]):
                IdvSrc_MasedByIdvTgt = IdvSrc_MaskedByTgts[tgt_ind]  # (T*(H/W),)
                tgt = tgts[tgt_ind]  # (T*(H/W),)

                sum_MaskedSrc = IdvSrc_MasedByIdvTgt.sum()
                sum_tgt = tgt.sum()

                cost_IdvMaskedSrc_to_IdvTgt = sum_MaskedSrc + sum_tgt
                cost_IdvSrc_to_Tgts.append(cost_IdvMaskedSrc_to_IdvTgt)

            cost_IdvSrc_to_Tgts = torch.stack(cost_IdvSrc_to_Tgts, dim=0)  # (G,)
            cost_mtx.append(cost_IdvSrc_to_Tgts)

        cost_mtx = torch.stack(cost_mtx, dim=0)
        return cost_mtx

    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = calculate_cost_mtrx(inputs_MaskedByTgts, targets)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def calculate_fg_projection_dice(out_masks, tgt_box_masks, axis):
    """
    calculate fg box projection cost
    :param out_masks: (N, T, H, W)
    :param tgt_box_masks: (M, T, H, W)
    :param axis
    :return: projection dice cost matrix of shape: (N, M)
    """
    out_masks = out_masks.sigmoid()

    out_masks_MaskedByTgts = []
    for src_ind in range(out_masks.shape[0]):
        out_IdvMask = out_masks[src_ind][None, :]
        out_IdvMask_MaskedByTgts = out_IdvMask * tgt_box_masks  # (M, T, H, W)
        out_masks_MaskedByTgts.append(out_IdvMask_MaskedByTgts)
    out_masks_MaskedByTgts = torch.stack(out_masks_MaskedByTgts, dim=0)  # (N, M, T, H, W)

    # (Q, G, T, H, W) -> (Q, G, T, (1/H), (1/W)) -> (Q, G, T*(W/H))
    out_masks_MaskedByTgts_axis = out_masks_MaskedByTgts.max(dim=axis+1, keepdim=True)[0].flatten(2, 4).float()

    # (Q, T, H, W) -> (Q, T, (1/H), (1/W)) -> (Q, T*(W/H)))
    out_masks_axis = out_masks.max(dim=axis, keepdim=True)[0].flatten(1, 3).float()
    tgt_box_masks_axis = tgt_box_masks.max(dim=axis, keepdim=True)[0].flatten(1, 3).float()

    cost_fg_proj = batch_fg_projection_dice(out_masks_axis, out_masks_MaskedByTgts_axis, tgt_box_masks_axis)
    return cost_fg_proj


def calculate_projection_dice(out_masks, tgt_box_masks, axis, bg_projection):
    """
    calculate fg box projection cost
    :param out_masks: (N, T, H, W)
    :param tgt_box_masks: (M, T, H, W)
    :param axis: int, axis to perform projection
    :param type: perform on foreground/background
    :return: projection dice cost matrix of shape: (N, M)
    """
    out_masks = out_masks.sigmoid()

    if bg_projection:
        out_masks = 1. - out_masks  # invert p to (1-p) as negative possibility
        tgt_box_masks = 1. - tgt_box_masks  # invert fg/bg label

    out_masks_MaskedByTgts = []
    for src_ind in range(out_masks.shape[0]):
        out_IdvMask = out_masks[src_ind][None, :]
        out_IdvMask_MaskedByTgts = out_IdvMask * tgt_box_masks  # (M, T, H, W)
        out_masks_MaskedByTgts.append(out_IdvMask_MaskedByTgts)
    out_masks_MaskedByTgts = torch.stack(out_masks_MaskedByTgts, dim=0)  # (N, M, T, H, W)

    # (Q, G, T, H, W) -> (Q, G, T, (1/H), (1/W)) -> (Q, G, T*(W/H))
    out_masks_MaskedByTgts_axis = out_masks_MaskedByTgts.max(dim=axis + 1, keepdim=True)[0].flatten(2, 4).float()

    # (Q, T, H, W) -> (Q, T, (1/H), (1/W)) -> (Q, T*(W/H)))
    out_masks_axis = out_masks.max(dim=axis, keepdim=True)[0].flatten(1, 3).float()
    tgt_box_masks_axis = tgt_box_masks.max(dim=axis, keepdim=True)[0].flatten(1, 3).float()

    cost_proj = batch_projection_dice(out_masks_axis, out_masks_MaskedByTgts_axis, tgt_box_masks_axis)
    return cost_proj


class VideoHungarianMatcherProjMask(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_projection: float = 1):
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
            cost_class = -out_prob[:, tgt_ids]  # (num_queries, num_classes) --> (num_queries, num_gt)

            out_mask = outputs["pred_masks"][b]  # [num_queries, T, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_box_mask = targets[b]["box_masks"].to(out_mask)  # [num_gts, T, H_pred, W_pred], 有可能有空的mask(dummy)

            if tgt_ids.shape[0] > 0:
                with autocast(enabled=False):
                    # original projection
                    # cost_projection = calculate_axis_projection(out_mask, tgt_box_mask, 3) + \
                    #                   calculate_axis_projection(out_mask, tgt_box_mask, 2)
                                      # calculate_axis_projection(out_mask, tgt_box_mask, 1)

                    # fg/bg projection
                    cost_projection = calculate_projection_dice(out_mask, tgt_box_mask, 3, False) + \
                                      calculate_projection_dice(out_mask, tgt_box_mask, 2, False) + \
                                      calculate_projection_dice(out_mask, tgt_box_mask, 3, True) + \
                                      calculate_projection_dice(out_mask, tgt_box_mask, 2, True)


            else:
                cost_projection = torch.zeros((100, 0), dtype=torch.float32, device=out_prob.device)

            # Final cost matrix
            C = (self.cost_class * cost_class + self.cost_projection * cost_projection)  # (num_query, num_gt)
            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))  # [( query_idxs: ndarray(num_gt,), gt_idxs: ndarray(num_gt,))]

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