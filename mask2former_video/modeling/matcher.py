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
from detectron2.structures.masks import BitMasks


#############################################
############ original projection ############
#############################################
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
    # inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_axis_projection(out_mask, tgt_box_mask, axis):
    """
    :param out_mask: (N, T, H, W)
    :param tgt_box_mask: (N, T, H, W)
    :param axis: axis to project
    :return:
    """
    out_mask = out_mask.sigmoid()
    src_mask_axis = out_mask.max(dim=axis, keepdim=True)[0].flatten(1, 3).float()  # (N, T*(H or W))
    tgt_box_mask_axis = tgt_box_mask.max(dim=axis, keepdim=True)[0].flatten(1, 3).float()  # (N, T*(H or W))

    return batch_dice_loss_jit(src_mask_axis, tgt_box_mask_axis)


#############################################
######### projection limited sample #########
#############################################
def batch_limited_sample_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    :param inputs: (Q, G, C)
    :param targets: (G, C)
    :return: loss: (Q, G)
    """
    numerator = 2 * torch.einsum("qgc,gc->qg", inputs, targets)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_limited_sample_dice_loss_jit = torch.jit.script(
    batch_limited_sample_dice_loss
)  # type: torch.jit.ScriptModule


def batch_axis_projection_limited_sample(out_mask, tgt_boxmask, tgt_boxmask_limited, axis=-1):
    """
    :param out_mask: (Q, T, H, W)
    :param tgt_boxmask: (G, T, H, W)
    :param tgt_boxmask_limited: (G, T, H, W)
    :param axis: -1(y-proj) or -2(x_proj)
    :return:
    """

    Q = out_mask.shape[0]
    G = tgt_boxmask.shape[0]
    T = out_mask.shape[1]
    H = out_mask.shape[2]
    W = out_mask.shape[3]

    out_mask = out_mask.sigmoid().flatten(1)  # (Q, T, H, W) -> (Q, T*H*W)
    tgt_boxmask_limited = tgt_boxmask_limited.flatten(1)  # (Q, T, H, W) -> (Q, T*H*W)

    # (Q, T*H*W) x (G, T*H*W) -> (Q, G, T*H*W) -> (Q, G, T, H, W)
    out_to_tgt_limited = torch.einsum('qc,gc->qgc', out_mask, tgt_boxmask_limited).view(Q, G, T, H, W)

    # (Q, G, T, H, W) -> (Q, G, T*H or T*W)
    out_to_tgt_proj = out_to_tgt_limited.max(dim=axis, keepdim=True)[0].flatten(2)
    # (G, T, H, W) -> (G, T*H or T*W)
    tgt_proj = tgt_boxmask.max(dim=axis, keepdim=True)[0].flatten(1)

    cost = batch_limited_sample_dice_loss_jit(out_to_tgt_proj, tgt_proj)  # (Q, G)
    return cost


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
    T = out_mask.shape[1]
    C = T * tgt_first_bounds.shape[-1]

    # (Q, T, H, W) -> (Q, T, H or W)
    out_mask_proj, proj_inds_axis = out_mask.max(dim=axis, keepdim=True)
    # (Q, T, H or W) -> (Q, C) -> (Q, C*G) -> (Q*G, C)
    out_mask_proj = out_mask_proj.flatten(1).repeat(1, G).view(Q*G, C)
    proj_inds_axis = proj_inds_axis.flatten(1).repeat(1, G).view(Q*G, C)

    # (G, T, H, W) -> (G, T, H or W) -> (G, C) -> (Q*G, C)
    tgt_boxmask_proj = tgt_boxmask.max(dim=axis, keepdim=True)[0].flatten(1).repeat(Q, 1)

    # (G, T, H or W) -> (G, C) -> (Q*G, C)
    tgt_first_bounds = tgt_first_bounds.flatten(1).repeat(Q, 1)
    tgt_second_bounds = tgt_second_bounds.flatten(1).repeat(Q, 1)

    # bool, (Q*G, C)
    flag_first = proj_inds_axis >= tgt_first_bounds
    flag_second = proj_inds_axis < tgt_second_bounds
    flag_proj = flag_first * flag_second  # (Q*G, C)

    tgt_boxmask_proj *= flag_proj

    cost = batch_limited_label_dice_loss_jit(out_mask_proj, tgt_boxmask_proj).view(Q, G)
    return cost

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
            tgt_boxmask = targets[b]["box_masks"].to(out_mask)  # [num_gts, T, H_pred, W_pred], 有可能有空的mask(dummy)
            # tgt_boxmask_limited_x = targets[b]["box_masks"].to(out_mask)  # [num_gts, T, H_pred, W_pred]
            # tgt_boxmask_limited_y = targets[b]["box_masks"].to(out_mask)  # [num_gts, T, H_pred, W_pred]
            tgt_left_bounds = targets[b]["left_bounds"].to(out_mask)
            tgt_right_bounds = targets[b]["right_bounds"].to(out_mask)
            tgt_top_bounds = targets[b]["top_bounds"].to(out_mask)
            tgt_bottom_bounds = targets[b]["bottom_bounds"].to(out_mask)

            if tgt_ids.shape[0] > 0:
                with autocast(enabled=False):
                    # original projection
                    # cost_projection = batch_axis_projection(out_mask, tgt_boxmask, 3) + \
                    #                   batch_axis_projection(out_mask, tgt_boxmask, 2)

                    # limited projection
                    # cost_projection = \
                    #     batch_axis_projection_limited_sample(out_mask, tgt_boxmask, tgt_boxmask_limited_x, axis=-1) + \
                    #     batch_axis_projection_limited_sample(out_mask, tgt_boxmask, tgt_boxmask_limited_y, axis=-2)

                    # projection limited label
                    cost_projection = \
                        batch_axis_projection_limited_label(
                            out_mask, tgt_boxmask, tgt_left_bounds, tgt_right_bounds, axis=-1
                        ) + \
                        batch_axis_projection_limited_label(
                            out_mask, tgt_boxmask, tgt_top_bounds, tgt_bottom_bounds, axis=-2
                        )
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