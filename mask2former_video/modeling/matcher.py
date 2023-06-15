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

from ..utils.weaksup_utils import unfold_wo_center

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


#############################################
############# color similarity ##############
#############################################
def batch_pairwise_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    :param inputs: (Q, T, k*k-1, H, W)
    :param targets: (G, T, k*k-1, H, W)
    :return: cost matrix: (Q, G)
    """
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = torch.einsum("nc,mc->nm", inputs, targets)  # (Q, G)
    denominator = targets.sum(dim=1)[None, :].clamp(min=1.0)  # (1, G)
    loss = numerator / denominator
    return loss


batch_parirwise_loss_jit = torch.jit.script(
    batch_pairwise_loss
)  # type: torch.jit.ScriptModule


def calculate_similarity_cost_video(
    out_mask,
    tgt_box_mask,
    tgt_similarities,
    color_thr,
    kernel_size,
    dilation,
):
    """
    :param out_mask: (Q, T, H, W)
    :param tgt_box_mask: (G, T, H, W)
    :param tgt_similarities: (G, T, k*k-1, H, W)
    :param color_thr: float
    :param kernel_size: int
    :param dilation: int
    :return:
    """
    # only use pairwise color similarities inside the GT box
    assert len(tgt_box_mask.shape) == 4
    tgt_similarities = (tgt_similarities >= color_thr).float() * tgt_box_mask[:, :, None].float()  # (G, T, k*k-1, H, W)

    # Prepare data for pairwise loss
    log_fg_prob = F.logsigmoid(out_mask)  # (Q, T, H, W)
    log_bg_prob = F.logsigmoid(-out_mask)  # (Q, T, H, W)
    # 此处取出的值是每个点周围k*k个点的概率（前/背景）
    log_fg_prob_unfold = unfold_wo_center(
        log_fg_prob, kernel_size=kernel_size,
        dilation=dilation
    )  # (Q, T, k*k-1, H, W)
    log_bg_prob_unfold = unfold_wo_center(
        log_bg_prob, kernel_size=kernel_size,
        dilation=dilation
    )  # (Q, T, k*k-1, H, W)

    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    # (Q, T, H, W) -> (Q, T, 1, H, W), (Q, T, 1, H, W) + (Q, T, k*k-1, H, W) ->(Q, T, k*k-1, H, W)
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold  # (N, T, k*k-1, H, W)
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold  # (N, T, k*k-1, H, W)

    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)  # (N, T, k*k-1, H, W)
    src_similarities = -(torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_)  # (N, T, k*k-1, H, W)

    # return batch_parirwise_loss_jit(src_similarities.float(), tgt_similarities)

    cost = []
    T = tgt_box_mask.shape[1]
    for f_i in range(T):
        # (N, k*k-1, H, W) -> (N, E)
        src_sim_frame = src_similarities[:, f_i].flatten(1).float()
        tgt_sim_frame = tgt_similarities[:, f_i].flatten(1)
        numerator = torch.einsum("nc,mc->nm", src_sim_frame, tgt_sim_frame)  # (Q, G)
        denominator = tgt_sim_frame.sum(dim=1)[None, :].clamp(min=1.0)  # (1, G)
        cost_frame = numerator / denominator  # (Q, G)
        cost.append(cost_frame)  # (Q, G) -> (1, G, Q)

    return torch.stack(cost, dim=0).mean(0)


class VideoHungarianMatcherProjPair(nn.Module):
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

            out_mask = outputs["pred_masks"][b]  # (Q, T, H, W)

            # gt masks are already padded when preparing target
            tgt_boxmask = targets[b]["box_masks"].to(out_mask)  # (G, T, H, W), 有可能有空的mask(dummy)
            # tgt_left_bounds = targets[b]["left_bounds"].to(out_mask)  # (G, T, H)
            # tgt_right_bounds = targets[b]["right_bounds"].to(out_mask)  # (G, T, H)
            # tgt_top_bounds = targets[b]["top_bounds"].to(out_mask)  # (G, T, W)
            # tgt_bottom_bounds = targets[b]["bottom_bounds"].to(out_mask)  # (G, T, W)
            tgt_similarities = targets[b]["color_similarities"].to(out_mask)  # (G, T, k*k-1, H, W)

            if tgt_ids.shape[0] > 0:
                with autocast(enabled=False):
                    ##### projection #####
                    # original projection
                    cost_projection = batch_axis_projection(out_mask, tgt_boxmask, 3) +  \
                                      batch_axis_projection(out_mask, tgt_boxmask, 2)

                    # projection limited label
                    # cost_projection = \
                    #     batch_axis_projection_limited_label(
                    #         out_mask, tgt_boxmask, tgt_left_bounds, tgt_right_bounds, axis=-1
                    #     ) + \
                    #     batch_axis_projection_limited_label(
                    #         out_mask, tgt_boxmask, tgt_top_bounds, tgt_bottom_bounds, axis=-2
                    #     )

                    ##### color similarity #####
                    warmup_factor = min(self._iter.item() / float(self.pairwise_warmup_iters), 1.0)
                    cost_pairwise = calculate_similarity_cost_video(
                        out_mask, tgt_boxmask, tgt_similarities,
                        self.pairwise_color_thresh, self.pairwise_size, self.pairwise_dilation
                    ) * warmup_factor
            else:
                cost_projection = torch.zeros((num_queries, 0), dtype=torch.float32, device=out_prob.device)
                cost_pairwise = torch.zeros((num_queries, 0), dtype=torch.float32, device=out_prob.device)

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


class VideoHungarianMatcherProj(nn.Module):
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
            # tgt_left_bounds = targets[b]["left_bounds"].to(out_mask)
            # tgt_right_bounds = targets[b]["right_bounds"].to(out_mask)
            # tgt_top_bounds = targets[b]["top_bounds"].to(out_mask)
            # tgt_bottom_bounds = targets[b]["bottom_bounds"].to(out_mask)

            if tgt_ids.shape[0] > 0:
                with autocast(enabled=False):
                    # original projection
                    cost_projection = batch_axis_projection(out_mask, tgt_boxmask, 3) + \
                                      batch_axis_projection(out_mask, tgt_boxmask, 2)

                    # projection limited label
                    # cost_projection = \
                    #     batch_axis_projection_limited_label(
                    #         out_mask, tgt_boxmask, tgt_left_bounds, tgt_right_bounds, axis=-1
                    #     ) + \
                    #     batch_axis_projection_limited_label(
                    #         out_mask, tgt_boxmask, tgt_top_bounds, tgt_bottom_bounds, axis=-2
                    #     )
            else:
                cost_projection = torch.zeros((num_queries, 0), dtype=torch.float32, device=out_prob.device)

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

class VideoHungarianMatcher(nn.Module):
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

            out_mask = outputs["pred_masks"][b]  # [num_queries, T, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)  # [num_gts, T, H_pred, W_pred]

            # out_mask = out_mask[:, None]
            # tgt_mask = tgt_mask[:, None]
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).flatten(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).flatten(1)

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