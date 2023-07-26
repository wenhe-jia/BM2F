import torch
import torch.nn.functional as F

from detectron2.projects.point_rend.point_features import point_sample
from detectron2.layers import cat

import numpy as np
from skimage import color


def unfold_wo_center(x, kernel_size, dilation):
    # 在每个点的周围取一系列值，图像之外pad的部分就是0了
    assert x.dim() == 4
    assert kernel_size % 2 == 1

    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    )  # (N, C, H, W) --unfold--> (N, C*k*k, H*W)

    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    )  # (N, C*k*k, H*W) --> (N, C, k*k, H, W)

    # remove the center pixels
    size = kernel_size ** 2
    unfolded_x = torch.cat((
        unfolded_x[:, :, :size // 2],
        unfolded_x[:, :, size // 2 + 1:]
    ), dim=2)  # (N, C, k*k-1, H, W)

    return unfolded_x


def get_images_color_similarity(images, image_masks, kernel_size, dilation):
    # images: (1, 3, H/4, W/4)
    # image_masks: torch.ones(H/4, W/4)
    # kernel_size: 3
    # dilation: 2
    assert images.dim() == 4
    assert images.size(0) == 1

    unfolded_images = unfold_wo_center(
        images, kernel_size=kernel_size, dilation=dilation
    )  # (1, 3, k*k-1, h, w)

    diff = images[:, :, None] - unfolded_images
    # diff(edge): (1, 3, 1, H/4, W/4) - (1, 3, k*k-1, H/4, W/4) -> edges: (1, 3, k*k-1, H/4, W/4)
    # 一个像素和周边8个像素的在lab空间3通道的减法（差值）
    similarity = torch.exp(-torch.norm(diff, dim=1) * 0.5)  # edge: (1, k*k-1, H/4, W/4)

    # for COCO
    unfolded_weights = unfold_wo_center(
        image_masks[None, None], kernel_size=kernel_size,
        dilation=dilation
    )  # (1, 1, k*k-1, H/4, W/4)
    unfolded_weights = torch.max(unfolded_weights, dim=1)[0]  # (1, k*k-1, H/4, W/4)

    return similarity * unfolded_weights


def get_inconstant_point_coords_with_randomness(
    coarse_logits, variance_func, num_points, oversample_ratio, importance_sample_ratio
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.

    Args:
        coarse_logits (Tensor): A tensor of shape (N, k**2-1, Hmask, Wmask)
        uncertainty_func: A function that takes a Tensor of shape (N, k**2-1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.

    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)  # (num_ins, 3*num_p, 2)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)  # (N, k**2-1, 3*num_p)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = variance_func(point_logits)  #（N, 1, 3*num_p)
    num_uncertain_points = int(importance_sample_ratio * num_points)  # 0.75*num_p
    num_random_points = num_points - num_uncertain_points  # num_p - int(0.25*num_p)
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]  # (num_ins, 0.75*num_p)
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)  # (num_ins)
    idx += shift[:, None]  # (num_ins, 0.75*num_p)
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords


def filter_temporal_pairs_by_color_similarity(
    coords_curr,
    coords_next,
    frame_curr,
    frame_next,
    color_similarity_threshold=0.3,
    input_image=False
):
    """

    :param coords_curr: (k, 2)
    :param coords_next: (k, 2)
    :param frame_curr: tensor(3, h, w)
    :param frame_next: tensor(3, h, w)
    :param color_similarity_threshold: float
    :param input_image: bool
    :return: coords_curr, coords_next
    """

    if input_image:
        frame_lab_curr = torch.as_tensor(
            color.rgb2lab(frame_curr.byte().permute(1, 2, 0).cpu().numpy()),
            device=frame_curr.device, dtype=torch.float32
        ).permute(2, 0, 1)
        frame_lab_next = torch.as_tensor(
            color.rgb2lab(frame_next.byte().permute(1, 2, 0).cpu().numpy()),
            device=frame_next.device, dtype=torch.float32
        ).permute(2, 0, 1)
    else:
        frame_lab_curr = frame_curr
        frame_lab_next = frame_next

    # （3, k)
    lab_pix_curr = frame_lab_curr[:, coords_curr[:, 1].long(), coords_curr[:, 0].long()]
    lab_pix_next = frame_lab_next[:, coords_next[:, 1].long(), coords_next[:, 0].long()]

    diff = lab_pix_curr - lab_pix_next
    similarity = torch.exp(-torch.norm(diff, dim=0) * 0.5)  # (k,)
    # keep_ind = torch.where(similarity >= color_similarity_threshold)[0].cpu()
    keep_ind = torch.where(similarity >= color_similarity_threshold)[0]  # DEBUG
    return coords_curr[keep_ind].to(dtype=torch.int16), coords_next[keep_ind].to(dtype=torch.int16)