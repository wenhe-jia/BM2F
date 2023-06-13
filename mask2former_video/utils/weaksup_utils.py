from skimage import color

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from detectron2.projects.point_rend.point_features import point_sample
from detectron2.layers import cat

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


def get_obj_feats(feats_4x, boxes_4x):
    """

    :param feats_4x: (2, D, h, w)
    :param boxes_4x: (2, 4)
    :return:
    """
    # (D, h, w)
    obj_curr_feats = feats_4x[0, :, boxes_4x[0, 1]:boxes_4x[0, 3], boxes_4x[0, 0]:boxes_4x[0, 2]]
    obj_next_feats = feats_4x[1, :, boxes_4x[1, 1]:boxes_4x[1, 3], boxes_4x[1, 0]:boxes_4x[1, 2]]
    return obj_curr_feats, obj_next_feats


def generate_grid_coords(height, width):
    h_coords = torch.linspace(0, height - 1, height, dtype=torch.int32)
    w_coords = torch.linspace(0, width - 1, width, dtype=torch.int32)

    ys, xs = torch.meshgrid(h_coords, w_coords)  # (h, w)

    # xs = xs.transpose(0, 1)[:, :, None]
    # ys = ys.transpose(0, 1)[:, :, None]
    xs = xs[:, :, None]
    ys = ys[:, :, None]

    # (h, w, 2) -> (h*w, 2)
    return torch.cat([xs, ys], dim=-1).flatten(0, 1)


def calculate_patch_matching(
    obj_curr_feats,
    obj_next_feats,
    boxes_curr_and_next,
    topk_match=1,
):
    """

    :param obj_curr_feats: (D, h_curr, w_curr)
    :param obj_next_feats: (D, h_next, w_enxt)
    :param boxes_curr_and_next: (2, 4)
    :param topk_match: int
    :return:
    """
    next_feat_num = obj_next_feats.shape[1] * obj_next_feats.shape[2]
    if next_feat_num < topk_match:
        topk_match = next_feat_num

    # (obj_curr_h * obj_curr_w, 2)
    # grid_coords_obj_curr = generate_grid_coords(
    #     obj_curr_feats.shape[1], obj_curr_feats.shape[2]
    # ).to(device=obj_curr_feats.device)  # DEBUG
    grid_coords_obj_curr = generate_grid_coords(obj_curr_feats.shape[1], obj_curr_feats.shape[2])
    grid_coords_obj_curr[:, 0] += boxes_curr_and_next[0, 0]
    grid_coords_obj_curr[:, 1] += boxes_curr_and_next[0, 1]

    # grid_coords_obj_next = generate_grid_coords(
    #     obj_next_feats.shape[1], obj_next_feats.shape[2]
    # ).to(device=obj_curr_feats.device)  # DEBUG
    grid_coords_obj_next = generate_grid_coords(obj_next_feats.shape[1], obj_next_feats.shape[2])
    grid_coords_obj_next[:, 0] += boxes_curr_and_next[1, 0]
    grid_coords_obj_next[:, 1] += boxes_curr_and_next[1, 1]

    obj_curr_feats = obj_curr_feats.flatten(1, 2).permute(1, 0)  # (obj_curr_h * obj_curr_w, C)
    obj_next_feats = obj_next_feats.flatten(1, 2).permute(1, 0)  # (obj_next_h * obj_next_w, C)

    # select one point in current frame
    # random_ind = random.sample(range(grid_coords_obj_curr.shape[0]), 3 if 3 <= grid_coords_obj_curr.shape[0] else grid_coords_obj_curr.shape[0])
    # grid_coords_obj_curr = grid_coords_obj_curr[random_ind]
    # obj_curr_feats = obj_curr_feats[random_ind]

    # with autocast(enabled=True):
    # mem_bef_mtrx = torch.cuda.memory_allocated()
    # dist_mtrx = -torch.cdist(obj_curr_feats, obj_next_feats, p=2).half()  # DEBUG
    dist_mtrx = torch.cdist(obj_curr_feats, obj_next_feats, p=2).cpu()  # (obj_curr_h * obj_curr_w, obj_next_h * obj_next_w)
    # mem_af_mtrx = torch.cuda.memory_allocated()
    # print("after mtrx:", (torch.cuda.memory_allocated() - mem_bef_mtrx) / 1024**2, "MB", dist_mtrx.shape, dist_mtrx.dtype)
    # match_inds = torch.argsort(dist_mtrx, dim=1)[:, :topk_match]  # (im1_fg_num, im2_fg_num)
    match_inds = dist_mtrx.topk(k=topk_match, dim=1)[1]
    # print("after sort:", (torch.cuda.memory_allocated() - mem_af_mtrx) / 1024 ** 2, "MB", match_inds.dtype)

    # XY
    obj_curr_matched_coords = grid_coords_obj_curr[:, None].repeat(1, topk_match, 1).flatten(0, 1)

    obj_next_candidate_coords = grid_coords_obj_next[None].repeat(grid_coords_obj_curr.shape[0], 1, 1)
    obj_next_candidate_xs = obj_next_candidate_coords[:, :, 0]
    obj_next_candidate_ys = obj_next_candidate_coords[:, :, 1]

    obj_next_xs = obj_next_candidate_xs.gather(1, match_inds)
    obj_next_ys = obj_next_candidate_ys.gather(1, match_inds)
    obj_next_matched_coords = torch.cat([obj_next_xs[:, :, None], obj_next_ys[:, :, None]], dim=2).flatten(0, 1)

    return obj_curr_matched_coords.int(), obj_next_matched_coords.int()


def get_instance_temporal_pairs(feats, boxes, k=1):
    obj_curr_feats, obj_next_feats = get_obj_feats(feats, boxes)

    curr_matched_coords, next_matched_coords = calculate_patch_matching(
        obj_curr_feats, obj_next_feats, boxes, topk_match=k
    )
    assert curr_matched_coords.shape == next_matched_coords.shape

    return curr_matched_coords, next_matched_coords


def filter_temporal_pairs_by_color_similarity(
    coords_curr,
    coords_next,
    frame_lab_curr,
    frame_lab_next,
    color_similarity_threshold=0.3,
):
    """

    :param coords_curr: (k, 2)
    :param coords_next: (k, 2)
    :param image_curr: tensor(3, h, w)
    :param image_next: tensor(3, h, w)
    :param color_similarity_threshold:
    :return: coords_curr, coords_next
    """

    # (h, w, 3)
    # frame_lab_curr = color.rgb2lab(image_curr.byte().permute(1, 2, 0).cpu().numpy())
    # frame_lab_next = color.rgb2lab(image_next.byte().permute(1, 2, 0).cpu().numpy())
    # torch.as_tensor(frame_lab, device=downsampled_images.device, dtype=torch.float32)

    # （3, k)
    lab_pix_curr = frame_lab_curr[:, coords_curr[:, 1].long(), coords_curr[:, 0].long()]
    lab_pix_next = frame_lab_next[:, coords_next[:, 1].long(), coords_next[:, 0].long()]

    diff = lab_pix_curr - lab_pix_next
    similarity = torch.exp(-torch.norm(diff, dim=0) * 0.5)  # (k,)
    keep_ind = torch.where(similarity >= color_similarity_threshold)[0].cpu()
    # keep_ind = torch.where(similarity >= color_similarity_threshold)[0]  # DEBUG

    return coords_curr[keep_ind], coords_next[keep_ind]