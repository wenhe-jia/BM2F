import torch
import torch.nn.functional as F

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
    )  # (1, 3*k*k, h*w )

    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    )  # (1, 3, k*k, h, w)

    # remove the center pixels
    size = kernel_size ** 2
    unfolded_x = torch.cat((
        unfolded_x[:, :, :size // 2],
        unfolded_x[:, :, size // 2 + 1:]
    ), dim=2)  # (1, 3, k*k-1, h, w)

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
    # diff: (1, 3, 1, H/4, W/4) - (1, 3, k*k-1, H/4, W/4) -> edges: (1, 3, k*k-1, H/4, W/4)
    # 一个像素和周边8个像素的在lab空间3通道的减法（差值）
    similarity = torch.exp(-torch.norm(diff, dim=1) * 0.5)  # (1, k*k-1, H/4, W/4)

    unfolded_weights = unfold_wo_center(
        image_masks[None, None], kernel_size=kernel_size,
        dilation=dilation
    )  # (1, 1, k*k-1, H/4, W/4)
    unfolded_weights = torch.max(unfolded_weights, dim=1)[0]  # (1, k*k-1, H/4, W/4)

    return similarity * unfolded_weights