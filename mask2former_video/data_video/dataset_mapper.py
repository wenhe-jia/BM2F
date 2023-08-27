# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import copy
import logging
import random
import json
import numpy as np
from typing import List, Union
import torch

from detectron2.config import configurable
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
)

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from .augmentation import build_augmentation

__all__ = ["YTVISDatasetMapperWithCoords", "YTVISDatasetMapper", "CocoClipDatasetMapper"]


def filter_empty_instances(instances, by_box=True, by_mask=True, box_threshold=1e-5):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty

    Returns:
        Instances: the filtered instances.
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x

    instances.gt_ids[~m] = -1
    return instances


def _get_dummy_anno(num_classes):
    return {
        "iscrowd": 0,
        "category_id": num_classes,
        "id": -1,
        "bbox": np.array([0, 0, 0, 0]),
        "bbox_mode": BoxMode.XYXY_ABS,
        "segmentation": [np.array([0.0] * 6)]
    }


def ytvis_annotations_to_instances(annos, image_size):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes", "gt_ids",
            "gt_masks", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)

    classes = [int(obj["category_id"]) for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    ids = [int(obj["id"]) for obj in annos]
    ids = torch.tensor(ids, dtype=torch.int64)
    target.gt_ids = ids

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        masks = []
        for segm in segms:
            assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                segm.ndim
            )
            # mask array
            masks.append(segm)
        # torch.from_numpy does not support array with negative stride.
        masks = BitMasks(
            torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
        )
        target.gt_masks = masks

    return target


class YTVISDatasetMapperWithCoords:
    """
    A callable which takes a dataset dict in YouTube-VIS Dataset format,
    and map it into a format used by the model.
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        sampling_frame_num: int = 2,
        sampling_frame_range: int = 5,
        sampling_frame_shuffle: bool = False,
        num_classes: int = 40,
        temporal_topk: int = 1,
        temp_dist_thr: float = 0.9,
        use_input_resolution_for_temp: bool = False,
        fixed_sampling_interval: bool = False,
        pca_split_match: bool = False,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
        """
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.sampling_frame_num     = sampling_frame_num
        self.sampling_frame_range   = sampling_frame_range
        self.sampling_frame_shuffle = sampling_frame_shuffle
        self.num_classes            = num_classes

        self.temporal_topk                 = temporal_topk
        self.temp_dist_thr                 = temp_dist_thr
        self.use_input_resolution_for_temp = use_input_resolution_for_temp
        self.fixed_sampling_interval       = fixed_sampling_interval
        self.pca_split_match               = pca_split_match

        # matching_file_path = "matching_coords_vitg_ShortRange_NormDist_top{}_frame{}".format(
        #     temporal_topk,
        #     sampling_frame_num
        # )
        matching_file_path = "matching_coords_vitg_ShortRange_NormDist_top10_frame3"
        self.matching_file_path = matching_file_path
        assert self.temporal_topk >= 1

        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = build_augmentation(cfg, is_train)

        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
        sampling_frame_range = cfg.INPUT.SAMPLING_FRAME_RANGE
        sampling_frame_shuffle = cfg.INPUT.SAMPLING_FRAME_SHUFFLE

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "sampling_frame_num": sampling_frame_num,
            "sampling_frame_range": sampling_frame_range,
            "sampling_frame_shuffle": sampling_frame_shuffle,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "temporal_topk": cfg.MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.TEMPORAL_TOPK,
            "temp_dist_thr": cfg.MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.TEMPORAL_DIST_THRESH,
            "use_input_resolution_for_temp": cfg.MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.USE_INPUT_RESOLUTION,
            "fixed_sampling_interval": cfg.INPUT.FIXED_SAMPLING_INTERVAL,
            "pca_split_match": cfg.MODEL.MASK_FORMER_VIDEO.WEAK_SUPERVISION.PCA_SPLIT_MATCH,
        }

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one video, in YTVIS Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # TODO consider examining below deepcopy as it costs huge amount of computations.
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        video_length = dataset_dict["length"]
        if self.is_train:

            if self.fixed_sampling_interval:
                # frame2, fix interval sampling
                if self.sampling_frame_num == 2:
                    if 0 < video_length <= 8:
                        interval = 2
                    elif 8 < video_length <= 16:
                        interval = 3
                    else:
                        interval = 5

                    # if 0 < video_length <= 10:
                    #     interval = 4
                    # elif 10 < video_length <= 20:
                    #     interval = 10
                    # elif 20 < video_length <= 30:
                    #     interval = 15
                    # elif 30 < video_length <= 40:
                    #     interval = 20
                    # elif 40 < video_length:
                    #     interval = 36
                    # else:
                    #     raise ValueError("video length is not valid")

                    ref_frame = random.randrange(video_length - interval)
                    tgt_frame = ref_frame + interval
                    selected_idx = [ref_frame, tgt_frame]

                # frame 3, fix interval sampling
                elif self.sampling_frame_num == 3:
                    if 0 < video_length <= 8:
                        interval = 2
                    elif 8 < video_length <= 16:
                        interval = 3

                    else:
                        interval = 5

                    # elif 16 < video_length <= 24:
                    #     interval = 7
                    # elif 24 < video_length <= 32:
                    #     interval = 10
                    # elif 32 < video_length <= 40:
                    #     interval = 14
                    # elif 40 < video_length <= 48:
                    #     interval = 18
                    # elif 48 < video_length:
                    #     interval = 23
                    # else:
                    #     raise ValueError("video length is not valid")

                    ref_frame = random.randrange(video_length - 2 * interval)
                    first_tgt_frame = ref_frame + interval
                    second_tgt_frame = first_tgt_frame + interval
                    selected_idx = [ref_frame, first_tgt_frame, second_tgt_frame]

                else:
                    raise NotImplementedError
            else:
                # random adaptive sampling
                ref_frame = random.randrange(video_length)

                start_idx = max(0, ref_frame-self.sampling_frame_range)
                end_idx = min(video_length, ref_frame+self.sampling_frame_range + 1)

                selected_idx = np.random.choice(
                    np.array(list(range(start_idx, ref_frame)) + list(range(ref_frame+1, end_idx))),
                    self.sampling_frame_num - 1,
                )
                selected_idx = selected_idx.tolist() + [ref_frame]
                selected_idx = sorted(selected_idx)
                if self.sampling_frame_shuffle:
                    random.shuffle(selected_idx)
        else:
            selected_idx = range(video_length)

        video_annos = dataset_dict.pop("annotations", None)
        file_names = dataset_dict.pop("file_names", None)

        if self.is_train:
            _ids = set()
            for frame_idx in selected_idx:
                _ids.update([anno["id"] for anno in video_annos[frame_idx]])
            ids = dict()
            for i, _id in enumerate(_ids):
                ids[_id] = i
            # ids: {dataset_ins_id: video_ins_id}
            # print(ids)

            ##### load matching coordinates from json file #####
            match_coords_video = {}
            for gt_id, _id in ids.items():
                match_coords_video.update(
                    {
                        gt_id: [
                            {
                                "curr_pts": torch.zeros((0, 2), dtype=torch.int16),
                                "next_pts": torch.zeros((0, 2), dtype=torch.int16),
                            } for _ in range(self.sampling_frame_num - 1)
                        ]
                    }
                )
            dataset_dict['match_coords'] = match_coords_video


        dataset_dict["image"] = []
        dataset_dict["instances"] = []
        dataset_dict["file_names"] = []
        for f_i, frame_idx in enumerate(selected_idx):
            dataset_dict["file_names"].append(file_names[frame_idx])

            # Read image
            image = utils.read_image(file_names[frame_idx], format=self.image_format)
            utils.check_image_size(dataset_dict, image)

            aug_input = T.AugInput(image)
            transforms = self.augmentations(aug_input)
            image = aug_input.image

            image_shape = image.shape[:2]  # h, w
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))

            if (video_annos is None) or (not self.is_train):
                continue

            # NOTE copy() is to prevent annotations getting changed from applying augmentations
            _frame_annos = []  # single frame annotations
            for anno in video_annos[frame_idx]:
                _anno = {}
                for k, v in anno.items():
                    _anno[k] = copy.deepcopy(v)
                _frame_annos.append(_anno)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in _frame_annos
                if obj.get("iscrowd", 0) == 0
            ]
            # 先为每个instance生成一个dummy obj
            sorted_annos = [_get_dummy_anno(self.num_classes) for _ in range(len(ids))]

            for _anno in annos:
                idx = ids[_anno["id"]]  # 这个 video 有这个实例，但是在这一帧不一定有，没有就是 dummy obj
                sorted_annos[idx] = _anno
            _gt_ids = [_anno["id"] for _anno in sorted_annos]

            # check whether horizontal flip is applied
            do_hflip = sum(isinstance(t, T.HFlipTransform) for t in transforms.transforms) % 2 == 1
            if f_i < len(selected_idx) - 1:
                matching_file_name = file_names[frame_idx].replace(
                    "JPEGImages", self.matching_file_path
                ).replace(
                    "{}.jpg".format(file_names[frame_idx].split("/")[-1].split(".")[0]),
                    "frame_{}_{}.json".format(
                        file_names[frame_idx].split("/")[-1].split(".")[0],
                        file_names[selected_idx[f_i + 1]].split("/")[-1].split(".")[0],
                    )
                )

                with open(matching_file_name, "r") as f:
                    matching_info = json.load(f)
                    assert matching_info["interval"] == interval

                    if matching_info["tfmd_size"] is not None:
                        # calculate scale factor
                        org_shape = (matching_info["tfmd_size"][0], matching_info["tfmd_size"][1])
                        scale_factor = image_shape[0] / org_shape[0]

                        ins_matches = matching_info["matching"]
                        match_ins_ids = []
                        for i, ins_match in enumerate(ins_matches):
                            for _id, _match_coords in ins_match.items():
                                match_ins_ids.append(int(_id))
                        assert len(ins_matches) <= len(_gt_ids), "{} | {}".format(match_ins_ids, _gt_ids)

                        # 当前两帧中所有实例的匹配关系，每个实例的匹配关系是一个dict，包含id和coords，
                        # 每个实例的匹配关系是一个dict，包含curr_pts和next_pts
                        for i, ins_match in enumerate(ins_matches):
                            for _id, _match_contents in ins_match.items():
                                _id = int(_id)
                                assert _id in _gt_ids

                                if self.pca_split_match:
                                    # PCA split matching
                                    # main
                                    main_contents = _match_contents["main"]
                                    main_curr_pts_ins = torch.as_tensor(
                                        np.ascontiguousarray(np.array(main_contents["curr_pts"]) * scale_factor)
                                    )
                                    main_next_pts_ins = torch.as_tensor(
                                        np.ascontiguousarray(np.array(main_contents["next_pts"]) * scale_factor)
                                    )
                                    main_dists = torch.as_tensor(
                                        np.ascontiguousarray(np.array(main_contents["dist"]))
                                    )
                                    main_saved_topk = main_contents["topk"]
                                    assert main_curr_pts_ins.shape == main_next_pts_ins.shape

                                    if main_curr_pts_ins.shape[0] != 0:
                                        if do_hflip:
                                            main_curr_pts_ins[:, 0] = image_shape[1] - (main_curr_pts_ins[:, 0] + 1)
                                            main_next_pts_ins[:, 0] = image_shape[1] - (main_next_pts_ins[:, 0] + 1)

                                        if not self.use_input_resolution_for_temp:
                                            main_curr_pts_ins *= 0.25
                                            main_next_pts_ins *= 0.25

                                        assert main_curr_pts_ins.shape[0] % main_saved_topk == 0
                                        main_pt_num = int(main_curr_pts_ins.shape[0] / main_saved_topk)

                                        main_curr_pts_ins = main_curr_pts_ins.reshape((main_pt_num, main_saved_topk, 2))
                                        main_next_pts_ins = main_next_pts_ins.reshape((main_pt_num, main_saved_topk, 2))
                                        main_dists = main_dists.reshape((main_pt_num, main_saved_topk))

                                        if main_saved_topk > self.temporal_topk:
                                            main_keep_num = self.temporal_topk
                                        else:
                                            main_keep_num = main_saved_topk

                                        _main_curr_pts_ins = main_curr_pts_ins[:, :main_keep_num, :].flatten(0, 1)
                                        _main_next_pts_ins = main_next_pts_ins[:, :main_keep_num, :].flatten(0, 1)
                                        _main_dists = main_dists[:, :main_keep_num].flatten(0, 1)

                                        # # filter matching by correlations
                                        # main_keep_inds = torch.where(_main_dists < self.temp_dist_thr)[0]
                                        #
                                        # _main_curr_pts_ins = _main_curr_pts_ins[main_keep_inds]
                                        # _main_next_pts_ins = _main_next_pts_ins[main_keep_inds]
                                    else:
                                        _main_curr_pts_ins = torch.zeros((0, 2))
                                        _main_next_pts_ins = torch.zeros((0, 2))

                                    # sub
                                    sub_contents = _match_contents["sub"]
                                    sub_curr_pts_ins = torch.as_tensor(
                                        np.ascontiguousarray(np.array(sub_contents["curr_pts"]) * scale_factor)
                                    )
                                    sub_next_pts_ins = torch.as_tensor(
                                        np.ascontiguousarray(np.array(sub_contents["next_pts"]) * scale_factor)
                                    )
                                    sub_dists = torch.as_tensor(
                                        np.ascontiguousarray(np.array(sub_contents["dist"]))
                                    )
                                    sub_saved_topk = sub_contents["topk"]
                                    assert sub_curr_pts_ins.shape == sub_next_pts_ins.shape

                                    if sub_curr_pts_ins.shape[0] != 0:
                                        if do_hflip:
                                            sub_curr_pts_ins[:, 0] = image_shape[1] - (sub_curr_pts_ins[:, 0] + 1)
                                            sub_next_pts_ins[:, 0] = image_shape[1] - (sub_next_pts_ins[:, 0] + 1)

                                        if not self.use_input_resolution_for_temp:
                                            sub_curr_pts_ins *= 0.25
                                            sub_next_pts_ins *= 0.25

                                        assert sub_curr_pts_ins.shape[0] % sub_saved_topk == 0
                                        sub_pt_num = int(sub_curr_pts_ins.shape[0] / sub_saved_topk)

                                        sub_curr_pts_ins = sub_curr_pts_ins.reshape((sub_pt_num, sub_saved_topk, 2))
                                        sub_next_pts_ins = sub_next_pts_ins.reshape((sub_pt_num, sub_saved_topk, 2))
                                        sub_dists = sub_dists.reshape((sub_pt_num, sub_saved_topk))

                                        if sub_saved_topk > self.temporal_topk:
                                            sub_keep_num = self.temporal_topk
                                        else:
                                            sub_keep_num = sub_saved_topk

                                        _sub_curr_pts_ins = sub_curr_pts_ins[:, :sub_keep_num, :].flatten(0, 1)
                                        _sub_next_pts_ins = sub_next_pts_ins[:, :sub_keep_num, :].flatten(0, 1)
                                        _sub_dists = sub_dists[:, :sub_keep_num].flatten(0, 1)

                                        # # filter matching by correlations
                                        # sub_keep_inds = torch.where(_sub_dists < self.temp_dist_thr)[0]
                                        #
                                        # _sub_curr_pts_ins = _sub_curr_pts_ins[sub_keep_inds]
                                        # _sub_next_pts_ins = _sub_next_pts_ins[sub_keep_inds]
                                    else:
                                        _sub_curr_pts_ins = torch.zeros((0, 2))
                                        _sub_next_pts_ins = torch.zeros((0, 2))

                                    _curr_pts_ins = torch.cat([_main_curr_pts_ins, _sub_curr_pts_ins], dim=0)
                                    _next_pts_ins = torch.cat([_main_next_pts_ins, _sub_next_pts_ins], dim=0)

                                else:
                                    # grab the matching coordinates and apply transform
                                    curr_pts_ins = torch.as_tensor(
                                        np.ascontiguousarray(np.array(_match_contents["curr_pts"]) * scale_factor)
                                    )
                                    next_pts_ins = torch.as_tensor(
                                        np.ascontiguousarray(np.array(_match_contents["next_pts"]) * scale_factor)
                                    )
                                    dists = torch.as_tensor(
                                        np.ascontiguousarray(np.array(_match_contents["corr"]))
                                    )
                                    assert curr_pts_ins.shape == next_pts_ins.shape

                                    if do_hflip:
                                        curr_pts_ins[:, 0] = image_shape[1] - (curr_pts_ins[:, 0] + 1)
                                        next_pts_ins[:, 0] = image_shape[1] - (next_pts_ins[:, 0] + 1)

                                    if not self.use_input_resolution_for_temp:
                                        curr_pts_ins *= 0.25
                                        next_pts_ins *= 0.25

                                    # 选取topk个点
                                    saved_topk = _match_contents["topk"]
                                    assert curr_pts_ins.shape[0] % saved_topk == 0
                                    pt_num = int(curr_pts_ins.shape[0] / saved_topk)

                                    curr_pts_ins = curr_pts_ins.reshape((pt_num, saved_topk, 2))
                                    next_pts_ins = next_pts_ins.reshape((pt_num, saved_topk, 2))
                                    dists = dists.reshape((pt_num, saved_topk))

                                    if saved_topk > self.temporal_topk:
                                        keep_num = self.temporal_topk
                                    else:
                                        keep_num = saved_topk

                                    _curr_pts_ins = curr_pts_ins[:, :keep_num, :].flatten(0, 1)
                                    _next_pts_ins = next_pts_ins[:, :keep_num, :].flatten(0, 1)
                                    _dists = dists[:, :keep_num].flatten(0, 1)

                                    # # filter matching by correlations
                                    # keep_inds = torch.where(_dists < self.temp_dist_thr)[0]
                                    #
                                    # _curr_pts_ins = _curr_pts_ins[keep_inds]
                                    # _next_pts_ins = _next_pts_ins[keep_inds]

                                dataset_dict['match_coords'][_id][f_i]["curr_pts"] = _curr_pts_ins.ceil().to(
                                    dtype=torch.int16
                                )
                                dataset_dict['match_coords'][_id][f_i]["next_pts"] = _next_pts_ins.ceil().to(
                                    dtype=torch.int16
                                )

            instances = utils.annotations_to_instances(sorted_annos, image_shape, mask_format="bitmask")
            instances.gt_ids = torch.tensor(_gt_ids)
            if instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                instances = filter_empty_instances(instances)
            else:
                # 在选的两帧上都没有object，所有的sorted_annos, ids, _frame_annos都是空的
                instances.gt_masks = BitMasks(torch.empty((0, *image_shape)))  # tensor([]), shape in (0, h, w)
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()  # tensor([]), shape in (0, 4)

            dataset_dict["instances"].append(instances)

        return dataset_dict


class YTVISDatasetMapper:
    """
    A callable which takes a dataset dict in YouTube-VIS Dataset format,
    and map it into a format used by the model.
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        sampling_frame_num: int = 2,
        sampling_frame_range: int = 5,
        sampling_frame_shuffle: bool = False,
        num_classes: int = 40,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
        """
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.sampling_frame_num     = sampling_frame_num
        self.sampling_frame_range   = sampling_frame_range
        self.sampling_frame_shuffle = sampling_frame_shuffle
        self.num_classes            = num_classes
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = build_augmentation(cfg, is_train)

        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
        sampling_frame_range = cfg.INPUT.SAMPLING_FRAME_RANGE
        sampling_frame_shuffle = cfg.INPUT.SAMPLING_FRAME_SHUFFLE

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "sampling_frame_num": sampling_frame_num,
            "sampling_frame_range": sampling_frame_range,
            "sampling_frame_shuffle": sampling_frame_shuffle,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
        }

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one video, in YTVIS Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # TODO consider examining below deepcopy as it costs huge amount of computations.
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        video_length = dataset_dict["length"]
        if self.is_train:

            # frame2, fix interval sampling
            # if 0 < video_length <= 10:
            #     interval = 4
            # elif 10 < video_length <= 20:
            #     interval = 10
            # elif 20 < video_length <= 30:
            #     interval = 15
            # elif 30 < video_length <= 40:
            #     interval = 20
            # elif 40 < video_length:
            #     interval = 36
            # else:
            #     raise ValueError("video length is not valid")
            #
            # ref_frame = random.randrange(video_length - interval)
            # tgt_frame = ref_frame + interval
            # selected_idx = [ref_frame, tgt_frame]

            # frame 3, fix interval sampling
            if 0 < video_length <= 8:
                interval = 2
            elif 8 < video_length <= 16:
                interval = 3

            # else:
            #     interval = 5

            elif 16 < video_length <= 24:
                interval = 7
            elif 24 < video_length <= 32:
                interval = 10
            elif 32 < video_length <= 40:
                interval = 14
            elif 40 < video_length <= 48:
                interval = 18
            elif 48 < video_length:
                interval = 23
            else:
                raise ValueError("video length is not valid")

            ref_frame = random.randrange(video_length - 2 * interval)
            first_tgt_frame = ref_frame + interval
            second_tgt_frame = first_tgt_frame + interval
            selected_idx = [ref_frame, first_tgt_frame, second_tgt_frame]


            # random adaptive sampling
            # ref_frame = random.randrange(video_length)
            #
            # start_idx = max(0, ref_frame-self.sampling_frame_range)
            # end_idx = min(video_length, ref_frame+self.sampling_frame_range + 1)
            #
            # selected_idx = np.random.choice(
            #     np.array(list(range(start_idx, ref_frame)) + list(range(ref_frame+1, end_idx))),
            #     self.sampling_frame_num - 1,
            # )
            # selected_idx = selected_idx.tolist() + [ref_frame]
            # selected_idx = sorted(selected_idx)
            # if self.sampling_frame_shuffle:
            #     random.shuffle(selected_idx)
        else:
            selected_idx = range(video_length)

        video_annos = dataset_dict.pop("annotations", None)
        file_names = dataset_dict.pop("file_names", None)

        if self.is_train:
            _ids = set()
            for frame_idx in selected_idx:
                _ids.update([anno["id"] for anno in video_annos[frame_idx]])
            ids = dict()
            for i, _id in enumerate(_ids):
                ids[_id] = i

        dataset_dict["image"] = []
        dataset_dict["instances"] = []
        dataset_dict["file_names"] = []
        for frame_idx in selected_idx:
            dataset_dict["file_names"].append(file_names[frame_idx])

            # Read image
            image = utils.read_image(file_names[frame_idx], format=self.image_format)
            utils.check_image_size(dataset_dict, image)

            aug_input = T.AugInput(image)
            transforms = self.augmentations(aug_input)
            image = aug_input.image

            image_shape = image.shape[:2]  # h, w
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))

            if (video_annos is None) or (not self.is_train):
                continue

            # NOTE copy() is to prevent annotations getting changed from applying augmentations
            _frame_annos = []  # single frame annotations
            for anno in video_annos[frame_idx]:
                _anno = {}
                for k, v in anno.items():
                    _anno[k] = copy.deepcopy(v)
                _frame_annos.append(_anno)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in _frame_annos
                if obj.get("iscrowd", 0) == 0
            ]
            # 先为每个instance生成一个dummy obj
            sorted_annos = [_get_dummy_anno(self.num_classes) for _ in range(len(ids))]

            for _anno in annos:
                idx = ids[_anno["id"]]
                sorted_annos[idx] = _anno
            _gt_ids = [_anno["id"] for _anno in sorted_annos]

            instances = utils.annotations_to_instances(sorted_annos, image_shape, mask_format="bitmask")
            instances.gt_ids = torch.tensor(_gt_ids)
            if instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                instances = filter_empty_instances(instances)
            else:
                # 在选的两帧上都没有object，所有的sorted_annos, ids, _frame_annos都是空的
                instances.gt_masks = BitMasks(torch.empty((0, *image_shape)))  # tensor([]), shape in (0, h, w)
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()  # tensor([]), shape in (0, 4)

            dataset_dict["instances"].append(instances)

        return dataset_dict


class CocoClipDatasetMapper:
    """
    A callable which takes a COCO image which converts into multiple frames,
    and map it into a format used by the model.
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        sampling_frame_num: int = 2,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
        """
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.sampling_frame_num     = sampling_frame_num
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = build_augmentation(cfg, is_train)

        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "sampling_frame_num": sampling_frame_num,
        }

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        img_annos = dataset_dict.pop("annotations", None)
        file_name = dataset_dict.pop("file_name", None)
        original_image = utils.read_image(file_name, format=self.image_format)

        dataset_dict["image"] = []
        dataset_dict["instances"] = []
        dataset_dict["file_names"] = [file_name] * self.sampling_frame_num
        for _ in range(self.sampling_frame_num):
            utils.check_image_size(dataset_dict, original_image)

            aug_input = T.AugInput(original_image)
            transforms = self.augmentations(aug_input)
            image = aug_input.image

            image_shape = image.shape[:2]  # h, w
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))

            if (img_annos is None) or (not self.is_train):
                continue

            _img_annos = []
            for anno in img_annos:
                _anno = {}
                for k, v in anno.items():
                    _anno[k] = copy.deepcopy(v)
                _img_annos.append(_anno)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in _img_annos
                if obj.get("iscrowd", 0) == 0
            ]
            _gt_ids = list(range(len(annos)))
            for idx in range(len(annos)):
                if len(annos[idx]["segmentation"]) == 0:
                    annos[idx]["segmentation"] = [np.array([0.0] * 6)]

            instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
            instances.gt_ids = torch.tensor(_gt_ids)
            if instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                instances = filter_empty_instances(instances)
            else:
                instances.gt_masks = BitMasks(torch.empty((0, *image_shape)))
            dataset_dict["instances"].append(instances)

        return dataset_dict
