# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

from .dataset_mapper import YTVISDatasetMapper, CocoClipDatasetMapper
from .build import *

from .datasets import *
from .ytvis_eval import YTVISEvaluator

from .dataset_mapper_w_feat import YTVISDatasetWithFeatsMapper
# from .dataset_mapper_w_coord import YTVISDatasetWithCoordMapper