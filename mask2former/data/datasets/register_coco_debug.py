import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances

_PREDEFINED_SPLITS_COCO_DEBUG = {
    "coco_train_debug": ("coco-debug", "coco-debug/coco-debug-train.json"),
    "coco_val_debug": ("coco-debug", "coco-debug/coco-debug-val.json")
}

def register_coco_debug(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_COCO_DEBUG.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_builtin_metadata("coco"),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_coco_debug(_root)