from enum import Enum
from detectron2.data.datasets import register_coco_instances


class DatasetName(Enum):
    Day = 1
    Night = 2
    DayAndNight = 3


def prepare_dataset(dataset_name: DatasetName):
    if dataset_name == DatasetName.Day:
        name = DatasetName.Day.name.lower()
        base_path = './data/detectron2/bdd.v1i.coco'
    elif dataset_name == DatasetName.Night:
        name = DatasetName.Night.name.lower()
        base_path = './data/detectron2/rmsw_5k_night.v2i.coco'
    else:
        name = DatasetName.DayAndNight.name.lower()
        base_path = './data/detectron2/day_and_night'

    register_coco_instances(name + "_train", {}, base_path + "/train/_annotations.coco.json", base_path + "/train")
    register_coco_instances(name + "_val", {}, base_path + "/valid/_annotations.coco.json", base_path + "/valid")
    register_coco_instances(name + "_test", {}, base_path + "/test/_annotations.coco.json", base_path + "/test")

    return name
