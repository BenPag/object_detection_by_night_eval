import os
from enum import Enum


class DatasetName(Enum):
    Day = 1
    Night = 2
    Both = 3  # DayAndNight


dataset = DatasetName.Day
model = DatasetName.Day
batch = '--batch 16'
project = '--project runs/yolo/test'

if dataset == DatasetName.Day:
    data = '--data data/yolo/bdd-1/data.yaml'
elif dataset == DatasetName.Night:
    data = '--data data/yolo/rmsw_5k_night-2/data.yaml'
else:
    data = '--data data/yolo/day_and_night/data.yaml'

if model == DatasetName.Day:
    name = '--name 03_day_on_' + dataset.name.lower()
    weights = '--weights runs/yolo/train/day_v7_b4_e200/weights/best.pt'
elif model == DatasetName.Night:
    name = '--name 03_night_on' + dataset.name.lower()
    weights = '--weights runs/yolo/train/night_v7_b4_e200/weights/best.pt'
else:
    name = '--name 03_both_on' + dataset.name.lower()
    weights = '--weights runs/yolo/train/day_and_night_v7_b4_e200/weights/best.pt'

os.system('python yolov7/test.py --device 0 ' + batch + data + weights + name)
