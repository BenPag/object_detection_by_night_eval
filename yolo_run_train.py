import os
from enum import Enum


class DatasetName(Enum):
    Day = 1
    Night = 2
    DayAndNight = 3


dataset = DatasetName.Day
weights = '--weights runs/train/day_v7_b4_e50/weights/epoch_049.pt'
batch = '--batch 4'
epochs = '--epochs 200'
project = '--project runs/yolo/train'

if dataset == DatasetName.Day:
    data = '--data data/yolo/bdd-1/data.yaml'
    name = '--name day_v7_b4_e200'
elif dataset == DatasetName.Night:
    data = '--data data/yolo/rmsw_5k_night-2/data.yaml'
    name = '--name night_v7_b4_e200'
else:
    data = '--data data/yolo/day_and_night/data.yaml'
    name = '--name day_and_night_v7_b4_e200'

os.system('python yolov7/train.py --device 0 ' + batch + epochs + data + weights + name)

# test
# DAY and NIGHT
# python yolov7/test.py --data data/yolo/day_and_night/data.yaml --batch 16 --device 0 --weights 'runs\train\day_and_night_v7_b4_e100\weights\best.pt' --name 03_both_on_both
# python yolov7/test.py --data data/yolo/bdd-1/data.yaml --batch 16 --device 0 --weights 'runs\train\day_and_night_v7_b4_e100\weights\best.pt' --name 03_both_on_day
# python yolov7/test.py --data data/yolo/rmsw_5k_night-2/data.yaml --batch 16 --device 0 --weights 'runs\train\day_and_night_v7_b4_e100\weights\best.pt' --name 03_both_on_night

# DAY
# python yolov7/test.py --data data/yolo/day_and_night/data.yaml --batch 16 --device 0 --weights 'runs\train\day_v7_b4_e200\weights\best.pt' --name 01_day_on_both
# python yolov7/test.py --data data/yolo/bdd-1/data.yaml --batch 16 --device 0 --weights 'runs\train\day_v7_b4_e200\weights\best.pt' --name 01_day_on_day
# python yolov7/test.py --data data/yolo/rmsw_5k_night-2/data.yaml --batch 16 --device 0 --weights 'runs\train\day_v7_b4_e200\weights\best.pt' --name 01_day_on_night

# NIGHT
# python yolov7/test.py --data data/yolo/day_and_night/data.yaml --batch 16 --device 0 --weights 'runs\train\night_v7_b4_e200\weights\best.pt' --name 02_night_on_both
# python yolov7/test.py --data data/yolo/bdd-1/data.yaml --batch 16 --device 0 --weights 'runs\train\night_v7_b4_e200\weights\best.pt' --name 02_night_on_day
# python yolov7/test.py --data data/yolo/rmsw_5k_night-2/data.yaml --batch 16 --device 0 --weights 'runs\train\night_v7_b4_e200\weights\best.pt' --name 02_night_on_night
