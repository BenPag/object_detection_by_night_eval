from roboflow import Roboflow
from glob import glob
import os
import shutil
import re_label


skipDownload = True

if skipDownload is False:
    rf = Roboflow(api_key="<insert_your_roboflow_api_key>")
    # https://universe.roboflow.com/khermz2012-gmail-com/bdd-ujnwc/dataset/1
    dayDataProject = rf.workspace("khermz2012-gmail-com").project("bdd-ujnwc")
    dayDataProject.version(1).download("yolov7", None, 'yolo', False)
    dayDataProject.version(1).download("coco", None, 'detectron2', False)

    # https://universe.roboflow.com/uni-koped/rmsw_5k_night/dataset/2
    nightDataProject = rf.workspace("uni-koped").project("rmsw_5k_night")
    nightDataProject.version(2).download("yolov7", None, 'yolo', False)
    nightDataProject.version(2).download("coco", None, 'detectron2', False)

    # for detector in ['yolo', 'detectron2']: yolo only unless reassignment of annotations is fixed
    for detector in ['yolo']:
        base_dir = './' + detector
        dist_dir = base_dir + '/day_and_night_2'
        shutil.copytree(glob(base_dir + '/bdd*')[-1], dist_dir + '/.', dirs_exist_ok=True)
        shutil.copytree(glob(base_dir + '/rmsw_5k_night*')[-1], dist_dir + '/.', dirs_exist_ok=True)
        if detector == 'yolo':
            os.remove(dist_dir + '/data.yaml')
            shutil.copyfile('day-and-night_data.yaml', dist_dir + '/data.yaml')

re_label.relabel_yolo_data()
re_label.relabel_detectron2_data()
# re_label.merge_annotations()
