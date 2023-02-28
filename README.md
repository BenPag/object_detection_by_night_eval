# BCG

## Requirements
* Python >3.9
* CUDA >10.3 & <=11.7
* OpenCV

## Instructions
1. Clone Git Repository (with submodules)
2. Install pip requirements for yolo and detectron2
3. insert your roboflow api key in `data/get-data.py`
4. run `cd data && python ./get-data.py`
5. Download [`yolov7_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt) from GitHub and place it into root dir. 
6. Start training 
   * For Detectron2 see `detectron2_run.py` 
   * For Yolo see `yolo_run_train.py` 
 