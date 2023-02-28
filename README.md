# Evaluation of object detection by night
As part of the Image-Based Computer Graphics seminar at the TH Köln (University of Applied Sciences), 
we investigated how object detectors behave at night with and without explicit training.  
The project documentation (written in german) can be found [here](./Projektdokumentation%20Langer,%20Pagelsdorf%20-%20Verhalten%20von%20Objektdetektoren%20bei%20Nacht.pdf).

## Requirements
* Python >3.9
* CUDA >10.3 & <=11.7
* OpenCV

## Instructions
1. Clone Git Repository (with submodules)
2. Install pip requirements for yolo and detectron2
3. insert your roboflow API key in `data/get-data.py`
4. run `cd data && python ./get-data.py`
5. Download [`yolov7_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt) from GitHub and place it into root dir. 
6. Start training 
   * For Detectron2 see `detectron2_run.py` 
   * For Yolo see `yolo_run_train.py` 
 
## Results
Some training results especially for YOLOv7 can be found here:
* `runs/yolo/train`
* `runs/yolo/test`
* `runs/detectron2/train`
* `runs/detectron2/test`

All model are trained with a batch size of 4 over 200 epochs, respectively to Detectron2 over 100 epochs. 

For YOLO there exists a `__old` directory which contains failed intermediate results e.g. trying freezing layers.