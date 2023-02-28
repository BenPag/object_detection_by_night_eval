import os
import time
import glob
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.data import build_detection_test_loader
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2_data import DatasetName, prepare_dataset


def provide_config(dataset_name: DatasetName, resume_training=True):
    setup_logger()

    if dataset_name == DatasetName.Day:
        name = DatasetName.Day.name.lower()
        number_of_train_images = 3664
    elif dataset_name == DatasetName.Night:
        name = DatasetName.Night.name.lower()
        number_of_train_images = 3435
    else:
        name = DatasetName.DayAndNight.name.lower()
        number_of_train_images = 3664 + 3435

    base_model = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
    batch_size = 4
    epochs = 100

    # epochs = MAX_ITER * BATCH_SIZE / TOTAL_NUM_IMAGES
    # MAX_ITER = epochs * TOTAL_NUM_IMAGES  / BATCH_SIZE
    iterations_per_epoch = int(number_of_train_images / batch_size)

    # Config reference: https://detectron2.readthedocs.io/en/latest/modules/config.html
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(base_model))
    cfg.DATASET_NAME = name
    cfg.DATASETS.TRAIN = (name + "_train",)
    cfg.DATASETS.TEST = (name + "_val",)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(base_model)  # Let training initialize from model zoo

    cfg.SOLVER.CHECKPOINT_PERIOD = iterations_per_epoch * 5
    cfg.SOLVER.WARMUP_ITERS = iterations_per_epoch
    cfg.SOLVER.WARMUP_FACTOR = 1 / iterations_per_epoch
    cfg.SOLVER.MAX_ITER = epochs * iterations_per_epoch
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.STEPS = (int(cfg.SOLVER.MAX_ITER * 0.75), int(cfg.SOLVER.MAX_ITER * 0.9))

    # RoI minibatch size *per image* (number of regions of interest [ROIs]) during training
    # Total number of RoIs per training minibatch =
    #   ROI_HEADS.BATCH_SIZE_PER_IMAGE * SOLVER.IMS_PER_BATCH
    # E.g., a common configuration is: 512 * 16 = 8192
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
    cfg.TEST.EVAL_PERIOD = iterations_per_epoch * 2
    cfg.INPUT.MIN_SIZE_TEST = 640
    cfg.OUTPUT_DIR = './runs/detectron2/train/' + name + time.strftime("%d%m-%H%M%S", time.localtime())

    if resume_training:
        dirs = glob.glob('./runs/detectron2/train/' + name + '*')
        dirs.sort()
        cfg.OUTPUT_DIR = dirs[-1]

    return cfg


class CocoTrainer(DefaultTrainer):

    def __init__(self, dataset_name: DatasetName, resume_training=True):
        cfg = provide_config(dataset_name, resume_training)
        super().__init__(cfg)
        if resume_training is False:
            os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = cfg.OUTPUT_DIR + "/eval"
            os.makedirs(output_folder, exist_ok=True)

        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    def run_test(self, dataset_name: DatasetName):
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.65
        self.resume_or_load()

        test_name = dataset_name.name.lower()
        if test_name != self.cfg.DATASET_NAME:
            prepare_dataset(dataset_name)

        output_test_dir = "./runs/detectron2/test/" + self.cfg.DATASET_NAME + '_on_' + test_name
        os.makedirs(output_test_dir, exist_ok=True)

        dataset = test_name + '_test'
        evaluator = COCOEvaluator(dataset, self.cfg, False, output_dir=output_test_dir)
        val_loader = build_detection_test_loader(self.cfg, dataset)
        inference_on_dataset(self._trainer.model, val_loader, evaluator)
