import random
import cv2
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2_trainer import CocoTrainer
from detectron2_data import DatasetName, prepare_dataset

resume_training = False
dataset_name = DatasetName.Day
dataset_test = DatasetName.Night
name = prepare_dataset(dataset_name)


def do_train():
    trainer = CocoTrainer(dataset_name)
    trainer.resume_or_load(resume=resume_training)
    trainer.train()


def do_test():
    trainer = CocoTrainer(dataset_name, True)
    trainer.run_test(dataset_test)


def do_visualize():
    train_metadata = MetadataCatalog.get(name + "_train")
    dataset_dicts = DatasetCatalog.get(name + "_train")

    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=train_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow(str(d["image_id"]), vis.get_image()[:, :, ::-1])

    cv2.waitKey(0)


if __name__ == '__main__':
    # do_test()
    do_train()
    # do_visualize()
