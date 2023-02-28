import glob
import json


def relabel_yolo_data():
    print('relabel yolo data')
    forbidden_label_ids = ('5', '6', '7', '8', '9', '10', '11', '12', '13', '14')
    labels = glob.glob('yolo/*/*/labels/*.txt')
    for label in labels:
        print(label)
        file = open(label, "r")
        content = file.read()
        file.close()

        file = open(label, "w")
        file.write(
            '\n'.join(
                filter(lambda l: l.startswith(forbidden_label_ids) is False, content.splitlines())))
        file.close()


def relabel_detectron2_data():
    print('relabel detectron2 data')
    forbidden_label_ids = ('6', '7', '8', '9', '10', '11', '12', '13', '14')
    labels = glob.glob('detectron2/[!day_and]*/*/_annotations.coco.json')
    has_forbidden_labels = False

    for label in labels:
        print(label)
        file = open(label, "r")
        data = json.load(file)

        for category in data['categories']:
            if int(category['id']) > 5 or str(category['id']) in forbidden_label_ids:
                has_forbidden_labels = True
                del data['categories'][data['categories'].index(category)]

        for annotation in data['annotations']:
            if int(annotation['category_id']) > 5 or str(annotation['category_id']) in forbidden_label_ids:
                has_forbidden_labels = True
                del data['annotations'][data['annotations'].index(annotation)]

        file.close()

        with open(label, 'w') as f:
            json.dump(data, f)

    if has_forbidden_labels:
        relabel_detectron2_data()


def merge_annotations():
    print('merge detectron2 annotations')
    relabel_detectron2_data()
    for split_type in ['test', 'train', 'valid']:
        labels = glob.glob('detectron2/[!day_and]*/' + split_type + '/_annotations.coco.json')
        merged_annotations: any = None
        for label in labels:
            with open(label, 'r') as f:
                if merged_annotations is None:
                    merged_annotations = json.load(f)
                else:
                    data = json.load(f)
                    for image in data['images']:
                        merged_annotations['images'].append(image)
                    for annotation in data['annotations']:
                        merged_annotations['annotations'].append(annotation)

        cnt = 0
        for image in merged_annotations['images']:
            image['id'] = cnt
            cnt += 1

        # ToDo remapping of annotations to new image_id
        with open('detectron2/day_and_night/' + split_type + '/_annotations.coco.json', 'w') as f:
            json.dump(merged_annotations, f)
