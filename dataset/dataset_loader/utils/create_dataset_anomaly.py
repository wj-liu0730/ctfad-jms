import os
import cv2
import numpy as np
from tqdm import tqdm

image_folder = "dataset/weldingspot/dataset_whole/images"
label_folder = "dataset/weldingspot/dataset_whole/labels"

crop_image_save_folder = "dataset/weldingspot/dataset_anomaly/images"
crop_label_save_folder = "dataset/weldingspot/dataset_anomaly/labels"

if not os.path.exists(crop_image_save_folder):
    os.makedirs(crop_image_save_folder)

if not os.path.exists(crop_label_save_folder):
    os.makedirs(crop_label_save_folder)

image_name_all = os.listdir(image_folder)


for img_name in tqdm(image_name_all):
    image = cv2.imread(os.path.join(image_folder, img_name))
    label_path = os.path.join(label_folder, img_name.replace('.png', '.txt'))
    labels = np.loadtxt(label_path)
    h, w, _ = image.shape
    class_ids = labels[:, 0]
    boxes = np.array(labels[:, 1:])
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] * w - boxes[:, 2] * w * 0.5
    boxes_xyxy[:, 1] = boxes[:, 1] * h - boxes[:, 3] * h * 0.5
    boxes_xyxy[:, 2] = boxes[:, 0] * w + boxes[:, 2] * w * 0.5
    boxes_xyxy[:, 3] = boxes[:, 1] * h + boxes[:, 3] * h * 0.5

    for box_id, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = box
        crop = image[int(y1):int(y2), int(x1):int(x2)]
        cv2.imwrite('{}/{}_{}.png'.format(crop_image_save_folder, img_name.split('.')[0], box_id), crop)
        class_type = int(class_ids[box_id])
        text_file = open('{}/{}_{}.txt'.format(crop_label_save_folder, img_name.split('.')[0], box_id), 'w')
        text_file.write(f"{class_type} ")
        text_file.write("\n")
        text_file.close()