import os
import cv2
import numpy as np
from tqdm import tqdm
import keras_cv


def start_points(size, split_size, overlap=0.25):
    points = [0]
    stride = int(split_size * (1 - overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            if split_size == size:
                break
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        # print(points)
        counter += 1
    return points


def calculate_intersection_area(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate the coordinates of the intersection rectangle
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)

    # Check if there is an intersection
    if x_inter_min < x_inter_max and y_inter_min < y_inter_max:
        # Calculate the width and height of the intersection rectangle
        intersection_width = x_inter_max - x_inter_min
        intersection_height = y_inter_max - y_inter_min
        # Calculate the intersection area
        intersection_area = intersection_width * intersection_height
    else:
        # No intersection
        intersection_area = 0

    return intersection_area


image_folder = "dataset/weldingspot/dataset_whole/images"
label_folder = "dataset/weldingspot/dataset_whole/labels"

split_image_save_folder = "dataset/weldingspot/dataset_split/images"
if not os.path.exists(split_image_save_folder):
    os.makedirs(split_image_save_folder)

split_label_save_folder = "dataset/weldingspot/dataset_split/labels"
if not os.path.exists(split_label_save_folder):
    os.makedirs(split_label_save_folder)

image_name_all = os.listdir(image_folder)

split_width = 640
split_height = 640

for img_name in tqdm(image_name_all):
    image = cv2.imread(os.path.join(image_folder, img_name))
    label_path = os.path.join(label_folder, img_name.replace('.png', '.txt'))
    labels = np.loadtxt(label_path)
    h, w, _ = image.shape

    classes = np.array(labels[:, 0])
    boxes = np.array(labels[:, 1:])

    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] * w - boxes[:, 2] * w * 0.5
    boxes_xyxy[:, 1] = boxes[:, 1] * h - boxes[:, 3] * h * 0.5
    boxes_xyxy[:, 2] = boxes[:, 0] * w + boxes[:, 2] * w * 0.5
    boxes_xyxy[:, 3] = boxes[:, 1] * h + boxes[:, 3] * h * 0.5

    X_points = start_points(w, split_width, 0.25)
    Y_points = start_points(h, split_height, 0.25)

    count = 0
    frmt = 'png'

    for i in Y_points:
        for j in X_points:
            split = image[i:i + split_height, j:j + split_width]
            image_coord = [j, i, j + split_width, i + split_height]
            count += 1
            label_list = []
            for k, box in enumerate(boxes_xyxy):
                inter_area = calculate_intersection_area(box, image_coord)
                if inter_area > 0:
                    box_in_split_image = [max(box[0], image_coord[0]) - j,
                                          max(box[1], image_coord[1]) - i,
                                          min(box[2], image_coord[2]) - j,
                                          min(box[3], image_coord[3]) - i]

                    class_id = classes[k]
                    w_box_in_split_image = box_in_split_image[2] - box_in_split_image[0]
                    h_box_in_split_image = box_in_split_image[3] - box_in_split_image[1]
                    label_for_split_image = [int(class_id),
                                             0.5 * (box_in_split_image[0] + box_in_split_image[2]) / split_width,
                                             0.5 * (box_in_split_image[1] + box_in_split_image[3]) / split_height,
                                             w_box_in_split_image / split_width,
                                             h_box_in_split_image / split_height,
                                             image_coord[0],  # global coordinate
                                             image_coord[1]]
                    label_list.append(label_for_split_image)

                if len(label_list) > 0:
                    cv2.imwrite('{}/{}_{}.png'.format(split_image_save_folder, img_name.split('.')[0], count), split)
                    text_file = open('{}/{}_{}.txt'.format(split_label_save_folder, img_name.split('.')[0], count), 'w')
                    for line in label_list:
                        for item in line:
                            text_file.write(f"{item} ")
                        text_file.write("\n")
                    text_file.close()
