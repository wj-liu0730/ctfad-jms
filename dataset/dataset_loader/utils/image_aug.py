import copy
import tensorflow as tf
import keras
import keras_cv
import cv2
import os
import numpy as np
from keras_cv import visualization
from dataset.dataset_loader.WeldingSpotYolo import CLASS_MAPPING

image_path = "data/weldingspot/dataset_split/images/a59dc443-008_35.png"
label_path = "data/weldingspot/dataset_split/labels/a59dc443-008_35.txt"

image_list = []
boxes_list = []
classes_list = []
image = cv2.imread(os.path.join(image_path))
labels = np.loadtxt(label_path)
h, w, _ = image.shape

boxes = labels[:, 1:5]
boxes = np.array(boxes)
boxes[:, 0] *= image.shape[1]  # w
boxes[:, 1] *= image.shape[0]  # h
boxes[:, 2] *= image.shape[1]  # w
boxes[:, 3] *= image.shape[0]  # h

classes = np.array(labels[:, 0])

# # add original image
# image_list.append(image)
# classes_list.append(classes)
# boxes_list.append(boxes)

# add_hue_image
random_hue = keras_cv.layers.RandomHue(factor=(0.6, 0.6), value_range=(0, 255))  # factor \in [0, 0.6]
rand_hue_image = random_hue(copy.deepcopy(image))
image_list.append(rand_hue_image)
classes_list.append(classes)
boxes_list.append(boxes)

#add_saturation_image
random_saturation = keras_cv.layers.RandomSaturation(factor=(0.6, 0.6))  # factor \in [0, 0.8]
ran_sat_image = random_saturation(copy.deepcopy(image))
image_list.append(ran_sat_image)
classes_list.append(classes)
boxes_list.append(boxes)

#add_contrast_image
random_contrast = keras.layers.RandomContrast(factor=(0.8, 0.8))  # factor \in [0, 0.8]
ran_contrast_image = random_contrast(copy.deepcopy(image))
image_list.append(ran_contrast_image)
classes_list.append(classes)
boxes_list.append(boxes)

#add_brightness_image
random_brightness = keras.layers.RandomBrightness(factor=(0.4, 0.4), value_range=(0, 255))  # factor \in [0, 0.6]
ran_bright_image = random_brightness(copy.deepcopy(image))
image_list.append(ran_bright_image)
classes_list.append(classes)
boxes_list.append(boxes)

#add flip image
image_lr_flip = tf.image.flip_left_right(copy.deepcopy(image))
flipped_box = np.array(copy.deepcopy(boxes))
flipped_box[:, 0] = w - boxes[:, 0]
image_list.append(image_lr_flip)
classes_list.append(classes)
boxes_list.append(flipped_box)

#add rotate image
k = 1
rot_image = tf.image.rot90(copy.deepcopy(image), k=1)  # rotates k*90 degrees
rot_box = np.array(copy.deepcopy(boxes))

cos_angle = np.cos(-1 * k * np.pi / 2)
sin_angle = np.sin(-1 * k * np.pi / 2)
rotated_boxes = []

for box in boxes:
    center_x, center_y, width, height = box

    # Translate box center to origin
    center_x -= w / 2
    center_y -= h / 2

    # Rotate the center point
    rotated_center_x = center_x * cos_angle - center_y * sin_angle
    rotated_center_y = center_x * sin_angle + center_y * cos_angle

    # Translate back to the image center
    rotated_center_x += w / 2
    rotated_center_y += h / 2
    rotated_boxes.append([rotated_center_x, rotated_center_y, width, height])

image_list.append(rot_image)
classes_list.append(classes)
boxes_list.append(rotated_boxes)

# add rescaling image
zoom_factor = -0.5
ran_zoom = keras.layers.RandomZoom(height_factor=(zoom_factor, zoom_factor),
                                   fill_mode="constant")  # factor \in [0.8, 1.2]
ran_zoom_image = ran_zoom(copy.deepcopy(image))
image_list.append(ran_zoom_image)
boxes = copy.deepcopy(boxes)
scaled_boxes = []
rescale_factor = 1 - zoom_factor

for box in boxes:
    center_x, center_y, width, height = box
    center_x -= w / 2
    center_y -= h / 2

    rescaled_center_x = center_x * rescale_factor + w / 2
    rescaled_center_y = center_y * rescale_factor + h / 2
    rescaled_width = width * rescale_factor
    rescaled_height = height * rescale_factor

    scaled_boxes.append([rescaled_center_x, rescaled_center_y, rescaled_width, rescaled_height])

classes_list.append(classes)
boxes_list.append(scaled_boxes)

# add random translation image
trans_factor_h = 0.2
trans_factor_w = 0.2
random_trans = keras.layers.RandomTranslation(height_factor=(trans_factor_h, trans_factor_h),
                                              width_factor=(trans_factor_w, trans_factor_w))
translated = random_trans(copy.deepcopy(image))
image_list.append(translated)

translated_box = []
x_offset = trans_factor_w * w
y_offset = trans_factor_h * h

for box in boxes:
    center_x, center_y, width, height = box
    center_x += x_offset
    center_y += y_offset
    translated_box.append([center_x, center_y, width, height])
classes_list.append(classes)
boxes_list.append(translated_box)


bounding_box = {"boxes": boxes_list, "classes": classes_list}

num_image = len(image_list)
rows = 2
cols = int(num_image / rows)

visualization.plot_bounding_box_gallery(
    np.array(image_list),
    value_range=(0, 255),
    rows=rows,
    cols=cols,
    y_true=bounding_box,
    show=True,
    scale=5,
    font_scale=1.5,
    bounding_box_format='center_xywh',
    class_mapping=CLASS_MAPPING)
