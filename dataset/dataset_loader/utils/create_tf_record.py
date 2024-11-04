"""This script preprocessing data into tf.records for WeldingSpot dataset"""

import os
import numpy as np
import sys
from tqdm import tqdm
NUM_SAMPLES_PER_RECORD = 50
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from dataset.dataset_loader.utils.tf_data_format import *


def create_data_point(image, path, label_info):
    feature = {
        "image": image_feature(image),
        "path": bytes_feature(path),
        "boxes": bbox_feature(label_info["boxes"]),
        "classes": int64_feature_list(label_info["classes"]),
        "image_id": bytes_feature(label_info["image_id"]),
        "global_coord": float_feature_list(label_info["global_coord"]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def generate_tfrecords_dataset(dataset_name="dataset_whole", mode="train"):
    data_root = f'dataset/weldingspot/{dataset_name}'
    image_root = os.path.join(data_root, 'images')
    label_root = os.path.join(data_root, 'labels')

    tfrecords_dir = os.path.join(data_root, f'tf_records/{mode}')
    image_ids = np.loadtxt(os.path.join(data_root, f'{mode}.txt'), dtype=str)

    n_samples = len(image_ids)
    pbar = tqdm(total=n_samples)


    if not os.path.exists(tfrecords_dir):
        os.makedirs(tfrecords_dir)

    num_tfrecords = n_samples // NUM_SAMPLES_PER_RECORD

    if n_samples % NUM_SAMPLES_PER_RECORD:
        num_tfrecords += 1  # add one record if there are any remaining sample

    for tfrec_num in range(num_tfrecords):
        # samples = annotations[(tfrec_num * NUM_SAMPLES_PER_RECORD): ((tfrec_num + 1) * num_samples)]

        sample_names = image_ids[(tfrec_num * NUM_SAMPLES_PER_RECORD): ((tfrec_num + 1) * NUM_SAMPLES_PER_RECORD)]

        with tf.io.TFRecordWriter(tfrecords_dir + "/tfr_%.2i-%i.tfrec" % (tfrec_num, n_samples)) as writer:
            for sample in sample_names:
                image_path = os.path.join(image_root, sample + '.png')
                ano_path = os.path.join(label_root, sample + '.txt')
                image = tf.io.decode_png(tf.io.read_file(image_path))
                h, w, _ = image.shape

                annotations = np.loadtxt(ano_path)
                if len(annotations.shape) == 1:
                    annotations = annotations.reshape(1, -1)

                boxes = annotations[:, 1:5]
                boxes = np.array(boxes)
                boxes[:, 0] *= image.shape[1]  # w
                boxes[:, 1] *= image.shape[0]  # h
                boxes[:, 2] *= image.shape[1]  # w
                boxes[:, 3] *= image.shape[0]  # h

                class_id = [int(i) for i in annotations[:, 0]]
                global_coord = annotations[0, 5:]
                image_id = sample.split('.')[0]
                # image = tf.expand_dims(image, axis=0)

                if len(boxes) < 50:  # we pad the boxes and class_id to have a fixed size of 50 for parallel processing
                    padded_box = np.zeros((50 - len(boxes), 4), dtype=np.float32)
                    boxes = np.concatenate([boxes, padded_box], axis=0)
                    class_id = class_id + [-1] * (50 - len(class_id))
                    # all boxes and gloabel_coord are already in absolute resolution
                label_info = {"boxes": boxes, "classes": class_id, "image_id": image_id,
                              "global_coord": global_coord}
                data_point = create_data_point(image, image_path, label_info)
                writer.write(data_point.SerializeToString())
                pbar.update(1)
    pbar.close()


if __name__ == "__main__":

    dataset_name = ["dataset_whole", "dataset_split"]
    mode_list = ["train", "val"]

    for dataset_name in dataset_name:
        for mode in mode_list:
            print(f"Generating tfrecords for {dataset_name}--{mode}")
            generate_tfrecords_dataset(dataset_name, mode)
