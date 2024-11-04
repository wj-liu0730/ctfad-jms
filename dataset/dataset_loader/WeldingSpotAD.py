import os
import dataclasses
from pathlib import Path

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


@dataclasses.dataclass
class WeldingSpotADConfig:
    """Data Configuration"""
    dataset_name: str = 'dataset_anomaly'
    data_root: str = 'dataset/weldingspot/'
    train_batch_size: int = 8
    val_batch_size: int = 4
    test_batch_size: int = 6
    shuffle: bool = True
    task_type: str = 'reconstruction'  # 'reconstruction ' or 'classification'
    edge_preprocess: bool = True


class WeldingSpotAD:
    def __init__(self, params: WeldingSpotADConfig):
        super(WeldingSpotAD).__init__()
        self.params = params
        self.dataset_root = Path(self.params.data_root) / f"{self.params.dataset_name}/"
        self.resize_shape = (400, 400)
        if self.params.edge_preprocess:
            self.input_shape = (400, 400, 1)
        else:
            self.input_shape = (400, 400, 3)

    def load_train_dataset(self):
        if self.params.task_type == "reconstruction":
            return self._load_dataset_only_normal().batch(self.params.train_batch_size,
                                                          drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        else:
            return self._load_dataset('train').batch(self.params.train_batch_size, drop_remainder=True).prefetch(
                tf.data.AUTOTUNE)

    def load_test_dataset(self):
        # here the test dataset is the validation dataset
        return self._load_dataset('val').batch(self.params.test_batch_size, drop_remainder=True).prefetch(
            tf.data.AUTOTUNE)

    def load_val_dataset(self):
        return self._load_dataset('val').batch(self.params.val_batch_size, drop_remainder=True).prefetch(
            tf.data.AUTOTUNE)

    def _load_dataset(self, mode):
        data_folder = os.path.join(self.dataset_root)
        train_ids = np.loadtxt(os.path.join(data_folder, f'{mode}.txt'), dtype=str)
        all_image_list = [os.path.join(data_folder, 'images', f'{image_id}.png') for image_id in train_ids]
        all_labels_list = [os.path.join(data_folder, 'labels', f'{image_id}.txt') for image_id in train_ids]
        resize_shape = [self.resize_shape] * len(all_image_list)
        edge_preprocess = [self.params.edge_preprocess] * len(all_image_list)
        tf_dataset = tf.data.Dataset.from_tensor_slices(
            (all_image_list, all_labels_list, resize_shape, edge_preprocess))
        dataset = tf_dataset.map(tf_parse_fn)
        if mode != "train":
            return dataset
        return dataset.shuffle(buffer_size=dataset.cardinality())

    def _load_dataset_only_normal(self):
        data_folder = os.path.join(self.dataset_root)
        train_ids = np.loadtxt(os.path.join(data_folder, 'train.txt'), dtype=str)
        normal_image_list = []

        for image_id in train_ids:
            if np.loadtxt(os.path.join(data_folder, 'labels', f'{image_id}.txt'), dtype=int) == 0:
                normal_image_list.append(os.path.join(data_folder, "images", f'{image_id}.png'))

        print(f"Number of positive samples: {len(normal_image_list)} / {len(train_ids)}")
        resize_shape = [self.resize_shape] * len(normal_image_list)
        edge_preprocess = [self.params.edge_preprocess] * len(normal_image_list)
        tf_dataset = tf.data.Dataset.from_tensor_slices((normal_image_list, resize_shape, edge_preprocess))
        dataset = tf_dataset.map(tf_parse_fn_rec)
        return dataset.shuffle(buffer_size=dataset.cardinality())


def tf_parse_fn_rec(image_path, resize_shape, edge_preprocess):
    training_sample = {}
    image = tf.io.decode_png(tf.io.read_file(image_path), channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)  # normalize the image to [0,1] range
    if edge_preprocess:
        image = sobel_edge_filter(image)
        image = tf.expand_dims(image, axis=-1)
    training_sample['image'] = tf.image.resize_with_pad(image, resize_shape[0], resize_shape[1])

    return training_sample


def tf_parse_fn(image_path, label_path, resize_shape, edge_preprocess):
    training_sample = {}
    image = tf.io.decode_png(tf.io.read_file(image_path), channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)  # normalize the image to [0,1] range
    if edge_preprocess:
        image = sobel_edge_filter(image)
        image = tf.expand_dims(image, axis=-1)
    training_sample['image'] = tf.image.resize_with_pad(image, resize_shape[0], resize_shape[1])
    training_sample['label'] = int(tf.io.read_file(label_path))
    return training_sample


def sobel_edge_filter(image):
    image = tf.expand_dims(image, axis=0)
    image = tf.image.rgb_to_grayscale(image)
    sobel_edges = tf.image.sobel_edges(image)
    sobel_x = sobel_edges[..., 0]
    sobel_y = sobel_edges[..., 1]
    image = tf.math.sqrt(tf.math.square(sobel_x) + tf.math.square(sobel_y))
    image = tf.squeeze(image)
    return image


def visualize_dataset(images, labels, num_to_visualize):
    """plot original image and labels using subplots"""
    images = images[:num_to_visualize]
    labels = labels[:num_to_visualize]
    fig, ax = plt.subplots(1, len(images), figsize=(20, 20))
    for i, (image, label) in enumerate(zip(images, labels)):
        ax[i].imshow(image)
        ax[i].set_title(f"Label: {label}", fontsize=30)
        ax[i].axis('off')
    plt.show()


if __name__ == "__main__":
    dataset = WeldingSpotAD(WeldingSpotADConfig())
    dataset = dataset.load_test_dataset()
    for batch in dataset:
        images = batch['image']
        labels = batch['label']
        visualize_dataset(images, labels, num_to_visualize=4)
        break
