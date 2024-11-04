import os
import dataclasses
from pathlib import Path
import tensorflow as tf
from keras_cv import visualization
import numpy as np

ClASS_IDS = [
    "normal",
    "abnormal",
]
CLASS_MAPPING = dict(zip(range(len(ClASS_IDS)), ClASS_IDS))

BBOX_FORMAT = "center_xywh"


@dataclasses.dataclass
class WeldingSpotConfig:
    """Data Configuration"""
    dataset_name: str = 'dataset_whole'
    data_root: str = 'dataset/weldingspot/'
    train_batch_size: int = 8
    val_batch_size: int = 4
    test_batch_size: int = 1
    num_workers: int = 4
    shuffle: bool = True
    task_type: str = 'object_detection'


class WeldingSpot:

    def __init__(self, params: WeldingSpotConfig):
        super(WeldingSpot).__init__()
        self.params = params
        self.num_classes = len(ClASS_IDS)
        self.dataset_root = Path(self.params.data_root) / f"{self.params.dataset_name}/"
        self.test_dataset_root = Path(self.params.data_root) / "dataset_whole/"
        self.train_val_ratio = 0.8

    def load_train_dataset(self):
        return self._load_dataset('train'). \
            batch(self.params.train_batch_size, drop_remainder=True). \
            prefetch(tf.data.AUTOTUNE)

    def load_test_dataset(self):
        # here the test dataset is the validation dataset
        return self._load_dataset('test'). \
            batch(self.params.test_batch_size, drop_remainder=True). \
            prefetch(tf.data.AUTOTUNE)

    def load_val_dataset(self):
        return self._load_dataset('val'). \
            batch(self.params.val_batch_size, drop_remainder=True). \
            prefetch(tf.data.AUTOTUNE)

    def _load_dataset(self, mode):
        if mode == 'test':
            data_folder = os.path.join(self.test_dataset_root, "tf_records", "val")
        else:
            data_folder = os.path.join(self.dataset_root, "tf_records", mode)

        all_files = []
        for name in os.listdir(data_folder):
            all_files.append(os.path.join(data_folder, name))
        dataset = tf.data.TFRecordDataset(all_files)
        dataset = dataset.map(self._parse_tfrecord_fn)

        if mode == 'test':
            return dataset
        return dataset.shuffle(buffer_size=dataset.cardinality())

    def _parse_tfrecord_fn(self, data_point_tfr):
        feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "path": tf.io.FixedLenFeature([], tf.string),
            "boxes": tf.io.FixedLenFeature([], tf.string),
            "classes": tf.io.VarLenFeature(tf.int64),
            "image_id": tf.io.VarLenFeature(tf.string),
            "global_coord": tf.io.VarLenFeature(tf.float32),
        }
        data_point = tf.io.parse_single_example(data_point_tfr, feature_description)

        data_point["image"] = tf.io.decode_png(data_point["image"], channels=3)

        if self.params.dataset_name == 'dataset_whole':
            data_point["image"] = tf.image.resize_with_pad(data_point["image"], 4032, 3036)  # 3024

        data_point["boxes"] = tf.RaggedTensor.from_tensor(tf.io.parse_tensor(data_point["boxes"], out_type=tf.float64))
        data_point["classes"] = tf.sparse.to_dense(data_point["classes"])
        data_point["global_coord"] = tf.sparse.to_dense(data_point["global_coord"])
        bbounding_box = {"boxes": data_point["boxes"].to_tensor(), "classes": data_point["classes"]}
        data_point["bounding_box"] = bbounding_box
        return data_point


def visualize_dataset(dataset, value_range, rows, cols, bounding_box_format=BBOX_FORMAT, path=None):
    inputs = next(iter(dataset.take(1)))
    image, bounding_boxes = inputs["image"], inputs["bounding_box"]

    visualization.plot_bounding_box_gallery(
        image,
        value_range=value_range,
        rows=rows,
        cols=cols,
        true_color=(0, 255, 0),
        y_true=bounding_boxes,
        show=False,
        scale=5,
        dpi=300,
        font_scale=3.5,
        bounding_box_format=bounding_box_format,
        class_mapping=CLASS_MAPPING,
        path=path
    )
    return


def load_split_image(image_id):
    path = "dataset/weldingspot/dataset_split/images"
    label_path = 'dataset/weldingspot/dataset_split/labels'
    all_files = os.listdir(path)
    images = []
    coor = []
    image_id = np.array(image_id.values).astype(str).squeeze()

    for file_name in all_files:
        image_name = file_name.split('.')[0].split("_")[:-1]
        image_name = "_".join(image_name)
        if image_name == image_id:
            image_path = os.path.join(path, file_name)
            image = tf.io.decode_png(tf.io.read_file(image_path))
            images.append(image)
            ano_name = file_name.split('.')[0] + '.txt'
            ano_path = os.path.join(label_path, ano_name)
            with open(ano_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    x_coor = float(line.split(' ')[-3])
                    y_coor = float(line.split(' ')[-2])
                    coor.append([x_coor, y_coor])
                    break

    return np.array(images), np.array(coor)


def crop_images_from_boxes(image, boxes, box_format='center_xywh'):
    assert box_format == 'center_xywh', "Only center_xywh format is supported for now"

    crops = []
    for box in boxes:
        x, y, w, h = box
        y0 = int(y-h/2)
        x0 = int(x-w/2)
        h = int(h)
        w = int(w)
        if h < 0:
            crop = tf.zeros(shape=(400, 400, 3))  # add a dummy crop if the bounding box is invalid
        else:
            crop = image[y0:y0 + h, x0:x0 + w, :]
            crop = tf.image.resize_with_pad(crop, 400, 400)
        crops.append(crop)
    return np.array(crops)


if __name__ == "__main__":
    dataset = WeldingSpot(WeldingSpotConfig())
    dataset = dataset.load_train_dataset()
    visualize_dataset(dataset, value_range=(0, 255), rows=1, cols=1, bounding_box_format=BBOX_FORMAT, path="test.png")
