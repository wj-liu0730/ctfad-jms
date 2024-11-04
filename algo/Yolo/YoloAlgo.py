import logging

from dataclasses import dataclass

import tensorflow as tf
import os
import keras_cv


from algo.framework import AlgoConfig, Algo
from keras_cv import visualization
import math
import numpy as np


@dataclass
class YoloAlgoConfig(AlgoConfig):
    model_name: str = 'yolo_v8'
    backbone_type: str = "yolo_v8_s_backbone_coco"  # all backbones are here https://keras.io/api/keras_cv/models/
    backbone_freeze: bool = False
    global_clipnorm: float = 10
    box_loss_weight: float = 7.5
    classification_loss_weight: float = 0.5
    fpn_depth: int = 1


class YoloAlgo(Algo):
    def __init__(self, params: YoloAlgoConfig, class_mapping, bbox_format="center_xywh"):
        super(YoloAlgo).__init__()
        self.params = params
        self._class_mapping = class_mapping
        self._num_class = len(class_mapping)
        self._bbox_format = bbox_format
        self._backbone = keras_cv.models.YOLOV8Backbone.from_preset(self.params.backbone_type)
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=self.params.learning_rate)
        self.model = self._build_model()
        self.evaluattion_metrics = keras_cv.metrics.BoxCOCOMetrics(bounding_box_format=self._bbox_format,
                                                                   evaluate_freq=1,)

    def _build_model(self) -> tf.keras.Model:
        yolo = keras_cv.models.YOLOV8Detector(
            num_classes=self._num_class,
            bounding_box_format=self._bbox_format,
            backbone=self._backbone,
            fpn_depth=self.params.fpn_depth   # the depth of the feature pyramid network options: 1,2,3
        )
        yolo.compile(
            classification_loss='binary_crossentropy',
            box_loss='ciou',
            optimizer=self._optimizer,
            box_loss_weight=self.params.box_loss_weight,
            classification_loss_weight=self.params.classification_loss_weight,
            jit_compile=False,
        )  # here we use the model wrapper to compile the model.

        if self.params.pretrained_model_path != '':
            yolo.load_weights(self.params.pretrained_model_path)
            logging.info("Pretrained model loaded ...")

        yolo.backbone.trainable = self.params.backbone_freeze

        return yolo

    def optimize(self, train_minibatch):
        inputs = train_minibatch['image']
        labels = train_minibatch['bounding_box']
        loss = self._optimize(inputs, labels)
        return loss

    def _optimize(self, inputs, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(inputs)
            loss = self.model.compute_loss(x=inputs, y=labels, y_pred=predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def validate(self, val_minibatch):
        inputs = val_minibatch['image']
        labels = val_minibatch['bounding_box']
        loss = self._validate(inputs, labels)
        return loss.numpy().mean()

    def _validate(self, inputs, labels):
        predictions = self.model(inputs)
        loss = self.model.compute_loss(x=inputs, y=labels, y_pred=predictions)
        return loss

    def predict(self, inputs):
        return self.model.predict(inputs)  # filtered result after non-max suppression

    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        path = path + f"{self.params.model_name}.keras"
        self.model.save(path)

    def performance_eval(self, eval_minibatch):
        inputs = eval_minibatch['image']
        labels = eval_minibatch['bounding_box']
        predictions = self.predict(inputs)
        self.evaluattion_metrics.update_state(labels, predictions)

    def get_visualization_result(self, minibatch):
        inputs = minibatch['image']
        labels = minibatch['bounding_box']
        predictions = self.predict(inputs)
        fig = visualize_detections(inputs, predictions, labels, self._bbox_format, self._class_mapping)
        return fig

    def get_performance_result(self):
        return self.evaluattion_metrics.result()


    @property
    def class_mapping(self):
        return self._class_mapping

    @property
    def bbox_format(self):
        return self._bbox_format


def visualize_detections(image, y_pred, y_true, bounding_box_format, class_mapping, col=4):
    if image.shape[0] < col:
        col = image.shape[0]
    row = math.ceil(image.shape[0] / col)
    y_pred = bounding_box_to_ragged(y_pred)
    fig = visualization.plot_bounding_box_gallery(
        image,
        value_range=(0, 255),
        bounding_box_format=bounding_box_format,
        y_true=y_true,
        y_pred=y_pred,
        scale=4,
        rows=row,
        cols=col,
        show=None,
        path=None,
        font_scale=0.7,
        dpi=300,
        class_mapping=class_mapping,
    )
    return fig


def bounding_box_to_ragged(bounding_boxes, sentinel=-1, dtype=tf.float32):
    boxes = bounding_boxes.get("boxes")
    classes = bounding_boxes.get("classes")
    confidence = bounding_boxes.get("confidence", None)

    mask = classes != sentinel

    boxes = tf.ragged.boolean_mask(boxes, mask)
    classes = tf.ragged.boolean_mask(classes, mask)
    if confidence is not None:
        confidence = tf.ragged.boolean_mask(confidence, mask)

    if isinstance(boxes, tf.Tensor):
        boxes = tf.RaggedTensor.from_tensor(boxes)

    if isinstance(classes, tf.Tensor) and len(classes.shape) > 1:
        classes = tf.RaggedTensor.from_tensor(classes)

    if confidence is not None:
        if isinstance(confidence, tf.Tensor) and len(confidence.shape) > 1:
            confidence = tf.RaggedTensor.from_tensor(confidence)

    result = bounding_boxes.copy()
    result["boxes"] = tf.cast(boxes, dtype)
    result["classes"] = tf.cast(classes, dtype)

    if confidence is not None:
        result["confidence"] = tf.cast(confidence, dtype)

    return result





