import tensorflow as tf
import dataclasses
from dataclasses import field
import os
from algo.Anomaly.Models.ResNet import ResNetConfig, ResNetModel
from algo.Anomaly.Models.Transformer import TransformerModel, TransformerConfig
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from algo.Anomaly.utils import plot_curves_result


@dataclasses.dataclass
class AlgoConfig:
    epochs: int = 100
    evaluation_period: int = 5
    learning_rate: float = 0.001
    loss_function: str = "binary_crossentropy"


class ClsAlgo:
    def __init__(self, params, input_shape, output_shape):
        self.params = params
        self.model_params = self.params.model_config
        self.input_shape = input_shape
        self._loss = self._build_loss_function()
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=self.params.learning_rate)
        self.model = None

    def _build_loss_function(self):
        if self.params.loss_function == "mean_squared_error":
            return tf.keras.losses.MeanSquaredError()
        elif self.params.loss_function == "binary_crossentropy":
            return tf.keras.losses.BinaryCrossentropy()
        else:
            raise NotImplementedError

    def train_step(self, train_dataset_batch):
        inputs = train_dataset_batch["image"]
        labels = train_dataset_batch["label"]
        train_loss = self._optimize(inputs, labels)
        train_info = {"train_loss": train_loss.numpy()}
        return train_info

    @tf.function
    def _optimize(self, inputs, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(inputs)
            loss = self._loss(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def validate_step(self, val_dataset_batch):
        inputs = val_dataset_batch["image"]
        labels = val_dataset_batch["label"]
        predictions = self.model(inputs)
        val_loss = self._loss(labels, predictions)
        validation_info = {"val_loss": val_loss.numpy()}
        return validation_info, predictions

    def performance_eval(self, labels, predictions):
        # here we can calculate the performance metrics
        rec_error = tf.square(labels - predictions)
        rec_error = tf.reduce_mean(rec_error, axis=(1, 2, 3))
        return rec_error

    def predict(self, inputs):
        return self.model(inputs)

    def get_visualization_result(self, inputs, outputs, num_to_visualize=5):
        input_images = inputs[:num_to_visualize]
        predicted_labels = outputs[:num_to_visualize]
        fig, ax = plt.subplots(1, len(input_images), figsize=(20, 10))
        for i, (input_image, label) in enumerate(zip(input_images, predicted_labels)):
            ax[i].imshow(input_image)
            ax[i].axis('off')
            ax[i].set_title(f"Predicted Label: {label}")
        return fig

    def decode_result(self, pre_prob, threshold=0.5):
        pre_prob = pre_prob.numpy().squeeze()
        return (pre_prob > threshold).astype(int)

    def get_performance_info(self, all_inputs, all_predictions, all_labels):
        all_inputs = np.array(all_inputs)
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        acc_plot, best_accuracy = self.get_accuracy_stat(all_predictions, all_labels)
        auc_plot, best_threshold_roc, prc_auc, roc_auc = plot_curves_result(all_labels, all_predictions)

        best_performance = roc_auc
        performance_info = {"best_performance": best_performance,
                            "best_threshold_roc": best_threshold_roc,
                            "prc_auc": prc_auc,
                            "roc_auc": roc_auc,
                            "acc_plot": acc_plot,
                            "auc_plot": auc_plot}

        return performance_info

    def get_accuracy_stat(self, predictions, all_labels):
        thresholds = np.linspace(np.min(predictions), np.max(predictions), 100)
        accuracies = []
        for threshold in thresholds:
            # Classify as positive (1) if the error is below the threshold
            predictions = (predictions > threshold).astype(int)
            # Calculate accuracy based on how well predictions match true labels
            accuracy = accuracy_score(all_labels, predictions)
            accuracies.append(accuracy)

        fig = plt.figure(figsize=(9, 6))
        plt.plot(thresholds, accuracies, label='Accuracy')
        plt.title('Accuracy vs Threshold')
        plt.xlabel('Threshold')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()

        best_accuracy = np.max(accuracies)
        return fig, best_accuracy

    def save_weights(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save_weights(os.path.join(path, "model.weights.h5"))


@dataclasses.dataclass
class ResNetClsAlgoConfig(AlgoConfig):
    model_config: ResNetConfig = field(default_factory=ResNetConfig)


class ResNetClsAlgo(ClsAlgo):
    def __init__(self, params: ResNetClsAlgoConfig, input_shape, output_shape):
        super().__init__(params, input_shape, output_shape)
        self.model = ResNetModel(params.model_config, input_shape, output_shape).model


@dataclasses.dataclass
class TransformerClsAlgoConfig(AlgoConfig):
    model_config: TransformerConfig = field(default_factory=TransformerConfig)


class TransformerClsAlgo(ClsAlgo):
    def __init__(self, params: TransformerClsAlgoConfig, input_shape, output_shape):
        super().__init__(params, input_shape, output_shape)
        self.model = TransformerModel(params.model_config, input_shape, output_shape).model
