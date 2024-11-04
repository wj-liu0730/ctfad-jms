import tensorflow as tf
import dataclasses
from dataclasses import field
import os
from algo.Anomaly.Models.AutoEncoder import AutoEncoderConfig, AutoEncoderModel
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from algo.Anomaly.utils import plot_curves_result


@dataclasses.dataclass
class CnnReconAlgoConfig:
    model_config: AutoEncoderConfig = field(default_factory=AutoEncoderConfig)
    epochs: int = 100
    evaluation_period: int = 5
    learning_rate: float = 0.001
    loss_function: str = "mean_squared_error"  # this is the default loss function in tf.keras.losses


class CnnReconAlgo:
    def __init__(self, params: CnnReconAlgoConfig, input_shape, output_shape):
        self.params = params
        self.model_params = self.params.model_config
        self.input_shape = input_shape
        self._loss = self._build_loss_function()
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=self.params.learning_rate)
        self.model = AutoEncoderModel(self.model_params, input_shape, output_shape).model

    def _build_loss_function(self):
        if self.params.loss_function == "mean_squared_error":
            return tf.keras.losses.MeanSquaredError()
        elif self.params.loss_function == "binary_crossentropy":
            return tf.keras.losses.BinaryCrossentropy()
        else:
            raise NotImplementedError

    def train_step(self, train_dataset_batch):
        inputs = train_dataset_batch["image"]
        # here the labels are the same as the inputs for reconstruction task
        train_loss = self._optimize(inputs, inputs)
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

    def predict(self, inputs):
        return self.model(inputs)

    def decode_result(self, rec_error, threshold=0.1):
        # The threshold is obtained from the validation dataset
        predicted_labels = (rec_error > threshold).astype(int)
        return predicted_labels

    def validate_step(self, val_dataset_batch):
        inputs = val_dataset_batch["image"]
        predictions = self.model(inputs)
        val_loss = self._loss(inputs, predictions)
        validation_info = {"val_loss": val_loss.numpy()}
        return validation_info, predictions

    def get_rec_error(self, inputs, predictions):
        rec_error = np.square(inputs - predictions)
        rec_error = np.mean(rec_error, axis=(1, 2, 3))
        return rec_error

    def get_visualization_result(self, inputs, outputs, num_to_visualize=5):
        input_images = inputs[:num_to_visualize]
        outputs_images = outputs[:num_to_visualize]
        fig, ax = plt.subplots(2, len(input_images), figsize=(20, 10))
        for i, (input_image, out_image) in enumerate(zip(input_images, outputs_images)):
            ax[0][i].imshow(input_image)
            ax[1][i].imshow(out_image)
            ax[0][i].axis('off')
            ax[1][i].axis('off')
        return fig

    def get_performance_info(self, all_inputs, all_predictions, all_labels):
        all_inputs = np.array(all_inputs)
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        rec_error = self.get_rec_error(all_inputs, all_predictions)
        acc_plot, best_accuracy = self.get_accuracy_stat(rec_error, all_labels)

        auc_plot, best_threshold_roc, prc_auc, roc_auc = plot_curves_result(all_labels, rec_error)

        best_performance = roc_auc
        performance_info = {"best_performance": best_performance,
                            "best_threshold_roc": best_threshold_roc,
                            "prc_auc": prc_auc,
                            "roc_auc": roc_auc,
                            "acc_plot": acc_plot,
                            "auc_plot": auc_plot}

        return performance_info

    def get_accuracy_stat(self, reconstruction_error, all_labels):
        thresholds = np.linspace(np.min(reconstruction_error), np.max(reconstruction_error), 100)
        accuracies = []
        for threshold in thresholds:
            # Classify as positive (1) if the error is below the threshold
            predictions = (reconstruction_error > threshold).astype(int)
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
