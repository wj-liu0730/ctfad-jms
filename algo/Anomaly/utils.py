import logging

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix


def train_loop(algo, tf_training_dataset, tf_val_dataset, writer, cfg):
    progress = tqdm(total=cfg.AlgoParams.epochs, desc="Training Process")
    best_performance = -100
    for ep in range(cfg.AlgoParams.epochs):
        training_info_dict = {}
        step_counter = 0

        if ep % cfg.AlgoParams.evaluation_period == 0:
            performance = val_loop(algo, tf_val_dataset, writer, ep)
            if performance > best_performance:
                best_performance = performance
                algo.save_weights(cfg.JobParams.output_path + '/tf_model/best_model/')

        for iter, batch in enumerate(tf_training_dataset):
            training_info = algo.train_step(batch)
            step_counter += 1

            for key, value in training_info.items():
                training_info_dict[key] = value \
                    if key not in training_info_dict else training_info_dict[key] + value

        for key, value in training_info_dict.items():
            writer.add_scalar("training/{}".format(key), value / step_counter, ep)
            print(f"Epoch {ep}, {key}={value / step_counter}")

        progress.update(1)
    algo.save_weights(cfg.JobParams.output_path + '/tf_model/final_model/')


def val_loop(algo, tf_eval_dataset, writer, ep):
    # the val_loop should be implemented differently for different algorithms and cases
    validation_info_dict = {}
    step_counter = 0
    all_labels = []
    all_predictions = []
    all_inputs = []
    visualization = None

    for batch in tf_eval_dataset:
        labels = batch["label"]
        inputs = batch["image"]
        validation_info, predictions = algo.validate_step(batch)

        all_labels.extend(labels.numpy().tolist())
        all_predictions.extend(predictions.numpy().tolist())
        all_inputs.extend(inputs.numpy().tolist())

        for key, value in validation_info.items():
            validation_info_dict[key] = value \
                if key not in validation_info_dict else validation_info_dict[key] + value

        if visualization is None:
            visualization = algo.get_visualization_result(inputs, predictions)
            writer.add_figure(f"validation/visualization", visualization, ep)

        step_counter += 1

    for key, value in validation_info_dict.items():
        writer.add_scalar("validation/{}".format(key), value / step_counter, ep)
        print(f"Epoch {ep}, {key}={value / step_counter}")

    # store the whole dataset is not mem efficient
    performance_info = algo.get_performance_info(all_inputs, all_predictions, all_labels)

    for key, value in performance_info.items():

        if "plot" in key:
            writer.add_figure(f"validation/{key}", value, ep)
        else:
            writer.add_scalar(f"validation/{key}", value, ep)

    return performance_info["best_performance"]


def test_loop(algo, test_dataset, writer, test_batch=2):
    for i, batch in enumerate(test_dataset):
        fig = algo.get_visualization_result(batch)
        writer.add_figure(f"test/{i}th_batch", fig)
        if (i + 1) >= test_batch:
            break
    logging.info("Finish Testing Process")


def plot_curves_result(y_true, y_pred_prob):
    fpr, tpr, roc_thres = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    precision, recall, prc_thres = precision_recall_curve(y_true, y_pred_prob)
    prc_auc = average_precision_score(y_true, y_pred_prob)

    # Best cutoff according to ROC
    roc_thres[0] -= 1
    distances = np.sqrt((1 - tpr) ** 2 + fpr ** 2)
    best_threshold_roc = roc_thres[np.argmin(distances)]

    # Plotting ROC and PRC
    fig = plt.figure(figsize=(12, 5))

    # ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Annotate ROC Curve
    roc_indices = np.round(np.linspace(0, len(roc_thres) - 1, 10)).astype(int)
    for i in roc_indices:
        plt.annotate(f'{roc_thres[i]:.4f}', (fpr[i], tpr[i]), textcoords="offset points", xytext=(10, -10))

    # Precision-Recall Curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2, label='PR curve (area = %0.2f)' % prc_auc)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")

    # Annotate PRC
    prc_indices = np.round(np.linspace(0, len(prc_thres) - 1, 10)).astype(int)
    for i in prc_indices[:-1]:  # Last point (recall=1) might not have a corresponding threshold
        plt.annotate(f'{prc_thres[i]:.4f}', (recall[i], precision[i]), textcoords="offset points", xytext=(10, 10))

    plt.tight_layout()

    # plt.savefig(save_path, dpi=300)
    # plt.show()
    return fig, best_threshold_roc, prc_auc, roc_auc