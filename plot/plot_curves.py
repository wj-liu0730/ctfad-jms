import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import json

data_index = {"ensemble": {"loss": ["ae_loss", "resnet_loss", "trans_loss"],
                           "prc": ["ae_prc", "resnet_prc", "trans_prc"],
                           "roc": ["ae_roc", "resnet_roc", "trans_roc"]
                           },

              "whole_split": [{"loss": ["split", "whole"],
                               "map_50": ["split", "whole"],
                               "map_75": ["split", "whole"]},
                              {"recall_large": ["split", "whole"],
                               "recall_small": ["split", "whole"],
                               "detection_100": ["split", "whole"]}],
              }


def plot_ensemble():
    labels = ["AutoEncoder", "ResNet", "Transformer"]
    y_labels = ["Training Loss", "PRC AUC", "ROC AUC"]
    data_folder = "plot/ensemble"
    sns.set_style('darkgrid')
    span_value = 3
    steps_to_keep = 50
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, metric in enumerate(data_index["ensemble"]):
        for j, runs in enumerate(data_index["ensemble"][metric]):
            data_frame = pd.read_csv(os.path.join(data_folder, f"{metric}/{runs}.csv"))
            num_rows = len(data_frame)
            skip_rows = list(range(steps_to_keep, num_rows))
            data_frame = data_frame.drop(skip_rows)
            num_rows = len(data_frame)
            data_frame['Steps'] = range(num_rows)
            data_frame['Mean'] = data_frame['Value'].ewm(span=span_value).mean()
            data_frame['Std'] = data_frame['Value'].ewm(span=span_value).std()
            row_std_error = data_frame['Std'] / np.sqrt(span_value)
            data_frame['Mean-Std-error'] = data_frame['Mean'] - row_std_error
            data_frame['Mean+Std-error'] = data_frame['Mean'] + row_std_error

            sns.lineplot(x='Steps', y='Value', data=data_frame, ax=axes[i], label=labels[j], legend=False,
                         linewidth=3.5)
            axes[i].fill_between(data_frame['Steps'], data_frame['Mean-Std-error'], data_frame['Mean+Std-error'],
                                 alpha=0.3)
            axes[i].set_xlabel("Epochs", fontsize=20)
            axes[i].set_ylabel(y_labels[i], fontsize=20)
            axes[i].tick_params(axis='both', which='major', labelsize=15)

        if metric == "loss":
            axes[i].set_ylim([0, 0.5])

    # axes[1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), fancybox=False, shadow=False, ncol=3, fontsize=18)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.52, 0), fancybox=False, shadow=False, ncol=3,
               fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(f"ensemble.png", dpi=300)


def plot_whole_split():
    labels = ["Split", "Whole"]
    y_labels = [["Training Loss", "Map (IoU = 0.5)", "Map (IoU = 0.7) "],
                ["Recall (L)", "Recall (S)", "Max Detections"]]
    data_folder = "plot/whole_split"
    sns.set_style('darkgrid')
    span_value = 5
    steps_to_keep = 200
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for k, group in enumerate(data_index["whole_split"]):
        for i, metric in enumerate(group):
            for j, runs in enumerate(group[metric]):
                data_frame = pd.read_csv(os.path.join(data_folder, f"{metric}/{runs}.csv"))
                num_rows = len(data_frame)
                skip_rows = list(range(steps_to_keep, num_rows))
                data_frame = data_frame.drop(skip_rows)
                data_frame['Steps'] = data_frame['Step']
                data_frame['Mean'] = data_frame['Value'].ewm(span=span_value).mean()
                data_frame['Std'] = data_frame['Value'].ewm(span=span_value).std()
                row_std_error = data_frame['Std'] / np.sqrt(span_value)
                data_frame['Mean-Std-error'] = data_frame['Mean'] - row_std_error
                data_frame['Mean+Std-error'] = data_frame['Mean'] + row_std_error

                sns.lineplot(x='Steps', y='Value', data=data_frame, ax=axes[k, i], label=labels[j], legend=False,
                             linewidth=3.5)
                axes[k, i].fill_between(data_frame['Steps'], data_frame['Mean-Std-error'], data_frame['Mean+Std-error'],
                                     alpha=0.3)
                axes[k, i].set_xlabel("Epochs", fontsize=20)
                axes[k, i].set_ylabel(y_labels[k][i], fontsize=20)
                axes[k, i].tick_params(axis='both', which='major', labelsize=15)

            if metric == "loss":
                axes[k, i].set_ylim([0, 50])
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, ['Split image', 'Whole image'], loc="lower center", bbox_to_anchor=(0.52, 0), fancybox=False, shadow=False, ncol=3,
               fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(f"plot/whole_split.png", dpi=300)


if __name__ == "__main__":
    plot_ensemble()
    plot_whole_split()
