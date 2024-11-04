import os
import sys
import time

import hydra
import random
import numpy as np
import dataclasses
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algo.Yolo.YoloAlgo import YoloAlgoConfig, YoloAlgo, visualize_detections
from job.job_config import JobConfig
from hydra.core.config_store import ConfigStore

from dataset.dataset_loader.WeldingSpotYolo import WeldingSpot as DatasetLoader
from dataset.dataset_loader.WeldingSpotYolo import WeldingSpotConfig as DataConfig
from dataset.dataset_loader.WeldingSpotYolo import CLASS_MAPPING, BBOX_FORMAT, load_split_image
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
@dataclasses.dataclass
class AllConfig:
    AlgoParams: YoloAlgoConfig = dataclasses.field(default_factory=YoloAlgoConfig)
    DataParams: DataConfig = dataclasses.field(default_factory=DataConfig)
    JobParams: JobConfig = dataclasses.field(default_factory=JobConfig)


@hydra.main(version_base=None, config_path="../config", config_name='YoloWelding_Config')
def run(cfg: AllConfig) -> None:
    if not cfg.JobParams.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            exit("GPU allocated failed")

    cfg.JobParams.output_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    dataset = DatasetLoader(cfg.DataParams)

    assert cfg.AlgoParams.pretrained_model_path != '', "Pretrained model path is required"

    algo = YoloAlgo(cfg.AlgoParams, class_mapping=CLASS_MAPPING, bbox_format=BBOX_FORMAT)
    algo.model.summary()
    random.seed(cfg.JobParams.seed)
    np.random.seed(cfg.JobParams.seed)
    writer = SummaryWriter(os.path.join(cfg.JobParams.output_path, cfg.JobParams.job_name))
    tf_test_dataset = dataset.load_test_dataset()
    test_loop(cfg, algo, tf_test_dataset, writer)


def test_loop(cfg, algo, tf_test_dataset, writer):

    # layer_outputs = [layer.output for layer in algo.model.layers]

    # Create a new model that will return these outputs
    intermodel_1 = Model(inputs=algo.model.input, outputs=algo.model.get_layer('pa_fpn_p4p5_pre_bn').output)
    intermodel_2 = Model(inputs=algo.model.input, outputs=algo.model.get_layer('pa_fpn_p4p5_output_conv').output)
    intermodel_3 = Model(inputs=algo.model.input, outputs=algo.model.get_layer('pa_fpn_p3p4p5_output_conv').output)
    intermodel_4 = Model(inputs=algo.model.input, outputs=algo.model.get_layer('pa_fpn_p3p4p5_output').output)
    intermodel_6 = Model(inputs=algo.model.input, outputs=algo.model.get_layer('yolo_v8_head_1_class_1_conv').output)
    intermodel_7 = Model(inputs=algo.model.input, outputs=algo.model.get_layer('yolo_v8_head_1_box_1_conv').output)
    intermodel_8 = Model(inputs=algo.model.input, outputs=algo.model.get_layer('yolo_v8_head_1_class_2_conv').output)
    intermodel_9 = Model(inputs=algo.model.input, outputs=algo.model.get_layer('yolo_v8_head_1_box_2_conv').output)
    intermodel_10 = Model(inputs=algo.model.input, outputs=algo.model.get_layer('yolo_v8_head_1_class_3_conv').output)
    intermodel_11 = Model(inputs=algo.model.input, outputs=algo.model.get_layer('yolo_v8_head_1_box_3_conv').output)

    all_models = [intermodel_1, intermodel_2, intermodel_3, intermodel_4, intermodel_6, intermodel_7, intermodel_8,
                  intermodel_9, intermodel_10, intermodel_11]

    for j, test_data in enumerate(tf_test_dataset):
        if cfg.DataParams.dataset_name == 'dataset_split':
            os.makedirs(f"outputs/plot/feature_maps_{j}", exist_ok=True)
            image_id = test_data['image_id']
            inputs, _ = load_split_image(image_id)
            inputs = tf.expand_dims(inputs[0], axis=0)  # take the first image
            plt.imshow(inputs[0])
            plt.axis('off')
            plt.savefig(f"outputs/plot/feature_maps_{j}/input.png", dpi=300, bbox_inches='tight')
            plt.close()
            for i, int_model in enumerate(all_models):
                outputs = int_model(inputs)
                visualize_feature_maps(outputs, save_path=f"outputs/plot/feature_maps_{j}/{i}.png")



def visualize_feature_maps(feature_maps, save_path):
    plt.imshow(feature_maps[0, :, :, 0], cmap='coolwarm', interpolation='nearest')  # Assuming batch size = 1
    plt.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    run()
