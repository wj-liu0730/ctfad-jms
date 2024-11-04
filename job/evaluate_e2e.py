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
    writer = SummaryWriter(os.path.join(cfg.JobParams.output_path, cfg.JobParams.job_name))

    dataset = DatasetLoader(cfg.DataParams)

    assert cfg.AlgoParams.pretrained_model_path != '', "Pretrained model path is required"

    algo = YoloAlgo(cfg.AlgoParams, class_mapping=CLASS_MAPPING, bbox_format=BBOX_FORMAT)
    random.seed(cfg.JobParams.seed)
    np.random.seed(cfg.JobParams.seed)

    tf_test_dataset = dataset.load_test_dataset()
    test_loop(cfg, algo, tf_test_dataset, writer)

def test_loop(cfg, algo, tf_test_dataset, writer):

    algo.evaluattion_metrics.reset_state()
    visualization = None
    time_all = []

    # warm up
    warm_up_dataset = tf_test_dataset.take(3)
    for warm_up_data in warm_up_dataset:
        inputs = warm_up_data['image']
        if cfg.DataParams.dataset_name == 'dataset_split':
            image_id = warm_up_data['image_id']
            inputs, coor = load_split_image(image_id)
        algo.predict(inputs)

    top_n = 100  # take the top 100 boxes as the final result

    for test_data in tf_test_dataset:
        start_time = time.perf_counter()

        image = test_data['image']
        labels = test_data['bounding_box']

        if cfg.DataParams.dataset_name == 'dataset_split':
            image_id = test_data['image_id']
            inputs, coor = load_split_image(image_id)
            predictions = algo.predict(inputs)
            bs, num_box, c = predictions['boxes'].shape
            coor = np.expand_dims(coor, axis=1).repeat(num_box, axis=1)
            predictions['boxes'][:, :, :2] = predictions['boxes'][:, :, :2] + coor
            for key, value in predictions.items():
                # aggregate all results from split images to the whole image
                if key != 'num_detections':
                    predictions[key] = np.concatenate(value, axis=0)[np.newaxis, :]

            predictions['number_detections'] = (1,)
            # sort the boxes by confidence and the size of the boxes
            sort_ind = np.argsort(predictions['confidence'] * -1, axis=-1)  # * -1 for descending order
            predictions['confidence'] = np.take_along_axis(predictions['confidence'], sort_ind, axis=-1)
            predictions['classes'] = np.take_along_axis(predictions['classes'], sort_ind, axis=-1)
            predictions['boxes'] = predictions['boxes'][:, sort_ind[0], :]

            box_areas = predictions['boxes'][:, :, 2] * predictions['boxes'][:, :, 3]
            box_sort_ind = np.argsort(box_areas * -1, axis=-1)
            predictions['confidence'] = np.take_along_axis(predictions['confidence'], box_sort_ind, axis=-1)
            predictions['classes'] = np.take_along_axis(predictions['classes'], box_sort_ind, axis=-1)
            predictions['boxes'] = predictions['boxes'][:, box_sort_ind[0], :]
        else:
            predictions = algo.predict(image)

        # we take the top_n boxes as the final result
        predictions['boxes'] = predictions['boxes'][:, :top_n, :]
        predictions['confidence'] = predictions['confidence'][:, :top_n]
        predictions['classes'] = predictions['classes'][:, :top_n]

        used_time = time.perf_counter() - start_time
        time_all.append(used_time)

        algo.evaluattion_metrics.update_state(labels, predictions)

        if visualization is None:
            visualization \
                = visualize_detections(image, predictions, labels, algo.bbox_format, algo.class_mapping)
            writer.add_figure(f"val/visualization", visualization, 0)

    test_result = algo.get_performance_result()

    for key, value in test_result.items():
        writer.add_scalar(f"val/{key}", value.numpy(), 0)

    total_params = algo.model.count_params()
    writer.add_scalar(f"val/total_params", total_params, 0)

    time_mean = np.mean(time_all)
    time_std = np.std(time_all)
    writer.add_scalar(f"val/time_mean", time_mean, 0)
    writer.add_scalar(f"val/time_std", time_std, 0)


if __name__ == '__main__':
    run()
