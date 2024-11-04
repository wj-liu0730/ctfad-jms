import os
import sys
import hydra
import random
import numpy as np
import dataclasses
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algo.Anomaly.ADAlgoRecon import CnnReconAlgoConfig, CnnReconAlgo
from algo.Anomaly.utils import plot_curves_result
from job.job_config import JobConfig
from hydra.core.config_store import ConfigStore

from dataset.dataset_loader.WeldingSpotAD import WeldingSpotADConfig as DataConfig
from dataset.dataset_loader.WeldingSpotAD import WeldingSpotAD as DatasetLoader
from tensorboardX import SummaryWriter


@dataclasses.dataclass
class AllConfig:
    AlgoParams: CnnReconAlgoConfig = dataclasses.field(default_factory=CnnReconAlgoConfig)
    DataParams: DataConfig = dataclasses.field(default_factory=DataConfig)
    JobParams: JobConfig = dataclasses.field(default_factory=JobConfig)


@hydra.main(version_base=None, config_path="../config", config_name='CnnRecon_Config')
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
    if cfg.DataParams.edge_preprocess:
        cfg.AlgoParams.model_config.output_activation = "relu"
    algo = CnnReconAlgo(cfg.AlgoParams, dataset.input_shape, dataset.input_shape)

    random.seed(cfg.JobParams.seed)
    np.random.seed(cfg.JobParams.seed)

    tf_test_dataset = dataset.load_test_dataset()
    test_loop(algo, tf_test_dataset, writer)

def test_loop(algo, test_dataset, writer, test_batch=2):
    for i, batch in enumerate(test_dataset):
        y_predict = algo.predict(batch)

        fig = algo.get_visualization_result(batch)
        writer.add_figure(f"test/{i}th_batch", fig)
        if (i + 1) >= test_batch:
            break

    y_predict = None
    y_true = None
    result_plot = plot_curves_result(y_true, y_predict)
    writer.add_figure(f"test/AUROC", result_plot)


if __name__ == '__main__':

    run()
