import os
import sys
import hydra
import random
import numpy as np
import dataclasses
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algo.Anomaly.ADAlgoClass import ResNetClsAlgo, ResNetClsAlgoConfig
from algo.Anomaly.utils import train_loop, test_loop
from job.job_config import JobConfig
from hydra.core.config_store import ConfigStore

from dataset.dataset_loader.WeldingSpotAD import WeldingSpotADConfig as DataConfig
from dataset.dataset_loader.WeldingSpotAD import WeldingSpotAD as DatasetLoader
from tensorboardX import SummaryWriter


@dataclasses.dataclass
class AllConfig:
    AlgoParams: ResNetClsAlgoConfig = dataclasses.field(default_factory=ResNetClsAlgoConfig)
    DataParams: DataConfig = dataclasses.field(default_factory=DataConfig)
    JobParams: JobConfig = dataclasses.field(default_factory=JobConfig)


@hydra.main(version_base=None, config_path="../config", config_name='Classification_ResNet')
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
    algo = ResNetClsAlgo(cfg.AlgoParams, dataset.input_shape, 1) # output probability

    random.seed(cfg.JobParams.seed)
    np.random.seed(cfg.JobParams.seed)

    if cfg.JobParams.run_mode == 'train':
        tf_training_data = dataset.load_train_dataset()
        tf_val_dataset = dataset.load_val_dataset()
        train_loop(algo, tf_training_data, tf_val_dataset, writer, cfg)
    else:
        tf_test_dataset = dataset.load_test_dataset()
        test_loop(algo, tf_test_dataset, writer)


if __name__ == '__main__':

    run()
