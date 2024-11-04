import copy
import logging
import os
import sys
import time

import hydra
import random
import numpy as np
import dataclasses
import tensorflow as tf
import keras_cv
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algo.Yolo.YoloAlgo import YoloAlgoConfig, YoloAlgo, visualize_detections
from algo.Anomaly.ADAlgoRecon import CnnReconAlgoConfig, CnnReconAlgo
from algo.Anomaly.ADAlgoClass import TransformerClsAlgoConfig, TransformerClsAlgo, ResNetClsAlgo, ResNetClsAlgoConfig

from job.job_config import JobConfig
from hydra.core.config_store import ConfigStore

from dataset.dataset_loader.WeldingSpotYolo import WeldingSpot as ObjDatasetLoader
from dataset.dataset_loader.WeldingSpotYolo import WeldingSpotConfig as OBJDataConfig
from dataset.dataset_loader.WeldingSpotYolo import CLASS_MAPPING, BBOX_FORMAT, load_split_image, crop_images_from_boxes
from tensorboardX import SummaryWriter
from dataset.dataset_loader.WeldingSpotAD import WeldingSpotADConfig as ADDataConfig
from dataset.dataset_loader.WeldingSpotAD import WeldingSpotAD as ADDataLoader


@dataclasses.dataclass
class AllConfig:
    YoloParams: YoloAlgoConfig = dataclasses.field(default_factory=YoloAlgoConfig)
    AutoEncoderParams: CnnReconAlgoConfig = dataclasses.field(default_factory=CnnReconAlgoConfig)
    TransformerParams: TransformerClsAlgoConfig = dataclasses.field(default_factory=TransformerClsAlgoConfig)
    ResNetParams: ResNetClsAlgoConfig = dataclasses.field(default_factory=ResNetClsAlgoConfig)
    OBJDataParams: OBJDataConfig = dataclasses.field(default_factory=OBJDataConfig)
    ADDataParams: ADDataConfig = dataclasses.field(default_factory=ADDataConfig)
    JobParams: JobConfig = dataclasses.field(default_factory=JobConfig)


@hydra.main(version_base=None, config_path="../config", config_name='Ctfad_Config')
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

    Objdataset = ObjDatasetLoader(cfg.OBJDataParams)
    ADdataset = ADDataLoader(cfg.ADDataParams)
    assert cfg.YoloParams.pretrained_model_path != '', "Pretrained model path is required"
    assert cfg.AutoEncoderParams.model_config.model_path != '', "Pretrained model path is required"
    assert cfg.TransformerParams.model_config.model_path != '', "Pretrained model path is required"
    assert cfg.ResNetParams.model_config.model_path != '', "Pretrained model path is required"

    yolo = YoloAlgo(cfg.YoloParams, class_mapping=CLASS_MAPPING, bbox_format=BBOX_FORMAT)

    autoencoder = CnnReconAlgo(cfg.AutoEncoderParams, ADdataset.input_shape, ADdataset.input_shape)
    transformer = TransformerClsAlgo(cfg.TransformerParams, ADdataset.input_shape, 1)
    resnet = ResNetClsAlgo(cfg.ResNetParams, ADdataset.input_shape, 1)

    yolo.model.summary()
    autoencoder.model.summary()
    transformer.model.summary()
    resnet.model.summary()

    random.seed(cfg.JobParams.seed)
    np.random.seed(cfg.JobParams.seed)

    obj_test_dataset = Objdataset.load_test_dataset()
    ad_test_dataset = ADdataset.load_test_dataset()
    test_loop(cfg, yolo, autoencoder, transformer, resnet, obj_test_dataset, ad_test_dataset, writer)


def test_loop(cfg, yolo, autoencoder, transformer, resnet, obj_test_dataset, ad_test_dataset, writer):

    # warm up
    warm_obj_dataset = obj_test_dataset.take(3)
    warm_ad_test_dataset = ad_test_dataset.take(3)

    for warm_up_obj_data, warm_up_ad_data in zip(warm_obj_dataset, warm_ad_test_dataset):
        obj_inputs = warm_up_obj_data['image']
        if cfg.OBJDataParams.dataset_name == 'dataset_split':
            image_id = warm_up_obj_data['image_id']
            obj_inputs, coor = load_split_image(image_id)
        yolo.predict(obj_inputs)
        ad_inputs = warm_up_ad_data['image']
        autoencoder.predict(ad_inputs)
        transformer.predict(ad_inputs)
        resnet.predict(ad_inputs)


    all_labels = []
    all_predictions_ae = []
    all_predictions_resnet = []
    all_predictions_transformer = []
    ens_predictions = []
    all_inputs = []

    for test_data in ad_test_dataset:
        ad_inputs = test_data['image']
        ad_labels = test_data['label']

        ae_predictions = autoencoder.predict(ad_inputs)
        ae_predictions = autoencoder.get_rec_error(ad_inputs, ae_predictions)
        trans_predictions = transformer.predict(ad_inputs)
        resnet_predictions = resnet.predict(ad_inputs)

        all_labels.extend(ad_labels.numpy().tolist())
        all_inputs.extend(ad_inputs.numpy().tolist())

        ae_res = autoencoder.decode_result(ae_predictions, threshold=13835.884882387907)
        tran_res = transformer.decode_result(trans_predictions, threshold=0.7390993328556751)
        resnet_res = resnet.decode_result(resnet_predictions, threshold=0.2710454774939496)

        all_predictions_ae.extend(ae_res.tolist())
        all_predictions_resnet.extend(resnet_res.tolist())
        all_predictions_transformer.extend(tran_res.tolist())

        res_mean = np.mean((ae_res, tran_res, resnet_res), axis=0)
        final_res = (res_mean >= 0.66).astype(int)  # majority 2 / 3
        ens_predictions.extend(final_res.tolist())

    ae_performance_info = resnet.get_performance_info(all_inputs, all_predictions_ae, all_labels)
    resnet_performance = resnet.get_performance_info(all_inputs, all_predictions_resnet, all_labels)
    transformer_performance = resnet.get_performance_info(all_inputs, all_predictions_transformer, all_labels)
    ens_performance = resnet.get_performance_info(all_inputs, ens_predictions, all_labels)

    for key, value in ae_performance_info.items():
        if "plot" in key:
            writer.add_figure(f"AutoEncoder/{key}", value, 0)
        else:
            writer.add_scalar(f"AutoEncoder/{key}", value, 0)

    for key, value in resnet_performance.items():
        if "plot" in key:
            writer.add_figure(f"ResNet/{key}", value, 0)
        else:
            writer.add_scalar(f"ResNet/{key}", value, 0)

    for key, value in transformer_performance.items():
        if "plot" in key:
            writer.add_figure(f"Transformer/{key}", value, 0)
        else:
            writer.add_scalar(f"Transformer/{key}", value, 0)

    for key, value in ens_performance.items():
        if "plot" in key:
            writer.add_figure(f"Ensemble/{key}", value, 0)
        else:
            writer.add_scalar(f"Ensemble/{key}", value, 0)

    autoencoder_params = autoencoder.model.count_params()
    transformer_params = transformer.model.count_params()
    resnet_params = resnet.model.count_params()
    total_params = autoencoder_params + transformer_params + resnet_params

    writer.add_text("validation/params", f"AutoEncoder: {autoencoder_params}, "
                                         f"Transformer: {transformer_params}, "
                                         f"ResNet: {resnet_params}, Total: {total_params}", 0)


if __name__ == '__main__':
    run()
