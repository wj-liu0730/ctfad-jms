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

    random.seed(cfg.JobParams.seed)
    np.random.seed(cfg.JobParams.seed)

    obj_test_dataset = Objdataset.load_test_dataset()
    ad_test_dataset = ADdataset.load_test_dataset()
    test_loop(cfg, yolo, autoencoder, transformer, resnet, obj_test_dataset, ad_test_dataset, writer)


def test_loop(cfg, yolo, autoencoder, transformer, resnet, obj_test_dataset, ad_test_dataset, writer):
    yolo.evaluattion_metrics.reset_state()
    visualization = None
    time_all = []

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

    top_n = 50
    n_threshold = 300
    all_labels = []

    all_ae_predictions = []
    all_resnet_predictions = []
    all_transformer_predictions = []

    all_origin_predictions = []
    valid_nums = []
    ae_time = []
    resnet_time = []
    transformer_time = []

    for test_data in obj_test_dataset:
        start_time = time.perf_counter()

        image = test_data['image']
        labels = test_data['bounding_box']
        all_labels.append(labels)

        if cfg.OBJDataParams.dataset_name == 'dataset_split':
            image_id = test_data['image_id']
            inputs, coor = load_split_image(image_id)
            predictions = yolo.predict(inputs)
            bs, num_box, c = predictions['boxes'].shape
            coor = np.expand_dims(coor, axis=1).repeat(num_box, axis=1)
            predictions['boxes'][:, :, :2] = predictions['boxes'][:, :, :2] + coor
            for key, value in predictions.items():
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
            predictions = yolo.predict(image)

        # we take the top_n boxes as the final result
        predictions['boxes'] = predictions['boxes'][:, :top_n, :]
        predictions['confidence'] = predictions['confidence'][:, :top_n]
        predictions['classes'] = predictions['classes'][:, :top_n]

        invalid_num = np.count_nonzero(predictions['classes'] == -1)
        valid_pre = top_n - invalid_num
        valid_boxes = predictions['boxes'].squeeze()[:valid_pre, :]

        crops = crop_images_from_boxes(image.numpy().squeeze(), valid_boxes, box_format=BBOX_FORMAT)
        crops = tf.image.convert_image_dtype(crops, tf.float32)  # [0, 1]

        t0 = time.perf_counter()
        ae_predictions = autoencoder.predict(crops)
        ae_predictions = autoencoder.get_rec_error(crops, ae_predictions)
        t1 = time.perf_counter()
        trans_predictions = transformer.predict(crops)
        t2 = time.perf_counter()
        resnet_predictions = resnet.predict(crops)
        t3 = time.perf_counter()

        ae_time.append(t1 - t0)
        transformer_time.append(t2 - t1)
        resnet_time.append(t3 - t2)

        all_ae_predictions.append(ae_predictions)
        all_transformer_predictions.append(trans_predictions)
        all_resnet_predictions.append(resnet_predictions)
        valid_nums.append(valid_pre)
        all_origin_predictions.append(predictions)

        used_time = time.perf_counter() - start_time
        time_all.append(used_time)

        if visualization is None:
            visualization \
                = visualize_detections(image, predictions, labels, yolo.bbox_format, yolo.class_mapping)
            writer.add_figure(f"val/visualization", visualization, 0)

    min_list = []
    max_list = []
    for pre in all_ae_predictions:
        min = pre.min()
        max = pre.max()
        min_list.append(min)
        max_list.append(max)
    min_ae_predictions = np.min(min_list)
    max_ae_predictions = np.max(max_list)
    thres_list_ae = np.linspace(min_ae_predictions, max_ae_predictions, n_threshold)

    min_list = []
    max_list = []
    for pre in all_resnet_predictions:
        min = pre.numpy().min()
        max = pre.numpy().max()
        min_list.append(min)
        max_list.append(max)
    min_resnet_predictions = np.min(min_list)
    max_resnet_predictions = np.max(max_list)
    thres_list_resnet = np.linspace(min_resnet_predictions, max_resnet_predictions, n_threshold)

    min_list = []
    max_list = []
    for pre in all_transformer_predictions:
        min = pre.numpy().min()
        max = pre.numpy().max()
        min_list.append(min)
        max_list.append(max)
    min_transformer_predictions = np.min(min_list)
    max_transformer_predictions = np.max(max_list)
    thres_list_trans = np.linspace(min_transformer_predictions, max_transformer_predictions, n_threshold)

    ae_map_result_list = []
    tran_map_result_list = []
    resnet_map_result_list = []
    ctfad_evaluate = keras_cv.metrics.BoxCOCOMetrics(bounding_box_format=BBOX_FORMAT, evaluate_freq=1)

    for j in tqdm(range(n_threshold)):
        thres_ae, thres_resnet, thres_transformer = thres_list_ae[j], thres_list_resnet[j], thres_list_trans[j]
        ae_evaluate = keras_cv.metrics.BoxCOCOMetrics(bounding_box_format=BBOX_FORMAT, evaluate_freq=1)
        transformer_evaluate = keras_cv.metrics.BoxCOCOMetrics(bounding_box_format=BBOX_FORMAT, evaluate_freq=1)
        resnet_evaluate = keras_cv.metrics.BoxCOCOMetrics(bounding_box_format=BBOX_FORMAT, evaluate_freq=1)

        for i in range(len(all_labels)):
            labels = all_labels[i]
            tran_predictions = all_transformer_predictions[i]
            res_predictions = all_resnet_predictions[i]
            ae_predictions = all_ae_predictions[i]
            ae_res = autoencoder.decode_result(ae_predictions, threshold=thres_ae)
            tran_res = transformer.decode_result(tran_predictions, threshold=thres_transformer)
            resnet_res = resnet.decode_result(res_predictions, threshold=thres_resnet)
            predictions = all_origin_predictions[i]
            valid_pre = valid_nums[i]

            ae_bbox = copy.deepcopy(predictions)
            ae_bbox['classes'][:, :valid_pre] = ae_res
            ae_bbox['confidence'][:, :valid_pre] = ae_predictions

            res_bbox = copy.deepcopy(predictions)
            res_bbox['classes'][:, :valid_pre] = resnet_res
            res_bbox['confidence'][:, :valid_pre] = np.ones_like(res_predictions.numpy().squeeze())

            tran_bbox = copy.deepcopy(predictions)
            tran_bbox['classes'][:, :valid_pre] = tran_res
            tran_bbox['confidence'][:, :valid_pre] = np.ones_like(res_predictions.numpy().squeeze())

            ae_evaluate.update_state(labels, ae_bbox)
            transformer_evaluate.update_state(labels, tran_bbox)
            resnet_evaluate.update_state(labels, res_bbox)

        ae_result = ae_evaluate.result()
        tran_result = transformer_evaluate.result()
        resnet_result = resnet_evaluate.result()

        ae_map_result = ae_result['MaP'].numpy()
        tran_map_result = tran_result['MaP'].numpy()
        resnet_map_result = resnet_result['MaP'].numpy()

        ae_map_result_list.append(ae_map_result)
        tran_map_result_list.append(tran_map_result)
        resnet_map_result_list.append(resnet_map_result)

        for key, value in ae_result.items():
            writer.add_scalar(f"val_ae/{key}", value.numpy(), j)

        for key, value in tran_result.items():
            writer.add_scalar(f"val_tran/{key}", value.numpy(), j)

        for key, value in resnet_result.items():
            writer.add_scalar(f"val_res/{key}", value.numpy(), j)

    # ======================= Evaluate ctfad ======================================
    for i in range(len(all_labels)):
        labels = all_labels[i]
        tran_predictions = all_transformer_predictions[i]
        res_predictions = all_resnet_predictions[i]
        ae_predictions = all_ae_predictions[i]

        ae_res = autoencoder.decode_result(ae_predictions, threshold=13835.884882387907)
        tran_res = transformer.decode_result(tran_predictions, threshold=0.7390993328556751)
        resnet_res = resnet.decode_result(res_predictions, threshold=0.2710454774939496)
        res_mean = np.mean((ae_res, tran_res, resnet_res), axis=0)
        final_res = (res_mean >= 0.66).astype(int)  # majority 2 / 3
        predictions = all_origin_predictions[i]
        valid_pre = valid_nums[i]
        predictions['classes'][:, :valid_pre] = final_res
        predictions['confidence'][:, :valid_pre] = np.ones_like(res_predictions.numpy().squeeze())
        ctfad_evaluate.update_state(labels, predictions)

    ctfad_result = ctfad_evaluate.result()

    for key, value in ctfad_result.items():
        writer.add_scalar(f"val_ctfad/{key}", value.numpy(), 0)

    best_perf_ae_id = np.argmax(ae_map_result_list)
    best_perf_tran_id = np.argmax(tran_map_result_list)
    best_perf_resnet_id = np.argmax(resnet_map_result_list)

    best_perf_ae_threshold = thres_list_ae[best_perf_ae_id]
    best_perf_tran_threshold = thres_list_trans[best_perf_tran_id]
    best_perf_resnet_threshold = thres_list_resnet[best_perf_resnet_id]

    logging.info(f"Threshold_ae: {best_perf_ae_threshold} \n"
                 f"Threshold_tran: {best_perf_tran_threshold} \n"
                 f"Threshold_resnet: {best_perf_resnet_threshold}")

    yolo_params = yolo.model.count_params()
    autoencoder_params = autoencoder.model.count_params()
    transformer_params = transformer.model.count_params()
    resnet_params = resnet.model.count_params()
    total_params = yolo_params + autoencoder_params + transformer_params + resnet_params

    writer.add_scalar(f"val/total_params", total_params, 0)

    time_mean = np.mean(time_all)
    time_std = np.std(time_all)

    writer.add_scalar(f"val/time_mean", time_mean, 0)
    writer.add_scalar(f"val/time_std", time_std, 0)

    writer.add_scalar(f"val_ae/time_mean", np.mean(ae_time), 0)
    writer.add_scalar(f"val_res/time_mean", np.mean(resnet_time), 0)
    writer.add_scalar(f"val_tran/time_mean", np.mean(transformer_time), 0)


if __name__ == '__main__':
    run()
