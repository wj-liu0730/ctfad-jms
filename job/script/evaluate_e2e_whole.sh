python job/evaluate_e2e.py \
DataParams.dataset_name=dataset_whole \
DataParams.train_batch_size=1 \
DataParams.test_batch_size=1 \
DataParams.val_batch_size=1 \
JobParams.job_name=eval_e2e_whole \
AlgoParams.pretrained_model_path=outputs/yolo_e2e_whole/tf_model/best_model/yolo_v8.keras
