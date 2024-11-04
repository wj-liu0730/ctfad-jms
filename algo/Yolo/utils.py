import numpy as np
import logging
from tqdm import tqdm



def train_loop(algo, tf_training_dataset, tf_val_dataset, writer, cfg):
    logging.info("Start Training Process for YOLO")
    progress = tqdm(total=cfg.AlgoParams.training_epochs, desc="Training Process")
    best_performance = -100
    for ep in range(cfg.AlgoParams.training_epochs):
        if ep % cfg.AlgoParams.evaluation_period == 0:
            performance = val_loop(algo, tf_val_dataset, writer, ep)
            if performance > best_performance:
                best_performance = performance
                algo.save_model(cfg.JobParams.output_path + '/tf_model/best_model/')

        loss_list = []
        for iter, batch in enumerate(tf_training_dataset):
            training_loss = algo.optimize(batch)
            loss_list.append(training_loss)
            progress.set_description(f"Status: Epochs {ep}--Iters {iter}, loss={training_loss:.4f}")

        writer.add_scalar("train/loss", np.mean(loss_list), ep)
        progress.update(1)
    algo.save_model(cfg.JobParams.output_path + '/tf_model/final_model/')



def val_loop(algo, tf_eval_dataset, writer, ep):
    # the val_loop should be implemented differently for different algorithms and cases
    validation_loss = []
    algo.evaluattion_metrics.reset_state()
    visualization = None
    for batch in tf_eval_dataset:
        loss = algo.validate(batch)
        algo.performance_eval(batch)
        validation_loss.append(loss)
        if visualization is None:
            visualization = algo.get_visualization_result(batch)  # Visualization for the first batch
            writer.add_figure(f"val/visualization", visualization, ep)
    performance = algo.get_performance_result()
    map_result = performance['MaP'].numpy()
    for key, value in performance.items():
        writer.add_scalar(f"val/{key}", value.numpy(), ep)
    writer.add_scalar("val/loss", np.mean(validation_loss), ep)
    return map_result


def test_loop(algo, test_dataset, writer, test_batch=2):
    for i, batch in enumerate(test_dataset):
        fig = algo.get_visualization_result(batch)
        writer.add_figure(f"test/{i}th_batch", fig)
        if (i + 1) >= test_batch:
            break
    logging.info("Finish Testing Process")
