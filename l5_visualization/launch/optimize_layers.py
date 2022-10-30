import sys

sys.path.append("/workspaces/lectures")

import hydra
from pathlib import Path
import hydra
from tensorflow import keras
from l5_visualization.optimizators import ActivationOptimizer
import cv2
import numpy as np
import tensorflow as tf


@hydra.main(config_path="../configs/hydra", config_name=Path(__file__).stem + "_config")
def run(cfg):
    model = keras.models.load_model(cfg.model_path)
    # define optimizator

    out_dir = Path("layers_vis")
    print(f"saving visualization to {str(out_dir.absolute())}")
    for layer_name in cfg.layers_to_optimize:
        layer = model.get_layer(layer_name)

        if "conv2d" not in layer.name:
            continue

        for filter_index in range(layer.filters):
            ao = ActivationOptimizer(
                model=model,
                layer_name=layer.name,
                activation_index=(slice(None), slice(None), filter_index),
                steps=cfg.steps,
                step_size=cfg.step_size,
                reg_coef=cfg.reg_coef,
            )
            for i in range(cfg.n_images_per_filter):
                random_image = tf.random.uniform(
                    [32, 32, 3],
                    minval=0,
                    maxval=None,
                    dtype=tf.dtypes.float32,
                    seed=None,
                    name=None,
                )
                loss, image_raw = ao(random_image)

                image = 255 * (image_raw)
                image = tf.cast(image, tf.uint8)
                image = np.array(image)
                # resize
                image = cv2.resize(image, cfg.resize_image_to)

                image_out_dir = out_dir / layer.name / f"filter_{filter_index}"
                image_out_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    str(image_out_dir / f"sample_{i}") + ".png",
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                )


if __name__ == "__main__":
    # load model

    run()
