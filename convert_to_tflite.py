import itertools
import os

import click

import tensorflow as tf
from dataset.generators import get_dataset_mobilenet
from models.mobilenet_model import get_mobilenet_model

alpha = 1.0
rows = 224


def load_from_checkpoint(checkpoint_dir, checkpoint_path=None):
    model = get_mobilenet_model(alpha, rows)
    ckpt = tf.train.Checkpoint(net=model)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=1)

    if checkpoint_path:
        ckpt.restore(checkpoint_path)
    else:
        ckpt.restore(manager.latest_checkpoint)

    return model


def load_from_weights(path):
    model = get_mobilenet_model(alpha, rows)
    model.load_weights(path)

    return model


def representative_dataset_gen(annot_path_val, img_dir_val, batch_size):
    ds, ds_size = get_dataset_mobilenet(annot_path_val, img_dir_val, batch_size)

    def dataset_gen(num_calibration_steps=100):
        for inputs, _ in itertools.islice(ds, 0, num_calibration_steps):
            yield inputs

    return dataset_gen


def export_to_tflite(
    saved_model_dir, output_path, annot_path_val, img_dir_val, batch_size=10
):
    converter = tf.lite.TFLiteConverter.from_saved_model(
        saved_model_dir=saved_model_dir
    )

    converter.representative_dataset = representative_dataset_gen(
        annot_path_val, img_dir_val, batch_size=10
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    tflite_model = converter.convert()

    open(output_path, "wb").write(tflite_model)


@click.command()
@click.option("-p", "--prefix", type=str, default="1")
@click.option("-s", "--saved-model-dir", type=click.Path(exists=True))
@click.option("-t", "--tflite-output-path", type=click.Path(exists=False))
@click.option("-c", "--checkpoint", type=click.Path(exists=True))
@click.option("-a", "--annot-path-val", type=click.Path(exists=True))
@click.option("-i", "--img-dir-val", type=click.Path(exists=True))
def main(
    prefix, saved_model_dir, tflite_output_path, checkpoint, annot_path_val, img_dir_val
):
    # save model as saved_model
    model = load_from_checkpoint(checkpoint)
    # model = load_from_weights('./weights.best.mobilenet.h5')

    tf.saved_model.save(model, saved_model_dir)

    # export model to tflite
    os.makedirs(os.path.dirname(tflite_output_path))
    export_to_tflite(saved_model_dir, tflite_output_path)


if __name__ == "__main__":
    main()
