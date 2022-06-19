from typing import List, Text
from absl import logging

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow import keras
from tfx_bsl.public import tfxio
from tfx import v1 as tfx

LONG_TAIL_VARIABLES = ['tempo_desperd']
NUMERICAL_VARS = ['escalacoes', 'norm_escalacao', 'anos_como_pro']
CEIL_VARS = ['avg_3', 'avg_4']

LABEL_KEY = 'pro_target'

_FEATURE_KEYS = ['tempo_desperd', 'escalacoes',
                 'norm_escalacao', 'anos_como_pro', 'avg_3', 'avg_4']

_TRAIN_BATCH_SIZE = 20
_EVAL_BATCH_SIZE = 10


def _fill_in_missing(x):
    """Replace missing values in a SparseTensor.

    Fills in missing values of `x` with '' or 0, and converts to a dense tensor.

    Args:
      x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
        in the second dimension.

    Returns:
      A rank 1 tensor where missing values of `x` have been filled in.
    """
    if not isinstance(x, tf.sparse.SparseTensor):
        return x

    default_value = '' if x.dtype == tf.string else 0
    return tf.squeeze(
        tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
            default_value),
        axis=1)


def _get_log(value):
    apply_log = tf.reduce_any(
        tf.logical_or(tf.equal(tf.size(value), 0), tf.less(value, 0)))

    log_tensor = tf.where(apply_log, tf.constant(
        [float('nan')], dtype=tf.float32), tf.math.log(tf.cast(value, dtype=tf.float32)))

    return log_tensor


def _set_max_value(value):
    avg_tensor = tf.where(tf.greater(value, 1), tf.ones_like(value), value)

    return avg_tensor


def _transformed_name(key, transform_name):
    return key
    return transform_name + key


def _input_fn(file_pattern: List[Text],
              data_accessor: tfx.components.DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
    """Generates features and label for tuning/training.

    Args:
      file_pattern: List of paths or patterns of input tfrecord files.
      data_accessor: DataAccessor for converting input to RecordBatch.
      tf_transform_output: A TFTransformOutput.
      batch_size: representing the number of consecutive elements of returned
        dataset to combine in a single batch

    Returns:
      A dataset that contains (features, indices) tuple where features is a
        dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    dataset = data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(batch_size=batch_size),
        schema=tf_transform_output.raw_metadata.schema)

    transform_layer = tf_transform_output.transform_features_layer()

    def apply_transform(raw_features):
        return _apply_preprocessing(raw_features, transform_layer)

    return dataset.map(apply_transform).repeat()


def preprocessing_fn(inputs):
    outputs = {}

    for key in LONG_TAIL_VARIABLES:
        outputs[key] = tft.scale_to_z_score(_get_log(
            _fill_in_missing(inputs[key])))

    for key in CEIL_VARS:
        outputs[key] = tft.scale_to_z_score(_set_max_value(
            _fill_in_missing(inputs[key])))

    for key in NUMERICAL_VARS:
        outputs[key] = tft.scale_to_z_score(
            _fill_in_missing(inputs[key]))

    outputs[LABEL_KEY] = _fill_in_missing(inputs[LABEL_KEY])

    return outputs


def _apply_preprocessing(raw_features, tft_layer):
    transformed_features = tft_layer(raw_features)
    if LABEL_KEY in raw_features:
        transformed_label = transformed_features.pop(LABEL_KEY)
        return transformed_features, transformed_label

    return transformed_features, None


def _get_serve_tf_examples_fn(model, tf_transform_output):
    # adding transformations in model layer
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()

        # Only add the necessary features to the model
        required_feature_spec = {
            k: v for k, v in feature_spec.items() if k in _FEATURE_KEYS
        }
        parsed_features = tf.io.parse_example(serialized_tf_examples,
                                              required_feature_spec)

        transformed_features, _ = _apply_preprocessing(parsed_features,
                                                       model.tft_layer)
        return model(transformed_features)

    return serve_tf_examples_fn


def _build_keras_model() -> tf.keras.Model:
    inputs = [keras.layers.Input(shape=(1,), name=feat)
              for feat in _FEATURE_KEYS]
    input_layer = keras.layers.concatenate(inputs)
    outputs = keras.layers.Dense(1, activation='sigmoid')(input_layer)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-2),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy()])

    model.summary(print_fn=logging.info)
    return model


# TFX Trainer will call this function.
def run_fn(fn_args: tfx.components.FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size=_TRAIN_BATCH_SIZE)
    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size=_EVAL_BATCH_SIZE)

    model = _build_keras_model()
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps)

    signatures = {
        'serving_default': _get_serve_tf_examples_fn(model, tf_transform_output),
    }

    model.save(fn_args.serving_model_dir,
               save_format='tf', signatures=signatures)
