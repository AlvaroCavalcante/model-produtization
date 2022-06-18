from typing import List, Text
from absl import logging

import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_model_analysis as tfma
from tensorflow_transform.tf_metadata import schema_utils
from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx_bsl.tfxio import dataset_options
from tensorflow import estimator as tf_estimator
from tensorflow import keras
from tfx import v1 as tfx
from tfx_bsl.public import tfxio
from tensorflow_metadata.proto.v0 import schema_pb2

LONG_TAIL_VARIABLES = ['tempo_desperd']
NUMERICAL_VARS = ['escalacoes', 'norm_escalacao', 'anos_como_pro']
CEIL_VARS = ['avg_3', 'avg_4']

LABEL_KEY = 'pro_target'

_FEATURE_KEYS = ['log_tempo_desperd', 'zscore_escalacoes',
                 'zscore_norm_escalacao', 'zscore_anos_como_pro', 'ceil_avg_3', 'ceil_avg_4']

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
    return transform_name + key


def _transformed_names(keys, transform_name):
    return [_transformed_name(key, transform_name) for key in keys]


def _input_fn(file_pattern: List[str],
              data_accessor: tfx.components.DataAccessor,
              schema: schema_pb2.Schema,
              batch_size: int = 200) -> tf.data.Dataset:
    """Generates features and label for training.

    Args:
      file_pattern: List of paths or patterns of input tfrecord files.
      data_accessor: DataAccessor for converting input to RecordBatch.
      schema: schema of the input data.
      batch_size: representing the number of consecutive elements of returned
        dataset to combine in a single batch

    Returns:
      A dataset that contains (features, indices) tuple where features is a
        dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(
            batch_size=batch_size, label_key=LABEL_KEY),
        schema=schema).repeat()


def preprocessing_fn(inputs):
    outputs = {}
    print(f'Showing inputs {inputs}')

    for key in LONG_TAIL_VARIABLES:
        outputs[_transformed_name(key, 'log_')] = tft.scale_to_z_score(_get_log(
            _fill_in_missing(inputs[key])))

    for key in CEIL_VARS:
        outputs[_transformed_name(key, 'ceil_')] = tft.scale_to_z_score(_set_max_value(
            _fill_in_missing(inputs[key])))

    for key in NUMERICAL_VARS:
        outputs[_transformed_name(key, 'zscore_')] = tft.scale_to_z_score(
            _fill_in_missing(inputs[key]))

    outputs[LABEL_KEY] = _fill_in_missing(inputs[LABEL_KEY])

    print(f'Showing outputs: {outputs}')
    return outputs


def _apply_preprocessing(raw_features, tft_layer):
    transformed_features = tft_layer(raw_features)
    if LABEL_KEY in raw_features:
        transformed_label = transformed_features.pop(LABEL_KEY)
        return transformed_features, transformed_label
    else:
        return transformed_features, None


def _get_serve_tf_examples_fn(model, tf_transform_output):
    # We must save the tft_layer to the model to ensure its assets are kept and
    # tracked.
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def serve_tf_examples_fn(serialized_tf_examples):
        # Expected input is a string which is serialized tf.Example format.
        feature_spec = tf_transform_output.raw_feature_spec()
        # Only add the necessary features to the model
        required_feature_spec = {
            k: v for k, v in feature_spec.items() if k in _FEATURE_KEYS
        }
        parsed_features = tf.io.parse_example(serialized_tf_examples,
                                              required_feature_spec)

        # Preprocess parsed input with transform operation defined in
        # preprocessing_fn().
        transformed_features, _ = _apply_preprocessing(parsed_features,
                                                       model.tft_layer)
        # Run inference with ML model.
        return model(transformed_features)

    return serve_tf_examples_fn


def _build_keras_model() -> tf.keras.Model:
    """Creates a DNN Keras model for classifying penguin data.

    Returns:
      A Keras Model.
    """
    # The model below is built with Functional API, please refer to
    # https://www.tensorflow.org/guide/keras/overview for all API options.
    inputs = [keras.layers.Input(shape=(1,), name=f) for f in _FEATURE_KEYS]
    d = keras.layers.concatenate(inputs)
    for _ in range(2):
        d = keras.layers.Dense(8, activation='relu')(d)
    outputs = keras.layers.Dense(3)(d)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-2),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()])

    model.summary(print_fn=logging.info)
    return model


# TFX Trainer will call this function.
def run_fn(fn_args: tfx.components.FnArgs):
    """Train the model based on given args.

    Args:
      fn_args: Holds args used to train the model as name/value pairs.
    """

    # This schema is usually either an output of SchemaGen or a manually-curated
    # version provided by pipeline author. A schema can also derived from TFT
    # graph if a Transform component is used. In the case when either is missing,
    # `schema_from_feature_spec` could be used to generate schema from very simple
    # feature_spec, but the schema returned would be very primitive.

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

    # The result of the training should be saved in `fn_args.serving_model_dir`
    # directory.
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)


# # Tf.Transform considers these features as "raw"
# def _get_raw_feature_spec(schema):
#     return schema_utils.schema_as_feature_spec(schema).feature_spec


# def _build_estimator(config, hidden_units=None, warm_start_from=None):
#     """Build an estimator for predicting the tipping behavior of taxi riders.

#     Args:
#       config: tf.estimator.RunConfig defining the runtime environment for the
#         estimator (including model_dir).
#       hidden_units: [int], the layer sizes of the DNN (input layer first)
#       warm_start_from: Optional directory to warm start from.

#     Returns:
#       A dict of the following:
#         - estimator: The estimator that will be used for training and eval.
#         - train_spec: Spec for training.
#         - eval_spec: Spec for eval.
#         - eval_input_receiver_fn: Input function for eval.
#     """
#     print('Building estimator')

#     real_valued_columns = [
#         tf.feature_column.numeric_column(key, shape=())
#         for key in _transformed_names(LONG_TAIL_VARIABLES, 'log_')
#     ]

#     real_valued_columns += [
#         tf.feature_column.numeric_column(key, shape=())
#         for key in _transformed_names(NUMERICAL_VARS, 'zscore_')
#     ]

#     real_valued_columns += [
#         tf.feature_column.numeric_column(key, shape=())
#         for key in _transformed_names(CEIL_VARS, 'ceil_')
#     ]

#     print(f'Real vars {real_valued_columns}')

#     return tf_estimator.DNNLinearCombinedClassifier(
#         config=config,
#         dnn_feature_columns=real_valued_columns,
#         dnn_hidden_units=hidden_units or [100, 70, 50, 25],
#         warm_start_from=warm_start_from)


# def _input_fn(file_pattern: List[str],
#               data_accessor: DataAccessor,
#               tf_transform_output: tft.TFTransformOutput,
#               batch_size: int = 200) -> tf.data.Dataset:
#     """Generates features and label for tuning/training.

#     Args:
#       file_pattern: List of paths or patterns of input tfrecord files.
#       data_accessor: DataAccessor for converting input to RecordBatch.
#       tf_transform_output: A TFTransformOutput.
#       batch_size: representing the number of consecutive elements of returned
#         dataset to combine in a single batch

#     Returns:
#       A dataset that contains (features, indices) tuple where features is a
#         dictionary of Tensors, and indices is a single Tensor of label indices.
#     """
#     return data_accessor.tf_dataset_factory(
#         file_pattern,
#         dataset_options.TensorFlowDatasetOptions(
#             batch_size=batch_size, label_key=LABEL_KEY),
#         tf_transform_output.transformed_metadata.schema)


# def _example_serving_receiver_fn(tf_transform_output, schema):
#     """Build the serving in inputs.

#     Args:
#       tf_transform_output: A TFTransformOutput.
#       schema: the schema of the input data.

#     Returns:
#       Tensorflow graph which parses examples, applying tf-transform to them.
#     """
#     raw_feature_spec = _get_raw_feature_spec(schema)
#     raw_feature_spec.pop(LABEL_KEY)

#     raw_input_fn = tf_estimator.export.build_parsing_serving_input_receiver_fn(
#         raw_feature_spec, default_batch_size=None)
#     serving_input_receiver = raw_input_fn()

#     transformed_features = tf_transform_output.transform_raw_features(
#         serving_input_receiver.features)

#     return tf_estimator.export.ServingInputReceiver(
#         transformed_features, serving_input_receiver.receiver_tensors)


# def _eval_input_receiver_fn(tf_transform_output, schema):
#     """Build everything needed for the tf-model-analysis to run the model.

#     Args:
#       tf_transform_output: A TFTransformOutput.
#       schema: the schema of the input data.

#     Returns:
#       EvalInputReceiver function, which contains:
#         - Tensorflow graph which parses raw untransformed features, applies the
#           tf-transform preprocessing operators.
#         - Set of raw, untransformed features.
#         - Label against which predictions will be compared.
#     """
#     # Notice that the inputs are raw features, not transformed features here.
#     raw_feature_spec = _get_raw_feature_spec(schema)

#     serialized_tf_example = tf.compat.v1.placeholder(
#         dtype=tf.string, shape=[None], name='input_example_tensor')

#     # Add a parse_example operator to the tensorflow graph, which will parse
#     # raw, untransformed, tf examples.
#     features = tf.io.parse_example(
#         serialized=serialized_tf_example, features=raw_feature_spec)

#     # Now that we have our raw examples, process them through the tf-transform
#     # function computed during the preprocessing step.
#     transformed_features = tf_transform_output.transform_raw_features(
#         features)

#     # The key name MUST be 'examples'.
#     receiver_tensors = {'examples': serialized_tf_example}

#     # NOTE: Model is driven by transformed features (since training works on the
#     # materialized output of TFT, but slicing will happen on raw features.
#     features.update(transformed_features)

#     return tfma.export.EvalInputReceiver(
#         features=features,
#         receiver_tensors=receiver_tensors,
#         labels=transformed_features[LABEL_KEY])


# # TFX will call this function
# def trainer_fn(trainer_fn_args, schema):
#     """Build the estimator using the high level API.

#     Args:
#       trainer_fn_args: Holds args used to train the model as name/value pairs.
#       schema: Holds the schema of the training examples.

#     Returns:
#       A dict of the following:
#         - estimator: The estimator that will be used for training and eval.
#         - train_spec: Spec for training.
#         - eval_spec: Spec for eval.
#         - eval_input_receiver_fn: Input function for eval.
#     """
#     # Number of nodes in the first layer of the DNN
#     first_dnn_layer_size = 100
#     num_dnn_layers = 4
#     dnn_decay_factor = 0.7

#     train_batch_size = 40
#     eval_batch_size = 40

#     print("Starting training...")

#     tf_transform_output = tft.TFTransformOutput(
#         trainer_fn_args.transform_output)

#     print(tf_transform_output)

#     def train_input_fn(): return _input_fn(  # pylint: disable=g-long-lambda
#         trainer_fn_args.train_files,
#         trainer_fn_args.data_accessor,
#         tf_transform_output,
#         batch_size=train_batch_size)

#     def eval_input_fn(): return _input_fn(  # pylint: disable=g-long-lambda
#         trainer_fn_args.eval_files,
#         trainer_fn_args.data_accessor,
#         tf_transform_output,
#         batch_size=eval_batch_size)

#     train_spec = tf_estimator.TrainSpec(  # pylint: disable=g-long-lambda
#         train_input_fn,
#         max_steps=trainer_fn_args.train_steps)

#     print('Creating receiver')

#     def serving_receiver_fn(): return _example_serving_receiver_fn(  # pylint: disable=g-long-lambda
#         tf_transform_output, schema)

#     exporter = tf_estimator.FinalExporter('cartola-pro', serving_receiver_fn)
#     eval_spec = tf_estimator.EvalSpec(
#         eval_input_fn,
#         steps=trainer_fn_args.eval_steps,
#         exporters=[exporter],
#         name='cartola-pro-eval')

#     run_config = tf_estimator.RunConfig(
#         save_checkpoints_steps=999, keep_checkpoint_max=5)

#     run_config = run_config.replace(
#         model_dir=trainer_fn_args.serving_model_dir)
#     warm_start_from = trainer_fn_args.base_model

#     estimator = _build_estimator(
#         # Construct layers sizes with exponetial decay
#         hidden_units=[
#             max(2, int(first_dnn_layer_size * dnn_decay_factor**i))
#             for i in range(num_dnn_layers)
#         ],
#         config=run_config,
#         warm_start_from=warm_start_from)

#     # Create an input receiver for TFMA processing
#     def receiver_fn(): return _eval_input_receiver_fn(  # pylint: disable=g-long-lambda
#         tf_transform_output, schema)

#     return {
#         'estimator': estimator,
#         'train_spec': train_spec,
#         'eval_spec': eval_spec,
#         'eval_input_receiver_fn': receiver_fn
#     }
