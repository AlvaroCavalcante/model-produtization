import tensorflow as tf
import tensorflow_transform as tft

LONG_TAIL_VARIABLES = ['tempo_desperd']
NUMERICAL_VARS = ['escalacoes', 'norm_escalacao', 'anos_como_pro']
CEIL_VARS = ['avg_3', 'avg_4']

LABEL_KEY = 'pro_target'


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
