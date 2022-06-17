import tensorflow as tf

LONG_TAIL_VARIABLES = ['anos_desde_criacao', 'instagram_num', 'facebook_num',
                       'min_camp', 'interacoes_g1', 'tempo_desperd',
                       'iteracao_volei', 'iteracao_atletismo']

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
    def log_f(): return tf.math.log(tf.cast(value, dtype=tf.float32))
    def nan_f(): return tf.constant([float('nan')], dtype=tf.float32)

    apply_log = tf.reduce_any(
        tf.logical_or(tf.equal(tf.size(value), 0), tf.less(value, 0)))

    print(apply_log)

    log_tensor = tf.where(apply_log, tf.constant(
        [float('nan')], dtype=tf.float32), tf.math.log(tf.cast(value, dtype=tf.float32)))

    # tf.cond(apply_log, nan_f, log_f)

    print(log_tensor)
    return log_tensor


def _transformed_name(key, transform_name):
    return transform_name + key


def preprocessing_fn(inputs):
    outputs = {}
    print(f'Showing inputs {inputs}')

    for key in LONG_TAIL_VARIABLES:
        print(f'Key: {key}')
        print(f'New inputs: {inputs[key]}')

        outputs[_transformed_name(key, 'log_')] = _get_log(
            _fill_in_missing(inputs[key]))

    return outputs