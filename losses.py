import tensorflow as tf


class CustomCategoricalCrossEntropy(tf.keras.losses.Loss):
    def __init__(
        self,
        from_logits=False,
        sparse=False,
        label_smoothing=0.0,
        axis=-1,
        epsilon=1e-10,
        name="eps_categorical_crossentropy",
        reduction="sum_over_batch_size",
    ):
        super().__init__(reduction=reduction, name=name)
        self.label_smoothing = label_smoothing
        self.from_logits = from_logits
        self.sparse = sparse
        self.axis = axis
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        if self.from_logits:  # Apply softmax if utilities are given
            y_pred = tf.nn.softmax(y_pred, axis=self.axis)
        else:
            y_pred = tf.convert_to_tensor(y_pred)
        if self.sparse:  # Create OneHot labels if sparse labels are given
            y_true = tf.one_hot(y_true, depth=tf.shape(y_pred)[self.axis])
        else:
            y_true = tf.cast(y_true, y_pred.dtype)

        # Smooth labels
        if self.label_smoothing > 0:
            label_smoothing = tf.convert_to_tensor(
                self.label_smoothing, dtype=y_pred.dtype
            )
            num_classes = tf.cast(tf.shape(y_true)[self.axis], y_pred.dtype)
            y_true = y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)

        # Apply label clipping to avoid log(0) and such issues
        y_pred = tf.clip_by_value(y_pred, self.epsilon, 1.0 - self.epsilon)
        return -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=self.axis)
