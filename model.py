import tqdm
import time
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from abc import abstractmethod


from losses import CustomCategoricalCrossEntropy


def softmax_with_availabilities(
    items_logit_by_choice,
    available_items_by_choice,
    axis=-1,
    normalize_exit=False,
    eps=1e-5,
):
    # Substract max utility to avoid overflow
    normalizer = tf.reduce_max(items_logit_by_choice, axis=axis, keepdims=True)
    numerator = tf.exp(items_logit_by_choice - normalizer)

    # Set unavailable products utility to 0
    numerator = tf.multiply(numerator, available_items_by_choice)
    # Sum of total available utilities
    denominator = tf.reduce_sum(numerator, axis=axis, keepdims=True)

    # Add 1 to the denominator to take into account the exit choice
    if normalize_exit:
        denominator += tf.exp(-normalizer)
    # Avoid division by 0 when only unavailable items have highest utilities
    elif eps:
        denominator += eps

    # Compute softmax
    return numerator / denominator


class Model:
    def __init__(
        self,
        label_smoothing=0.0,
        add_exit_choice=False,
        optimizer="lbfgs",
        lbfgs_tolerance=1e-8,
        lbfgs_parallel_iterations=4,
        callbacks=None,
        lr=0.001,
        epochs=1000,
        batch_size=32,
        regularization=None,
        regularization_strength=0.0,
    ):

        self.add_exit_choice = add_exit_choice
        self.label_smoothing = label_smoothing
        self.stop_training = False

        self.loss = CustomCategoricalCrossEntropy(
            from_logits=False, label_smoothing=self.label_smoothing
        )
        self.exact_nll = CustomCategoricalCrossEntropy(
            from_logits=False,
            label_smoothing=0.0,
            sparse=False,
            axis=-1,
            epsilon=1e-35,
            name="exact_categorical_crossentropy",
            reduction="sum_over_batch_size",
        )
        self.callbacks = tf.keras.callbacks.CallbackList(
            callbacks, add_history=True, model=None
        )
        self.callbacks.set_model(self)

        self.optimizer_name = optimizer
        if optimizer.lower() == "adam":
            self.optimizer = tf.keras.optimizers.Adam(lr)
        elif optimizer.lower() == "sgd":
            self.optimizer = tf.keras.optimizers.SGD(lr)
        elif optimizer.lower() == "adamax":
            self.optimizer = tf.keras.optimizers.Adamax(lr)
        elif optimizer.lower() == "lbfgs" or optimizer.lower() == "l-bfgs":
            print("Using L-BFGS optimizer, setting up .fit() function")
            self.optimizer = "lbfgs"
            self.fit = self._fit_with_lbfgs
        else:
            print(f"Optimizer {optimizer} not implemented, switching for default Adam")
            self.optimizer = tf.keras.optimizers.Adam(lr)

        self.epochs = epochs
        self.batch_size = batch_size
        self.lbfgs_tolerance = lbfgs_tolerance
        self.lbfgs_parallel_iterations = lbfgs_parallel_iterations

        if regularization is not None:
            if np.sum(regularization_strength) <= 0:
                raise ValueError(
                    "Regularization Strength must be positive if it's set."
                )
            if regularization.lower() == "l1":
                self.regularizer = tf.keras.regularizers.L1(l1=regularization_strength)
            elif regularization.lower() == "l2":
                self.regularizer = tf.keras.regularizers.L2(l2=regularization_strength)
            elif regularization.lower() == "l1l2":
                if isinstance(regularization_strength, (list, tuple)):
                    self.regularizer = tf.keras.regularizers.L1L2(
                        l1=regularization_strength[0], l2=regularization_strength[1]
                    )
                else:
                    self.regularizer = tf.keras.regularizers.L1L2(
                        l1=regularization_strength, l2=regularization_strength
                    )
            else:
                raise ValueError(
                    "Regularization type not recognized, choose among l1, l2 and l1l2."
                )
            self.regularization = regularization
            self.regularization_strength = regularization_strength
        else:
            self.regularization_strength = 0.0
            self.regularization = None

    @abstractmethod
    def compute_batch_utility(
        self,
        shared_features_by_choice,
        items_features_by_choice,
        available_items_by_choice,
        choices,
    ):
        pass

    @tf.function
    def train_step(
        self,
        shared_features_by_choice,
        items_features_by_choice,
        available_items_by_choice,
        choices,
        sample_weight=None,
    ):
        with tf.GradientTape() as tape:
            utilities = self.compute_batch_utility(
                shared_features_by_choice=shared_features_by_choice,
                items_features_by_choice=items_features_by_choice,
                available_items_by_choice=available_items_by_choice,
                choices=choices,
            )

            probabilities = softmax_with_availabilities(
                items_logit_by_choice=utilities,
                available_items_by_choice=available_items_by_choice,
                normalize_exit=self.add_exit_choice,
                axis=-1,
            )
            # Negative Log-Likelihood
            neg_loglikelihood = self.loss(
                y_pred=probabilities,
                y_true=tf.one_hot(choices, depth=probabilities.shape[1]),
                sample_weight=sample_weight,
            )
            if self.regularization is not None:
                regularization = tf.reduce_sum(
                    [self.regularizer(w) for w in self.trainable_weights]
                )
                neg_loglikelihood += regularization

        grads = tape.gradient(neg_loglikelihood, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return neg_loglikelihood

    def fit(
        self,
        choice_dataset,
        sample_weight=None,
        val_dataset=None,
        verbose=0,
    ):
        if hasattr(self, "instantiated"):
            if not self.instantiated:
                raise ValueError(
                    "Model not instantiated. Please call .instantiate() first."
                )
        epochs = self.epochs
        batch_size = self.batch_size

        losses_history = {"train_loss": []}
        if verbose >= 0 and verbose < 2:
            t_range = tqdm.trange(epochs, position=0)
        else:
            t_range = range(epochs)

        self.callbacks.on_train_begin()
        # Iterate of epochs
        for epoch_nb in t_range:
            if verbose >= 2:
                print(f"Start Epoch {epoch_nb}")
            self.callbacks.on_epoch_begin(epoch_nb)
            t_start = time.time()
            train_logs = {"train_loss": []}
            val_logs = {"val_loss": []}
            epoch_losses = []

            if sample_weight is not None:
                if verbose > 0:
                    inner_range = tqdm.tqdm(
                        choice_dataset.iter_batch(
                            shuffle=True,
                            sample_weight=sample_weight,
                            batch_size=batch_size,
                        ),
                        total=int(len(choice_dataset) / np.max([1, batch_size])),
                        position=1,
                        leave=False,
                    )
                else:
                    inner_range = choice_dataset.iter_batch(
                        shuffle=True, sample_weight=sample_weight, batch_size=batch_size
                    )

                for batch_nb, (
                    (
                        shared_features_batch,
                        items_features_batch,
                        available_items_batch,
                        choices_batch,
                    ),
                    weight_batch,
                ) in enumerate(inner_range):
                    self.callbacks.on_train_batch_begin(batch_nb)

                    neg_loglikelihood = self.train_step(
                        shared_features_batch,
                        items_features_batch,
                        available_items_batch,
                        choices_batch,
                        sample_weight=weight_batch,
                    )

                    train_logs["train_loss"].append(neg_loglikelihood)
                    temps_logs = {k: tf.reduce_mean(v) for k, v in train_logs.items()}
                    self.callbacks.on_train_batch_end(batch_nb, logs=temps_logs)

                    # Optimization Steps
                    epoch_losses.append(neg_loglikelihood)

                    if verbose > 0:
                        inner_range.set_description(
                            f"Epoch Negative-LogLikeliHood: {np.mean(epoch_losses):.4f}"
                        )

            # In this case we do not need to batch the sample_weights
            else:
                if verbose > 0:
                    inner_range = tqdm.tqdm(
                        choice_dataset.iter_batch(shuffle=True, batch_size=batch_size),
                        total=int(len(choice_dataset) / np.max([batch_size, 1])),
                        position=1,
                        leave=False,
                    )
                else:
                    inner_range = choice_dataset.iter_batch(
                        shuffle=True, batch_size=batch_size
                    )
                for batch_nb, (
                    shared_features_batch,
                    items_features_batch,
                    available_items_batch,
                    choices_batch,
                ) in enumerate(inner_range):
                    self.callbacks.on_train_batch_begin(batch_nb)
                    neg_loglikelihood = self.train_step(
                        shared_features_batch,
                        items_features_batch,
                        available_items_batch,
                        choices_batch,
                    )
                    train_logs["train_loss"].append(neg_loglikelihood)
                    temps_logs = {k: tf.reduce_mean(v) for k, v in train_logs.items()}
                    self.callbacks.on_train_batch_end(batch_nb, logs=temps_logs)

                    # Optimization Steps
                    epoch_losses.append(neg_loglikelihood)

                    if verbose > 0:
                        inner_range.set_description(
                            f"Epoch Negative-LogLikeliHood: {np.mean(epoch_losses):.4f}"
                        )

            # Take into account the fact that the last batch may have a
            # different length for the computation of the epoch loss.
            if batch_size != -1:
                last_batch_size = available_items_batch.shape[0]
                coefficients = tf.concat(
                    [tf.ones(len(epoch_losses) - 1) * batch_size, [last_batch_size]],
                    axis=0,
                )
                epoch_losses = tf.multiply(epoch_losses, coefficients)
                epoch_loss = tf.reduce_sum(epoch_losses) / len(choice_dataset)
            else:
                epoch_loss = tf.reduce_mean(epoch_losses)
            losses_history["train_loss"].append(epoch_loss)
            print_loss = losses_history["train_loss"][-1].numpy()
            desc = f"Epoch {epoch_nb} Train Loss {print_loss:.4f}"
            if verbose > 1:
                print(
                    f"Loop {epoch_nb} Time:",
                    f"{time.time() - t_start:.4f}",
                    f"Loss: {print_loss:.4f}",
                )

            # Test on val_dataset if provided
            if val_dataset is not None:
                test_losses = []
                for batch_nb, (
                    shared_features_batch,
                    items_features_batch,
                    available_items_batch,
                    choices_batch,
                ) in enumerate(
                    val_dataset.iter_batch(shuffle=False, batch_size=batch_size)
                ):
                    self.callbacks.on_batch_begin(batch_nb)
                    self.callbacks.on_test_batch_begin(batch_nb)
                    test_losses.append(
                        self.batch_predict(
                            shared_features_batch,
                            items_features_batch,
                            available_items_batch,
                            choices_batch,
                        )[0]["optimized_loss"]
                    )
                    val_logs["val_loss"].append(test_losses[-1])
                    temps_logs = {k: tf.reduce_mean(v) for k, v in val_logs.items()}
                    self.callbacks.on_test_batch_end(batch_nb, logs=temps_logs)

                test_loss = tf.reduce_mean(test_losses)
                if verbose > 1:
                    print("Test Negative-LogLikelihood:", test_loss.numpy())
                    desc += f", Test Loss {np.round(test_loss.numpy(), 4)}"
                losses_history["test_loss"] = losses_history.get("test_loss", []) + [
                    test_loss.numpy()
                ]
                train_logs = {**train_logs, **val_logs}

            temps_logs = {k: tf.reduce_mean(v) for k, v in train_logs.items()}
            self.callbacks.on_epoch_end(epoch_nb, logs=temps_logs)
            if self.stop_training:
                print("Early Stopping taking effect")
                break
            t_range.set_description(desc)
            t_range.refresh()

        temps_logs = {k: tf.reduce_mean(v) for k, v in train_logs.items()}
        self.callbacks.on_train_end(logs=temps_logs)
        return losses_history

    @tf.function()
    def batch_predict(
        self,
        shared_features_by_choice,
        items_features_by_choice,
        available_items_by_choice,
        choices,
        sample_weight=None,
    ):
        # Compute utilities from features
        utilities = self.compute_batch_utility(
            shared_features_by_choice,
            items_features_by_choice,
            available_items_by_choice,
            choices,
        )
        # Compute probabilities from utilities & availabilties
        probabilities = softmax_with_availabilities(
            items_logit_by_choice=utilities,
            available_items_by_choice=available_items_by_choice,
            normalize_exit=self.add_exit_choice,
            axis=-1,
        )

        # Compute loss from probabilities & actual choices
        # batch_loss = self.loss(probabilities, c_batch, sample_weight=sample_weight)
        batch_loss = {
            "optimized_loss": self.loss(
                y_pred=probabilities,
                y_true=tf.one_hot(choices, depth=probabilities.shape[1]),
                sample_weight=sample_weight,
            ),
            "Exact-NegativeLogLikelihood": self.exact_nll(
                y_pred=probabilities,
                y_true=tf.one_hot(choices, depth=probabilities.shape[1]),
                sample_weight=sample_weight,
            ),
        }
        return batch_loss, probabilities

    def predict_probas(self, choice_dataset, batch_size=-1):
        stacked_probabilities = []
        for (
            shared_features_by_choice,
            items_features_by_choice,
            available_items_by_choice,
            choices,
        ) in choice_dataset.iter_batch(batch_size=batch_size):
            _, probabilities = self.batch_predict(
                shared_features_by_choice=shared_features_by_choice,
                items_features_by_choice=items_features_by_choice,
                available_items_by_choice=available_items_by_choice,
                choices=choices,
            )
            stacked_probabilities.append(probabilities)

        return tf.concat(stacked_probabilities, axis=0)

    def evaluate(self, choice_dataset, sample_weight=None, batch_size=-1, mode="eval"):
        batch_losses = []
        if sample_weight is not None:
            for (
                shared_features_by_choice,
                items_features_by_choice,
                available_items_by_choice,
                choices,
            ), batch_sample_weight in choice_dataset.iter_batch(
                batch_size=batch_size, sample_weight=sample_weight
            ):
                loss, _ = self.batch_predict(
                    shared_features_by_choice=shared_features_by_choice,
                    items_features_by_choice=items_features_by_choice,
                    available_items_by_choice=available_items_by_choice,
                    choices=choices,
                    sample_weight=batch_sample_weight,
                )
                if mode == "eval":
                    batch_losses.append(loss["Exact-NegativeLogLikelihood"])
                elif mode == "optim":
                    batch_losses.append(loss["optimized_loss"])
        else:
            for (
                shared_features_by_choice,
                items_features_by_choice,
                available_items_by_choice,
                choices,
            ) in choice_dataset.iter_batch(batch_size=batch_size):
                loss, _ = self.batch_predict(
                    shared_features_by_choice=shared_features_by_choice,
                    items_features_by_choice=items_features_by_choice,
                    available_items_by_choice=available_items_by_choice,
                    choices=choices,
                )
                if mode == "eval":
                    batch_losses.append(loss["Exact-NegativeLogLikelihood"])
                elif mode == "optim":
                    batch_losses.append(loss["optimized_loss"])
        if batch_size != -1:
            last_batch_size = available_items_by_choice.shape[0]
            coefficients = tf.concat(
                [tf.ones(len(batch_losses) - 1) * batch_size, [last_batch_size]], axis=0
            )
            batch_losses = tf.multiply(batch_losses, coefficients)
            batch_loss = tf.reduce_sum(batch_losses) / len(choice_dataset)
        else:
            batch_loss = tf.reduce_mean(batch_losses)
        return batch_loss

    def _lbfgs_train_step(self, choice_dataset, sample_weight=None):
        # obtain the shapes of all trainable parameters in the model
        shapes = tf.shape_n(self.trainable_weights)
        n_tensors = len(shapes)

        # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
        # prepare required information first
        count = 0
        idx = []  # stitch indices
        part = []  # partition indices

        for i, shape in enumerate(shapes):
            n = np.prod(shape)
            idx.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape))
            part.extend([i] * n)
            count += n

        part = tf.constant(part)

        @tf.function
        def assign_new_model_parameters(params_1d):
            params = tf.dynamic_partition(params_1d, part, n_tensors)
            for i, (shape, param) in enumerate(zip(shapes, params)):
                self.trainable_weights[i].assign(tf.reshape(param, shape))

        # now create a function that will be returned by this factory
        @tf.function
        def f(params_1d):
            # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
            with tf.GradientTape() as tape:
                # update the parameters in the model
                assign_new_model_parameters(params_1d)
                # calculate the loss
                loss_value = self.evaluate(
                    choice_dataset,
                    sample_weight=sample_weight,
                    batch_size=-1,
                    mode="eval",
                )
                if self.regularization is not None:
                    regularization = tf.reduce_sum(
                        [self.regularizer(w) for w in self.trainable_weights]
                    )
                    loss_value += regularization

            # calculate gradients and convert to 1D tf.Tensor
            grads = tape.gradient(loss_value, self.trainable_weights)

            # Replace None gradients with zeros to avoid dynamic_stitch errors
            # This happens when some weights don't contribute to the loss (e.g., nest-specific
            # coefficients for closure types that don't exist in certain nests)
            grads_filtered = []
            for i, grad in enumerate(grads):
                if grad is None:
                    # Create a zero tensor with the same shape as the corresponding weight
                    grads_filtered.append(tf.zeros_like(self.trainable_weights[i]))
                else:
                    grads_filtered.append(grad)

            grads = tf.dynamic_stitch(idx, grads_filtered)
            f.iter.assign_add(1)

            # store loss value so we can retrieve later
            tf.py_function(f.history.append, inp=[loss_value], Tout=[])

            return loss_value, grads

        # store these information as members so we can use them outside the scope
        f.iter = tf.Variable(0)
        f.idx = idx
        f.part = part
        f.shapes = shapes
        f.assign_new_model_parameters = assign_new_model_parameters
        f.history = []
        return f

    def _fit_with_lbfgs(self, choice_dataset, sample_weight=None, verbose=0):
        epochs = self.epochs
        func = self._lbfgs_train_step(
            choice_dataset=choice_dataset, sample_weight=sample_weight
        )

        # convert initial model parameters to a 1D tf.Tensor
        init_params = tf.dynamic_stitch(func.idx, self.trainable_weights)
        # train the model with L-BFGS solver
        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=func,
            initial_position=init_params,
            max_iterations=epochs,
            tolerance=self.lbfgs_tolerance,
            f_absolute_tolerance=-1,
            f_relative_tolerance=-1,
            parallel_iterations=self.lbfgs_parallel_iterations,
        )

        func.assign_new_model_parameters(results.position)
        if results[1].numpy():
            print("Error: L-BFGS Optimization failed.")
        if verbose > 0:
            print("L-BFGS Opimization finished:")
            print("---------------------------------------------------------------")
            print(f"Number of iterations: {results[2].numpy()}")
            print(
                f"Converged before reaching max iterations: {results[0].numpy()}",
            )
        return {"train_loss": func.history}
