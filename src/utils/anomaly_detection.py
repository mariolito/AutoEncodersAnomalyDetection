from sklearn.metrics import mean_squared_error
import tensorflow as tf
from src.utils.tf_layer import set_input, add_layer, create_layer_config
import math
import numpy as np
from tqdm import trange
import warnings
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s  - %(message)s', level=logging.INFO)
warnings.filterwarnings('ignore')


class AnomalyDetector1D(object):

    def __init__(self, config):
        self.num_epochs = config['num_epochs']
        self.verbose = config['verbose']
        self.learning_rate = config['learning_rate']
        self.beta1 = config['beta1']
        self.mini_batch_size = config['mini_batch_size']
        self.layers = config['layers']
        self.config = config
        self.save_model = True

    def make_model(self):
        prev_name = ""
        model = tf.keras.Sequential()
        model = set_input(model, {'input_shape': self.input_shape})
        for layer_num_nodes in self.layers:
            config_layer = {"name": "dense", "units": layer_num_nodes}
            config_layer = create_layer_config(config_general=self.config, config_layer=config_layer)
            model = add_layer(model, config_layer, prev_name)
            prev_name = config_layer['name']
        config_layer.update({
            "name": "dense",
            "batch_normalization": False,
            "dropout": False,
            "units": self.k
        })
        model = add_layer(model, config_layer, prev_name)
        return model

    def random_mini_batches(self, X):
        mini_batches = []
        permutation = list(np.random.permutation(self.m))
        shuffled_X = X[permutation, :]
        num_complete_minibatches = math.floor(self.m / self.mini_batch_size)
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[
                           k * self.mini_batch_size: k * self.mini_batch_size +
                                                     self.mini_batch_size, :]
            mini_batch = mini_batch_X
            mini_batches.append(mini_batch)
        # Handling the end case (last mini-batch < mini_batch_size)
        if self.m % self.mini_batch_size != 0:
            mini_batch_X = shuffled_X[
                           num_complete_minibatches * self.mini_batch_size:
                           self.m, :]
            mini_batch = mini_batch_X
            mini_batches.append(mini_batch)
        return mini_batches

    def train(self, X_train, X_test_Legit=None, X_test_Fraud=None, validation=False, name=None, dir_store=None):

        result = {}

        (self.m, self.k) = X_train.shape

        self.input_shape = X_train.shape[1:]


        model = self.make_model()

        loss_function = tf.keras.losses.MeanSquaredError()

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta1)

        loss_l = []

        t = trange(self.num_epochs, desc="Epoch back-propagation", leave=True)
        for epoch in t:
            minibatches = self.random_mini_batches(X_train)
            for minibatch in minibatches:
                minibatch_X = minibatch
                with tf.GradientTape() as tape:

                    activations = model(minibatch_X)

                    loss = loss_function(minibatch_X, activations)

                grads = tape.gradient(loss, model.trainable_variables)

                optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))

                msg = "Loss: {}".format(str(round(loss.numpy(), 4)))
                t.set_description(msg)
                t.refresh()
            if epoch % 10 == 0:
                loss_l.append(loss.numpy())
        result.update({
            "model": model,
            "loss": loss_l
        })
        if validation:
            pred_test_Fraud = model(X_test_Fraud).numpy()
            test_mse_Fraud = mean_squared_error(pred_test_Fraud, X_test_Fraud)

            pred_test_Legit = model(X_test_Legit).numpy()
            test_mse_Legit = mean_squared_error(pred_test_Legit, X_test_Legit)
            result.update({
                "pred_test_Fraud": pred_test_Fraud,
                "test_mse_Fraud": test_mse_Fraud,
                "pred_test_Legit": pred_test_Legit,
                "test_mse_Legit": test_mse_Legit
            })
            if self.verbose == 2:
                logging.info(
                    'Train completed.'
                    + ' test_mse_Legit: '
                    + str(round(test_mse_Legit, 4))
                    + ' test_mse_Fraud: '
                    + str(round(test_mse_Fraud, 4))
                )
        return result
