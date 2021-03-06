import jax
import jax.numpy as jnp
import numpy as np

from absl import app
from absl import flags
from absl import logging

from tqdm import trange

import matplotlib.pyplot as plt

import time

import training_utils

"""
Simple app for implementing a vanilla linear regression model with JAX.

Config your flags in flags.cfg.
"""

flags.DEFINE_float('lr', 0.01, help='learning rate')
flags.DEFINE_float('loc_data_noise', 0.0, help='the mean of the guassian noise added to data')
flags.DEFINE_float('std_data_noise', 0.01, help='standard deviation of the guassian noise added to data')
flags.DEFINE_float('w', 3.0, help='weight coeff')
flags.DEFINE_float('b', 10.0, help='bias')
flags.DEFINE_float('init_w', 2.2, help='initial weight coeff')
flags.DEFINE_float('init_b', 3.7, help='initial bias')
flags.DEFINE_integer('num_train_data', 100, help='number of training data points')
flags.DEFINE_integer('epochs', 60, help='number of epochs')
flags.DEFINE_bool('use_jit', False, help='Use just in time compilation or not')

# flag validators
def lr_checker(number):
    return number > 0
def std_data_noise_checker(number):
    return number > 0
def epoch_checker(number):
    return number > 0
def num_train_data_checker(number):
    return number > 0
       

flags.register_validator('lr', checker=lr_checker, message='lr must be a positive float.')
flags.register_validator('std_data_noise', checker=std_data_noise_checker, message='std must be a positive float.')
flags.register_validator('num_train_data', checker=num_train_data_checker, message='number of traininng data must be more than zero')
flags.register_validator('epochs', checker=epoch_checker, message='epochs is a positive integer')

def main(argv):
    del argv

    learing_rate = flags.FLAGS.lr
    loc = flags.FLAGS.loc_data_noise
    std = flags.FLAGS.std_data_noise
    w = flags.FLAGS.w
    b = flags.FLAGS.b
    w0 = flags.FLAGS.init_w
    b0 = flags.FLAGS.init_b
    num_data = flags.FLAGS.num_train_data
    epochs = flags.FLAGS.epochs
    use_jit = flags.FLAGS.use_jit
    # function for generating training data
    def func(input):
        """
        Description:

        This is the real model, and we use it to
        generate training data.

        input -- an scalar
        """
        return w*input + b

    # function for generating training data (vectorized version)
    def batched_func(batched_input):
        """
        Description:

        This is the real model, and we use it to
        generate training data (Vectorized version)

        batched_input -- a batch of scalars
        """
        return jax.vmap(func, in_axes=(0))(batched_input)

    # Generating training data
    x_train = np.linspace(-10, 10, num_data).reshape(num_data, 1)
    noise = np.random.normal(loc, std, size=(num_data, 1))
    y_train = batched_func(x_train) + noise

    # initializing params

    params = training_utils.Params(w0, b0)
    param_history = []
    param_history.append(params)

    if use_jit:
        train_one_epoch_func = jax.jit(training_utils.train_one_epoch, static_argnums=(0, 3))
    else:
        train_one_epoch_func = training_utils.train_one_epoch

    # #registering out param container as pytree
    jax.tree_util.register_pytree_node(training_utils.Params,
     training_utils.flatten_MyContainer, training_utils.unflatten_MyContainer)



    # training loop
    t_start = time.time()
    for _ in trange(0, epochs):

        params = train_one_epoch_func(training_utils.batched_predict, params,
         (x_train, y_train), training_utils.loss_fn, learing_rate)
        param_history.append(params)
    t_end = time.time()

    elapsed_time = t_end - t_start

    # writing results and plotting model
    log_file_object = open("log.txt","w")

    for item in param_history:
        log_file_object.write('w = ' + str(item.w) + ', b = ' + str(item.b) + '\n')
    log_file_object.write('training time (seconds): ' + str(elapsed_time))

    log_file_object.close()

    fig, ax = plt.subplots()
    ax.scatter(x_train, y_train)
    ax.plot(x_train, training_utils.batched_predict(param_history[-1], x_train), color='brown')
    ax.set_title('training data & trained model')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    ax.set_facecolor('lightgray')

    fig.savefig('trained_model.png')
    print('training time (seconds): ' + str(elapsed_time))


# Entry point of exec
if __name__ == '__main__':
    app.run(main)