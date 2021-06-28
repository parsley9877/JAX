import jax
import jax.numpy as jnp
import numpy as np


# function for generating training data
def func(input):
    """
    Description:

    This is the real model, and we use it to
    generate training data.

    input -- an scalar
    """
    return 3*input + 10

# function for generating training data (vectorized version)
def batched_func(batched_input):
    """
    Description:

    This is the real model, and we use it to
    generate training data (Vectorized version)

    batched_input -- a batch of scalars
    """
    return jax.vmap(func, in_axes=(0))(batched_input)

# model to be trained with params as trainable values
def predict(params, input):
    """
    Description:

    This is the model to be trained. Pay close
    attention to the signature of this function. We use functional approach
    in JAX, unlike other libraries.

    input -- an scalar
    params -- a pytree data structure consists of parameters to be trained
    """
    w, b = params
    return w*input + b

# model to be trained with params as trainable values (vectorized version)
def batched_predict(params, batched_input):
    """
    Description:

    This is the model to be trained (Vectorized version). Pay close
    attention to the signature of this function. We use functional approach
    in JAX, unlike other libraries.

    batched_input -- a batch of scalars
    params -- a pytree data structure consists of parameters to be trained
    """
    return jax.vmap(predict, in_axes=(None, 0))(params, batched_input)

# MSE loss function
def loss_fn(params, model, data):
    """
    Description:

    This is MSE loss function, again pay close attention to
    function signature as this is the function which is going to be differentiated, so
    params must be in its inputs. we do not need to vectorize this function as it is written
    with batching considerations.

    params -- pytree of trainable parameters
    model --  model to be trained
    data -- a tuple of training data --> (x_train, y_train)
    """

    x, y = data
    return jnp.mean((model(params, x) - y) ** 2)

# update rule
def update(params, grads, lr):
    """
    Description:

    This is a funtion that updates current params, and input new params.
    (SGD is used here for updating)

    params -- pytree of trainable parameters
    grads -- grads of params (it is a pytree with exactly the same structure of params)
    lr -- learning rate
    """
    return params - lr * grads


def train_one_epoch(model, params, data, loss_fn, lr):
    """
    Description:

    One loop of training (Full batch).
    """
    grad = jax.grad(loss_fn, argnums=(0))
    return update(params, grad(params, model, data), lr)