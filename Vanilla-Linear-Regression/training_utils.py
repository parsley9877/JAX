import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Iterable


class Params():
    """
    Contaoner for model params
    """
    def __init__(self, w: float, b: float):
        self.w = w
        self.b = b

def flatten_MyContainer(container: Params) -> Tuple[Iterable[int], str]:
    """Returns an iterable over container contents, and aux data.
    We use this func to register our param container as pytree node
    """
    flat_contents = [container.w, container.b]
    return flat_contents, ''

def unflatten_MyContainer(aux: str, flat_contents: Iterable[int]) -> Params:
    """Converts aux data and the flat contents into a MyContainer.
    We use this func to register our param container as pytree node
    """
    return Params(*flat_contents)

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
    w, b = params.w, params.b
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
    return jax.tree_multimap(lambda p, g: p - lr * g, params, grads)


def train_one_epoch(model, params, data, loss_fn, lr):
    """
    Description:

    One loop of training (Full batch).
    """
    grad = jax.grad(loss_fn, argnums=(0))
    return update(params, grad(params, model, data), lr)