import jax
import jax.numpy as jnp
import numpy as np

from absl import app
from absl import flags
from absl import logging

from stateful_classes import BadMatMultiplier, GoodMatMultiplier, State, flatten_MyContainer, unflatten_MyContainer

flags.DEFINE_list('initial_vector', [1.0, 2.0, 3.0], help='initial vector to start')
flags.DEFINE_integer('num',  10, help='number of time multiplication is done')
flags.DEFINE_bool('bad_results', False, help='choose to see wrong stateful approach, or correct functional approach results')


def main(argv):

    del argv

    vec = flags.FLAGS.initial_vector
    num = flags.FLAGS.num
    bad_results = flags.FLAGS.bad_results
    

    vec = jnp.array([float(x) for x in vec])
    mat = jnp.stack([vec, 2*vec, 3*vec])

    # #registering out param container as pytree
    jax.tree_util.register_pytree_node(State, flatten_MyContainer, unflatten_MyContainer)

    initial_state = State(mat, 0)

    if bad_results:
        stateful_class = BadMatMultiplier(initial_state)
    else:
        stateful_class = GoodMatMultiplier()

    # a valid function for using with jit, must have valid pytree inputs, so we registered State class as pytree
    jit_func = jax.jit(stateful_class.mat_mul)

    for i in range(0, num):
        if bad_results:
            jit_func(stateful_class.state.mat)
        else:
            initial_state = jit_func(initial_state, mat)

    print(jax.tree_util.tree_leaves(initial_state))

    


# Entry point of exec
if __name__ == '__main__':
    app.run(main)

