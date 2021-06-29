import jax.numpy as jnp
from typing import Tuple, Iterable

class State():

    """
    A building block for showing the state of a class
    """

    def __init__(self, mat, num) -> None:
        self.num = num
        self.mat =mat
    def update_mat(self, new_mat) -> None:
        self.mat = new_mat
    def update_num(self, new_num) -> None:
        self.num = new_num

def flatten_MyContainer(container: State) -> Tuple[Iterable[int], str]:
    """Returns an iterable over container contents, and aux data.
    We use this func to register our param container as pytree node
    """
    flat_contents = [container.mat, container.num]
    return flat_contents, ''

def unflatten_MyContainer(aux: str, flat_contents: Iterable[int]) -> State:
    """Converts aux data and the flat contents into a MyContainer.
    We use this func to register our param container as pytree node
    """
    return State(*flat_contents)

class BadMatMultiplier():
    """
    Not compatiable with jax.jit()
    """

    def __init__(self, init_state) -> None:
        self.state = init_state

    # Main function to be used with jax.jit
    def mat_mul(self, mat):
        current_mat = self.state.mat @ mat
        current_num = self.state.num + 1
        self.state.update_mat(current_mat)
        self.state.update_num(current_num)
        # print('Current State: ', self.state.mat)
        # print('Current Count: ', self.state.num)

class GoodMatMultiplier():
    """
    Compatiable with jax.jit()

    The class is just a namespace in this situation!
    We delete states, and use states as inputs of the function, and get new states as outputs of the function,
    it is functional programming approach.
    """

    def __init__(self) -> None:
        pass
    
    # Main function to be used with jax.jit()
    def mat_mul(self, state, mat):

        current_mat = state.mat @ mat
        num_so_far = state.num + 1

        state.update_mat(current_mat)
        state.update_num(num_so_far)

        # print('Current State: ', self.state.mat)
        # print('Current Count: ', self.state.num)

        return state