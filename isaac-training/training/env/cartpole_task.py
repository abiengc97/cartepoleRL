# training/envs/cartpole_task.py

# Use upstream Cartpole logic but keep the class under our repo.
from omniisaacgymenvs.tasks.cartpole import CartpoleTask as _UpstreamCartpoleTask

class CartpoleTask(_UpstreamCartpoleTask):
    # Example of where you'd tweak behavior later by overriding methods.
    pass
