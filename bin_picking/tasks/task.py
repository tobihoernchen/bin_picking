import mujoco
import numpy as np


class Task:
    def __init__(self):
        self.sim_model, self.sim_data = None, None

    def register(self, sim: tuple["mujoco.MjModel", "mujoco.MjData"]):
        self.sim_model, self.sim_data = sim

    def task_body_xml(self):
        raise NotImplementedError

    def task_actuators_xml(self):
        raise NotImplementedError

    def get_reward(self):
        raise NotImplementedError

    def add_to_spec(self, spec):
        return spec

    def position_of(self, body_name: str) -> np.ndarray:
        """Get the position of a body in the simulation.

        Args:
            body_name (str): Name of the body.

        Returns:
            np.ndarray: Position of the body as a 3D vector.
        """
        return self.sim_data.body(body_name).xpos
